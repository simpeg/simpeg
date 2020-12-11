from __future__ import division
import numpy as np
from .code_utils import deprecate_method

from discretize.utils import (
    Zero,
    Identity,
    mkvc,
    sdiag,
    sdInv,
    speye,
    kron3,
    spzeros,
    ddx,
    av,
    av_extrap,
    ndgrid,
    ind2sub,
    sub2ind,
    getSubArray,
    inv3X3BlockDiagonal,
    inv2X2BlockDiagonal,
    TensorType,
    makePropertyTensor,
    invPropertyTensor,
)

avExtrap = deprecate_method(av_extrap, "avExtrap", removal_version="0.15.0")


def diagEst(matFun, n, k=None, approach="Probing"):
    """
        Estimate the diagonal of a matrix, A. Note that the matrix may be a
        function which returns A times a vector.

        Three different approaches have been implemented:

        1. Probing: cyclic permutations of vectors with 1's and 0's (default)
        2. Ones: random +/- 1 entries
        3. Random: random vectors

        :param callable matFun: takes a (numpy.ndarray) and multiplies it by a matrix to estimate the diagonal
        :param int n: size of the vector that should be used to compute matFun(v)
        :param int k: number of vectors to be used to estimate the diagonal
        :param str approach: approach to be used for getting vectors
        :rtype: numpy.ndarray
        :return: est_diag(A)

        Based on Saad http://www-users.cs.umn.edu/~saad/PDF/umsi-2005-082.pdf,
        and https://www.cita.utoronto.ca/~niels/diagonal.pdf
    """

    if type(matFun).__name__ == "ndarray":
        A = matFun

        def matFun(v):
            return A.dot(v)

    if k is None:
        k = np.floor(n / 10.0)

    if approach.upper() == "ONES":

        def getv(n, i=None):
            v = np.random.randn(n)
            v[v < 0] = -1.0
            v[v >= 0] = 1.0
            return v

    elif approach.upper() == "RANDOM":

        def getv(n, i=None):
            return np.random.randn(n)

    else:  # if approach == 'Probing':

        def getv(n, i):
            v = np.zeros(n)
            v[i:n:k] = 1.0
            return v

    Mv = np.zeros(n)
    vv = np.zeros(n)

    for i in range(0, k):
        vk = getv(n, i)
        Mv += matFun(vk) * vk
        vv += vk * vk

    d = Mv / vv

    return d


def uniqueRows(M):
    b = np.ascontiguousarray(M).view(np.dtype((np.void, M.dtype.itemsize * M.shape[1])))
    _, unqInd = np.unique(b, return_index=True)
    _, invInd = np.unique(b, return_inverse=True)
    unqM = M[unqInd]
    return unqM, unqInd, invInd


def eigenvalue_by_power_iteration(combo_objfct, model, n_pw_iter=4, fields_list=None, seed=None):
    """
    Estimate the highest eigenvalue of any objective function term or combination thereof
    (data_misfit, regularization or ComboObjectiveFunction) for a given model.
    The highest eigenvalue is estimated by power iterations and Rayleigh quotient.

    Parameters
    ----------

    :param SimPEG.BaseObjectiveFunction combo_objfct: objective function term of which to estimate the highest eigenvalue
    :param numpy.ndarray model: current geophysical model to estimate the objective function derivatives at
    :param int n_pw_iter: number of power iterations to estimate the highest eigenvalue
    :param list fiels_list: (optional) list of fields for each data misfit term in combo_objfct. If none given,
                            they will be evaluated within the function. If combo_objfct mixs data misfit and regularization
                            terms, the list should contains SimPEG.fields for the data misfit terms and None for the
                            regularization term.
    :param int seed: Random seed for the initial random guess of eigenvector.

    Return
    ------

    :return float eigenvalue: estimated value of the highest eigenvalye

    """

    if seed is not None:
        np.random.seed(seed)

    # Initial guess for eigen-vector
    x0 = np.random.rand(*model.shape)
    x0 = x0 / np.linalg.norm(x0)

    # transform to ComboObjectiveFunction if required
    if getattr(combo_objfct, "objfcts", None) is None:
        combo_objfct = 1. * combo_objfct

    # create Field for data misfit if necessary and not provided
    if fields_list is None:
        fields_list = []
        for k, obj in enumerate(combo_objfct.objfcts):
            if hasattr(obj, "simulation"):
                fields_list += [obj.simulation.fields(model)]
            else:
                # required to put None to conserve it in the list
                # The idea is that the function can have a mixed of dmis and reg terms
                # (see test)
                fields_list += [None]
    elif not isinstance(fields_list, (list, tuple, np.ndarray)):
            fields_list = [fields_list]

    #Power iteration: estimate eigenvector
    for i in range(n_pw_iter):
        x1 = 0.
        for j, (mult, obj) in enumerate(zip(combo_objfct.multipliers, combo_objfct.objfcts)):
            if hasattr(obj, "simulation"): # if data misfit term
                aux = obj.deriv2(model, v=x0, f=fields_list[j])
                if not isinstance(aux, Zero):
                    x1 += mult * aux
            else:
                aux = obj.deriv2(model, v=x0)
                if not isinstance(aux, Zero):
                    x1 += mult * aux
        x0 = x1 / np.linalg.norm(x1)

    # Compute highest eigenvalue from estimated eigenvector
    eigenvalue=0.
    for j, (mult, obj) in enumerate(zip(combo_objfct.multipliers, combo_objfct.objfcts)):
        if hasattr(obj, "simulation"): # if data misfit term
            eigenvalue += mult * x0.dot(obj.deriv2(model, v=x0, f=fields_list[j]))
        else:
            eigenvalue += mult * x0.dot(obj.deriv2(model, v=x0,))

    return eigenvalue


def cartesian2spherical(m):
    """ Convert from cartesian to spherical """

    # nC = int(len(m)/3)

    x = m[:, 0]
    y = m[:, 1]
    z = m[:, 2]

    a = (x ** 2.0 + y ** 2.0 + z ** 2.0) ** 0.5

    t = np.zeros_like(x)
    t[a > 0] = np.arcsin(z[a > 0] / a[a > 0])

    p = np.zeros_like(x)
    p[a > 0] = np.arctan2(y[a > 0], x[a > 0])

    m_atp = np.r_[a, t, p]

    return m_atp


def spherical2cartesian(m):
    """ Convert from spherical to cartesian """

    a = m[:, 0] + 1e-8
    t = m[:, 1]
    p = m[:, 2]

    m_xyz = np.r_[a * np.cos(t) * np.cos(p), a * np.cos(t) * np.sin(p), a * np.sin(t)]

    return m_xyz


def dip_azimuth2cartesian(dip, azm_N):
    """
    dip_azimuth2cartesian(dip,azm_N)

    Function converting degree angles for dip and azimuth from north to a
    3-components in cartesian coordinates.

    INPUT
    dip     : Value or vector of dip from horizontal in DEGREE
    azm_N   : Value or vector of azimuth from north in DEGREE

    OUTPUT
    M       : [n-by-3] Array of xyz components of a unit vector in cartesian

    Created on Dec, 20th 2015

    @author: dominiquef
    """

    azm_N = np.asarray(azm_N)
    dip = np.asarray(dip)

    # Number of elements
    nC = azm_N.size

    M = np.zeros((nC, 3))

    # Modify azimuth from North to cartesian-X
    azm_X = (450.0 - np.asarray(azm_N)) % 360.0
    inc = -np.deg2rad(np.asarray(dip))
    dec = np.deg2rad(azm_X)

    M[:, 0] = np.cos(inc) * np.cos(dec)
    M[:, 1] = np.cos(inc) * np.sin(dec)
    M[:, 2] = np.sin(inc)

    return M


def coterminal(theta):
    """
    Compute coterminal angle so that [-pi < theta < pi]
    """

    sub = theta[np.abs(theta) >= np.pi]
    sub = -np.sign(sub) * (2 * np.pi - np.abs(sub))

    theta[np.abs(theta) >= np.pi] = sub

    return theta
