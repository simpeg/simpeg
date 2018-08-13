from __future__ import division
import numpy as np

from discretize.utils import (
    Zero, Identity, mkvc, sdiag, sdInv, speye, kron3, spzeros, ddx, av,
    av_extrap, ndgrid, ind2sub, sub2ind, getSubArray, inv3X3BlockDiagonal,
    inv2X2BlockDiagonal, TensorType, makePropertyTensor, invPropertyTensor,
)


def avExtrap(**kwargs):
    raise Exception("avExtrap has been depreciated. Use av_extrap instead.")


def diagEst(matFun, n, k=None, approach='Probing'):
    """
        Estimate the diagonal of a matrix, A. Note that the matrix may be a
        function which returns A times a vector.

        Three different approaches have been implemented:

        1. Probing: cyclic permutations of vectors with 1's and 0's (default)
        2. Ones: random +/- 1 entries
        3. Random: random vectors

        :param callable matFun: takes a (numpy.array) and multiplies it by a matrix to estimate the diagonal
        :param int n: size of the vector that should be used to compute matFun(v)
        :param int k: number of vectors to be used to estimate the diagonal
        :param str approach: approach to be used for getting vectors
        :rtype: numpy.array
        :return: est_diag(A)

        Based on Saad http://www-users.cs.umn.edu/~saad/PDF/umsi-2005-082.pdf,
        and http://www.cita.utoronto.ca/~niels/diagonal.pdf
    """

    if type(matFun).__name__ == 'ndarray':
        A = matFun

        def matFun(v):
            return A.dot(v)

    if k is None:
        k = np.floor(n/10.)

    if approach.upper() == 'ONES':
        def getv(n, i=None):
            v = np.random.randn(n)
            v[v < 0] = -1.
            v[v >= 0] = 1.
            return v

    elif approach.upper() == 'RANDOM':
        def getv(n, i=None):
            return np.random.randn(n)

    else:  # if approach == 'Probing':
        def getv(n, i):
            v = np.zeros(n)
            v[i:n:k] = 1.
            return v

    Mv = np.zeros(n)
    vv = np.zeros(n)

    for i in range(0, k):
        vk = getv(n, i)
        Mv += matFun(vk)*vk
        vv += vk*vk

    d = Mv/vv

    return d


def uniqueRows(M):
    b = np.ascontiguousarray(M).view(np.dtype(
        (np.void, M.dtype.itemsize * M.shape[1]))
    )
    _, unqInd = np.unique(b, return_index=True)
    _, invInd = np.unique(b, return_inverse=True)
    unqM = M[unqInd]
    return unqM, unqInd, invInd

def atp2xyz(m):
    """ Convert from spherical to cartesian """

    a = m[:, 0] + 1e-8
    t = m[:, 1]
    p = m[:, 2]

    m_xyz = np.r_[a*np.cos(t)*np.cos(p),
                  a*np.cos(t)*np.sin(p),
                  a*np.sin(t)]

    return m_xyz


def xyz2pst(m, param):
    """
    Rotates from cartesian to pst
    pst coordinates along the primary field H0

    INPUT:
        m : nC-by-3 array for [x,y,z] components
        param: List of parameters [A, I, D] as given by survey.SrcList.param
    """

    Rz = np.vstack((np.r_[np.cos(np.deg2rad(-param[2])),
                          -np.sin(np.deg2rad(-param[2])), 0],
                   np.r_[np.sin(np.deg2rad(-param[2])),
                         np.cos(np.deg2rad(-param[2])), 0],
                   np.r_[0, 0, 1]))

    Rx = np.vstack((np.r_[1, 0, 0],
                   np.r_[0, np.cos(np.deg2rad(-param[1])),
                         -np.sin(np.deg2rad(-param[1]))],
                   np.r_[0, np.sin(np.deg2rad(-param[1])),
                         np.cos(np.deg2rad(-param[1]))]))

    yvec = np.c_[0, 1, 0]
    pvec = np.dot(Rz, np.dot(Rx, yvec.T))

    xvec = np.c_[1, 0, 0]
    svec = np.dot(Rz, np.dot(Rx, xvec.T))

    zvec = np.c_[0, 0, 1]
    tvec = np.dot(Rz, np.dot(Rx, zvec.T))

    m_pst = np.r_[np.dot(pvec.T, m.T),
                  np.dot(svec.T, m.T),
                  np.dot(tvec.T, m.T)].T

    return m_pst


def pst2xyz(m, param):
    """
    Rotates from pst to cartesian
    pst coordinates along the primary field H0

    INPUT:
        m : nC-by-3 array for [x,y,z] components
        param: List of parameters [A, I, D] as given by survey.SrcList.param
    """

    nC = int(len(m)/3)

    Rz = np.vstack((np.r_[np.cos(np.deg2rad(-param[2])),
                          -np.sin(np.deg2rad(-param[2])), 0],
                   np.r_[np.sin(np.deg2rad(-param[2])),
                         np.cos(np.deg2rad(-param[2])), 0],
                   np.r_[0, 0, 1]))

    Rx = np.vstack((np.r_[1, 0, 0],
                   np.r_[0, np.cos(np.deg2rad(-param[1])),
                         -np.sin(np.deg2rad(-param[1]))],
                   np.r_[0, np.sin(np.deg2rad(-param[1])),
                         np.cos(np.deg2rad(-param[1]))]))

    yvec = np.c_[0, 1, 0]
    pvec = np.dot(Rz, np.dot(Rx, yvec.T))

    xvec = np.c_[1, 0, 0]
    svec = np.dot(Rz, np.dot(Rx, xvec.T))

    zvec = np.c_[0, 0, 1]
    tvec = np.dot(Rz, np.dot(Rx, zvec.T))

    pst_mat = np.c_[pvec, svec, tvec]

    m_xyz = np.dot(m, pst_mat.T)

    return m_xyz


def xyz2atp(m):
    """ Convert from cartesian to spherical """

    # nC = int(len(m)/3)

    x = m[:, 0]
    y = m[:, 1]
    z = m[:, 2]

    a = (x**2. + y**2. + z**2.)**0.5

    t = np.zeros_like(x)
    t[a > 0] = np.arcsin(z[a > 0]/a[a > 0])

    p = np.zeros_like(x)
    p[a > 0] = np.arctan2(y[a > 0], x[a > 0])

    m_atp = np.r_[a, t, p]

    return m_atp


def dipazm_2_xyz(dip, azm_N):
    """
    dipazm_2_xyz(dip,azm_N)

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

    if isinstance(azm_N, float):
        nC = 1

    else:
        nC = len(azm_N)

    M = np.zeros((nC, 3))

    # Modify azimuth from North to cartesian-X
    azm_X = (450. - np.asarray(azm_N)) % 360.

    dec = np.deg2rad(np.asarray(dip))
    inc = np.deg2rad(azm_X)

    M[:, 0] = np.cos(dec) * np.cos(inc)
    M[:, 1] = np.cos(dec) * np.sin(inc)
    M[:, 2] = np.sin(dec)

    return M
