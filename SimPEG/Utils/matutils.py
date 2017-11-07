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

    nC = int(len(m)/3)

    a = m[:nC]
    t = m[nC:2*nC]
    p = m[2*nC:]

    m_xyz = np.r_[a*np.cos(t)*np.cos(p),
                  a*np.cos(t)*np.sin(p),
                  a*np.sin(t)]

    return m_xyz


def xyz2atp(m):
    """ Convert from cartesian to spherical """

    nC = int(len(m)/3)

    x = m[:nC]
    y = m[nC:2*nC]
    z = m[2*nC:]

    a = (x**2. + y**2. + z**2.)**0.5

    t = np.zeros(nC)
    t[a > 0] = np.arcsin(z[a > 0]/a[a > 0])

    p = np.zeros(nC)
    p[a > 0] = np.arctan2(y[a > 0], x[a > 0])

    m_atp = np.r_[a, t, p]

    return m_atp
