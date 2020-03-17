from __future__ import division
import numpy as np
from scipy.spatial import ConvexHull
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

    azm_N = np.asarray(azm_N)
    dip = np.asarray(dip)

    # Number of elements
    nC = azm_N.size

    M = np.zeros((nC, 3))

    # Modify azimuth from North to cartesian-X
    azm_X = (450. - np.asarray(azm_N)) % 360.
    inc = -np.deg2rad(np.asarray(dip))
    dec = np.deg2rad(azm_X)

    M[:, 0] = np.cos(inc) * np.cos(dec)
    M[:, 1] = np.cos(inc) * np.sin(dec)
    M[:, 2] = np.sin(inc)

    return M


def calculate_2D_trend(points, values, order=0, method='all'):
    """
    detrend2D(points, values, order=0, method='all')

    Function to remove a trend from 2D scatter points with values

    Parameters:
    ----------

    points: ndarray or floats, shape(npoints, 2)
        Coordinates of input points

    values: ndarray of floats, shape(npoints,)
        Values to be detrended

    order: int
        Order of the polynomial to be used

    method: str
        Method to be used for the detrending
            "all": USe all points
            "corners": Only use points on the convex hull


    Returns
    -------

    trend: ndarray of floats, shape(npoints,)
        Calculated trend

    coefficients: ndarray of floats, shape(order+1)
        Coefficients for the polynomial describing the trend
        trend = c[0] + points[:, 0] * c[1] +  points[:, 1] * c[2]

    """

    assert method in ['all', 'corners'], (
        "method must be 'all', or 'corners'"
    )

    assert order in [0, 1, 2], "order must be 0, 1, or 2"

    if method == "corners":
        hull = ConvexHull(points[:, :2])
        # Extract only those points that make the ConvexHull
        pts = np.c_[points[hull.vertices, :2], values[hull.vertices]]
    else:
        # Extract all points
        pts = np.c_[points[:, :2], values]

    if order == 0:
        data_trend = np.mean(pts[:, 2]) * np.ones(points[:, 0].shape)
        print('Removed data mean: {0:.6g}'.format(data_trend[0]))
        C = np.r_[0, 0, data_trend]

    elif order == 1:
        # best-fit linear plane
        A = np.c_[pts[:, 0], pts[:, 1], np.ones(pts.shape[0])]
        C, _, _, _ = np.linalg.lstsq(A, pts[:, 2], rcond=None)    # coefficients

        # evaluate at all data locations
        data_trend = C[0]*points[:, 0] + C[1]*points[:, 1] + C[2]
        print('Removed linear trend with mean: {0:.6g}'.format(np.mean(data_trend)))

    elif order == 2:
        # best-fit quadratic curve
        A = np.c_[
            np.ones(pts.shape[0]), pts[:, :2],
            np.prod(pts[:, :2], axis=1),
            pts[:, :2]**2
        ]
        C, _, _, _ = np.linalg.lstsq(A, pts[:, 2], rcond=None)

        # evaluate at all data locations
        data_trend = np.dot(np.c_[
                np.ones(points[:, 0].shape),
                points[:, 0],
                points[:, 1],
                points[:, 0]*points[:, 1],
                points[:, 0]**2, points[:, 1]**2
                ], C).reshape(points[:, 0].shape)

        print('Removed polynomial trend with mean: {0:.6g}'.format(np.mean(data_trend)))
    return data_trend, C
