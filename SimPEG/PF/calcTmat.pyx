#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import numpy as np
from libc.math cimport log, atan, pow, floor
from cython.parallel import prange


def calcTmat(double[:,:] rxLoc,
             double[:,:] Xn,
             double[:,:] Yn,
             double[:,:] Zn):

    cdef int nC = Xn.shape[0]
    cdef double NewtG = 6.67408e-11*1e+8  # Convertion from mGal (1e-5) and g/cc (1e-3)
    cdef double eps = 1e-8  # add a small value to the locations to avoid
    cdef double val
    cdef int nD = rxLoc.shape[0]
    cdef double[:,:] F = np.zeros((nD, nC))
    cdef double[:] r = np.zeros(nC)
    cdef double[:] dx = np.zeros(2)
    cdef double[:] dy = np.zeros(2)
    cdef double[:] dz = np.zeros(2)
    cdef int aa, bb, cc, ii, jj
    cdef double[:] arg
    cdef double progress = 0

    # arg = np.linspace(0,nD,10, dtype=int)
    for ii in prange(nD, nofil=True):
        for jj in range(nC):

            dz[0] = rxLoc[ii, 2] - Zn[jj, 0]
            dz[1] = rxLoc[ii, 2] - Zn[jj, 1]

            dy[0] = Yn[jj, 0] - rxLoc[ii, 1]
            dy[1] = Yn[jj, 1] - rxLoc[ii, 1]

            dx[0] = Xn[jj, 0] - rxLoc[ii, 0]
            dx[1] = Xn[jj, 1] - rxLoc[ii, 0]

            for aa in range(2):
                for bb in range(2):
                    for cc in range(2):


                        r[jj] = pow(
                                pow(dx[aa],2.0) +
                                pow(dy[bb],2.0) +
                                pow(dz[cc],2.0), 0.50)


                        F[ii,jj] -= NewtG * pow(-1,aa) * pow(-1,bb) * pow(-1,cc) * (
                            dx[aa] * log(dy[bb] + r[jj] + eps) +
                            dy[bb] * log(dx[aa] + r[jj] + eps) -
                            dz[cc] * atan(dx[aa] * dy[bb] /
                                                  (dz[cc] * r[jj] + eps)))

        # if np.any(ii == arg):
        # print("Completed " + str(ii) + " of "+ str(nD) + "data")

    return np.asarray(F)
