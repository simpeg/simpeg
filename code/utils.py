from numpy import *
import numpy as np


def diff(A, d):
    if(d == 1):
        return A[1:, :, :] - A[:-1, :, :]
    elif(d == 2):
        return A[:, 1:, :] - A[:, :-1, :]
    else:
        return A[:, :, 1:] - A[:, :, :-1]
    #else:
    #    print('d must be 1,2 or 3')


def diffp(A, d1, d2):
    if(d1 == 1 and d2 == 2):
        return A[1:, 1:, :] - A[:-1, :-1, :]
    elif(d1 == 1 and d2 == 3):
        return A[1:, :, 1:] - A[:-1, :, :-1]
    else:
        return A[:, 1:, 1:] - A[:, :-1, :-1]


def diffm(A, d1, d2):
    if(d1 == 3 and d2 == 2):
        return A[:, :-1, 1:] - A[:, 1:, :-1]
    elif(d1 == 1 and d2 == 3):
        return A[1:, :, :-1] - A[:-1, :, 1:]
    elif(d1 == 2 and d2 == 1):
        return A[:-1, 1:, :] - A[1:, :-1, :]
    else:
        print('d must be 1, 2 or 3')


def ave(A, d):
    if(d == 1):
        return 0.5*(A[1:, :, :] + A[:-1, :, :])
    elif(d == 2):
        return 0.5*(A[:, 1:, :] + A[:, :-1, :])
    elif(d == 3):
        return 0.5*(A[:, :, 1:] + A[:, :, :-1])
    else:
        print('d must be 1,2 or 3')


def reshapeF(x, size):
    return np.reshape(x, size, order='F')


def mkvc(x, numDims=1):
    """Creates a vector with the number of dimension specified

    e.g.:

        a = np.array(1,2,3)

        mkvc(a, 1).shape
            > (3, )

        mkvc(a, 2).shape
            > (3, 1)

        mkvc(a, 3).shape
            > (3, 1, 1)

    """
    assert type(x) == np.ndarray, "Vector must be a numpy array"

    if numDims == 1:
        return x.flatten(order='F')
    elif numDims == 2:
        return x.flatten(order='F')[:, np.newaxis]
    elif numDims == 3:
        return x.flatten(order='F')[:, np.newaxis, np.newaxis]


def ndgrid(xin):
    """Form tensorial grid for 1, 2 and 3 dimensions. Return X1,X2,X3 arrays depending on the dimension"""

    if len(xin) == 1:
        return xin
    elif len(xin) == 2:
        X2, X1 = [mkvc(x) for x in np.broadcast_arrays(mkvc(xin[1], 1), mkvc(xin[0], 2))]
        return np.c_[X1, X2]
    elif len(xin) == 3:
        X3, X2, X1 = [mkvc(x) for x in np.broadcast_arrays(mkvc(xin[2], 1), mkvc(xin[1], 2), mkvc(xin[0], 3))]
        return np.c_[X1, X2, X3]


def flattenF(x):
    return np.flatten(x, order='F')


def printF(x):
    pass


def ind2sub(shape, ind):
    # From the given shape, returns the subscrips of the given index
    revshp = []
    revshp.extend(shape)
    mult = [1]
    for i in range(0, len(revshp)-1):
        mult.extend([mult[i]*revshp[i]])
    mult = array(mult).reshape(len(mult))

    sub = []

    for i in range(0, len(shape)):
        sub.extend([math.floor(ind / mult[i])])
        ind = ind - (math.floor(ind/mult[i]) * mult[i])
    return sub


def sub2ind(shape, subs):
    # From the given shape, returns the index of the given subscript
    revshp = list(shape)
    mult = [1]
    for i in range(0, len(revshp)-1):
        mult.extend([mult[i]*revshp[i]])
    mult = array(mult).reshape(len(mult), 1)

    idx = dot((subs), (mult))
    return idx


def mkmat(x):
    return reshape(matrix(x), (size(x), 1), 'F')


def hstack3(a, b, c):
    a = mkvc(a)
    b = mkvc(b)
    c = mkvc(c)
    a = mkmat(a)
    b = mkmat(b)
    c = mkmat(c)
    return hstack((hstack((a, b)), c))


if __name__ == '__main__':

    X, Y, Z = mgrid[0:4, 0:5, 0:6]

    print Z

    t = ave(X, 1)
    print t
