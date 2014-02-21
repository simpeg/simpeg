import numpy as np


def mkvc(x, numDims=1):
    """Creates a vector with the number of dimension specified

    e.g.::

        a = np.array([1, 2, 3])

        mkvc(a, 1).shape
            > (3, )

        mkvc(a, 2).shape
            > (3, 1)

        mkvc(a, 3).shape
            > (3, 1, 1)

    """
    if type(x) == np.matrix:
        x = np.array(x)

    assert type(x) == np.ndarray, "Vector must be a numpy array"

    if numDims == 1:
        return x.flatten(order='F')
    elif numDims == 2:
        return x.flatten(order='F')[:, np.newaxis]
    elif numDims == 3:
        return x.flatten(order='F')[:, np.newaxis, np.newaxis]


def ndgrid(*args, **kwargs):
    """
    Form tensorial grid for 1, 2, or 3 dimensions.

    Returns as column vectors by default.

    To return as matrix input:

        ndgrid(..., vector=False)

    The inputs can be a list or separate arguments.

    e.g.::

        a = np.array([1, 2, 3])
        b = np.array([1, 2])

        XY = ndgrid(a, b)
            > [[1 1]
               [2 1]
               [3 1]
               [1 2]
               [2 2]
               [3 2]]

        X, Y = ndgrid(a, b, vector=False)
            > X = [[1 1]
                   [2 2]
                   [3 3]]
            > Y = [[1 2]
                   [1 2]
                   [1 2]]

    """

    # Read the keyword arguments, and only accept a vector=True/False
    vector = kwargs.pop('vector', True)
    assert type(vector) == bool, "'vector' keyword must be a bool"
    assert len(kwargs) == 0, "Only 'vector' keyword accepted"

    # you can either pass a list [x1, x2, x3] or each seperately
    if type(args[0]) == list:
        xin = args[0]
    else:
        xin = args

    # Each vector needs to be a numpy array
    assert np.all([type(x) == np.ndarray for x in xin]), "All vectors must be numpy arrays."

    if len(xin) == 1:
        return xin[0]
    elif len(xin) == 2:
        XY = np.broadcast_arrays(mkvc(xin[1], 1), mkvc(xin[0], 2))
        if vector:
            X2, X1 = [mkvc(x) for x in XY]
            return np.c_[X1, X2]
        else:
            return XY[1], XY[0]
    elif len(xin) == 3:
        XYZ = np.broadcast_arrays(mkvc(xin[2], 1), mkvc(xin[1], 2), mkvc(xin[0], 3))
        if vector:
            X3, X2, X1 = [mkvc(x) for x in XYZ]
            return np.c_[X1, X2, X3]
        else:
            return XYZ[2], XYZ[1], XYZ[0]

def ind2sub(shape, inds):
    """From the given shape, returns the subscripts of the given index"""
    if type(inds) is not np.ndarray:
        inds = np.array(inds)
    assert len(inds.shape) == 1, 'Indexing must be done as a 1D row vector, e.g. [3,6,6,...]'
    return np.unravel_index(inds, shape, order='F')

def sub2ind(shape, subs):
    """From the given shape, returns the index of the given subscript"""
    if type(subs) is not np.ndarray:
        subs = np.array(subs)
    if subs.size == len(shape):
        subs = subs[np.newaxis,:]
    assert subs.shape[1] == len(shape), 'Indexing must be done as a column vectors. e.g. [[3,6],[6,2],...]'
    inds = np.ravel_multi_index(subs.T, shape, order='F')
    return mkvc(inds)

def getSubArray(A, ind):
    """subArray"""
    assert type(ind) == list, "ind must be a list of vectors"
    assert len(A.shape) == len(ind), "ind must have the same length as the dimension of A"

    if len(A.shape) == 2:
        return A[ind[0], :][:, ind[1]]
    elif len(A.shape) == 3:
        return A[ind[0], :, :][:, ind[1], :][:, :, ind[2]]
    else:
        raise Exception("getSubArray does not support dimension asked.")
