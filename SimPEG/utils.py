import numpy as np
from numpy import *


def mkvc(x, numDims=1):
    """Creates a vector with the number of dimension specified

    e.g.:

        a = np.array([1, 2, 3])

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


def ndgrid(*args, **kwargs):
    """
    Form tensorial grid for 1, 2, or 3 dimensions.

    Returns as column vectors by default.

    To return as matrix input:

        ndgrid(..., vector=False)

    The inputs can be a list or separate arguments.

    e.g.

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


def volTetra(xyz, A, B, C, D):
    """
    Returns the volume for tetrahedras volume specified by the indexes A to D.


    Input:
       xyz      - X,Y,Z vertex vector
       A,B,C,D  - vert index of the tetrahedra

    Output:
       V        - volume

    Algorithm:     http://en.wikipedia.org/wiki/Tetrahedron#Volume

       V = 1/3 A * h

       V = 1/6 | ( a - d ) o ( ( b - d ) X ( c - d ) ) |

    """

    AD = xyz[A, :] - xyz[D, :]
    BD = xyz[B, :] - xyz[D, :]
    CD = xyz[C, :] - xyz[D, :]

    V = (BD[:, 0]*CD[:, 1] - BD[:, 1]*CD[:, 0])*AD[:, 2] - (BD[:, 0]*CD[:, 2] - BD[:, 2]*CD[:, 0])*AD[:, 1] + (BD[:, 1]*CD[:, 2] - BD[:, 2]*CD[:, 1])*AD[:, 0]
    return V/6


def indexCube(nodes, nN):
    """
    Returns the index of nodes on the mesh.


    Input:
       nodes  - string of which nodes to return. e.g. 'ABCD'
       nN     - size of the nodal grid

    Output:
       index  - index in the order asked e.g. 'ABCD' --> (A,B,C,D)

      TWO DIMENSIONS:

      node(i,j)          node(i,j+1)
           A -------------- B
           |                |
           |    cell(i,j)   |
           |        I       |
           |                |
          D -------------- C
      node(i+1,j)        node(i+1,j+1)


      THREE DIMENSIONS:

            node(i,j,k+1)       node(i,j+1,k+1)
                E --------------- F
               /|               / |
              / |              /  |
             /  |             /   |
      node(i,j,k)         node(i,j+1,k)
           A -------------- B     |
           |    H ----------|---- G
           |   /cell(i,j)   |   /
           |  /     I       |  /
           | /              | /
           D -------------- C
      node(i+1,j,k)      node(i+1,j+1,k)


         @author Rowan Cockett

         Last modified on: 2013/07/26
    """

    assert type(nodes) == str, "Nodes must be a str variable: e.g. 'ABCD'"
    assert type(nN) == np.ndarray, "Number of nodes must be an ndarray"
    nodes = nodes.upper()
    # Make sure that we choose from the possible nodes.
    possibleNodes = 'ABCD' if nN.size == 2 else 'ABCDEFGH'
    for node in nodes:
        assert node in possibleNodes, "Nodes must be chosen from: '%s'" % possibleNodes
    dim = nN.size
    nC = nN - 1

    if dim == 2:
        ij = ndgrid(np.arange(nC[0]), np.arange(nC[1]))
        i, j = ij[:, 0], ij[:, 1]
    elif dim == 3:
        ijk = ndgrid(np.arange(nC[0]), np.arange(nC[1]), np.arange(nC[2]))
        i, j, k = ijk[:, 0], ijk[:, 1], ijk[:, 2]
    else:
        raise Exception('Only 2 and 3 dimensions supported.')

    nodeMap = {'A': [0, 0, 0], 'B': [0, 1, 0], 'C': [1, 1, 0], 'D': [1, 0, 0],
               'E': [0, 0, 1], 'F': [0, 1, 1], 'G': [1, 1, 1], 'H': [1, 0, 1]}
    out = ()
    for node in nodes:
        shift = nodeMap[node]
        if dim == 2:
            out += (sub2ind(nN, np.c_[i+shift[0], j+shift[1]]).flatten(), )
        elif dim == 3:
            out += (sub2ind(nN, np.c_[i+shift[0], j+shift[1], k+shift[2]]).flatten(), )

    return out


def faceInfo(xyz, A, B, C, D, average=True):
    """
    function [N] = faceInfo(y,A,B,C,D)

       Returns the averaged normal, area, and edge lengths for a given set of faces.

       If average option is FALSE then N is a cell array {nA,nB,nC,nD}


    Input:
       xyz          - X,Y,Z vertex vector
       A,B,C,D      - vert index of the face (counter clockwize)

    Options:
       average      - [true]/false, toggles returning all normals or the average

    Output:
       N            - average face normal or {nA,nB,nC,nD} if average = false
       area         - average face area
       edgeLengths  - exact edge Lengths, 4 column vector [AB, BC, CD, DA]

    see also testFaceNormal testFaceArea

    @author Rowan Cockett

    Last modified on: 2013/07/26

    """

    # compute normal that is pointing away from you.
    #
    #    A -------A-B------- B
    #    |                   |
    #    |                   |
    #   D-A       (X)       B-C
    #    |                   |
    #    |                   |
    #    D -------C-D------- C

    AB = xyz[B, :] - xyz[A, :]
    BC = xyz[C, :] - xyz[B, :]
    CD = xyz[D, :] - xyz[C, :]
    DA = xyz[A, :] - xyz[D, :]

    def cross(X, Y):
        return np.c_[X[1, :]*Y[2, :] - X[2, :]*Y[1, :],
                     X[2, :]*Y[0, :] - X[0, :]*Y[2, :],
                     X[0, :]*Y[1, :] - X[1, :]*Y[0, :]]

    nA = cross(AB, DA)
    nB = cross(BC, AB)
    nC = cross(CD, BC)
    nD = cross(DA, CD)

    length = lambda x: (x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2)**0.5
    normalize = lambda x: x/np.kron(np.ones((1, x.shape[1]), length(x), 1))
    if average:
        # average the normals at each vertex.
        N = (nA + nB + nC + nD)/4  # this is intrinsically weighted by area
        # normalize
        N = normalize(N)
    else:
        N = [normalize(nA), normalize(nB), normalize(nC), normalize(nD)]

    # Area calculation
    #
    # Approximate by 4 different triangles, and divide by 2.
    # Each triangle is one half of the length of the cross product
    #
    # So also could be viewed as the average parallelogram.
    area = (length(nA)+length(nB)+length(nC)+length(nD))/4

    # simple edge length calculations
    edgeLengths = [length(AB), length(BC), length(CD), length(DA)]

    return N, area, edgeLengths


def getSubArray(A, ind):
    """subArray"""
    return A[ind[0], :, :][:, ind[1], :][:, :, ind[2]]


def ind2sub(shape, ind):
    """From the given shape, returns the subscrips of the given index"""
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
    """From the given shape, returns the index of the given subscript"""
    revshp = list(shape)
    mult = [1]
    for i in range(0, len(revshp)-1):
        mult.extend([mult[i]*revshp[i]])
    mult = array(mult).reshape(len(mult), 1)

    idx = dot((subs), (mult))
    return idx
