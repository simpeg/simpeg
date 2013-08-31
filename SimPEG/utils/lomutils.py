import numpy as np
from scipy import sparse as sp
from matutils import mkvc, ndgrid, sub2ind
from sputils import sdiag


def exampleLomGird(nC, exType):
    assert type(nC) == list, "nC must be a list containing the number of nodes"
    assert len(nC) == 2 or len(nC) == 3, "nC must either two or three dimensions"
    exType = exType.lower()

    possibleTypes = ['rect', 'rotate']
    assert exType in possibleTypes, "Not a possible example type."

    if exType == 'rect':
        return ndgrid([np.cumsum(np.r_[0, np.ones(nx)/nx]) for nx in nC], vector=False)
    elif exType == 'rotate':
        if len(nC) == 2:
            X, Y = ndgrid([np.cumsum(np.r_[0, np.ones(nx)/nx]) for nx in nC], vector=False)
            amt = 0.5-np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
            amt[amt < 0] = 0
            return X + (-(Y - 0.5))*amt, Y + (+(X - 0.5))*amt
        elif len(nC) == 3:
            X, Y, Z = ndgrid([np.cumsum(np.r_[0, np.ones(nx)/nx]) for nx in nC], vector=False)
            amt = 0.5-np.sqrt((X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2)
            amt[amt < 0] = 0
            return X + (-(Y - 0.5))*amt, Y + (-(Z - 0.5))*amt, Z + (-(X - 0.5))*amt


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


def indexCube(nodes, gridSize, n=None):
    """
    Returns the index of nodes on the mesh.


    Input:
       nodes     - string of which nodes to return. e.g. 'ABCD'
       gridSize  - size of the nodal grid
       n         - number of nodes each i,j,k direction: [ni,nj,nk]


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
    assert type(gridSize) == np.ndarray, "Number of nodes must be an ndarray"
    nodes = nodes.upper()
    # Make sure that we choose from the possible nodes.
    possibleNodes = 'ABCD' if gridSize.size == 2 else 'ABCDEFGH'
    for node in nodes:
        assert node in possibleNodes, "Nodes must be chosen from: '%s'" % possibleNodes
    dim = gridSize.size
    if n is None:
        n = gridSize - 1

    if dim == 2:
        ij = ndgrid(np.arange(n[0]), np.arange(n[1]))
        i, j = ij[:, 0], ij[:, 1]
    elif dim == 3:
        ijk = ndgrid(np.arange(n[0]), np.arange(n[1]), np.arange(n[2]))
        i, j, k = ijk[:, 0], ijk[:, 1], ijk[:, 2]
    else:
        raise Exception('Only 2 and 3 dimensions supported.')

    nodeMap = {'A': [0, 0, 0], 'B': [0, 1, 0], 'C': [1, 1, 0], 'D': [1, 0, 0],
               'E': [0, 0, 1], 'F': [0, 1, 1], 'G': [1, 1, 1], 'H': [1, 0, 1]}
    out = ()
    for node in nodes:
        shift = nodeMap[node]
        if dim == 2:
            out += (sub2ind(gridSize, np.c_[i+shift[0], j+shift[1]]).flatten(), )
        elif dim == 3:
            out += (sub2ind(gridSize, np.c_[i+shift[0], j+shift[1], k+shift[2]]).flatten(), )

    return out


def faceInfo(xyz, A, B, C, D, average=True, normalizeNormals=True):
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
    assert type(average) is bool, 'average must be a boolean'
    assert type(normalizeNormals) is bool, 'normalizeNormals must be a boolean'
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
        return np.c_[X[:, 1]*Y[:, 2] - X[:, 2]*Y[:, 1],
                     X[:, 2]*Y[:, 0] - X[:, 0]*Y[:, 2],
                     X[:, 0]*Y[:, 1] - X[:, 1]*Y[:, 0]]

    nA = cross(AB, DA)
    nB = cross(BC, AB)
    nC = cross(CD, BC)
    nD = cross(DA, CD)

    length = lambda x: np.sqrt(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2)
    normalize = lambda x: x/np.kron(np.ones((1, x.shape[1])), mkvc(length(x), 2))
    if average:
        # average the normals at each vertex.
        N = (nA + nB + nC + nD)/4  # this is intrinsically weighted by area
        # normalize
        N = normalize(N)
    else:
        if normalizeNormals:
            N = [normalize(nA), normalize(nB), normalize(nC), normalize(nD)]
        else:
            N = [nA, nB, nC, nD]

    # Area calculation
    #
    # Approximate by 4 different triangles, and divide by 2.
    # Each triangle is one half of the length of the cross product
    #
    # So also could be viewed as the average parallelogram.
    #
    # WARNING: This does not compute correctly for concave quadrilaterals
    area = (length(nA)+length(nB)+length(nC)+length(nD))/4

    return N, area


def inv3X3BlockDiagonal(a11, a12, a13, a21, a22, a23, a31, a32, a33):
    """ B = inv3X3BlockDiagonal(a11, a12, a13, a21, a22, a23, a31, a32, a33)

    inverts a stack of 3x3 matrices

    Input:
     A   - a11, a12, a13, a21, a22, a23, a31, a32, a33

    Output:
     B   - inverse
     """

    a11 = mkvc(a11)
    a12 = mkvc(a12)
    a13 = mkvc(a13)
    a21 = mkvc(a21)
    a22 = mkvc(a22)
    a23 = mkvc(a23)
    a31 = mkvc(a31)
    a32 = mkvc(a32)
    a33 = mkvc(a33)

    detA = a31*a12*a23 - a31*a13*a22 - a21*a12*a33 + a21*a13*a32 + a11*a22*a33 - a11*a23*a32

    b11 = +(a22*a33 - a23*a32)/detA
    b12 = -(a12*a33 - a13*a32)/detA
    b13 = +(a12*a23 - a13*a22)/detA

    b21 = +(a31*a23 - a21*a33)/detA
    b22 = -(a31*a13 - a11*a33)/detA
    b23 = +(a21*a13 - a11*a23)/detA

    b31 = -(a31*a22 - a21*a32)/detA
    b32 = +(a31*a12 - a11*a32)/detA
    b33 = -(a21*a12 - a11*a22)/detA

    B = sp.vstack((sp.hstack((sdiag(b11), sdiag(b12),  sdiag(b13))),
                   sp.hstack((sdiag(b21), sdiag(b22),  sdiag(b23))),
                   sp.hstack((sdiag(b31), sdiag(b32),  sdiag(b33)))))

    return B


def inv2X2BlockDiagonal(a11, a12, a21, a22):
    """ B = inv2X2BlockDiagonal(a11, a12, a21, a22)

    Inverts a stack of 2x2 matrices by using the inversion formula

    inv(A) = (1/det(A)) * cof(A)^T

    Input:
    A   - a11, a12, a13, a21, a22, a23, a31, a32, a33

    Output:
    B   - inverse
    """

    a11 = mkvc(a11)
    a12 = mkvc(a12)
    a21 = mkvc(a21)
    a22 = mkvc(a22)

    # compute inverse of the determinant.
    detAinv = 1./(a11*a22 - a21*a12)

    b11 = +detAinv*a22
    b12 = -detAinv*a12
    b21 = -detAinv*a21
    b22 = +detAinv*a11

    B = sp.vstack((sp.hstack((sdiag(b11), sdiag(b12))),
                   sp.hstack((sdiag(b21), sdiag(b22)))))

    return B
