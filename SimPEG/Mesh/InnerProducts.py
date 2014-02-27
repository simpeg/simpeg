from scipy import sparse as sp
from SimPEG.Utils import sub2ind, ndgrid, mkvc, getSubArray, sdiag, inv3X3BlockDiagonal, inv2X2BlockDiagonal, makePropertyTensor, invPropertyTensor
import numpy as np


class InnerProducts(object):
    """
        This is a base for the SimPEG.Mesh classes. This mixIn creates the all the inner product matrices that you need!
    """
    def __init__(self):
        raise Exception('InnerProducts is a base class providing inner product matrices for meshes and cannot run on its own. Inherit to your favorite Mesh class.')

    def getFaceInnerProduct(self, materialProperty=None, returnP=False, invertProperty=False):
        """
            :param numpy.array materialProperty: material property (tensor properties are possible) at each cell center (nC, (1, 3, or 6))
            :param bool returnP: returns the projection matrices
            :param bool invertProperty: inverts the material property
            :rtype: scipy.csr_matrix
            :return: M, the inner product matrix (nF, nF)
        """
        if invertProperty:
            materialProperty = invPropertyTensor(self, materialProperty)

        Mu = makePropertyTensor(self, materialProperty)

        d = self.dim
        # We will multiply by sqrt on each side to keep symmetry
        V = sp.kron(sp.identity(d), sdiag(np.sqrt((2**(-d))*self.vol)))

        if d == 1:
            fP = _getFacePx(self)
            P000 = V*fP('fXm')
            P100 = V*fP('fXp')
        elif d == 2:
            fP = _getFacePxx(self)
            P000 = V*fP('fXm', 'fYm')
            P100 = V*fP('fXp', 'fYm')
            P010 = V*fP('fXm', 'fYp')
            P110 = V*fP('fXp', 'fYp')
        elif d == 3:
            fP = _getFacePxxx(self)
            P000 = V*fP('fXm', 'fYm', 'fZm')
            P100 = V*fP('fXp', 'fYm', 'fZm')
            P010 = V*fP('fXm', 'fYp', 'fZm')
            P110 = V*fP('fXp', 'fYp', 'fZm')
            P001 = V*fP('fXm', 'fYm', 'fZp')
            P101 = V*fP('fXp', 'fYm', 'fZp')
            P011 = V*fP('fXm', 'fYp', 'fZp')
            P111 = V*fP('fXp', 'fYp', 'fZp')

        A = P000.T*Mu*P000 + P100.T*Mu*P100
        P = [P000, P100]

        if d > 1:
            A = A + P010.T*Mu*P010 + P110.T*Mu*P110
            P += [P010, P110]
        if d > 2:
            A = A + P001.T*Mu*P001 + P101.T*Mu*P101 + P011.T*Mu*P011 + P111.T*Mu*P111
            P += [P001, P101, P011,  P111]
        if returnP:
            return A, P
        else:
            return A

    def getEdgeInnerProduct(self, materialProperty=None, returnP=False, invertProperty=False):
        """
            :param numpy.array materialProperty: material property (tensor properties are possible) at each cell center (nC, (1, 3, or 6))
            :param bool returnP: returns the projection matrices
            :param bool invertProperty: inverts the material property
            :rtype: scipy.csr_matrix
            :return: M, the inner product matrix (nE, nE)
        """
        if invertProperty:
            materialProperty = invPropertyTensor(self, materialProperty)

        Mu = makePropertyTensor(self, materialProperty)

        d = self.dim
        # We will multiply by sqrt on each side to keep symmetry
        V = sp.kron(sp.identity(d), sdiag(np.sqrt((2**(-d))*self.vol)))

        if d == 1:
            raise NotImplementedError('getEdgeInnerProduct not implemented for 1D')
        elif d == 2:
            eP = _getEdgePxx(self)
            P000 = V*eP('eX0', 'eY0')
            P100 = V*eP('eX0', 'eY1')
            P010 = V*eP('eX1', 'eY0')
            P110 = V*eP('eX1', 'eY1')
        elif d == 3:
            eP = _getEdgePxxx(self)
            P000 = V*eP('eX0', 'eY0', 'eZ0')
            P100 = V*eP('eX0', 'eY1', 'eZ1')
            P010 = V*eP('eX1', 'eY0', 'eZ2')
            P110 = V*eP('eX1', 'eY1', 'eZ3')
            P001 = V*eP('eX2', 'eY2', 'eZ0')
            P101 = V*eP('eX2', 'eY3', 'eZ1')
            P011 = V*eP('eX3', 'eY2', 'eZ2')
            P111 = V*eP('eX3', 'eY3', 'eZ3')

        Mu = makePropertyTensor(self, materialProperty)
        A = P000.T*Mu*P000 + P100.T*Mu*P100 + P010.T*Mu*P010 + P110.T*Mu*P110
        P = [P000, P100, P010, P110]
        if d == 3:
            A = A + P001.T*Mu*P001 + P101.T*Mu*P101 + P011.T*Mu*P011 + P111.T*Mu*P111
            P += [P001, P101, P011,  P111]
        if returnP:
            return A, P
        else:
            return A

# ------------------------ Geometries ------------------------------
#
#
#         node(i,j,k+1) ------ edge2(i,j,k+1) ----- node(i,j+1,k+1)
#              /                                    /
#             /                                    / |
#         edge3(i,j,k)     face1(i,j,k)        edge3(i,j+1,k)
#           /                                    /   |
#          /                                    /    |
#    node(i,j,k) ------ edge2(i,j,k) ----- node(i,j+1,k)
#         |                                     |    |
#         |                                     |   node(i+1,j+1,k+1)
#         |                                     |    /
#    edge1(i,j,k)      face3(i,j,k)        edge1(i,j+1,k)
#         |                                     |  /
#         |                                     | /
#         |                                     |/
#    node(i+1,j,k) ------ edge2(i+1,j,k) ----- node(i+1,j+1,k)


def _getFacePx(M):
    assert M._meshType == 'TENSOR', 'Only supported for a tensor mesh'
    return _getFacePx_Rectangular(M)

def _getFacePxx(M):
    if M._meshType == 'TREE':
        return M._getFacePxx

    return _getFacePxx_Rectangular(M)

def _getFacePxxx(M):
    if M._meshType == 'TREE':
        return M._getFacePxxx

    return _getFacePxxx_Rectangular(M)

def _getEdgePxx(M):
    if M._meshType == 'TREE':
        return M._getEdgePxx

    return _getEdgePxx_Rectangular(M)

def _getEdgePxxx(M):
    if M._meshType == 'TREE':
        return M._getEdgePxxx

    return _getEdgePxxx_Rectangular(M)

def _getFacePx_Rectangular(M):
    """Returns a function for creating projection matrices

    """
    ii = np.int64(range(M.nCx))

    def Px(xFace):
        """
            xFace is 'fXp' or 'fXm'
        """
        posFx = 0 if xFace == 'fXm' else 1
        IND = ii + posFx
        PX = sp.csr_matrix((np.ones(M.nC), (range(M.nC), IND)), shape=(M.nC, M.nF))
        return PX

    return Px

def _getFacePxx_Rectangular(M):
    """returns a function for creating projection matrices

        Mats takes you from faces a subset of all faces on only the
        faces that you ask for.

        These are centered around a single nodes.

        For example, if this was your entire mesh:

                        f3(Yp)
                  2_______________3
                  |               |
                  |               |
                  |               |
          f0(Xm)  |       x       |  f1(Xp)
                  |               |
                  |               |
                  |_______________|
                  0               1
                        f2(Ym)

        Pxx('fXm','fYm') = | 1, 0, 0, 0 |
                           | 0, 0, 1, 0 |

        Pxx('fXp','fYm') = | 0, 1, 0, 0 |
                           | 0, 0, 1, 0 |

        """
    i, j = np.int64(range(M.nCx)), np.int64(range(M.nCy))

    iijj = ndgrid(i, j)
    ii, jj = iijj[:, 0], iijj[:, 1]

    if M._meshType == 'LOM':
        fN1 = M.r(M.normals, 'F', 'Fx', 'M')
        fN2 = M.r(M.normals, 'F', 'Fy', 'M')

    def Pxx(xFace, yFace):
        """
            xFace is 'fXp' or 'fXm'
            yFace is 'fYp' or 'fYm'
        """
        # no | node      | f1     | f2
        # 00 | i  ,j     | i  , j | i, j
        # 10 | i+1,j     | i+1, j | i, j
        # 01 | i  ,j+1   | i  , j | i, j+1
        # 11 | i+1,j+1   | i+1, j | i, j+1

        posFx = 0 if xFace == 'fXm' else 1
        posFy = 0 if yFace == 'fYm' else 1

        ind1 = sub2ind(M.vnFx, np.c_[ii + posFx, jj])
        ind2 = sub2ind(M.vnFy, np.c_[ii, jj + posFy]) + M.nFx

        IND = np.r_[ind1, ind2].flatten()

        PXX = sp.csr_matrix((np.ones(2*M.nC), (range(2*M.nC), IND)), shape=(2*M.nC, M.nF))

        if M._meshType == 'LOM':
            I2x2 = inv2X2BlockDiagonal(getSubArray(fN1[0], [i + posFx, j]), getSubArray(fN1[1], [i + posFx, j]),
                                       getSubArray(fN2[0], [i, j + posFy]), getSubArray(fN2[1], [i, j + posFy]))
            PXX = I2x2 * PXX

        return PXX

    return Pxx

def _getFacePxxx_Rectangular(M):
    """returns a function for creating projection matrices

        Mats takes you from faces a subset of all faces on only the
        faces that you ask for.

        These are centered around a single nodes.
    """

    i, j, k = np.int64(range(M.nCx)), np.int64(range(M.nCy)), np.int64(range(M.nCz))

    iijjkk = ndgrid(i, j, k)
    ii, jj, kk = iijjkk[:, 0], iijjkk[:, 1], iijjkk[:, 2]

    if M._meshType == 'LOM':
        fN1 = M.r(M.normals, 'F', 'Fx', 'M')
        fN2 = M.r(M.normals, 'F', 'Fy', 'M')
        fN3 = M.r(M.normals, 'F', 'Fz', 'M')

    def Pxxx(xFace, yFace, zFace):
        """
            xFace is 'fXp' or 'fXm'
            yFace is 'fYp' or 'fYm'
            zFace is 'fZp' or 'fZm'
        """

        # no  | node        | f1        | f2        | f3
        # 000 | i  ,j  ,k   | i  , j, k | i, j  , k | i, j, k
        # 100 | i+1,j  ,k   | i+1, j, k | i, j  , k | i, j, k
        # 010 | i  ,j+1,k   | i  , j, k | i, j+1, k | i, j, k
        # 110 | i+1,j+1,k   | i+1, j, k | i, j+1, k | i, j, k
        # 001 | i  ,j  ,k+1 | i  , j, k | i, j  , k | i, j, k+1
        # 101 | i+1,j  ,k+1 | i+1, j, k | i, j  , k | i, j, k+1
        # 011 | i  ,j+1,k+1 | i  , j, k | i, j+1, k | i, j, k+1
        # 111 | i+1,j+1,k+1 | i+1, j, k | i, j+1, k | i, j, k+1

        posX = 0 if xFace == 'fXm' else 1
        posY = 0 if yFace == 'fYm' else 1
        posZ = 0 if zFace == 'fZm' else 1

        ind1 = sub2ind(M.vnFx, np.c_[ii + posX, jj, kk])
        ind2 = sub2ind(M.vnFy, np.c_[ii, jj + posY, kk]) + M.nFx
        ind3 = sub2ind(M.vnFz, np.c_[ii, jj, kk + posZ]) + M.nFx + M.nFy

        IND = np.r_[ind1, ind2, ind3].flatten()

        PXXX = sp.coo_matrix((np.ones(3*M.nC), (range(3*M.nC), IND)), shape=(3*M.nC, M.nF)).tocsr()

        if M._meshType == 'LOM':
            I3x3 = inv3X3BlockDiagonal(getSubArray(fN1[0], [i + posX, j, k]), getSubArray(fN1[1], [i + posX, j, k]), getSubArray(fN1[2], [i + posX, j, k]),
                                       getSubArray(fN2[0], [i, j + posY, k]), getSubArray(fN2[1], [i, j + posY, k]), getSubArray(fN2[2], [i, j + posY, k]),
                                       getSubArray(fN3[0], [i, j, k + posZ]), getSubArray(fN3[1], [i, j, k + posZ]), getSubArray(fN3[2], [i, j, k + posZ]))
            PXXX = I3x3 * PXXX

        return PXXX
    return Pxxx

def _getEdgePxx_Rectangular(M):
    i, j = np.int64(range(M.nCx)), np.int64(range(M.nCy))

    iijj = ndgrid(i, j)
    ii, jj = iijj[:, 0], iijj[:, 1]

    if M._meshType == 'LOM':
        eT1 = M.r(M.tangents, 'E', 'Ex', 'M')
        eT2 = M.r(M.tangents, 'E', 'Ey', 'M')

    def Pxx(xEdge, yEdge):
        # no | node      | e1      | e2
        # 00 | i  ,j     | i  ,j   | i  ,j
        # 10 | i+1,j     | i  ,j   | i+1,j
        # 01 | i  ,j+1   | i  ,j+1 | i  ,j
        # 11 | i+1,j+1   | i  ,j+1 | i+1,j
        posX = 0 if xEdge == 'eX0' else 1
        posY = 0 if yEdge == 'eY0' else 1

        ind1 = sub2ind(M.vnEx, np.c_[ii, jj + posX])
        ind2 = sub2ind(M.vnEy, np.c_[ii + posY, jj]) + M.nEx

        IND = np.r_[ind1, ind2].flatten()

        PXX = sp.coo_matrix((np.ones(2*M.nC), (range(2*M.nC), IND)), shape=(2*M.nC, M.nE)).tocsr()

        if M._meshType == 'LOM':
            I2x2 = inv2X2BlockDiagonal(getSubArray(eT1[0], [i, j + posX]), getSubArray(eT1[1], [i, j + posX]),
                                       getSubArray(eT2[0], [i + posY, j]), getSubArray(eT2[1], [i + posY, j]))
            PXX = I2x2 * PXX

        return PXX
    return Pxx

def _getEdgePxxx_Rectangular(M):
    i, j, k = np.int64(range(M.nCx)), np.int64(range(M.nCy)), np.int64(range(M.nCz))

    iijjkk = ndgrid(i, j, k)
    ii, jj, kk = iijjkk[:, 0], iijjkk[:, 1], iijjkk[:, 2]

    if M._meshType == 'LOM':
        eT1 = M.r(M.tangents, 'E', 'Ex', 'M')
        eT2 = M.r(M.tangents, 'E', 'Ey', 'M')
        eT3 = M.r(M.tangents, 'E', 'Ez', 'M')

    def Pxxx(xEdge, yEdge, zEdge):

        # no  | node        | e1          | e2          | e3
        # 000 | i  ,j  ,k   | i  ,j  ,k   | i  ,j  ,k   | i  ,j  ,k
        # 100 | i+1,j  ,k   | i  ,j  ,k   | i+1,j  ,k   | i+1,j  ,k
        # 010 | i  ,j+1,k   | i  ,j+1,k   | i  ,j  ,k   | i  ,j+1,k
        # 110 | i+1,j+1,k   | i  ,j+1,k   | i+1,j  ,k   | i+1,j+1,k
        # 001 | i  ,j  ,k+1 | i  ,j  ,k+1 | i  ,j  ,k+1 | i  ,j  ,k
        # 101 | i+1,j  ,k+1 | i  ,j  ,k+1 | i+1,j  ,k+1 | i+1,j  ,k
        # 011 | i  ,j+1,k+1 | i  ,j+1,k+1 | i  ,j  ,k+1 | i  ,j+1,k
        # 111 | i+1,j+1,k+1 | i  ,j+1,k+1 | i+1,j  ,k+1 | i+1,j+1,k

        posX = [0,0] if xEdge == 'eX0' else [1, 0] if xEdge == 'eX1' else [0,1] if xEdge == 'eX2' else [1,1]
        posY = [0,0] if yEdge == 'eY0' else [1, 0] if yEdge == 'eY1' else [0,1] if yEdge == 'eY2' else [1,1]
        posZ = [0,0] if zEdge == 'eZ0' else [1, 0] if zEdge == 'eZ1' else [0,1] if zEdge == 'eZ2' else [1,1]

        ind1 = sub2ind(M.vnEx, np.c_[ii, jj + posX[0], kk + posX[1]])
        ind2 = sub2ind(M.vnEy, np.c_[ii + posY[0], jj, kk + posY[1]]) + M.nEx
        ind3 = sub2ind(M.vnEz, np.c_[ii + posZ[0], jj + posZ[1], kk]) + M.nEx + M.nEy

        IND = np.r_[ind1, ind2, ind3].flatten()

        PXXX = sp.coo_matrix((np.ones(3*M.nC), (range(3*M.nC), IND)), shape=(3*M.nC, M.nE)).tocsr()

        if M._meshType == 'LOM':
            I3x3 = inv3X3BlockDiagonal(getSubArray(eT1[0], [i, j + posX[0], k + posX[1]]), getSubArray(eT1[1], [i, j + posX[0], k + posX[1]]), getSubArray(eT1[2], [i, j + posX[0], k + posX[1]]),
                                       getSubArray(eT2[0], [i + posY[0], j, k + posY[1]]), getSubArray(eT2[1], [i + posY[0], j, k + posY[1]]), getSubArray(eT2[2], [i + posY[0], j, k + posY[1]]),
                                       getSubArray(eT3[0], [i + posZ[0], j + posZ[1], k]), getSubArray(eT3[1], [i + posZ[0], j + posZ[1], k]), getSubArray(eT3[2], [i + posZ[0], j + posZ[1], k]))
            PXXX = I3x3 * PXXX

        return PXXX
    return Pxxx

if __name__ == '__main__':
    from TensorMesh import TensorMesh
    h = [np.array([1, 2, 3, 4]), np.array([1, 2, 1, 4, 2]), np.array([1, 1, 4, 1])]
    M = TensorMesh(h)
    mu = np.ones((M.nC, 6))
    A, P = M.getFaceInnerProduct(mu, returnP=True)
    B, P = M.getEdgeInnerProduct(mu, returnP=True)
