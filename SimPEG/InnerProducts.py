from scipy import sparse as sp
from sputils import sdiag
from utils import sub2ind, ndgrid, mkvc
import numpy as np


class InnerProducts(object):
    """
        Class creates the inner product matrices that you need!
    """
    def __init__(self):
        raise Exception('InnerProducts is a base class providing inner product matrices for meshes and cannot run on its own. Inherit to your favorite Mesh class.')

    def getFaceInnerProduct(self, mu):
        if self._meshType == 'TENSOR':
            pass
        elif self._meshType == 'LOM':
            pass  # todo: we should be doing something slightly different here!
        return getFaceInnerProduct(self, mu)

    def getEdgeInnerProduct(self, sigma):
        if self._meshType == 'TENSOR':
            pass
        elif self._meshType == 'LOM':
            pass  # todo: we should be doing something slightly different here!
        return getEdgeInnerProduct(self, sigma)


def getFaceInnerProduct(mesh, mu):

    m = np.array([mesh.nCx, mesh.nCy, mesh.nCz])
    nc = mesh.nC

    i, j, k = np.int64(range(m[0])), np.int64(range(m[1])), np.int64(range(m[2]))

    iijjkk = ndgrid(i, j, k)
    ii, jj, kk = iijjkk[:, 0], iijjkk[:, 1], iijjkk[:, 2]

    def Pxxx(pos):
        ind1 = sub2ind(mesh.nFx, np.c_[ii + pos[0][0], jj + pos[0][1], kk + pos[0][2]])
        ind2 = sub2ind(mesh.nFy, np.c_[ii + pos[1][0], jj + pos[1][1], kk + pos[1][2]]) + mesh.nF[0]
        ind3 = sub2ind(mesh.nFz, np.c_[ii + pos[2][0], jj + pos[2][1], kk + pos[2][2]]) + mesh.nF[0] + mesh.nF[1]

        IND = np.r_[ind1, ind2, ind3].flatten()

        return sp.coo_matrix((np.ones(3*nc), (range(3*nc), IND)), shape=(3*nc, np.sum(mesh.nF))).tocsr()

    #      node(i,j,k+1) ------ edge2(i,j,k+1) ----- node(i,j+1,k+1)
    #           /                                    /
    #          /                                    / |
    #      edge3(i,j,k)     face1(i,j,k)        edge3(i,j+1,k)
    #        /                                    /   |
    #       /                                    /    |
    # node(i,j,k) ------ edge2(i,j,k) ----- node(i,j+1,k)
    #      |                                     |    |
    #      |                                     |   node(i+1,j+1,k+1)
    #      |                                     |    /
    # edge1(i,j,k)      face3(i,j,k)        edge1(i,j+1,k)
    #      |                                     |  /
    #      |                                     | /
    #      |                                     |/
    # node(i+1,j,k) ------ edge2(i+1,j,k) ----- node(i+1,j+1,k)

    # no  | node        | f1        | f2        | f3
    # 000 | i  ,j  ,k   | i  , j, k | i, j  , k | i, j, k
    # 100 | i+1,j  ,k   | i+1, j, k | i, j  , k | i, j, k
    # 010 | i  ,j+1,k   | i  , j, k | i, j+1, k | i, j, k
    # 110 | i+1,j+1,k   | i+1, j, k | i, j+1, k | i, j, k
    # 001 | i  ,j  ,k   | i  , j, k | i, j  , k | i, j, k+1
    # 101 | i+1,j  ,k   | i+1, j, k | i, j  , k | i, j, k+1
    # 011 | i  ,j+1,k   | i  , j, k | i, j+1, k | i, j, k+1
    # 111 | i+1,j+1,k   | i+1, j, k | i, j+1, k | i, j, k+1
    P000 = Pxxx([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    P100 = Pxxx([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    P010 = Pxxx([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    P110 = Pxxx([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    P001 = Pxxx([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    P101 = Pxxx([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
    P011 = Pxxx([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
    P111 = Pxxx([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    if mu.size == mesh.nC:  # Isotropic!
        mu = mkvc(mu)  # ensure it is a vector.
        mu = sdiag(np.r_[mu, mu, mu])
    elif mu.shape[1] == 3:  # Diagonal tensor
        mu = sdiag(np.r_[mu[:, 0], mu[:, 1], mu[:, 2]])
    elif mu.shape[1] == 6:  # Fully anisotropic
        row1 = sp.hstack((sdiag(mu[:, 0]), sdiag(mu[:, 3]), sdiag(mu[:, 4])))
        row2 = sp.hstack((sdiag(mu[:, 3]), sdiag(mu[:, 1]), sdiag(mu[:, 5])))
        row3 = sp.hstack((sdiag(mu[:, 4]), sdiag(mu[:, 5]), sdiag(mu[:, 2])))
        mu = sp.vstack((row1, row2, row3))

    # Cell volume
    v = np.sqrt(mesh.vol)
    v3 = np.r_[v, v, v]
    V = sdiag(v3)*mu*sdiag(v3)  # to keep symmetry

    A = P000.T*V*P000 + P001.T*V*P001 + P010.T*V*P010 + P011.T*V*P011 + P100.T*V*P100 + P101.T*V*P101 + P110.T*V*P110 + P111.T*V*P111

    A = 0.125*A

    return A


def getEdgeInnerProduct(mesh, sigma):

    m = np.array([mesh.nCx, mesh.nCy, mesh.nCz])
    nc = mesh.nC

    i, j, k = np.int64(range(m[0])), np.int64(range(m[1])), np.int64(range(m[2]))

    iijjkk = ndgrid(i, j, k)
    ii, jj, kk = iijjkk[:, 0], iijjkk[:, 1], iijjkk[:, 2]

    def Pxxx(pos):
        ind1 = sub2ind(mesh.nEx, np.c_[ii + pos[0][0], jj + pos[0][1], kk + pos[0][2]])
        ind2 = sub2ind(mesh.nEy, np.c_[ii + pos[1][0], jj + pos[1][1], kk + pos[1][2]]) + mesh.nE[0]
        ind3 = sub2ind(mesh.nEz, np.c_[ii + pos[2][0], jj + pos[2][1], kk + pos[2][2]]) + mesh.nE[0] + mesh.nE[1]

        IND = np.r_[ind1, ind2, ind3].flatten()

        return sp.coo_matrix((np.ones(3*nc), (range(3*nc), IND)), shape=(3*nc, np.sum(mesh.nE))).tocsr()

    #      node(i,j,k+1) ------ edge2(i,j,k+1) ----- node(i,j+1,k+1)
    #           /                                    /
    #          /                                    / |
    #      edge3(i,j,k)     face1(i,j,k)        edge3(i,j+1,k)
    #        /                                    /   |
    #       /                                    /    |
    # node(i,j,k) ------ edge2(i,j,k) ----- node(i,j+1,k)
    #      |                                     |    |
    #      |                                     |   node(i+1,j+1,k+1)
    #      |                                     |    /
    # edge1(i,j,k)      face3(i,j,k)        edge1(i,j+1,k)
    #      |                                     |  /
    #      |                                     | /
    #      |                                     |/
    # node(i+1,j,k) ------ edge2(i+1,j,k) ----- node(i+1,j+1,k)

    # no  | node        | e1          | e2          | e3
    # 000 | i  ,j  ,k   | i  ,j  ,k   | i  ,j  ,k   | i  ,j  ,k
    # 100 | i+1,j  ,k   | i  ,j  ,k   | i+1,j  ,k   | i+1,j  ,k
    # 010 | i  ,j+1,k   | i  ,j+1,k   | i  ,j  ,k   | i  ,j+1,k
    # 110 | i+1,j+1,k   | i  ,j+1,k   | i+1,j  ,k   | i+1,j+1,k
    # 001 | i  ,j  ,k+1 | i  ,j  ,k+1 | i  ,j  ,k+1 | i  ,j  ,k
    # 101 | i+1,j  ,k+1 | i  ,j  ,k+1 | i+1,j  ,k+1 | i+1,j  ,k
    # 011 | i  ,j+1,k+1 | i  ,j+1,k+1 | i  ,j  ,k+1 | i  ,j+1,k
    # 111 | i+1,j+1,k+1 | i  ,j+1,k+1 | i+1,j  ,k+1 | i+1,j+1,k
    P000 = Pxxx([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    P100 = Pxxx([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
    P010 = Pxxx([[0, 1, 0], [0, 0, 0], [0, 1, 0]])
    P110 = Pxxx([[0, 1, 0], [1, 0, 0], [1, 1, 0]])
    P001 = Pxxx([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    P101 = Pxxx([[0, 0, 1], [1, 0, 1], [1, 0, 0]])
    P011 = Pxxx([[0, 1, 1], [0, 0, 1], [0, 1, 0]])
    P111 = Pxxx([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

    if sigma.size == mesh.nC:  # Isotropic!
        sigma = mkvc(sigma)  # ensure it is a vector.
        Sigma = sdiag(np.r_[sigma, sigma, sigma])
    elif sigma.shape[1] == 3:  # Diagonal tensor
        Sigma = sdiag(np.r_[sigma[:, 0], sigma[:, 1], sigma[:, 2]])
    elif sigma.shape[1] == 6:  # Fully anisotropic
        row1 = sp.hstack((sdiag(sigma[:, 0]), sdiag(sigma[:, 3]), sdiag(sigma[:, 4])))
        row2 = sp.hstack((sdiag(sigma[:, 3]), sdiag(sigma[:, 1]), sdiag(sigma[:, 5])))
        row3 = sp.hstack((sdiag(sigma[:, 4]), sdiag(sigma[:, 5]), sdiag(sigma[:, 2])))
        Sigma = sp.vstack((row1, row2, row3))

    # Cell volume
    v = np.sqrt(mesh.vol)
    v3 = np.r_[v, v, v]
    V = sdiag(v3)*Sigma*sdiag(v3)  # to keep symmetry

    A = P000.T*V*P000 + P001.T*V*P001 + P010.T*V*P010 + P011.T*V*P011 + P100.T*V*P100 + P101.T*V*P101 + P110.T*V*P110 + P111.T*V*P111

    A = 0.125*A
    P = sp.vstack((sdiag(v3)*P000,sdiag(v3)*P001,sdiag(v3)*P010,sdiag(v3)*P011,
                sdiag(v3)*P100,sdiag(v3)*P101,sdiag(v3)*P110,sdiag(v3)*P111))

    return A, P



if __name__ == '__main__':
    from TensorMesh import TensorMesh
    h = [np.array([1, 2, 3, 4]), np.array([1, 2, 1, 4, 2]), np.array([1, 1, 4, 1])]
    mesh = TensorMesh(h)
    mu = np.ones((mesh.nC, 6))
    A = getFaceInnerProduct(mesh,mu)
    B = getEdgeInnerProduct(mesh,mu)

