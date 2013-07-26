from scipy.sparse import linalg
from scipy import sparse
from sputils import *
from utils import *
from numpy import *
from TensorMesh import *

# [A] = getEdgeInnerProduct(X,Y,Z,sigma)
#

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


def subarray(T, i1, i2, i3):
    return take(take(take(T, i1, 0), i2, 1), i3, 2)


def getEdgeInnerProduct(mesh, sigma):

    h = mesh.h
    m  = array([size(h[0]), size(h[1]), size(h[2])])
    nc = prod(m)

    me1 = m + array([0, 1, 1]); ne1 = prod(me1)
    me2 = m + array([1, 0, 1]); ne2 = prod(me2)
    me3 = m + array([1, 1, 0]); ne3 = prod(me3)

    i = int64(linspace(0,m[0]-1,m[0]))
    j = int64(linspace(0,m[1]-1,m[1]))
    k = int64(linspace(0,m[2]-1,m[2]))

    ii,jj,kk = ndgrid(i,j,k,vector=False)
    ii = mkvc(ii); jj = mkvc(jj); kk = mkvc(kk)

    ## --------
    # no  | node        | e1          | e2          | e3
    # 000 | i  ,j  ,k   | i  ,j  ,k   | i  ,j  ,k   | i  ,j  ,k
    ind1 = sub2ind(me1,c_[ii,jj,kk])
    ind2 = sub2ind(me2,c_[ii,jj,kk]) + ne1
    ind3 = sub2ind(me3,c_[ii,jj,kk]) + ne1 + ne2

    IND = vstack((vstack((ind1,ind2)),ind3))
    IND = array(IND).flatten()

    P000 = sparse.coo_matrix((ones(3*nc),(linspace(0,3*nc-1,3*nc),IND)),shape=(3*nc,ne1+ne2+ne3)).tocsr()

    ## --------
    # no  | node        | e1          | e2          | e3
    # 100 | i+1,j  ,k   | i  ,j  ,k   | i+1,j  ,k   | i+1,j  ,k
    ind1 = sub2ind(me1,c_[ii,jj,kk])
    ind2 = sub2ind(me2,c_[ii+1,jj,kk]) + ne1
    ind3 = sub2ind(me3,c_[ii+1,jj,kk]) + ne1 + ne2

    IND = vstack((vstack((ind1,ind2)),ind3))
    IND = array(IND).flatten()

    P100 = sparse.coo_matrix((ones(3*nc),(linspace(0,3*nc-1,3*nc),IND)),shape=(3*nc,ne1+ne2+ne3)).tocsr()

    ## --------
    # no  | node        | e1          | e2          | e3
    # 010 | i  ,j+1,k   | i  ,j+1,k   | i  ,j  ,k   | i  ,j+1,k
    ind1 = sub2ind(me1,c_[ii,jj+1,kk])
    ind2 = sub2ind(me2,c_[ii,jj,kk]) + ne1
    ind3 = sub2ind(me3,c_[ii,jj+1,kk]) + ne1 + ne2

    IND = vstack((vstack((ind1,ind2)),ind3))
    IND = array(IND).flatten()

    P010 = sparse.coo_matrix((ones(3*nc),(linspace(0,3*nc-1,3*nc),IND)),shape=(3*nc,ne1+ne2+ne3)).tocsr()

    ## --------
    # no  | node        | e1          | e2          | e3
    # 110 | i+1,j+1,k   | i  ,j+1,k   | i+1,j  ,k   | i+1,j+1,k
    ind1 = sub2ind(me1,c_[ii,jj+1,kk])
    ind2 = sub2ind(me2,c_[ii+1,jj,kk]) + ne1
    ind3 = sub2ind(me3,c_[ii+1,jj+1,kk]) + ne1 + ne2

    IND = vstack((vstack((ind1,ind2)),ind3))
    IND = array(IND).flatten()

    P110 = sparse.coo_matrix((ones(3*nc),(linspace(0,3*nc-1,3*nc),IND)),shape=(3*nc,ne1+ne2+ne3)).tocsr()

    ######

    ## --------
    # no  | node        | e1          | e2          | e3
    # 001 | i  ,j  ,k+1   | i  ,j  ,k+1   | i  ,j  ,k+1   | i  ,j  ,k
    ind1 = sub2ind(me1,c_[ii,jj,kk+1])
    ind2 = sub2ind(me2,c_[ii,jj,kk+1]) + ne1
    ind3 = sub2ind(me3,c_[ii,jj,kk]) + ne1 + ne2

    IND = vstack((vstack((ind1,ind2)),ind3))
    IND = array(IND).flatten()

    P001 = sparse.coo_matrix((ones(3*nc),(linspace(0,3*nc-1,3*nc),IND)),shape=(3*nc,ne1+ne2+ne3)).tocsr()

    ## --------
    # no  | node          | e1            | e2            | e3
    # 101 | i+1,j  ,k+1   | i  ,j  ,k+1   | i+1,j  ,k+1   | i+1,j  ,k+1
    ind1 = sub2ind(me1,c_[ii,jj,kk+1])
    ind2 = sub2ind(me2,c_[ii+1,jj,kk+1]) + ne1
    ind3 = sub2ind(me3,c_[ii+1,jj,kk]) + ne1 + ne2

    IND = vstack((vstack((ind1,ind2)),ind3))
    IND = array(IND).flatten()

    P101 = sparse.coo_matrix((ones(3*nc),(linspace(0,3*nc-1,3*nc),IND)),shape=(3*nc,ne1+ne2+ne3)).tocsr()

    ## --------
    # no  | node          | e1             | e2            | e3
    # 011 | i  ,j+1,k+1   | i  ,j+1,k+1   | i  ,j  ,k+1   | i  ,j+1,k+1
    ind1 = sub2ind(me1,c_[ii,jj+1,kk+1])
    ind2 = sub2ind(me2,c_[ii,jj,kk+1]) + ne1
    ind3 = sub2ind(me3,c_[ii,jj+1,kk]) + ne1 + ne2

    IND = vstack((vstack((ind1,ind2)),ind3))
    IND = array(IND).flatten()

    P011 = sparse.coo_matrix((ones(3*nc),(linspace(0,3*nc-1,3*nc),IND)),shape=(3*nc,ne1+ne2+ne3)).tocsr()

    ## --------
    # no  | node          | e1            | e2            | e3
    # 111 | i+1,j+1,k+1   | i  ,j+1,k+1   | i+1,j  ,k+1   | i+1,j+1,k+1
    ind1 = sub2ind(me1,c_[ii,jj+1,kk+1])
    ind2 = sub2ind(me2,c_[ii+1,jj,kk+1]) + ne1
    ind3 = sub2ind(me3,c_[ii+1,jj+1,kk]) + ne1 + ne2

    IND = vstack((vstack((ind1,ind2)),ind3))
    IND = array(IND).flatten()

    P111 = sparse.coo_matrix((ones(3*nc),(linspace(0,3*nc-1,3*nc),IND)),shape=(3*nc,ne1+ne2+ne3)).tocsr()



    # Cell volume
    row1 = sp.hstack((sdiag(sigma[:, 0]), sdiag(sigma[:, 3]), sdiag(sigma[:, 4])))
    row2 = sp.hstack((sdiag(sigma[:, 3]), sdiag(sigma[:, 1]), sdiag(sigma[:, 5])))
    row3 = sp.hstack((sdiag(sigma[:, 4]), sdiag(sigma[:, 5]), sdiag(sigma[:, 2])))
    Sigma = sp.vstack((row1, row2, row3))

    v = sqrt(mesh.vol)
    v3 = r_[v, v, v]
    V = sdiag(v3)*Sigma*sdiag(v3)

    A = P000.T*V*P000 + P001.T*V*P001 + P010.T*V*P010 + P011.T*V*P011 + P100.T*V*P100 + P101.T*V*P101 + P110.T*V*P110 + P111.T*V*P111

    A = 0.125*A

    return A


if __name__ == '__main__':

    h = [array([1, 2, 3, 4]), array([1, 2, 1, 4, 2]), array([1, 1, 4, 1])]
    mesh = TensorMesh(h)
    sigma = ones((mesh.nC, 6))
    A = getEdgeInnerProduct(mesh, sigma)
