from scipy.sparse import linalg
from scipy import sparse
from sputils import *
from utils import *
from sputils import *
from numpy import *
from getEdgeTangent import *
from inv3X3BlockDiagonal import *

def volTetra(y,m,I,A,B,C,D):

    a11 = array(y[A,0]-y[B,0]); a12 = array(y[A,0]-y[C,0]); a13 = array(y[A,0]-y[D,0])
    a21 = array(y[A,1]-y[B,1]); a22 = array(y[A,1]-y[C,1]); a23 = array(y[A,1]-y[D,1])
    a31 = array(y[A,2]-y[B,2]); a32 = array(y[A,2]-y[C,2]); a33 = array(y[A,2]-y[D,2])

    return abs(a11*a22*a33 + a12*a23*a31 + a13*a21*a32 - a31*a22*a13 - a32*a23*a11 - a33*a21*a12)


def getCellVolume(X,Y,Z):

    m = array(shape(X))-1
    y = hstack3(mkvc(X),mkvc(Y),mkvc(Z))
    
    i = int64(linspace(0,m[0]-1,m[0]))
    j = int64(linspace(0,m[1]-1,m[1]))
    k = int64(linspace(0,m[2]-1,m[2]))
        
    ii,jj,kk = ndgrid(i,j,k) 
    ii = mkvc(ii); jj = mkvc(jj); kk = mkvc(kk)
    
    
    I  = int64(sub2ind(m,hstack3(ii,jj,kk)))
    A  = int64(sub2ind(m+1,hstack3(ii,jj,kk)))
    B  = int64(sub2ind(m+1,hstack3(ii,jj+1,kk)))
    C  = int64(sub2ind(m+1,hstack3(ii+1,jj+1,kk)))
    D  = int64(sub2ind(m+1,hstack3(ii+1,jj,kk)))
    E  = int64(sub2ind(m+1,hstack3(ii,jj,kk+1)))
    F  = int64(sub2ind(m+1,hstack3(ii,jj+1,kk+1)))
    G  = int64(sub2ind(m+1,hstack3(ii+1,jj+1,kk+1)))
    H  = int64(sub2ind(m+1,hstack3(ii+1,jj,kk+1)))
    
    v1 = volTetra(y,m,I,A,B,D,E)
    v2 = volTetra(y,m,I,B,E,F,G)
    v3 = volTetra(y,m,I,B,D,E,G)
    v4 = volTetra(y,m,I,B,C,D,G)
    v5 = volTetra(y,m,I,D,E,G,H)
    
    v  = 1.0/6.0 * ( v1 + v2 + v3 + v4 + v5 )
    return v.flatten()


if __name__ == '__main__':

    X,Y,Z = ndgrid(linspace(0,2,3),linspace(0,2,3),linspace(0,2,3))
    Z[2,2,2] = 2.5;   Z[0,0,0] = -0.5
    X[2,2,2] = 2.5; X[0,0,0] = -0.5

    v = getCellVolume(X,Y,Z)