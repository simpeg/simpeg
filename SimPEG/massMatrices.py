import numpy as np
from scipy import sparse as sp
from sputils import sdiag, speye, kron3, spzeros
from utils import mkvc



def getEdgeMassMatrix(sigma,mesh):
    """Get anisotropic mass matrix"""

    n = array([size(mesh.h[0]),size(mesh.h[1]),size(mesh.h[2])])
    nx = prod(n + [1, 0, 0])
    ex = reshape(arange(0,nx),[n[0]+1,n[1],n[2]])
    ny = prod(n + [0, 1, 0])
    ey = reshape(arange(0,ny),[n[0],n[1]+1,n[2]])  
    nz = prod(n + [0, 0, 1]);
    ez = reshape(arange(0,nz),[n[0],n[1],n[2]+1])  


    i = arange(0,n[0]-1); j = arange(0,n[1]-1); k = arange(0,n[2]-1)

    # corner i,j,k
    Px1 = take(ex,[i,j,k]);   Py1 = take(ey,[i,j,k]);    Pz1 = take(ez,[i,j,k])   
    # corner i+1,j,k
    Px2 = take(ex,[i,j,k]);   Py2 = take(ey,[i+1,j,k]);  Pz2 = take(ez,[i+1,j,k])
    # corner i,j+1,k
    Px3 = take(ex,[i,j+1,k]); Py3 = take(ey,[i,j,k]);    Pz3 = take(ez,[i,j+1,k])
    # corner i+1,j+1,k
    Px4 = take(ex,[i,j+1,k]); Py4 = take(ey,[i+1,j,k]);  Pz4 = take(ez,[i+1,j+1,k]);

    # corner i,j,k+1
    Px5 = take(ex,[i,j,k+1]);   Py5 = take(ey,[i,j,k+1]);    Pz5 = take(ez,[i,j,k])   
    # corner i+1,j,k+1
    Px6 = take(ex,[i,j,k+1]);   Py6 = take(ey,[i+1,j,k+1]);  Pz6 = take(ez,[i+1,j,k])
    # corner i,j+1,k+1
    Px7 = take(ex,[i,j+1,k+1]); Py7 = take(ey,[i,j,k+1]);    Pz7 = take(ez,[i,j+1,k])
    # corner i+1,j+1,k+1
    Px8 = take(ex,[i,j+1,k+1]); Py8 = take(ey,[i+1,j,k+1]);  Pz8 = take(ez,[i+1,j+1,k])


    nx1 = size(Px1); ny1 = size(Py1); nz1 = size(Pz1)
    #sparse.coo_matrix((V,(I,J)),shape=(4,4))
    P1 = block_diag(( sparse.coo_matrix(arange(0,nx1),Px1(:), e(nx1), nx1,nx),
                      sparse.coo_matrix(arange(0,ny1),Py1(:),e(ny1), ny1,ny),
                      sparse.coo_matrix(arange(0,nz1),Pz1(:),e(nz1), nz1,nz)))
       
    nx2 = numel(Px2); ny2 = numel(Py2); nz2 = numel(Pz2);   
    P2 = blkdiag( sparse(1:nx2,Px2(:), e(nx2), nx2,nx) , ...
              sparse(1:ny2,Py2(:),e(ny2), ny2,ny), ...
              sparse(1:nz2,Pz2(:),e(nz2), nz2,nz));
   
    nx3 = numel(Px3); ny3 = numel(Py3); nz3 = numel(Pz3);   
    P3 = blkdiag( sparse(1:nx3,Px3(:), e(nx3), nx3,nx) , ...
              sparse(1:ny3,Py3(:),e(ny3), ny3,ny), ...
              sparse(1:nz3,Pz3(:),e(nz3), nz3,nz));

    nx4 = numel(Px4); ny4 = numel(Py4); nz4 = numel(Pz4);   
    P4 = blkdiag( sparse(1:nx4,Px4(:), e(nx4), nx4,nx) , ...
              sparse(1:ny4,Py4(:), e(ny4), ny4,ny), ...
              sparse(1:nz4,Pz4(:), e(nz4), nz4,nz));
          
    nx5 = numel(Px5); ny5 = numel(Py5); nz5 = numel(Pz5);   
    P5 = blkdiag( sparse(1:nx5,Px5(:), e(nx5), nx5,nx) , ...
              sparse(1:ny5,Py5(:), e(ny5), ny5,ny), ...
              sparse(1:nz5,Pz5(:), e(nz5), nz5,nz));
          
    nx6 = numel(Px6); ny6 = numel(Py6); nz6 = numel(Pz6);   
    P6 = blkdiag( sparse(1:nx6,Px6(:), e(nx6), nx6,nx) , ...
              sparse(1:ny6,Py6(:), e(ny6), ny6,ny), ...
              sparse(1:nz6,Pz6(:), e(nz6), nz6,nz));

    nx7 = numel(Px7); ny7 = numel(Py7); nz7 = numel(Pz7);   
    P7 = blkdiag( sparse(1:nx7,Px7(:), e(nx7), nx7,nx) , ...
              sparse(1:ny7,Py7(:), e(ny7), ny7,ny), ...
              sparse(1:nz7,Pz7(:), e(nz7), nz7,nz));
          
    nx8 = numel(Px8); ny8 = numel(Py8); nz8 = numel(Pz8);   
    P8 = blkdiag( sparse(1:nx8,Px8(:), e(nx8), nx8,nx) , ...
              sparse(1:ny8,Py8(:), e(ny8), ny8,ny), ...
              sparse(1:nz8,Pz8(:), e(nz8), nz8,nz));
          
    V  = sdiag(sqrt([v(:); v(:); v(:)]));

    # generate the conductivity
    S  = [sdiag(sig(:,1)) , sdiag(sig(:,4)) , sdiag(sig(:,5)); ...
          sdiag(sig(:,4)) , sdiag(sig(:,2)) , sdiag(sig(:,6)); ...
          sdiag(sig(:,5)) , sdiag(sig(:,6))  ,  sdiag(sig(:,3))];

    # scale by the volume  
    S =  V*S*V;    
   
    M = 1/8*(P1'*S*P1 + P2'*S*P2 + P3'*S*P3 + P4'*S*P4 + ...
         P5'*S*P5 + P6'*S*P6 + P7'*S*P7 + P8'*S*P8);
 