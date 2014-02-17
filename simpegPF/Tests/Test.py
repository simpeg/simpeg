
from SimPEG.Utils.sputils import kron3, speye, sdiag, spzeros
import numpy as np
import scipy.sparse as sp

def ddxFaceDivBC(n, bc):

  ij   = (np.array([0, n-1]),np.array([0, 1]))
  vals = np.zeros(2)

  # Set the first side
  if(bc[0] == 'dirichlet'):
    vals[0] = 0
  elif(bc[0] == 'neumann'):
    vals[0] = -1

  # Set the second side
  if(bc[1] == 'dirichlet'):
    vals[1] = 0
  elif(bc[1] == 'neumann'):
    vals[1] = 1
  D = sp.csr_matrix((vals, ij), shape=(n,2))
  return D


def faceDivBC(mesh, BC, ind):
  """

  The facd divergence boundary condtion matrix

  """
  # The number of cell centers in each direction
  n = mesh.vnC
  # Compute faceDivergence operator on faces
  if(mesh.dim == 1):
    D = ddxFaceDivBC(n[0], BC[0])
  elif(mesh.dim == 2):
    D1 = sp.kron(speye(n[1]), ddxFaceDivBC(n[0], BC[0]))
    D2 = sp.kron(ddxFaceDivBC(n[1], BC[1]), speye(n[0]))
    D = sp.hstack((D1, D2), format="csr")
  elif(mesh.dim == 3):
    D1 = kron3(speye(n[2]), speye(n[1]), ddxFaceDivBC(n[0], BC[0]))
    D2 = kron3(speye(n[2]), ddxFaceDivBC(n[1], BC[1]), speye(n[0]))
    D3 = kron3(ddxFaceDivBC(n[2], BC[2]), speye(n[1]), speye(n[0]))
    D = sp.hstack((D1, D2, D3), format="csr")
  # Compute areas of cell faces & volumes
  S = mesh.area[ind]
  V = mesh.vol
  mesh._faceDiv = sdiag(1/V)*D*sdiag(S)

  return mesh._faceDiv


def faceBCind(mesh):
  """
  Find indices of boundary faces in each direction

  """
  if(mesh.dim==1):
    indxd = (mesh.gridFx==min(mesh.gridFx))
    indxu = (mesh.gridFx==max(mesh.gridFx))
    return indxd, indxu
  elif(mesh.dim==2):
    indxd = (mesh.gridFx[:,0]==min(mesh.gridFx[:,0]))
    indxu = (mesh.gridFx[:,0]==max(mesh.gridFx[:,0]))
    indyd = (mesh.gridFy[:,1]==min(mesh.gridFy[:,1]))
    indyu = (mesh.gridFy[:,1]==max(mesh.gridFy[:,1]))
    return indxd, indxu, indyd, indyu
  elif(mesh.dim==3):
    indxd = (mesh.gridFx[:,0]==min(mesh.gridFx[:,0]))
    indxu = (mesh.gridFx[:,0]==max(mesh.gridFx[:,0]))
    indyd = (mesh.gridFy[:,1]==min(mesh.gridFy[:,1]))
    indyu = (mesh.gridFy[:,1]==max(mesh.gridFy[:,1]))
    indzd = (mesh.gridFz[:,2]==min(mesh.gridFz[:,2]))
    indzu = (mesh.gridFz[:,2]==max(mesh.gridFz[:,2]))
    return indxd, indxu, indyd, indyu, indzd, indzu






def faceDivProj(mesh, flag):
    """"
    Construct divergence operator (face-stg to cell-centres).

    """
    # The number of cell centers in each direction
    n = mesh.vnC
    # Compute faceDivergence operator on faces
    if (flag=='in'):
        if(mesh.dim == 1):
            Pin = ddxPin(n[0])
        elif(mesh.dim == 2):
            P1 = sp.kron(speye(n[1]), ddxPin(n[0]))
            P2 = sp.kron(ddxPin(n[1]), speye(n[0]))
            Pin = sp.block_diag((P1, P2), format="csr")
        elif(mesh.dim == 3):
            P1 = kron3(speye(n[2]), speye(n[1]), ddxPin(n[0]))
            P2 = kron3(speye(n[2]), ddxPin(n[1]), speye(n[0]))
            P3 = kron3(ddxPin(n[2]), speye(n[1]), speye(n[0]))
            Pin = sp.block_diag((P1, P2, P3), format="csr")
        # Compute areas of cell faces & volumes
        return Pin
    elif(flag=='out'):
        if(mesh.dim == 1):
            Pout = ddxPout(n[0])
        elif(mesh.dim == 2):
            P1 = sp.kron(speye(n[1]), ddxPout(n[0]))
            P2 = sp.kron(ddxPout(n[1]), speye(n[0]))
            Pout = sp.block_diag((P1, P2), format="csr")
        elif(mesh.dim == 3):
            P1 = kron3(speye(n[2]), speye(n[1]), ddxPout(n[0]))
            P2 = kron3(speye(n[2]), ddxPout(n[1]), speye(n[0]))
            P3 = kron3(ddxPout(n[2]), speye(n[1]), speye(n[0]))
            Pout = sp.block_diag((P1, P2, P3), format="csr")
        # Compute areas of cell faces & volumes
        return Pout

def ddxPin(n):
    p0 =spzeros(n-1, 1)
    P = sdiag(np.ones(n-1))
    P = sp.hstack((p0, P, p0))

    return P

def ddxPout(n):
    ij   = (np.array([0, 1]),np.array([0, n]))
    vals = np.ones(2)
    P = sp.csr_matrix((vals, ij), shape=(2,n+1))
    return P
