import numpy as np, SimPEG as simpeg
from .MT1Danalytic import getEHfields
from scipy.constants import mu_0

def get1DEfields(m1d,sigma,freq,sourceAmp=1.0):
    """Function to get 1D electrical fields"""

    # Get the gradient
    G = m1d.nodalGrad
    # Mass matrices
    # Magnetic permeability
    Mmu = simpeg.Utils.sdiag(m1d.vol*(1.0/mu_0))
    # Conductivity
    Msig = m1d.getFaceInnerProduct(sigma)
    # Set up the solution matrix
    A = G.T*Mmu*G + 1j*2.*np.pi*freq*Msig
    # Define the inner part of the solution matrix
    Aii = A[1:-1,1:-1]
    # Define the outer part of the solution matrix
    Aio = A[1:-1,[0,-1]]

    # Set the boundary conditions
    Ed, Eu, Hd, Hu = getEHfields(m1d,sigma,freq,m1d.vectorNx)
    Etot = (Ed + Eu)
    if sourceAmp is not None:
        Etot = ((Etot/Etot[-1])*sourceAmp) # Scale the fields to be equal to sourceAmp at the top
    ## Note: The analytic solution is derived with e^iwt
    bc = np.r_[Etot[0],Etot[-1]]
    # The right hand side
    rhs = Aio*bc
    # Solve the system
    Aii_inv = simpeg.Solver(Aii)
    eii = Aii_inv*rhs
    # Assign the boundary conditions
    e = np.r_[bc[0],eii,bc[1]]
    # Return the electrical fields
    return e


if __name__ == '__main__':

    hz = [(100.,18)]
    M = simpeg.Mesh.TensorMesh([hz],'C')
    sig = np.zeros(M.nC) + 1e-8
    sig[M.vectorCCx<=0] = sigHalf
