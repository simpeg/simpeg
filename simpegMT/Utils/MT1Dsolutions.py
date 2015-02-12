import numpy as np, SimPEG as simpeg
from MT1Danalytic import getEHfields
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
    A = G.T*Mmu*G - 1j*2.*np.pi*freq*Msig
    # Define the inner part of the solution matrix
    Aii = A[1:-1,1:-1]
    # Define the outer part of the solution matrix
    Aio = A[1:-1,[0,-1]]

    # Set the boundary conditions
    Ed_low, Eu_low, Hd_low, Hu_low = getEHfields(m1d,sigma,freq,np.array([m1d.vectorNx[0]]))
    Etot_low = Ed_low + Eu_low
    ## Note: need to use conjugate of the analytic solution. It is derived with e^iwt
    bc = np.r_[Etot_low.conj(),sourceAmp]
    # The right hand side
    rhs = -Aio*bc
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
