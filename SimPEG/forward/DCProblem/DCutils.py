import numpy as np
import scipy.sparse as sp

def genTxRxmat(nelec, spacelec, surfloc, elecini, mesh):
    """ Generate projection matrix (Q) and """
    elecend = 0.5+spacelec*(nelec-1)
    elecLocR = np.linspace(elecini, elecend, nelec)
    elecLocT = elecLocR+1
    nrx = nelec-1
    ntx = nelec-1
    q = np.zeros((mesh.nC, ntx))
    Q = np.zeros((mesh.nC, nrx))

    for i in range(nrx):

        rxind1 = np.argwhere((mesh.gridCC[:,0]==surfloc) & (mesh.gridCC[:,1]==elecLocR[i]))
        rxind2 = np.argwhere((mesh.gridCC[:,0]==surfloc) & (mesh.gridCC[:,1]==elecLocR[i+1]))

        txind1 = np.argwhere((mesh.gridCC[:,0]==surfloc) & (mesh.gridCC[:,1]==elecLocT[i]))
        txind2 = np.argwhere((mesh.gridCC[:,0]==surfloc) & (mesh.gridCC[:,1]==elecLocT[i+1]))

        q[txind1,i] = 1
        q[txind2,i] = -1
        Q[rxind1,i] = 1
        Q[rxind2,i] = -1

    Q = sp.csr_matrix(Q)
    rxmidLoc = (elecLocR[0:nelec-1]+elecLocR[1:nelec])*0.5
    return q, Q, rxmidLoc
