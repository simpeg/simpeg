import SimPEG as simpeg, numpy as np

def homo1DModelSource(mesh,freq,m_back):
    '''
        Function that calculates and return background fields for a 3D mesh and model.
        The calculuations use 1D field solution for a vertical slice throught model (south-western most column),
        which is assigned at the fields everywhere for the respective polarizations.2

        :param Simpeg mesh object mesh: Holds information on the discretization
        :param float freq: The frequency to solve at
        :param np.array m_back: Background model of conductivity to base the calculations on.
        :rtype: numpy.ndarray (mesh.nE,2)
        :return: eBG_bp, E fields for the background model at both polarizations.

    '''

    # import
    from SimPEG.MT.Utils import get1DEfields
    # Get a 1d solution for a halfspace background
    mesh1d = simpeg.Mesh.TensorMesh([mesh.hz],np.array([mesh.x0[2]]))
    # Note: Everything is using e^iwt
    e0_1d = get1DEfields(mesh1d,mesh.r(m_back,'CC','CC','M')[0,0,:],freq)
    # Setup x (east) polarization (_x)
    ex_px = np.zeros(mesh.vnEx,dtype=complex)
    ey_px = np.zeros((mesh.nEy,1),dtype=complex)
    ez_px = np.zeros((mesh.nEz,1),dtype=complex)
    # Assign the source to ex_x
    for i in np.arange(mesh.vnEx[0]):
        for j in np.arange(mesh.vnEx[1]):
            ex_px[i,j,:] = -e0_1d
    eBG_px = np.vstack((simpeg.Utils.mkvc(ex_px,2),ey_px,ez_px))
    # Setup y (north) polarization (_py)
    ex_py = np.zeros((mesh.nEx,1), dtype='complex128')
    ey_py = np.zeros(mesh.vnEy, dtype='complex128')
    ez_py = np.zeros((mesh.nEz,1), dtype='complex128')
    # Assign the source to ey_py

    for i in np.arange(mesh.vnEy[0]):
        for j in np.arange(mesh.vnEy[1]):
            ey_py[i,j,:] = e0_1d
    # ey_py[1:-1,1:-1,1:-1] = 0
    eBG_py = np.vstack((ex_py,simpeg.Utils.mkvc(ey_py,2),ez_py))

    # Return the electric fields
    eBG_bp = np.hstack((eBG_px,eBG_py))
    return eBG_bp
