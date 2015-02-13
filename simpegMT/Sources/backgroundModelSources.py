def homo1DModelSource(mesh,bgMod):
	'''
		Function that calculates and return backround fields

	'''

	# import
    from simpegMT.Utils import get1DEfields
    # Get a 1d solution for a halfspace background
    mesh1d = simpeg.Mesh.TensorMesh([M.hz],np.array([M.x0[2]]))
    e0_1d = get1DEfields(mesh1d,M.r(sigBG,'CC','CC','M')[0,0,:],freq)
    # Setup x (east) polarization (_x)
    ex_px = np.zeros(M.vnEx,dtype=complex)
    ey_px = np.zeros((M.nEy,1),dtype=complex)
    ez_px = np.zeros((M.nEz,1),dtype=complex)
    # Assign the source to ex_x
    for i in arange(M.vnEx[0]):
        for j in arange(M.vnEx[2]):
            ex_px[i,j,:] = e0_1d
    eBG_px = np.vstack((simpeg.Utils.mkvc(M.r(ex_px,'Ex','Ex','V'),2),ey_px,ez_px))
    # Setup y (north) polarization (_py)
    ex_py = np.zeros(M.nEx, dtype='complex128')
	ey_py = np.zeros((M.vnEy), dtype='complex128')
	ez_py = np.zeros(M.nEz, dtype='complex128')
	# Assign the source to ey_py
	for i in arange(M.vnEy[0]):
	    for j in arange(M.vnEy[1]):
	        ey_py[i,j,:] = e0_1d 

    eBG_py = np.r_[ex_py,simpeg.Utils.mkvc(ey_py),ez_py]
	