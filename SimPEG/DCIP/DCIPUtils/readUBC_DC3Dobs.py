from SimPEG import np
import SimPEG.DCIP as DCIP

def genActiveindfromTopo(mesh, topo):        
    if mesh.dim==3:
        nCxy = mesh.nCx*mesh.nCy
        Zcc = mesh.gridCC[:,2].reshape((nCxy, mesh.nCz), order='F')
        Ftopo = NearestNDInterpolator(topo[:,:2], topo[:,2])
        XY = Utils.ndgrid(mesh.vectorCCx, mesh.vectorCCy)
        XY.shape
        topo = Ftopo(XY)
        actind = []                    
        for ixy in range(nCxy):
            actind.append(topo[ixy] <= Zcc[ixy,:])
    else:
        raise NotImplementedError("Only 3D is working")

    return actind

def gettopoCC(mesh, airind):
	mesh2D = Mesh.TensorMesh([mesh.hx, mesh.hy], mesh.x0[:2])
	zc = mesh.gridCC[:,2]
	AIRIND = airind.reshape((mesh.vnC[0]*mesh.vnC[1],mesh.vnC[2]), order='F')
	ZC = zc.reshape((mesh.vnC[0]*mesh.vnC[1], mesh.vnC[2]), order='F')
	topo = np.zeros(ZC.shape[0])
	topoCC = np.zeros(ZC.shape[0])
	for i in range(ZC.shape[0]):
	    ind  = np.argmax(ZC[i,:][~AIRIND[i,:]])
	    topo[i] = ZC[i,:][~AIRIND[i,:]].max() + mesh.hz[~AIRIND[i,:]][ind]*0.5
	    topoCC[i] = ZC[i,:][~AIRIND[i,:]].max() 
	XY = Utils.ndgrid(mesh.vectorCCx, mesh.vectorCCy)
	return topoCC

def readUBC_DC3Dobs(filename,mesh,topo,probType="CC"):
	text_file = open("./mill_dc.dat", "r")
	lines = text_file.readlines()
	text_file.close()
	SRC = []
	DATA = []
	srcLists = []
	isrc = 0
	airind = genActiveindfromTopo(mesh, topo)
	topoCC, topoCCind = gettopoCC(mesh, airind)	

	for line in lines:
	    if "!" in line.split(): continue
	    elif line == '\n': continue
	    elif line == ' \n': continue
	    temp =  map(float, line.split())
	    # Read a line for the current electrode
	    if len(temp) == 5: # SRC: Only X and Y are provided (assume no topography)
	        #TODO consider topography and assign the closest cell center in the earth                        
	        if isrc == 0:
	            DATA_temp = []
	        else:
	            DATA.append(np.asarray(DATA_temp))
	            DATA_temp = []
	            indM = Utils.closestPoints(mesh2D, DATA[isrc-1][:,1:3])
	            indN = Utils.closestPoints(mesh2D, DATA[isrc-1][:,3:5])
	            rx = DC.RxDipole(np.c_[DATA[isrc-1][:,1:3], topoCC[indM]], np.c_[DATA[isrc-1][:,3:5], topoCC[indN]])        
	            temp = np.asarray(temp)
	            if [SRC[isrc-1][0], SRC[isrc-1][1]] == [SRC[isrc-1][2], SRC[isrc-1][3]]:
	                indA = Utils.closestPoints(mesh2D, [SRC[isrc-1][0], SRC[isrc-1][1]])
	                tx = DC.SrcDipole([rx], [SRC[isrc-1][0], SRC[isrc-1][1], topoCC[indA]],[mesh.vectorCCx.max(), mesh.vectorCCy.max(), topoCC[-1]])
	            else:
	                indA = Utils.closestPoints(mesh2D, [SRC[isrc-1][0], SRC[isrc-1][1]])
	                indB = Utils.closestPoints(mesh2D, [SRC[isrc-1][2], SRC[isrc-1][3]])
	                tx = DC.SrcDipole([rx], [SRC[isrc-1][0], SRC[isrc-1][1], topoCC[indA]],[SRC[isrc-1][2], SRC[isrc-1][3], topoCC[indB]])
	            srcLists.append(tx)
	        SRC.append(temp)
	        isrc += 1
	    elif len(temp) == 7: # SRC: X, Y and Z are provided
	        SRC.append(temp)
	        isrc += 1
	    elif len(temp) == 6: # 
	        DATA_temp.append(np.r_[isrc, np.asarray(temp)])
	    elif len(temp) > 7:
	        DATA_temp.append(np.r_[isrc, np.asarray(temp)])
	        
	DATA.append(np.asarray(DATA_temp))
	DATA_temp = []
	indM = Utils.closestPoints(mesh2D, DATA[isrc-1][:,1:3])
	indN = Utils.closestPoints(mesh2D, DATA[isrc-1][:,3:5])
	rx = DCIP.RxDipole(np.c_[DATA[isrc-1][:,1:3], topoCC[indM]], np.c_[DATA[isrc-1][:,3:5], topoCC[indN]])        
	temp = np.asarray(temp)
	if [SRC[isrc-1][0], SRC[isrc-1][1]] == [SRC[isrc-1][2], SRC[isrc-1][3]]:
	    indA = Utils.closestPoints(mesh2D, [SRC[isrc-1][0], SRC[isrc-1][1]])
	    tx = DCIP.SrcDipole([rx], [SRC[isrc-1][0], SRC[isrc-1][1], topoCC[indA]],[mesh.vectorCCx.max(), mesh.vectorCCy.max(), topoCC[-1]])
	else:
	    indA = Utils.closestPoints(mesh2D, [SRC[isrc-1][0], SRC[isrc-1][1]])
	    indB = Utils.closestPoints(mesh2D, [SRC[isrc-1][2], SRC[isrc-1][3]])
	    tx = DCIP.SrcDipole([rx], [SRC[isrc-1][0], SRC[isrc-1][1], topoCC[indA]],[SRC[isrc-1][2], SRC[isrc-1][3], topoCC[indB]])
	srcLists.append(tx)	        
	text_file.close()
	survey = DCIP.SurveyDC(srcLists)

	# Do we need this?
	SRC = np.asarray(SRC)
	DATA = np.vstack(DATA)

	return {'DCsurvey':survey, 'airind':airind, 'topoCC':topoCC}
