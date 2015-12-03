import os

home_dir = 'C:\Users\dominiquef.MIRAGEOSCIENCE\Documents\GIT\SimPEG\simpegdc\simpegDCIP\Dev'

os.chdir(home_dir)


#%%
from SimPEG import np, Utils, Mesh, mkvc, SolverLU, sp
import simpegDCIP as DC
import pylab as plt
import time

# Load UBC mesh 3D
mesh = Utils.meshutils.readUBCTensorMesh('Mesh_40m.msh')

# Load model
model = Utils.meshutils.readUBCTensorModel('Synthetic.con',mesh)

#%%
# Display top section 
top = int(mesh.nCz)-1
mesh.plotSlice(model, ind=top, normal='Z', grid=True, pcolorOpts={'alpha':0.8})
ylim=(546000,546750)
xlim=(422900,423675)
# Takes two points from ginput and create survey
temp = plt.ginput(2)

# Add z coordinate
nz = mesh.vectorNz
endp = np.c_[np.asarray(temp),np.ones(2).T*nz[-1]]

# Create dipole survey receivers and plot
ab = 40
a = 20

# Evenly distribute transmitters for now and put on surface
dplen = np.sqrt( np.sum((endp[1,:] - endp[0,:])**2) ) 
dp_x = ( endp[1,0] - endp[0,0] ) / dplen
dp_y = ( endp[1,1] - endp[0,1] ) / dplen

nstn = np.floor( dplen / ab )
nrx = nstn-1

stn_x = endp[0,0] + np.cumsum( np.ones(nstn)*dp_x*ab )
stn_y = endp[0,1] + np.cumsum( np.ones(nstn)*dp_y*ab )

plt.scatter(stn_x,stn_y,s=100, c='w')

M = np.c_[stn_x-a*dp_x, stn_y-a*dp_y, np.ones(nstn).T*nz[-1]]
N = np.c_[stn_x+a*dp_x, stn_y+a*dp_y, np.ones(nstn).T*nz[-1]]

plt.scatter(M[:,0],M[:,1],s=10,c='r')
plt.scatter(N[:,0],N[:,1],s=10,c='b')

#%% Create system
#Set boundary conditions
mesh.setCellGradBC('neumann')

Div = mesh.faceDiv
Grad = mesh.cellGrad
Msig = Utils.sdiag(1./(mesh.aveF2CC.T*(1./model)))

A = Div*Msig*Grad

# Change one corner to deal with nullspace
A[0,0] = 1.

# Factor A matrix
Ainv = sp.linalg.splu(A)

#%% Forward model data
data = np.zeros( nstn*nrx )
problem = DC.ProblemDC_CC(mesh)
fig = plt.figure()




for ii in range(0, int(nstn)):
    start_time = time.time()
    
    rxloc_M = np.r_[M[0:ii,:],M[ii+1:,:]]
    rxloc_N = np.r_[N[0:ii,:],N[ii+1:,:]]
    
    
    Rx = DC.RxDipole(rxloc_M,rxloc_N)
    
    Tx = DC.SrcDipole([Rx], M[ii,:],N[ii,:])
    survey = DC.SurveyDC([Tx])

    problem.pair(survey)

    # Get the righthand side
    RHS = problem.getRHS()

    # Solve for phi
    P1 = mesh.getInterpolationMat(rxloc_M, 'CC')
    P2 = mesh.getInterpolationMat(rxloc_N, 'CC')

    phi = Ainv.solve(RHS) 
    d = P1*phi - P2*phi   
    
    data[(nrx*(ii)):nrx+(nrx*(ii))] = d.T#survey.dpred(model)
    
    # Plot pseudo section along line
    txmidx = endp[0,0] - np.mean(np.c_[M[ii,0],N[ii,0]])
    rxmidx = endp[0,0] - np.mean( np.c_[rxloc_M[:,0], rxloc_N[:,0]], axis=1 )
    
    txmidy = endp[0,1] - np.mean(np.c_[M[ii,1],N[ii,1]])
    rxmidy = endp[0,1] - np.mean( np.c_[rxloc_M[:,1], rxloc_N[:,1]], axis=1 )
    
    rxmid = np.sqrt(rxmidx**2 + rxmidy**2)
    txmid = np.sqrt(txmidx**2 + txmidy**2)
    
    midp = ( rxmid + txmid )/2
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    plt.scatter(midp,-np.abs(txmid-midp),s=50,c=data[(nrx*(ii)):nrx+(nrx*(ii))])


