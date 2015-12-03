import os

home_dir = 'C:\Users\dominiquef.MIRAGEOSCIENCE\Documents\GIT\SimPEG\simpegdc\simpegDCIP\Dev'

os.chdir(home_dir)


#%%
from SimPEG import np, Utils, Mesh, mkvc, SolverLU
import simpegDCIP as DC
import pylab as plt

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
nrx = 10
ab = 40
a = 20

# Evenly distribute transmitters for now and put on surface
dplen = np.sqrt( np.sum((endp[1,:] - endp[0,:])**2) ) 
dp_x = ( endp[1,0] - endp[0,0] ) / dplen
dp_y = ( endp[1,1] - endp[0,1] ) / dplen

nstn = np.floor( dplen / ab )

stn_x = endp[0,0] + np.cumsum( np.ones(nstn)*dp_x*ab )
stn_y = endp[0,1] + np.cumsum( np.ones(nstn)*dp_y*ab )

plt.scatter(stn_x,stn_y,s=100, c='w')

M = np.c_[stn_x-a*dp_x, stn_y-a*dp_y, np.ones(nstn).T*nz[-1]]
N = np.c_[stn_x+a*dp_x, stn_y+a*dp_y, np.ones(nstn).T*nz[-1]]

plt.scatter(M[:,0],M[:,1],s=10,c='r')
plt.scatter(N[:,0],N[:,1],s=10,c='b')

#%% Create inversion parameter

Rx = DC.RxDipole(M,N)
Tx = DC.SrcDipole([Rx], tx[0,:],tx[1,:])
survey = DC.SurveyDC([Tx])

problem = DC.ProblemDC_CC(mesh)
problem.pair(survey)

problem.Solver = SolverLU

data = survey.dpred(model)

#Set boundary conditions
mesh.setCellGradBC('neumann')

Div = mesh.faceDiv
Grad = mesh.cellGradBC
Msig = Utils.sdiag(1./(mesh.aveF2CC.T*(1./model)))

A = Div*Msig*Grad

# Change one corner to deal with nullspace
A[0,0] = 1.

# Get the righthand side
RHS = problem.getRHS

# Solve for phi
phi = SolverLU(A)*-RHS

