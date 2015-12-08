import os

home_dir = 'C:\Users\dominiquef.MIRAGEOSCIENCE\Documents\GIT\SimPEG\simpegdc\simpegDCIP\Dev'

os.chdir(home_dir)


#%%
from SimPEG import np, Utils, Mesh, mkvc, SolverLU, sp
import simpegDCIP as DC
import pylab as plt
import time
from scipy.interpolate import griddata
import numpy.matlib as npm

#from scipy.linalg import solve_banded

# Load UBC mesh 3D
mesh = Utils.meshutils.readUBCTensorMesh('Mesh_20m.msh')
#mesh = Utils.meshutils.readUBCTensorMesh('Mesh_40m.msh')

# Load model
model = Utils.meshutils.readUBCTensorModel('MtIsa_3D.con',mesh)
#model = Utils.meshutils.readUBCTensorModel('Synthetic.con',mesh)

#%% Create system
#Set boundary conditions
mesh.setCellGradBC('neumann')

Div = mesh.faceDiv
Grad = mesh.cellGrad
Msig = Utils.sdiag(1./(mesh.aveF2CC.T*(1./model)))

A = Div*Msig*Grad

# Change one corner to deal with nullspace
A[0,0] = 1.
A = sp.csc_matrix(A)

start_time = time.time()

# Factor A matrix
Ainv = sp.linalg.splu(A)

print("LU DECOMP--- %s seconds ---" % (time.time() - start_time))

#%%
# Display top section 
top = int(mesh.nCz)-1
mesh.plotSlice(model, ind=top, normal='Z', grid=True, pcolorOpts={'alpha':0.8})


# Takes two points from ginput and create survey
temp = plt.ginput(2, timeout = 0)

# Add z coordinate
nz = mesh.vectorNz
endp = np.c_[np.asarray(temp),np.ones(2).T*nz[-1]]

# Create dipole survey receivers and plot
a = 40
n = 8

# Evenly distribute transmitters for now and put on surface
dplen = np.sqrt( np.sum((endp[1,:] - endp[0,:])**2) ) 
dp_x = ( endp[1,0] - endp[0,0] ) / dplen
dp_y = ( endp[1,1] - endp[0,1] ) / dplen

nstn = np.floor( dplen / a )
nrx = nstn-1

stn_x = endp[0,0] + np.cumsum( np.ones(nstn)*dp_x*a )
stn_y = endp[0,1] + np.cumsum( np.ones(nstn)*dp_y*a )

plt.scatter(stn_x,stn_y,s=100, c='w')

M = np.c_[stn_x-a*dp_x/2, stn_y-a*dp_y/2, np.ones(nstn).T*nz[-1]]
N = np.c_[stn_x+a*dp_x/2, stn_y+a*dp_y/2, np.ones(nstn).T*nz[-1]]

plt.scatter(M[:,0],M[:,1],s=10,c='r')
plt.scatter(N[:,0],N[:,1],s=10,c='b')



#%% Forward model data
data = []#np.zeros( nstn*nrx )
problem = DC.ProblemDC_CC(mesh)
fig = plt.figure()
    
for ii in range(0, int(nstn)-2):
    start_time = time.time()
    
    # Select dipole locations for receiver: n || end of line
    
    idx = int( np.min([ii+n+1,nstn+1]) )
    rxloc_M = M[ii+2:ii+n+1,:]#np.r_[M[0:ii,:],M[ii+1:,:]]
    rxloc_N = N[ii+2:ii+n+1,:]#np.r_[N[0:ii,:],N[ii+1:,:]]
    
    nrx = rxloc_M.shape[0]
    
    Rx = DC.RxDipole(rxloc_M,rxloc_N)
    
    Tx = DC.SrcDipole([Rx], M[ii,:],N[ii,:])
    survey = DC.SurveyDC([Tx])

    problem.pair(survey)

    # Get the righthand side
    RHS_v1 = problem.getRHS()

    inds = Utils.closestPoints(mesh, np.c_[M[ii,:],N[ii,:]].T)
    RHS = mesh.getInterpolationMat(np.c_[M[ii,:],N[ii,:]].T, 'CC').T*( [-1,1] / mesh.vol[inds] )
    
    
    # Solve for phi
    P1 = mesh.getInterpolationMat(rxloc_M, 'CC')
    P2 = mesh.getInterpolationMat(rxloc_N, 'CC')

    #Direct Solve
    phi = Ainv.solve(RHS) 
    
    # Iterative Solve
    #Ainvb = sp.linalg.bicgstab(A,RHS, tol=1e-5)
    #phi = mkvc(Ainvb[0])
    
    # Compute potential at each electrode
    d = (P1*phi - P2*phi)*np.pi     
    
    # Convert 3D location to distance along survey line for 2D
    # Plot pseudo section along line
    
    rP1 = np.sqrt( np.sum( ( endp[0,:] - M[ii,:] )**2 , axis=0))
    rP2 = np.sqrt( np.sum( ( endp[0,:] - N[ii,:] )**2 , axis=0))
    rC1 = np.sqrt( np.sum( ( npm.repmat(endp[0,:],nrx, 1) - rxloc_M )**2 , axis=1))
    rC2 = np.sqrt( np.sum( ( npm.repmat(endp[0,:],nrx, 1) - rxloc_N )**2 , axis=1))

    if ii == 0:
        data = np.c_[np.ones(nrx)*rP1, np.ones(nrx)*rP2, rC1, rC2, mkvc(d), np.ones(nrx)*1e-2]
        
    else:
        temp = np.c_[np.ones(nrx)*rP1, np.ones(nrx)*rP2, rC1, rC2, mkvc(d), np.ones(nrx)*1e-2]
        data = np.r_[data,temp]#survey.dpred(model)
        
    
    print("--- %s seconds ---" % (time.time() - start_time))
         

    
    # Write data to UBC-2D format 
    #temp = np.c_[np.ones(nrx)*txmid-a/2, np.ones(nrx)*txmid+a/2,
    #    rxmid-a/2, rxmid+a/2, 
     #   mkvc(d) , np.ones(nrx)*1e-2]
        

# Modification for 2D problem
data[:,0:4] = data[:,0:4] + endp[0,0]

fid = open(home_dir + '\FWR_data.dat','w')
fid.write('SIMPEG FORWARD\n')   
np.savetxt(fid, data, fmt='%e',delimiter=' ',newline='\n')
fid.close()

fid = open(home_dir + '\OBS_LOC.dat','w')
fid.write('SIMPEG FORWARD\n')   
np.savetxt(fid, data[:,0:4], fmt='%e',delimiter=' ',newline='\n')
fid.close()

#%% Plot pseudo section
   
# Get distances between each poles
rC1P1 = data[:,0] - data[:,2] 
rC2P1 = data[:,0] - data[:,3]
rC1P2 = data[:,1] - data[:,2]
rC2P2 = data[:,1] - data[:,3]

# Compute apparent resistivity
rho = data[:,4] * 2*np.pi / ( 1/rC1P1 - 1/rC2P1 - 1/rC1P2 + 1/rC2P2 )

Cmid = (data[:,0] + data[:,1])/2
Pmid = (data[:,2] + data[:,3])/2

midp = ( Cmid + Pmid )/2
midz = -np.abs(Cmid-Pmid)

# Grid points
grid_x, grid_z = np.mgrid[np.min(midp):np.max(midp), np.min(midz):np.max(midz)]
grid_rho = griddata(np.c_[midp,midz], np.log10(abs(1/rho.T)), (grid_x, grid_z), method='linear')
plt.imshow(grid_rho.T, extent = (np.min(midp),np.max(midp),np.min(midz),np.max(midz)), origin='lower')
plt.colorbar()

# Plot apparent resistivity
plt.scatter(midp,midz,s=50,c=np.log10(abs(1/rho.T)))

#%% Export 2D mesh from section
fid = open(home_dir + '\Mesh_2D.msh','w')
fid.write('%i\n'% mesh.nCx)
fid.write('%f %f 1\n'% (mesh.vectorNx[0],mesh.vectorNx[1]))  
np.savetxt(fid, np.c_[mesh.vectorNx[2:],np.ones(mesh.nCx-1)], fmt='\t %e %i',delimiter=' ',newline='\n')
fid.write('\n')
fid.write('%i\n'% mesh.nCz)
fid.write('%f %f 1\n'%( -mesh.vectorNz[-1],-mesh.vectorNz[-2]))   
np.savetxt(fid, np.c_[-mesh.vectorNz[-3::-1],np.ones(mesh.nCz-1)], fmt='\t %e %i',delimiter=' ',newline='\n')
fid.close()

# Grab slice of model
m = np.reshape(model, (mesh.nCz, mesh.nCy, mesh.nCx))
m2D = m[::-1,9,:]
plt.figure()
plt.imshow(m2D)

fid = open(home_dir + '\MtIsa_2D.con','w')
fid.write('%i %i\n'% (mesh.nCx,mesh.nCz))
np.savetxt(fid, mkvc(m2D.T), fmt='%e',delimiter=' ',newline='\n')
fid.close()