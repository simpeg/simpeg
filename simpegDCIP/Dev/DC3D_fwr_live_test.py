import os

home_dir = 'C:\Users\dominiquef.MIRAGEOSCIENCE\Documents\GIT\SimPEG\simpegdc\simpegDCIP\Dev'

#os.chdir(home_dir)


#%%
from SimPEG import np, Utils, Mesh, mkvc, SolverLU, sp
import simpegDCIP as DC
import pylab as plt
import time
from scipy.interpolate import griddata
import numpy.matlib as npm
from readUBC_DC3Dobs import readUBC_DC3Dobs
from writeUBC_DC3Dobs import writeUBC_DC3Dobs
import scipy.interpolate as interpolation

#from scipy.linalg import solve_banded

# Load UBC mesh 3D
#mesh = Utils.meshutils.readUBCTensorMesh(home_dir + '\Mesh_20m.msh')
mesh = Utils.meshutils.readUBCTensorMesh('Mesh_40m.msh')

# Load model
#model = Utils.meshutils.readUBCTensorModel(home_dir + '\MtIsa_3D.con',mesh)
model = Utils.meshutils.readUBCTensorModel('Synthetic.con',mesh)

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

#%% Create survey
# Display top section 
top = int(mesh.nCz)-1
mesh.plotSlice(model, ind=top, normal='Z', grid=True, pcolorOpts={'alpha':0.8})

# Add z coordinate
nz = mesh.vectorNz

# Takes two points from ginput and create survey
temp = plt.ginput(2, timeout = 0)
endl = np.c_[np.asarray(temp),np.ones(2).T*nz[-1]]


#endl = np.c_[np.c_[[mesh.vectorCCx[21],mesh.vectorCCx[-21]],[mesh.vectorCCy[10],mesh.vectorCCy[10]]],np.ones(2).T*nz[-1]]

# Create dipole survey receivers and plot
a = 40
n = 8

# Evenly distribute transmitters for now and put on surface
dl_len = np.sqrt( np.sum((endl[1,:] - endl[0,:])**2) ) 
dl_x = ( endl[1,0] - endl[0,0] ) / dl_len
dl_y = ( endl[1,1] - endl[0,1] ) / dl_len
azm =  np.arctan(dl_y/dl_x)

nstn = np.floor( dl_len / a )
nrx = nstn-1

# Create dipole center location
stn_x = endl[0,0] + np.cumsum( np.ones(nstn)*dl_x*a )
stn_y = endl[0,1] + np.cumsum( np.ones(nstn)*dl_y*a )

# Create line of pole locations
M = np.c_[stn_x-a*dl_x/2, stn_y-a*dl_y/2, np.ones(nstn).T*nz[-1]]
N = np.c_[stn_x+a*dl_x/2, stn_y+a*dl_y/2, np.ones(nstn).T*nz[-1]]

Tx = []
Rx = []

for ii in range(0, int(nstn)-2):
    
    Tx.append(np.c_[M[ii,:],N[ii,:]])
    Rx.append(np.c_[M[ii+2:ii+n+1,:],N[ii+2:ii+n+1,:]])


# Plot stations along line   
#plt.scatter(stn_x,stn_y,s=100, c='w')



plt.scatter(M[:,0],M[:,1],s=10,c='r')
plt.scatter(N[:,0],N[:,1],s=10,c='b')



#%% Forward model data
data = []#np.zeros( nstn*nrx )
unct = []
problem = DC.ProblemDC_CC(mesh)
    
for ii in range(len(Tx)):
    start_time = time.time()
    
    # Select dipole locations for receiver: n || end of line
    
    idx = int( np.min([ii+n+1,nstn+1]) )
    rxloc_M = np.asarray(Rx[ii][:,0:3])#np.r_[M[0:ii,:],M[ii+1:,:]]
    rxloc_N = np.asarray(Rx[ii][:,3:])#np.r_[N[0:ii,:],N[ii+1:,:]]
    
    
    nrx = rxloc_M.shape[0]
    
    #Rx = DC.RxDipole(rxloc_M,rxloc_N)
    
    #Tx = DC.SrcDipole([Rx], M[ii,:],N[ii,:])
    #survey = DC.SurveyDC([Tx])

    #problem.pair(survey)

    # Get the righthand side
    #RHS_v1 = problem.getRHS()

    inds = Utils.closestPoints(mesh, np.asarray(Tx[ii]).T )
    RHS = mesh.getInterpolationMat(np.asarray(Tx[ii]).T, 'CC').T*( [-1,1] / mesh.vol[inds] )
    
    
    # Solve for phi
    P1 = mesh.getInterpolationMat(rxloc_M, 'CC')
    P2 = mesh.getInterpolationMat(rxloc_N, 'CC')

    #Direct Solve
    phi = Ainv.solve(RHS) 
    
    # Iterative Solve
    #Ainvb = sp.linalg.bicgstab(A,RHS, tol=1e-5)
    #phi = mkvc(Ainvb[0])
    
    # Compute potential at each electrode
    data.append((P1*phi - P2*phi)*np.pi)     
    unct.append(np.ones(nrx))
    # Convert 3D location to distance along survey line for 2D and plot pseudo
    # section along line
    


    #data.append(np.c_[np.ones(nrx)*rP1, np.ones(nrx)*rP2, rC1, rC2, mkvc(d), np.ones(nrx)*1e-2])
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    #fid.writelines("%e " % ii for ii in np.r_[M[ii,:],N[ii,:]] )
    #fid.write('%i\n'% nrx)
    #np.savetxt(fid, np.c_[rxloc_M,rxloc_N,mkvc(d)], fmt='%e',delimiter=' ',newline='\n')  


    # Write data to UBC-2D format 
    #temp = np.c_[np.ones(nrx)*txmid-a/2, np.ones(nrx)*txmid+a/2,
    #    rxmid-a/2, rxmid+a/2, 
     #   mkvc(d) , np.ones(nrx)*1e-2]

writeUBC_DC3Dobs(home_dir+'\FWR_data3D.dat',Tx,Rx,data,unct)     

# Create a 2D mesh along axis of end points and keep z-discretization
dx = np.min( [ np.min(mesh.hx), np.min(mesh.hy) ])
nc = np.ceil(dl_len/dx)

h1 = [(dx, 10, -1.3), (dx, nc), (dx, 10, 1.3)]
mesh2d = Mesh.TensorMesh([h1, mesh.hz], x0='CN')

# Create array of points for 2D mesh in space
x2d = mesh2d.vectorCCx + dx*nc/2 - dx/2
xx = endl[0,0] + x2d * np.cos(azm)
yy = endl[0,1] + x2d * np.sin(azm)
zz = mesh2d.vectorCCz

plt.scatter(xx,yy,s=20,c='y')


F = interpolation.NearestNDInterpolator(mesh.gridCC,model)
m2D = F(mesh2d.gridCC)

#data[:,0:4] = data[:,0:4] + endl[0,0]
fid = open(home_dir + '\FWR_data.dat','w')
fid.write('SIMPEG FORWARD\n')  

# Change coordinate system to distance along line

for ii in range(len(Tx)):
    
 
    rP1 = np.sqrt( np.sum( ( endl[0,:] - Tx[ii][:,0] )**2 , axis=0))
    rP2 = np.sqrt( np.sum( ( endl[0,:] - Tx[ii][:,1] )**2 , axis=0))
    rC1 = np.sqrt( np.sum( ( npm.repmat(endl[0,:],nrx, 1) - rxloc_M )**2 , axis=1))
    rC2 = np.sqrt( np.sum( ( npm.repmat(endl[0,:],nrx, 1) - rxloc_N )**2 , axis=1))
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

midl = ( Cmid + Pmid )/2
midz = -np.abs(Cmid-Pmid)

# Grid points
grid_x, grid_z = np.mgrid[np.min(midl):np.max(midl), np.min(midz):np.max(midz)]
grid_rho = griddata(np.c_[midl,midz], np.log10(abs(1/rho.T)), (grid_x, grid_z), method='linear')
plt.imshow(grid_rho.T, extent = (np.min(midl),np.max(midl),np.min(midz),np.max(midz)), origin='lower')
plt.colorbar()

# Plot apparent resistivity
plt.scatter(midl,midz,s=50,c=np.log10(abs(1/rho.T)))

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

# Export 2D model
fid = open(home_dir + '\MtIsa_2D.con','w')
fid.write('%i %i\n'% (mesh.nCx,mesh.nCz))
np.savetxt(fid, mkvc(m2D.T), fmt='%e',delimiter=' ',newline='\n')
fid.close()
#==============================================================================
# # Grab slice of model
# m = np.reshape(model, (mesh.nCz, mesh.nCy, mesh.nCx))
# m2D = m[::-1,9,:]
# plt.figure()
# plt.imshow(m2D)
#==============================================================================

#%%
tx, rx, d, wd = readUBC_DC3Dobs(home_dir + '\OBS_LOC_3D.dat')