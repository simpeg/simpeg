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
from plot_pseudoSection import plot_pseudoSection

#from scipy.linalg import solve_banded

# Load UBC mesh 3D
mesh = Utils.meshutils.readUBCTensorMesh(home_dir + '\Mesh_20m.msh')
#mesh = Utils.meshutils.readUBCTensorMesh('Mesh_40m.msh')

# Load model
model = Utils.meshutils.readUBCTensorModel(home_dir + '\MtIsa_3D.con',mesh)
#model = Utils.meshutils.readUBCTensorModel('Synthetic.con',mesh)

#%% Create system
#Set boundary conditions
mesh.setCellGradBC('neumann')

Div = mesh.faceDiv
Grad = mesh.cellGrad
Msig = Utils.sdiag(1./(mesh.aveF2CC.T*(1./model)))

A = Div*Msig*Grad

# Change one corner to deal with nullspace
A[0,0] = 1/mesh.vol[0]
A = sp.csc_matrix(A)

start_time = time.time()

# Factor A matrix
Ainv = sp.linalg.splu(A)

print("LU DECOMP--- %s seconds ---" % (time.time() - start_time))

#%% Create survey
# Display top section 
top = int(mesh.nCz)-1
mesh.plotSlice(model, ind=12, normal='Z', grid=True, pcolorOpts={'alpha':0.8})

# Add z coordinate
nz = mesh.vectorNz

# Takes two points from ginput and create survey
temp = plt.ginput(2, timeout = 0)
temp = np.c_[np.asarray(temp),np.ones(2).T*nz[-1]]

indx = Utils.closestPoints(mesh, temp )
endl = np.c_[mesh.gridCC[indx,0],mesh.gridCC[indx,1],np.ones(2).T*nz[-1]]

#endl = np.c_[np.asarray(temp),np.ones(2).T*nz[-1]]


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
M = np.c_[stn_x, stn_y, np.ones(nstn).T*nz[-1]]
N = np.c_[stn_x+a*dl_x, stn_y+a*dl_y, np.ones(nstn).T*nz[-1]]

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


#%% Load 3D data
[Tx, Rx, d, wd] = readUBC_DC3Dobs(home_dir + '\FWR_data3D.dat')


#%% Convert 3D obs to 2D and write to file
#data[:,0:4] = data[:,0:4] + endl[0,0]
#fid = open(home_dir + '\FWR_data2D.dat','w')
#fid.write('SIMPEG FORWARD\n')  

# Change coordinate system to distance along line
# Assume all data is acquired along line, and first transmitter pole is 
# at the origin

d2D = []

for ii in range(len(Tx)):
    
    if ii == 0:
        endp = Tx[0][0:2,0]
    
    nrx = Rx[ii].shape[0]
    
    for jj in range(nrx):
        
        rP1 = np.sqrt( np.sum( ( endp - Tx[ii][0:2,0] )**2 , axis=0))
        rP2 = np.sqrt( np.sum( ( endp - Tx[ii][0:2,1] )**2 , axis=0))
        rC1 = np.sqrt( np.sum( ( endp - Rx[ii][jj,0:2] )**2 , axis=0))
        rC2 = np.sqrt( np.sum( ( endp - Rx[ii][jj,3:5] )**2 , axis=0))
        
        d2D.append( np.r_[rP1, rP2, rC1, rC2, d[ii][jj], wd[ii][jj]] )
        #np.savetxt(fid, data, fmt='%e',delimiter=' ',newline='\n')
        
#%%
fid = open(home_dir + '\FWR_3D_2_2D.dat','w')
fid.write('SIMPEG FORWARD\n')   
for ii in range(len(d2D)): 
    fid.write('\n') 
    
    for jj in range(d2D[ii].shape[0]): 
        fid.write('%e ' % d2D[ii][jj])
        
fid.close()

#%% Create a 2D mesh along axis of end points and keep z-discretization
#==============================================================================
# dx = np.min( [ np.min(mesh.hx), np.min(mesh.hy) ])
# nc = np.ceil(dl_len/dx)+1
# 
# padx = dx*np.power(1.4,range(1,15))
# 
# # Creating padding cells
# h1 = np.r_[padx[::-1], np.ones(nc)*dx , padx]
# 
# # Create mesh with 0 coordinate centerer on the ginput points in cell center
# mesh2d = Mesh.TensorMesh([h1, mesh.hz], x0=(-np.sum(padx)-dx/2,mesh.x0[2]))
# 
# # Create array of points for interpolating from 3D to 2D mesh
# xx = endl[0,0] + mesh2d.vectorCCx * np.cos(azm)
# yy = endl[0,1] + mesh2d.vectorCCx * np.sin(azm)
# zz = mesh2d.vectorCCy
# 
# [XX,ZZ] = np.meshgrid(xx,zz)
# [YY,ZZ] = np.meshgrid(yy,zz)
# 
# xyz2d = np.c_[mkvc(XX),mkvc(YY),mkvc(ZZ)]
# 
# plt.scatter(xx,yy,s=20,c='y')
# 
# 
# F = interpolation.NearestNDInterpolator(mesh.gridCC,model)
# m2D = np.reshape(F(xyz2d),[mesh2d.nCx,mesh2d.nCy])
# 

#==============================================================================
 
# Create mesh with 0 coordinate centerer on the ginput points in cell center
mesh2d = Mesh.TensorMesh([mesh.hx, mesh.hz], x0=(mesh.x0[0]-endl[0,0],mesh.x0[2]))
m3D = np.reshape(model, (mesh.nCz, mesh.nCy, mesh.nCx))
m2D = m3D[:,1,:]

plt.figure()
axs = plt.subplot(1,1,1)
plt.pcolormesh(mesh2d.vectorNx,mesh2d.vectorNy,np.log10(m2D),alpha=0.5, cmap='gray')#axes = [mesh2d.vectorNx[0],mesh2d.vectorNx[-1],mesh2d.vectorNy[0],mesh2d.vectorNy[-1]])
#mesh2d.plotImage(mkvc(m2D), grid=True, ax=axs)

#%% Plot pseudo section

plot_pseudoSection(d2D,nz[-1])
#axs.axis([0,dl_len,mesh2d.vectorNy[-1]-dl_len/2,mesh2d.vectorNy[-1]])


#%% Export 2D mesh from section
fid = open(home_dir + '\Mesh_2D.msh','w')
fid.write('%i\n'% mesh2d.nCx)
fid.write('%f %f 1\n'% (mesh2d.vectorNx[0],mesh2d.vectorNx[1]))  
np.savetxt(fid, np.c_[mesh2d.vectorNx[2:],np.ones(mesh2d.nCx-1)], fmt='\t %e %i',delimiter=' ',newline='\n')
fid.write('\n')
fid.write('%i\n'% mesh2d.nCy)
fid.write('%f %f 1\n'%( 0,mesh2d.hy[-1]))   
np.savetxt(fid, np.c_[np.cumsum(mesh2d.hy[-2::-1])+mesh2d.hy[-1],np.ones(mesh2d.nCy-1)], fmt='\t %e %i',delimiter=' ',newline='\n')
fid.close()

# Export 2D model
fid = open(home_dir + '\MtIsa_2D.con','w')
fid.write('%i %i\n'% (mesh2d.nCx,mesh2d.nCy))
np.savetxt(fid, mkvc(m2D[::-1,:].T), fmt='%e',delimiter=' ',newline='\n')
fid.close()
#==============================================================================
# # Grab slice of model
# m = np.reshape(model, (mesh.nCz, mesh.nCy, mesh.nCx))
# m2D = m[::-1,9,:]
# plt.figure()
# plt.imshow(m2D)
#==============================================================================
