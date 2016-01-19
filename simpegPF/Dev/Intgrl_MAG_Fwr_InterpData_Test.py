#%%
from SimPEG import *
import matplotlib.pyplot as plt
import simpegPF as PF
import scipy.interpolate as interpolation
import time
from interpFFT import interpFFT
#from fwr_MAG_data import fwr_MAG_data

import os

home_dir = 'C:\\LC\\Private\\dominiquef\\Projects\\4414_Minsim\\Modeling\\MAG'

os.chdir(home_dir)

plt.close('all')

topofile = 'Gaussian.topo'

zoffset = 5
#%% Create survey
# Load in topofile or create flat surface
if not topofile:
 
    actv = np.ones(mesh.nC)   
    
else: 
    topo = np.genfromtxt(topofile,skip_header=1) 
    
    
B = np.array(([90.,0.,50000.]))
  
M = np.array(([90.,0.,315.]))

# Sphere radius
R = 25.
       
sclx = 100.
dx = 5.
#%% Loop through decreasing meshes and measure the residual
# Create mesh using simpeg and write out in GIF format
    
# # Or create juste a plane grid
xr = np.linspace(-102.5, 97.5, 41)
yr = np.linspace(-52.5, 47.5, 21)
X, Y = np.meshgrid(xr, yr)

nc = int(sclx/dx)

hxind = [(dx, 2*nc)]
hyind = [(dx, nc)]
hzind = [(dx, nc)]

mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CCN')

actv = PF.Magnetics.getActiveTopo(mesh,topo,'N')

# Drape observations on topo + offset
if not topofile:
    Z = np.ones((xr.size, yr.size)) * 2.5
    
else:
    F = interpolation.interp2d(topo[:,0],topo[:,1],topo[:,2])
    #F = interpolation.NearestNDInterpolator(topo[:,0:2],topo[:,2])
    Z = F(xr,yr) + zoffset
    
rxLoc = np.c_[Utils.mkvc(X.T), Utils.mkvc(Y.T), Utils.mkvc(Z.T)]

ndata = rxLoc.shape[0]

xn = mesh.vectorNx
yn = mesh.vectorNy
zn = mesh.vectorNz

mcell = mesh.nC

print 'Mesh size: ' + str(mcell)

#%% Create model
chibkg = 0.0001
chiblk = 0.01
model = np.ones(mcell)*chibkg

# Do a three sphere problem for more frequencies 
sph_ind = PF.MagAnalytics.spheremodel(mesh, 0., 0., -sclx/3, R)
model[sph_ind] = 0.5*chiblk

sph_ind = PF.MagAnalytics.spheremodel(mesh, -sclx/2., 0., -sclx/3., R/3.)
model[sph_ind] = 4.*chiblk

sph_ind = PF.MagAnalytics.spheremodel(mesh, sclx/2., 0., -sclx/2.5, R/2.5)
model[sph_ind] = 2.5*chiblk

# Zero out
model[actv==0] = -100
Utils.writeUBCTensorMesh('Mesh.msh',mesh)
Utils.writeUBCTensorModel('Model.sus',mesh,model)
Utils.writeUBCTensorModel('nullcell.dat',mesh,actv)
#actv = np.ones(mesh.nC)
#%% Forward mode ldata
d = PF.Magnetics.Intgrl_Fwr_Data(mesh,B,M,rxLoc,model,actv,'tmi')
#fwr_tmi = d[0:ndata]
#fwr_y = d[ndata:2*ndata]
#fwr_z = d[2*ndata:]
    

#%% Compute data on a line
xx = np.linspace(xr.min(), xr.max(), 200)
yy = np.zeros(len(xx))
zz = F(xx,0.) + zoffset

rxLoc = np.c_[xx,yy,zz]

d_line = PF.Magnetics.Intgrl_Fwr_Data(mesh,B,M,rxLoc,model,actv,'tmi')

d_iter = 4

l1_r = np.zeros((d_iter,5))
l2_r = np.zeros((d_iter,5))
linf_r = np.zeros((d_iter,5))
timer = np.zeros((d_iter,5))

d2d = np.reshape(d, (len(yr),len(xr)))
#%% Try different interpolation schemes
for ii in range(d_iter):    
    
    indx = ii+1
    
    
    dsub = d2d[::indx,::indx]
    
    xsub = xr[::indx]
    ysub = yr[::indx]
    
    # Nearest Neighbourg
    start_time = time.time()
    F = interpolation.NearestNDInterpolator(np.c_[mkvc(Y[::indx,::indx].T),mkvc(X[::indx,::indx].T)],mkvc(dsub.T))
    d_i2d_nnb = mkvc( F(0.,xx) )    
    l1_r[ii,0] = np.sum( np.abs(d_line - d_i2d_nnb) )**0.5
    l2_r[ii,0] = np.sum( (d_line - d_i2d_nnb)**2. )
    linf_r[ii,0] = np.max( np.abs(d_line - d_i2d_nnb) )
    timer[ii,0] = (time.time() - start_time)
    
    # Linear interpolation
    start_time = time.time()
    F = interpolation.interp2d(ysub,xsub,mkvc(dsub.T))
    d_i2d_lin = mkvc( F(0.,xx) )    
    l1_r[ii,1] = np.sum( np.abs(d_line - d_i2d_lin) )**0.5
    l2_r[ii,1] = np.sum( (d_line - d_i2d_lin)**2. )
    linf_r[ii,1] = np.max( np.abs(d_line - d_i2d_lin) )
    timer[ii,1] = (time.time() - start_time)
    
    
    # Cubic interpolation
    start_time = time.time()
    F = interpolation.interp2d(ysub,xsub,mkvc(dsub.T),kind='cubic')
    d_i2d_cub = mkvc( F(0.,xx) ) 
    l1_r[ii,2] = np.sum( np.abs(d_line - d_i2d_cub) )**0.5
    l2_r[ii,2] = np.sum( (d_line - d_i2d_cub)**2. )
    linf_r[ii,2] = np.max( np.abs(d_line - d_i2d_cub) )
    timer[ii,2] = (time.time() - start_time)    
    
    
    # Quintic interpolation
    start_time = time.time()
    F = interpolation.interp2d(ysub,xsub,mkvc(dsub.T),kind='quintic')
    d_i2d_qui = mkvc( F(0.,xx) )  
    l1_r[ii,3] = np.sum( np.abs(d_line - d_i2d_qui) )**0.5
    l2_r[ii,3] = np.sum( (d_line - d_i2d_qui)**2. )
    linf_r[ii,3] = np.max( np.abs(d_line - d_i2d_qui) )
    timer[ii,3] = (time.time() - start_time)
    
    # CloughTocher interpolation
    start_time = time.time()
    F = interpolation.CloughTocher2DInterpolator(np.c_[mkvc(Y[::indx,::indx].T),mkvc(X[::indx,::indx].T)],mkvc(dsub.T))
    d_i2d_CTI = mkvc( F(0.,xx) )  
    l1_r[ii,4] = np.sum( np.abs(d_line - d_i2d_CTI) )**0.5
    l2_r[ii,4] = np.sum( (d_line - d_i2d_CTI)**2. )
    linf_r[ii,4] = np.max( np.abs(d_line - d_i2d_CTI) )
    timer[ii,4] = (time.time() - start_time)
    

    
#==============================================================================
#     #%% FFT interpolation
#     d2d_out = interpFFT(xsub,ysub,dsub)
#     
#     # Create new distance vector
#     XX = np.linspace(np.min(xsub),np.max(xsub),d2d_out.shape[1])
#     YY = np.linspace(np.min(ysub),np.max(ysub),d2d_out.shape[0])
#     
#     start_time = time.time()
#     F = interpolation.interp2d(XX,YY,d2d_out)
#     d_i2d_fft = mkvc( F(xx,0.) )    
#     l1_r[ii,4] = np.sum( np.abs(d_line - d_i2d_nnb) )**0.5
#     l2_r[ii,4] = np.sum( (d_line - d_i2d_nnb)**2. )
#     linf_r[ii,4] = np.max( np.abs(d_line - d_i2d_nnb) )
#     timer[ii,4] = (time.time() - start_time)
#         
#     print("--- FFT completed in %s seconds ---" % (time.time() - start_time))
# 
#        
#     plt.figure()
#     ax = plt.subplot()
#     plt.imshow(d2d_out, interpolation="bicubic", extent=[xr.min(), xr.max(), yr.min(), yr.max() ], origin = 'lower')
#     plt.colorbar(fraction=0.04)
#==============================================================================
    

#%% Write predicted to file

#PF.Magnetics.writeUBCobs('Obsloc.loc',B,M,rxLoc,d,np.ones(len(d)))    

#%% Plot results
#==============================================================================
# print 'Residual between analytical sphere and integral forward' 
# for ii in range(d_iter):
#     nc = 3**(ii+1)
# 
#     print "||r||= " + str(lrl[ii]) + "\t dx= " + str(1./nc)
#==============================================================================
    

#%% Plot the forward solution from integral
plt.figure(figsize=[8,4])
ax = plt.subplot()
plt.imshow(d2d, interpolation="bicubic", extent=[xr.min(), xr.max(), yr.min(), yr.max() ], origin = 'lower')
plt.colorbar(fraction=0.02)
plt.contour(X,Y, d2d,10)
plt.scatter(X,Y, s=5)


plt.figure(figsize=[8,4])
plt.contour(X,Y, np.reshape(d,X.shape),10)
plt.scatter(X,Y, c=np.reshape(d,X.shape), s=10)
plt.scatter(xx,yy, c='k', s=20, marker='o')
ax.set_title('Numerical')


#%%

plt.figure(figsize=[7,9])
ax = plt.subplot()
plt.plot(xx,d_line,c='r', linewidth=3)
plt.plot(xx,d_i2d_lin)
plt.plot(xx,d_i2d_cub)
plt.plot(xx,d_i2d_qui)
plt.plot(xx,d_i2d_nnb)
plt.plot(xx,d_i2d_CTI)
plt.xlim(xx.min(),xx.max())

plt.legend(['True','linear','Cubic','Quintic','NearestN','CloughTorcher'],bbox_to_anchor=(0.75, 0.25))
# Plot interpolation from true value on line
F = interpolation.interp1d(xx,d_line)
dtrue = F(xr[::indx])
plt.plot(xr[::indx],dtrue,c='r',linewidth=0.,marker='o')
ax.set_title('Interpolated data profile')

#%% Write result to file
with file('Interp_residual.dat','w') as fid:
    fid.write('NearestN\tLinear\tCubic\tQuintic\tCloughTocher\n')
    fid.write('\nL2-norm\n')
    np.savetxt(fid, l2_r, fmt='%5.3e',delimiter='\t',newline='\n')
    fid.write('\nL1-norm\n')
    np.savetxt(fid, l1_r, fmt='%5.3e',delimiter='\t',newline='\n')
    fid.write('\nLinf-norm\n')
    np.savetxt(fid, linf_r, fmt='%5.3e',delimiter='\t',newline='\n')
