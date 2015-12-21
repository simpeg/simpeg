import os

home_dir = 'C:\Users\dominiquef.MIRAGEOSCIENCE\Documents\GIT\SimPEG\simpegpf\simpegPF\Dev'

os.chdir(home_dir)

#%%
from SimPEG import *
import matplotlib.pyplot as plt
import simpegPF as PF

#from fwr_MAG_data import fwr_MAG_data

plt.close('all')

#%% Create survey

B = np.array(([-45.,315.,50000.]))
  
M = np.array(([-45.,315.]))
 
# Sphere radius
R = 0.25
       
# # Or create juste a plane grid
xr = np.linspace(-2., 2., 5)
yr = np.linspace(-2., 2., 5)
X, Y = np.meshgrid(xr, yr)
Z = np.ones((xr.size, yr.size)) * 2.5
rxLoc = np.c_[Utils.mkvc(X), Utils.mkvc(Y), Utils.mkvc(Z)]

ndata = rxLoc.shape[0]

d_iter = 4
lrl = np.zeros(d_iter)
#%% Loop through decreasing meshes and measure the residual
# Create mesh using simpeg and write out in GIF format

for ii in range(d_iter):
       
    nc = 3**(ii+1)
    
    hxind = [(1./nc, nc)]
    hyind = [(1./nc, nc)]
    hzind = [(1./nc, nc)]
    
    mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CCC')
    
    xn = mesh.vectorNx
    yn = mesh.vectorNy
    zn = mesh.vectorNz
    
    mcell = mesh.nC
    
    print 'Mesh size: ' + str(mcell)
    
    sph_ind = PF.MagAnalytics.spheremodel(mesh, 0, 0, 0, R)
    
    chibkg = 0.
    chiblk = 0.01
    model = np.ones(mcell)*chibkg
    model[sph_ind] = chiblk
    
    #%% Forward mode ldata
    d = PF.Magnetics.Intgrl_Fwr_Data(mesh,B,M,rxLoc,model,'xyz')
    fwr_x = d[0:ndata]
    fwr_y = d[ndata:2*ndata]
    fwr_z = d[2*ndata:]
    
    #%% Get the analystical answer and compute the residual
    bxa,bya,bza = PF.MagAnalytics.MagSphereAnaFunA(rxLoc[:,0],rxLoc[:,1],rxLoc[:,2],R,0.,0.,0.,chiblk, np.array(([0.,0.,B[2]])),'secondary')
    Bd = (450.-float(B[1]))%360.
    Bi = B[0]; # Convert dip to horizontal to cartesian 
    
    Bx = np.cos(np.deg2rad(Bi)) * np.cos(np.deg2rad(Bd)) * B[2] 
    By = np.cos(np.deg2rad(Bi)) * np.sin(np.deg2rad(Bd)) * B[2] 
    Bz = np.sin(np.deg2rad(Bi)) * B[2] 
    
    Bo = np.c_[Bx, By, Bz]
        
    bxa,bya,bza = PF.MagAnalytics.MagSphereFreeSpace(rxLoc[:,0],rxLoc[:,1],rxLoc[:,2],R,0.,0.,0.,chiblk, Bo)
    #bxa,bya,bza = PF.MagAnalytics.MagSphereAnaFunA(rxLoc[:,0],rxLoc[:,1],rxLoc[:,2],R,0.,0.,0.,chiblk, np.array(([0.,0.,B[2]])),'secondary')
    
    r_Bx = fwr_x - bxa
    r_By = fwr_y - bya
    r_Bz = fwr_z - bza
    
    lrl[ii] = sum( r_Bx**2 + r_By**2 + r_Bz**2 ) **0.5
    

    
#%% Plot results
print 'Residual between analytical sphere and integral forward' 
for ii in range(d_iter):
    nc = 3**(ii+1)

    print "||r||= " + str(lrl[ii]) + "\t dx= " + str(1./nc)
    
#%% Plot fields

plt.figure(1)
ax = plt.subplot(221)
plt.imshow(np.reshape(bxa,X.shape).T, interpolation="bicubic", extent=[xr.min(), xr.max(), yr.min(), yr.max()], origin = 'lower')
plt.colorbar(fraction=0.04)
plt.contour(X,Y, np.reshape(bxa,X.shape).T,10)
plt.scatter(X,Y, c=np.reshape(bxa,X.shape).T, s=20)
ax.set_title('Sphere Ana Bx')

ax = plt.subplot(222)
plt.imshow(np.reshape(bya,X.shape).T, interpolation="bicubic", extent=[xr.min(), xr.max(), yr.min(), yr.max()], origin = 'lower')
plt.colorbar(fraction=0.04)
plt.contour(X,Y, np.reshape(bya,X.shape).T,10)
plt.scatter(X,Y, c=np.reshape(bya,X.shape).T, s=20)
ax.set_title('Sphere Ana By')

ax = plt.subplot(212)
plt.imshow(np.reshape(bza,X.shape).T, interpolation="bicubic", extent=[xr.min(), xr.max(), yr.min(), yr.max()], origin = 'lower')
plt.colorbar(fraction=0.04)
plt.contour(X,Y, np.reshape(bza,X.shape).T,10)
plt.scatter(X,Y, c=np.reshape(bza,X.shape).T, s=20)
ax.set_title('Sphere Ana Bz')

#%% Plot the forward solution from integral

plt.figure(2)
ax = plt.subplot(221)
plt.imshow(np.reshape(fwr_x,X.shape).T, interpolation="bicubic", extent=[xr.min(), xr.max(), yr.min(), yr.max() ], origin = 'lower')
plt.colorbar(fraction=0.04)
plt.contour(X,Y, np.reshape(fwr_x,X.shape).T,10)
plt.scatter(X,Y, c=np.reshape(fwr_x,X.shape).T, s=20)
ax.set_title('Sphere Ana Bx')

ax = plt.subplot(222)
plt.imshow(np.reshape(fwr_y,X.shape).T, interpolation="bicubic", extent=[xr.min(), xr.max(), yr.min(), yr.max()], origin = 'lower')
plt.colorbar(fraction=0.04)
plt.contour(X,Y, np.reshape(fwr_y,X.shape).T,10)
plt.scatter(X,Y, c=np.reshape(fwr_y,X.shape).T, s=20)
ax.set_title('Sphere Ana By')

ax = plt.subplot(212)
plt.imshow(np.reshape(fwr_z,X.shape).T, interpolation="bicubic", extent=[xr.min(), xr.max(), yr.min(), yr.max()], origin = 'lower')
plt.colorbar(fraction=0.04)
plt.contour(X,Y, np.reshape(fwr_z,X.shape).T,10)
plt.scatter(X,Y, c=np.reshape(fwr_z,X.shape).T, s=20)
ax.set_title('Sphere Ana Bz')


#%% Plot foward data
plt.figure(3)
ax = plt.subplot(221)
plt.imshow(np.reshape(r_Bx,X.shape).T, interpolation="bicubic", extent=[xr.min(), xr.max(), yr.min(), yr.max()], origin = 'lower')
plt.colorbar(fraction=0.04)
plt.contour(X,Y, np.reshape(r_Bx,X.shape).T,10)
plt.scatter(X,Y, c=np.reshape(r_Bx,X.shape).T, s=20)
ax.set_title('Sphere Ana Bx')

ax = plt.subplot(222)
plt.imshow(np.reshape(r_By,X.shape).T, interpolation="bicubic", extent=[xr.min(), xr.max(), yr.min(), yr.max()], origin = 'lower')
plt.colorbar(fraction=0.04)
plt.contour(X,Y, np.reshape(r_By,X.shape).T,10)
plt.scatter(X,Y, c=np.reshape(r_By,X.shape).T, s=20)
ax.set_title('Sphere Ana By')

ax = plt.subplot(212)
plt.imshow(np.reshape(r_Bz,X.shape).T, interpolation="bicubic", extent=[xr.min(), xr.max(), yr.min(), yr.max()], origin = 'lower')
plt.colorbar(fraction=0.04)
plt.contour(X,Y, np.reshape(r_Bz,X.shape).T,10)
plt.scatter(X,Y, c=np.reshape(r_Bz,X.shape).T, s=20)
ax.set_title('Sphere Ana Bz')