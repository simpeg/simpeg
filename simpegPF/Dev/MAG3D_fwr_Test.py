import os

home_dir = 'C:\Users\dominiquef.MIRAGEOSCIENCE\Documents\GIT\SimPEG\simpegpf\simpegPF\Dev'

os.chdir(home_dir)

#%%
from SimPEG import *
import matplotlib.pyplot as plt
import simpegPF as PF
from simpegPF import BaseMag
import matplotlib

from fwr_MAG_obs import fwr_MAG_obs

#%% Create survey

B = np.array(([90.,0.,50000.]))
  
M = np.array(([90.,0.]))
 
       
# # Or create juste a plane grid
xr = np.linspace(-1./2., 1./2., 10)
yr = np.linspace(-1./2., 1./2., 10)
X, Y = np.meshgrid(xr, yr)
Z = np.ones((xr.size, yr.size))*.75
rxLoc = np.c_[Utils.mkvc(X), Utils.mkvc(Y), Utils.mkvc(Z)]

ndata = rxLoc.shape[0]

#%%
# Create mesh using simpeg and write out in GIF format
nc = 30.

hxind = [(1./nc, nc)]
hyind = [(1./nc, nc)]
hzind = [(1./nc, nc)]

mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CCC')

xn = mesh.vectorNx
yn = mesh.vectorNy
zn = mesh.vectorNz

mcell = mesh.nC

sph_ind = PF.MagAnalytics.spheremodel(mesh, 0, 0, 0, 0.25)

chibkg = 0.
chiblk = 0.01
model = np.ones(mcell)*chibkg
model[sph_ind] = chiblk

#%% Forward mode ldata
d = fwr_MAG_obs(mesh,B,M,rxLoc,model)

#%% Get the analystical answer and compute the residual
bxa,bya,bza = PF.MagAnalytics.MagSphereAnaFunA(rxLoc[:,0],rxLoc[:,1],rxLoc[:,2],.25,0.,0.,0.,chiblk, np.array(([0.,0.,B[2]])),'secondary')

r_Bz = mkvc(d) - bza
lrl = sum( r_Bz**2 ) **0.5

print "Residual between analytical and integral= " + str(lrl)
#%% Plot fields

plt.figure(1)
ax = plt.subplot(131)
plt.imshow(np.reshape(bxa,X.shape), interpolation="bicubic", extent=[xr.min(), xr.max(), yr.min(), yr.max()])
plt.contour(X,Y, np.reshape(bxa,X.shape),10)
plt.scatter(X,Y, c='k', s=5)
ax.set_title('Sphere Ana Bx')

ax = plt.subplot(132)
plt.imshow(np.reshape(bya,X.shape), interpolation="bicubic", extent=[xr.min(), xr.max(), yr.min(), yr.max()])
plt.contour(X,Y, np.reshape(bya,X.shape),10)
plt.scatter(X,Y, c='k', s=5)
ax.set_title('Sphere Ana By')

ax = plt.subplot(133)
plt.imshow(np.reshape(bza,X.shape), interpolation="bicubic", extent=[xr.min(), xr.max(), yr.min(), yr.max()])
plt.contour(X,Y, np.reshape(bza,X.shape),10)
plt.scatter(X,Y, c='k', s=5)
ax.set_title('Sphere Ana Bz')
#%% Plot foward data
plt.figure(2)
plt.subplot(121)
d2D = np.reshape(d,X.shape)
plt.imshow(d2D, interpolation="bicubic", extent=[xr.min(), xr.max(), yr.min(), yr.max()])
plt.contour(X,Y, d2D,10)
plt.scatter(X,Y, c='k', s=5)

#%% Compare fields

plt.subplot(122)
plt.imshow(np.reshape(r_Bz,X.shape), interpolation="bicubic", extent=[xr.min(), xr.max(), yr.min(), yr.max()])
plt.contour(X,Y, np.reshape(dBz,X.shape),10)
plt.scatter(X,Y, c='k', s=5)