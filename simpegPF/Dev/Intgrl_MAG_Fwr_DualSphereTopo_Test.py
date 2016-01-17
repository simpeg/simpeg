import os

home_dir = 'C:\\LC\\Private\\dominiquef\\Projects\\4414_Minsim\\Modeling\\MAG'

os.chdir(home_dir)

#%%
from SimPEG import *
import matplotlib.pyplot as plt
import simpegPF as PF
import scipy.interpolate as interpolation
import time

#from fwr_MAG_data import fwr_MAG_data

plt.close('all')

topofile = 'Gaussian.topo'

zoffset = 2
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
       
# # Or create juste a plane grid
xr = np.linspace(-99., 99., 40)
yr = np.linspace(-49., 49., 20)
X, Y = np.meshgrid(xr, yr)



sclx = 100.
dx = np.asarray([15., 10., 5., 2.5])
 
d_iter = len(dx)
l1_r = np.zeros(d_iter)
l2_r = np.zeros(d_iter)
linf_r = np.zeros(d_iter)
timer = np.zeros(d_iter)
mcell = np.zeros(d_iter)
#%% Loop through decreasing meshes and measure the residual
# Create mesh using simpeg and write out in GIF format

for ii in range(d_iter):
    
    
    nc = int(sclx/dx[ii])
    
    hxind = [(dx[ii], 2*nc)]
    hyind = [(dx[ii], nc)]
    hzind = [(dx[ii], nc)]
    
    mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CCN')
    
    mcell[ii] = mesh.nC
    
    actv = PF.Magnetics.getActiveTopo(mesh,topo,'N')
    
    # Drape observations on topo + offset
    if not topofile:
        Z = np.ones((xr.size, yr.size)) * 2.5
        
    else:
        F = interpolation.NearestNDInterpolator(topo[:,0:2],topo[:,2])
        Z = F(X,Y) + zoffset
        
    rxLoc = np.c_[Utils.mkvc(X.T), Utils.mkvc(Y.T), Utils.mkvc(Z.T)]
    
    ndata = rxLoc.shape[0]

    xn = mesh.vectorNx
    yn = mesh.vectorNy
    zn = mesh.vectorNz
        
    print 'Mesh size: ' + str(mcell[ii])
    
    #%% Create model
    chibkg = 0.
    chiblk = 0.01
    model = np.ones(mcell[ii])*chibkg
    
    # Do a three sphere problem for more frequencies 
    sph_ind = PF.MagAnalytics.spheremodel(mesh, 0., 0., -sclx/3, R)
    model[sph_ind] = 0.5*chiblk
    
    sph_ind = PF.MagAnalytics.spheremodel(mesh, -sclx/2., 0., -sclx/3., R/3.)
    model[sph_ind] = 4.*chiblk
    
    sph_ind = PF.MagAnalytics.spheremodel(mesh, sclx/2., 0., -sclx/2.5, R/2.5)
    model[sph_ind] = 2.5*chiblk
    
    Utils.writeUBCTensorMesh('Mesh.msh',mesh)
    Utils.writeUBCTensorModel('Model.sus',mesh,model)
    #actv = np.ones(mesh.nC)
    #%% Forward mode ldata
    
    start_time = time.time()
    
    d = PF.Magnetics.Intgrl_Fwr_Data(mesh,B,M,rxLoc,model,actv,'tmi')
    
    timer[ii] = (time.time() - start_time)
    
    #fwr_tmi = d[0:ndata]
    #fwr_y = d[ndata:2*ndata]
    #fwr_z = d[2*ndata:]

    #%% Get the analystical answer and compute the residual
    #bxa,bya,bza = PF.MagAnalytics.MagSphereAnaFunA(rxLoc[:,0],rxLoc[:,1],rxLoc[:,2],R,0.,0.,0.,chiblk, np.array(([0.,0.,B[2]])),'secondary')
    Bd = (450.-float(B[1]))%360.
    Bi = B[0]; # Convert dip to horizontal to cartesian 
    
    Bx = np.cos(np.deg2rad(Bi)) * np.cos(np.deg2rad(Bd)) * B[2] 
    By = np.cos(np.deg2rad(Bi)) * np.sin(np.deg2rad(Bd)) * B[2] 
    Bz = np.sin(np.deg2rad(Bi)) * B[2] 
    
    Bo = np.c_[Bx, By, Bz]
    
    Ptmi = mkvc(np.r_[np.cos(np.deg2rad(Bi))*np.cos(np.deg2rad(Bd)),np.cos(np.deg2rad(Bi))*np.sin(np.deg2rad(Bd)),np.sin(np.deg2rad(Bi))],2).T;
    
    bxa,bya,bza = PF.MagAnalytics.MagSphereFreeSpace(rxLoc[:,0],rxLoc[:,1],rxLoc[:,2],R,0., 0., -sclx/3, 0.5*chiblk, Bo)
    bxb,byb,bzb = PF.MagAnalytics.MagSphereFreeSpace(rxLoc[:,0],rxLoc[:,1],rxLoc[:,2],R/3., -sclx/2., 0., -sclx/3.,4.*chiblk, Bo)
    bxc,byc,bzc = PF.MagAnalytics.MagSphereFreeSpace(rxLoc[:,0],rxLoc[:,1],rxLoc[:,2],R/2.5, sclx/2., 0., -sclx/2.5,2.5*chiblk, Bo)
    
    bx = bxa + bxb + bxc
    by = bya + byb + byc
    bz = bza + bzb + bzc
    
    b_tmi = mkvc(Ptmi.dot(np.c_[bx,by,bz].T))
    
    r_tmi = d - b_tmi
    #r_By = fwr_y - bya
    #r_Bz = fwr_z - bza
    
    l2_r[ii] = np.sum( r_tmi**2 ) **0.5
    l1_r[ii] = np.sum( np.abs( r_tmi ) )
    linf_r[ii] = np.max( np.abs( r_tmi ) )

#%% Write predicted to file

PF.Magnetics.writeUBCobs('Obsloc.loc',B,M,rxLoc,d,np.ones(len(d)))    

#%% Plot results
print 'Residual between analytical sphere and integral forward' 
print "dx \t nc \t l1 \t l2 \t linf \t Runtime"
for ii in range(d_iter):

    print str(dx[ii]) + "\t" + str(mcell[ii]) + "\t" + str(l1_r[ii]) + "\t" + str(l2_r[ii]) + "\t" + str(linf_r[ii]) + "\t" + str(timer[ii])
    
#%% Plot fields
plt.figure(1)
ax = plt.subplot()
plt.imshow(np.reshape(b_tmi,X.shape), interpolation="bicubic", extent=[xr.min(), xr.max(), yr.min(), yr.max()], origin = 'lower')
plt.colorbar(fraction=0.02)
plt.contour(X,Y, np.reshape(b_tmi,X.shape),10)
plt.scatter(X,Y, c=np.reshape(b_tmi,X.shape), s=20)
ax.set_title('Analytical')

#%% Plot the forward solution from integral
plt.figure(2)
ax = plt.subplot()
plt.imshow(np.reshape(d,X.shape), interpolation="bicubic", extent=[xr.min(), xr.max(), yr.min(), yr.max() ], origin = 'lower')
plt.colorbar(fraction=0.02)
plt.contour(X,Y, np.reshape(d,X.shape),10)
plt.scatter(X,Y, c=np.reshape(d,X.shape), s=20)
ax.set_title('Numerical')

#%% Plot residual data
plt.figure(3)
ax = plt.subplot()
plt.imshow(np.reshape(r_tmi,X.shape), interpolation="bicubic", extent=[xr.min(), xr.max(), yr.min(), yr.max()], origin = 'lower')
plt.colorbar(fraction=0.02)
plt.contour(X,Y, np.reshape(r_tmi,X.shape),10)
plt.scatter(X,Y, c=np.reshape(r_tmi,X.shape), s=20)
ax.set_title('Sphere Ana Bx')