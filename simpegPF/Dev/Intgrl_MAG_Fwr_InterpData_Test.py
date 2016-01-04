import os

home_dir = 'C:\\LC\\Private\\dominiquef\\Projects\\4414_Minsim\\Modeling\\MAG'

os.chdir(home_dir)

#%%
from SimPEG import *
import matplotlib.pyplot as plt
import simpegPF as PF
import scipy.interpolate as interpolation
import scipy.signal as sn
import time

#from fwr_MAG_data import fwr_MAG_data

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
chibkg = 0.000
chiblk = 0.01
model = np.ones(mcell)*chibkg

# Do a three sphere problem for more frequencies 
sph_ind = PF.MagAnalytics.spheremodel(mesh, 0., 0., -sclx/3, R)
model[sph_ind] = 0.5*chiblk

sph_ind = PF.MagAnalytics.spheremodel(mesh, -sclx/2., 0., -sclx/3., R/3.)
model[sph_ind] = 4.*chiblk

sph_ind = PF.MagAnalytics.spheremodel(mesh, sclx/2., 0., -sclx/2.5, R/2.5)
model[sph_ind] = 2.5*chiblk

Utils.writeUBCTensorMesh('Mesh.msh',mesh)
Utils.writeUBCTensorModel('Model.sus',mesh,model)
actv = np.ones(mesh.nC)
#%% Forward mode ldata
d = PF.Magnetics.Intgrl_Fwr_Data(mesh,B,M,rxLoc,model,actv,'tmi')
#fwr_tmi = d[0:ndata]
#fwr_y = d[ndata:2*ndata]
#fwr_z = d[2*ndata:]
    

#%% Compute data on a line
xx = np.linspace(-101., 99., 200)
yy = np.zeros(len(xx))
zz = F(xx,0.) + zoffset

rxLoc = np.c_[xx,yy,zz]

d_line = PF.Magnetics.Intgrl_Fwr_Data(mesh,B,M,rxLoc,model,actv,'tmi')

d_iter = 4

l1_r = np.zeros((d_iter,5))
l2_r = np.zeros((d_iter,5))
linf_r = np.zeros((d_iter,5))
timer = np.zeros((d_iter,5))

#%% Try different interpolation schemes
for ii in range(d_iter):    
    
    indx = ii+1
    
    d2d = np.reshape(d, (len(yr),len(xr)))
    d2d = mkvc(d2d[::indx,::indx].T)
    
    xsub = xr[::indx]
    ysub = yr[::indx]
    
    # Nearest Neighbourg
    start_time = time.time()
    F = interpolation.NearestNDInterpolator(np.c_[mkvc(Y[::indx,::indx].T),mkvc(X[::indx,::indx].T)],d2d)
    d_i2d_nnb = mkvc( F(0.,xx) )    
    l1_r[ii,0] = np.sum( np.abs(d_line - d_i2d_nnb) )**0.5
    l2_r[ii,0] = np.sum( (d_line - d_i2d_nnb)**2. )
    linf_r[ii,0] = np.max( np.abs(d_line - d_i2d_nnb) )
    timer[ii,0] = (time.time() - start_time)
    
    # Linear interpolation
    start_time = time.time()
    F = interpolation.interp2d(ysub,xsub,d2d)
    d_i2d_lin = mkvc( F(0.,xx) )    
    l1_r[ii,1] = np.sum( np.abs(d_line - d_i2d_lin) )**0.5
    l2_r[ii,1] = np.sum( (d_line - d_i2d_lin)**2. )
    linf_r[ii,1] = np.max( np.abs(d_line - d_i2d_lin) )
    timer[ii,1] = (time.time() - start_time)
    
    
    # Cubic interpolation
    start_time = time.time()
    F = interpolation.interp2d(ysub,xsub,d2d,kind='cubic')
    d_i2d_cub = mkvc( F(0.,xx) ) 
    l1_r[ii,2] = np.sum( np.abs(d_line - d_i2d_cub) )**0.5
    l2_r[ii,2] = np.sum( (d_line - d_i2d_cub)**2. )
    linf_r[ii,2] = np.max( np.abs(d_line - d_i2d_cub) )
    timer[ii,2] = (time.time() - start_time)    
    
    
    # Quintic interpolation
    start_time = time.time()
    F = interpolation.interp2d(ysub,xsub,d2d,kind='quintic')
    d_i2d_qui = mkvc( F(0.,xx) )  
    l1_r[ii,3] = np.sum( np.abs(d_line - d_i2d_qui) )**0.5
    l2_r[ii,3] = np.sum( (d_line - d_i2d_qui)**2. )
    linf_r[ii,3] = np.max( np.abs(d_line - d_i2d_qui) )
    timer[ii,3] = (time.time() - start_time)
    

    
    
#%% FFT interpolation

d2d = np.reshape(d, (len(yr),len(xr)))

# Add padding values by reflection (2**n)    
lenx = np.floor( np.log2( 2*len(xr) ) )
npadx = int(np.floor( ( 2**lenx - len(xr) ) /2. ))

#Create hemming taper
if np.mod(npadx*2+len(xr),2) != 0:
    oddx = 1
    
else:
    oddx = 0
    
tap0 = sn.hamming(npadx*2)    
tapl = sp.spdiags(tap0[0:npadx],0,npadx,npadx)
tapr = sp.spdiags(tap0[npadx:],0,npadx,npadx+oddx)
 
# Mirror the 2d data over the half lenght and apply 0-taper
d2dpad = np.hstack([np.fliplr(d2d[:,0:npadx]) * tapl, d2d, np.fliplr(d2d[:,-npadx:]) * tapr])    

# Repeat along the second dimension
leny = np.floor( np.log2( 2*len(yr) ) )
npady = int(np.floor( ( 2**leny - len(yr) ) /2. ))

#Create hemming taper
if np.mod(npady*2+len(yr),2) != 0:
    oddy = 1
    
else:
    oddy = 0
    
tap0 = sn.hamming(npady*2)
tapu = sp.spdiags(tap0[0:npady],0,npady,npady)
tapd = sp.spdiags(tap0[npady:],0,npady+oddy,npady)

d2dpad = np.vstack([tapu*np.flipud(d2dpad[0:npady,:]), d2dpad, tapd*np.flipud(d2dpad[-npady:,:])])

# Compute FFT
FFTd2d = np.fft.fft2(d2dpad)    

# Compute IFFT at a given location
# Do an FFT shift
#FFTshift = np.fft.fftshift(FFTd2d)
FFTd2d = np.hstack([FFTd2d[:,0:31],np.zeros((32,64)),FFTd2d[:,31:]])
FFTd2d = np.vstack([FFTd2d[0:15,:],np.zeros((32,128)),FFTd2d[15:,:]])
# Pad with zeros
#temp = np.zeros((FFTd2d.shape[0]*2,FFTd2d.shape[1]*2))


# Compute inverse FFT
IFFTd2d = np.fft.ifft2(FFTd2d)*FFTd2d.size/d2dpad.size



# Extract core
#d2d_out = np.real(IFFTd2d[npady*2:-(npady*2+oddy+1),npadx*2:-(npadx*2+oddx+1)])
d2d_out = np.real(IFFTd2d[:-1,:-1])
d2d_out = d2d_out[npady*2:-(npady*2+oddy),npadx*2:-(npadx*2+oddx)]

# Create new distance vector
XX = np.linspace(np.min(xr),np.max(xr),d2d_out.shape[1])
YY = np.linspace(np.min(yr),np.max(yr),d2d_out.shape[0])

start_time = time.time()
F = interpolation.interp2d(XX,YY,d2d_out)
d_i2d_fft = mkvc( F(xx,0.) )    
l1_r[ii,4] = np.sum( np.abs(d_line - d_i2d_nnb) )**0.5
l2_r[ii,4] = np.sum( (d_line - d_i2d_nnb)**2. )
linf_r[ii,4] = np.max( np.abs(d_line - d_i2d_nnb) )
timer[ii,4] = (time.time() - start_time)
    
print("--- FFT completed in %s seconds ---" % (time.time() - start_time))

plt.figure()
ax = plt.subplot()
plt.imshow(d2d_out, interpolation="bicubic", extent=[xr.min(), xr.max(), yr.min(), yr.max() ], origin = 'lower')
plt.colorbar(fraction=0.04)
    

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
plt.figure()
ax = plt.subplot()
plt.imshow(d2d, interpolation="bicubic", extent=[xr.min(), xr.max(), yr.min(), yr.max() ], origin = 'lower')
plt.colorbar(fraction=0.04)

plt.figure()
plt.contour(X,Y, np.reshape(d,X.shape),10)
plt.scatter(X,Y, c=np.reshape(d,X.shape), s=10)
plt.scatter(xx,yy, c='k', s=20, marker='o')
ax.set_title('Numerical')


#%%

plt.figure()
ax = plt.subplot()
plt.plot(xx,d_line,c='r', linewidth=3)
plt.plot(xx,d_i2d_lin,c='b')
plt.plot(xx,d_i2d_cub,c='g')
plt.plot(xx,d_i2d_qui,c='m')
plt.plot(xx,d_i2d_nnb,c='k')
plt.plot(xx,d_i2d_fft,c='c',marker='o')
plt.plot(xr[::indx],np.zeros(len(xr[::indx])),c='r',linewidth=0.,marker='o')
ax.set_title('Analytical')

#%% Write result to file
with file('l2_residual.dat','w') as fid:
    fid.write('NearestN \t Linear \t Cubic \t Quintic \t FFT\n')
    np.savetxt(fid, l2_r, fmt='%e',delimiter=' ',newline='\n')

with file('l1_residual.dat','w') as fid:
    fid.write('NearestN \t Linear \t Cubic \t Quintic \t FFT\n')
    np.savetxt(fid, l1_r, fmt='%e',delimiter=' ',newline='\n')