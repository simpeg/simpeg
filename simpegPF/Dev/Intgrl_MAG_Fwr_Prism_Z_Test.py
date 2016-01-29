"""
    Intgrl_MAG_Fwr_Prism_Z_test.py : Test the discretization error for a semi
    infinite prism extending from the surface.
    
    The test runs a sequence of forward calculations of magnetic over a 
    grid of points at varying height. The goal is to find a relation between
    observation height and mesh cell size.
    
    Since the integral equation is exact for a rectangular prism, the response 
    is first calculated for a semi-infinite prism aligned with the mesh with
    inducing field rotated 45d. The prism and data location is then rotate by
    -45d so prism is rotated with respect to base mesh. 
    
"""

#%%
from SimPEG import *
import matplotlib.pyplot as plt
import simpegPF as PF
import scipy.interpolate as interpolation
import time
import os

home_dir = 'C:\\LC\\Private\\dominiquef\\Projects\\4414_Minsim\\Modeling\\MAG'

os.chdir(home_dir)

#from fwr_MAG_data import fwr_MAG_data

plt.close('all')

topofile = []

zoffset = np.array([6., 5. ,3. , 2., 1.,0.5])

#%% Create survey
   
    
B = np.array(([90.,0.,50000.]))
  
M = np.array(([90.,0.,315.]))

# Block side lenght
R = 20.
       
# # Or create juste a plane grid
xr = np.linspace(-19.5, 19.5, 40)
yr = np.linspace(-19., 19., 40)
X, Y = np.meshgrid(xr, yr)



sclx = 90.
dx = np.asarray([5., 4. ,3. , 2., 1.])
 
d_iter = len(dx)
l1_r = np.zeros(d_iter)
l2_r = np.zeros(d_iter)
linf_r = np.zeros(d_iter)
timer = np.zeros(d_iter)
mcell = np.zeros(d_iter)
#%% First Loop through the observation heights and compute the true response
nc = int(sclx/30.)

hxind = [(30., nc)]
hyind = [(30., nc)]
hzind = [(1., 1)]

mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CCN')

actv = np.ones(mesh.nC)


# Pre-allocate true data
obs_x = np.zeros((X.size,zoffset.size))
obs_y = np.zeros((X.size,zoffset.size))
obs_z = np.zeros((X.size,zoffset.size))

plt.figure(1)
for ii in range(zoffset.size):
    
    # Drape observations on topo + offset
    if not topofile:
        Z = np.ones((xr.size, yr.size)) * zoffset[ii]
        
    else:
        F = interpolation.NearestNDInterpolator(topo[:,0:2],topo[:,2])
        Z = F(X,Y) + zoffset[ii]
        
    rxLoc = np.c_[Utils.mkvc(X.T), Utils.mkvc(Y.T), Utils.mkvc(Z.T)]
    
    ndata = rxLoc.shape[0]
        
    print 'Mesh size: ' + str(mesh.nC)
    
    #%% Create model
    chibkg = 0.
    chiblk = 0.01
    model = np.ones((mesh.nCx,mesh.nCy,mesh.nCz))*chibkg
    
    # Do a three sphere problem for more frequencies
    model[1:2,1:2,:] = chiblk    
    model = mkvc(model)
    
    Utils.writeUBCTensorMesh('Mesh.msh',mesh)
    Utils.writeUBCTensorModel('Model.sus',mesh,model)
    #actv = np.ones(mesh.nC)
    #%% Forward mode ldata
       
    temp = PF.Magnetics.Intgrl_Fwr_Data(mesh,B,M,rxLoc,model,actv,'xyz')
    
    obs_x[:,ii] = temp[0:ndata]
    obs_y[:,ii] = temp[ndata:2*ndata]
    obs_z[:,ii] = temp[2*ndata:]
    
    #%% Plot fields

    ax = plt.subplot(3,zoffset.size,ii+1)
    plt.imshow(np.reshape(obs_x[:,ii],X.shape), extent=[xr.min(), xr.max(), yr.min(), yr.max()], origin = 'lower')
    plt.colorbar(fraction=0.02)
    plt.contour(X,Y, np.reshape(obs_x[:,ii],X.shape),10)
    #plt.scatter(X,Y, c='k', s=10)
    
    ax = plt.subplot(3,zoffset.size,zoffset.size+ii+1)
    plt.imshow(np.reshape(obs_y[:,ii],X.shape), extent=[xr.min(), xr.max(), yr.min(), yr.max()], origin = 'lower')
    plt.colorbar(fraction=0.02)
    plt.contour(X,Y, np.reshape(obs_y[:,ii],X.shape),10)
    #plt.scatter(X,Y, c='k', s=10)
    
    ax = plt.subplot(3,zoffset.size,2*zoffset.size + ii+1)
    plt.imshow(np.reshape(obs_z[:,ii],X.shape), extent=[xr.min(), xr.max(), yr.min(), yr.max()], origin = 'lower')
    plt.colorbar(fraction=0.02)
    plt.contour(X,Y, np.reshape(obs_z[:,ii],X.shape),10)
    #plt.scatter(X,Y, c='k', s=10)

#%% Rotate the data and discretize the rotated block on various cell size

Rz = np.array([[np.cos(np.deg2rad(45)), -np.sin(np.deg2rad(45))],
               [np.sin(np.deg2rad(45)), np.cos(np.deg2rad(45))]])
   
temp = np.vstack([rxLoc[:,0].T,rxLoc[:,1].T])

ROTxy = np.dot(Rz,temp)      



    
#%%
    
blocksurf = ['Block_0.ts',True,True]

B = np.array(([90.,0.,50000.]))
  
M = np.array(([90.,0.,315.]))

l2 = np.zeros((dx.size,zoffset.size))
linf = np.zeros((dx.size,zoffset.size))

plt.figure()
for ii in range(dx.size):
    
    nc = int(sclx/dx[ii])
    
    hxind = [(dx[ii], nc)]
    hyind = [(dx[ii], nc)]
    hzind = [(1., 1)]
    
    mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CCN')
    
    actv = np.ones(mesh.nC)

    model = np.zeros(mesh.nC)
    
    indx = PF.BaseMag.gocad2vtk(blocksurf[0],mesh, bcflag = blocksurf[1], inflag = blocksurf[2])
    
    # Compensate for the volume difference
    scale = 1.#(30.**2.) / (len(indx)*dx[ii]**2.)
    model[indx] = chiblk * scale

    Utils.writeUBCTensorMesh('Mesh_' + str(int(dx[ii])) + 'm.msh',mesh)
    Utils.writeUBCTensorModel('Model_' + str(int(dx[ii])) + 'm.sus',mesh,model)
    
    
    for jj in range(zoffset.size): 
        
        # Drape observations on topo + offset
        if not topofile:
            Z = np.ones((xr.size, yr.size)) * zoffset[jj]
            
        else:
            F = interpolation.NearestNDInterpolator(topo[:,0:2],topo[:,2])
            Z = F(X,Y) + zoffset[jj]
        
        
        rxLoc_ROT = np.c_[ROTxy.T,mkvc(Z)]
        
        temp = PF.Magnetics.Intgrl_Fwr_Data(mesh,B,M,rxLoc_ROT,model,actv,'xyz')
        
        pre_x = temp[0:ndata]
        pre_y = temp[ndata:2*ndata]
        pre_z = temp[2*ndata:]
    
        res_z = (obs_z[:,jj] - pre_z) / np.max(np.abs(pre_z))
        l2[ii,jj] = np.sum( (res_z) ** 2. )**0.5 
        linf[ii,jj] = np.max( np.abs(res_z) )
        
        ax = plt.subplot(dx.size,zoffset.size,(ii*(zoffset.size))+jj+1)
        plt.imshow(np.reshape(res_z,X.shape), extent=[xr.min(), xr.max(), yr.min(), yr.max()], origin = 'lower')
        plt.colorbar(fraction=0.02)
        plt.contour(X,Y, np.reshape(res_z,X.shape),10)
        plt.title('Z:' + str(zoffset[jj]) + ' dx:' + str(dx[ii]))
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        #plt.scatter(rxLoc[:,0],rxLoc[:,1], c='k', s=10)
        
        
        


#%%
plt.figure()
plt.subplot(2,1,1)
for ii in range(l2.shape[0]):
    plt.plot(zoffset,l2[ii,:])
plt.xlabel('Z Heigth')
plt.ylabel('L2-residual')
plt.legend(dx.astype('str'))

plt.subplot(2,1,2)
for ii in range(l2.shape[1]):
    plt.plot(dx,l2[:,ii])
    
plt.legend(zoffset.astype('str'))
plt.xlabel('Cell Size')
plt.ylabel('L2-residual')


plt.figure()
plt.subplot(2,1,1)
for ii in range(linf.shape[0]):
    plt.plot(zoffset,linf[ii,:])
plt.xlabel('Z Heigth')
plt.ylabel('L_inf-residual')
plt.legend(dx.astype('str'))

plt.subplot(2,1,2)
for ii in range(linf.shape[1]):
    plt.plot(dx,linf[:,ii])
    
plt.legend(zoffset.astype('str'))
plt.xlabel('Cell Size')
plt.ylabel('Linf-residual')
#%% Plot results
#==============================================================================
# print 'Residual between analytical sphere and integral forward' 
# print "dx \t nc \t l1 \t l2 \t linf \t Runtime"
# for ii in range(d_iter):
# 
#     print str(dx[ii]) + "\t" + str(mcell[ii]) + "\t" + str(l1_r[ii]) + "\t" + str(l2_r[ii]) + "\t" + str(linf_r[ii]) + "\t" + str(timer[ii])
#     
#==============================================================================


#%% Plot the forward solution from integral
#==============================================================================
# plt.figure(2)
# ax = plt.subplot()
# plt.imshow(np.reshape(d,X.shape), interpolation="bicubic", extent=[xr.min(), xr.max(), yr.min(), yr.max() ], origin = 'lower')
# plt.colorbar(fraction=0.02)
# plt.contour(X,Y, np.reshape(d,X.shape),10)
# plt.scatter(X,Y, c=np.reshape(d,X.shape), s=20)
# ax.set_title('Numerical')
# 
# #%% Plot residual data
# plt.figure(3)
# ax = plt.subplot()
# plt.imshow(np.reshape(r_tmi,X.shape), interpolation="bicubic", extent=[xr.min(), xr.max(), yr.min(), yr.max()], origin = 'lower')
# plt.colorbar(fraction=0.02)
# plt.contour(X,Y, np.reshape(r_tmi,X.shape),10)
# plt.scatter(X,Y, c=np.reshape(r_tmi,X.shape), s=20)
# ax.set_title('Sphere Ana Bx')
#==============================================================================
