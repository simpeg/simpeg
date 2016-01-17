import os

home_dir = '.\Test_tria_2_grid'

inpfile = 'PYMAG3C_fwr.inp'

dsep = '\\'

os.chdir(home_dir)

#%%
from SimPEG import np, sp, Utils, Mesh 
import scipy.interpolate as interpolation
import simpegPF as PF
import pylab as plt
import time as tm

## New scripts to be added to basecode
#from fwr_MAG_data import fwr_MAG_data
#from read_MAGfwr_inp import read_MAGfwr_inp

#%%
chibkg = 0.001

# Read in topo surface
tsfile = 'Topo_Gaussian.ts'

# For now just read both
topofile = 'Gaussian.topo'

topo = np.genfromtxt(topofile,skip_header=1) 

# Offset data above topo 
zoffset = 2

# Load mesh file
B = np.array(([90.,0.,50000.]))
  
M = np.array(([90.,0.,315.]))

# Sphere radius
R = 25.


#%% Script starts here       
# # Create a grid of observations and offset the z from topo
xr = np.linspace(-99., 99., 40)
yr = np.linspace(-49., 49., 20)
X, Y = np.meshgrid(xr, yr)

F = interpolation.NearestNDInterpolator(topo[:,0:2],topo[:,2])
Z = F(X,Y) + zoffset
rxLoc = np.c_[Utils.mkvc(X.T), Utils.mkvc(Y.T), Utils.mkvc(Z.T)]
    
ndata = rxLoc.shape[0]
      
sclx = 100.

dx = 10

nc = int(sclx/dx)

hxind = [(dx, 2*nc)]
hyind = [(dx, nc)]
hzind = [(dx, nc)]

mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CCN')
    
# Load GOCAD surf
#[vrtx, trgl] = PF.BaseMag.read_GOCAD_ts(tsfile)
# Find active cells from surface
tin = tm.time()
print "Computing indices with VTK: "
[indx, bc] = PF.BaseMag.gocad2vtk(tsfile,mesh, bcflag = False, inflag = True)
print "VTK operation completed in " + str(tm.time() - tin)

actv = np.zeros(mesh.nC)
actv[indx] = 1

model= np.zeros(mesh.nC)
model[indx]= chibkg

Utils.meshutils.writeUBCTensorModel('VTKout.dat',mesh,model)

Utils.meshutils.writeUBCTensorMesh('Mesh_temp.msh',mesh)

start_time = tm.time()
    
d = PF.Magnetics.Intgrl_Fwr_Data(mesh,B,M,rxLoc,model,actv,'tmi')

timer = (tm.time() - start_time)

#%% Plot data
plt.figure(1)
ax = plt.subplot()
plt.imshow(np.reshape(d,X.shape), interpolation="bicubic", extent=[xr.min(), xr.max(), yr.min(), yr.max()], origin = 'lower')
plt.colorbar(fraction=0.02)
plt.contour(X,Y, np.reshape(d,X.shape),10)
plt.scatter(X,Y, c=np.reshape(d,X.shape), s=20)
ax.set_title('Forward data')

# Load model file
#model = Utils.meshutils.readUBCTensorModel(modfile,mesh)
  
# Load in topofile or create flat surface
#==============================================================================
# if topofile == 'null':
#  
#     actv = np.ones(mesh.nC)   
#     
# else: 
#     topo = np.genfromtxt(topofile,skip_header=1)
#     actv = PF.Magnetics.getActiveTopo(mesh,topo,'N')
# 
# 
# Utils.writeUBCTensorModel('nullcell.dat',mesh,actv)
#          
# # Load in observation file
# [B,M,dobs] = PF.BaseMag.readUBCmagObs(obsfile)
# 
# rxLoc = dobs[:,0:3]
# #rxLoc[:,2] += 5 # Temporary change for test
# ndata = rxLoc.shape[0]
#==============================================================================


#%% Run forward modeling
# Compute forward model using integral equation
#==============================================================================
# d = PF.Magnetics.Intgrl_Fwr_Data(mesh,B,M,rxLoc,model,actv,'tmi')
# 
# # Form data object with coordinates and write to file
# wd =  np.zeros((ndata,1))
# 
# # Save forward data to file
# PF.Magnetics.writeUBCobs(home_dir + dsep + 'FWR_data.dat',B,M,rxLoc,d,wd)
#==============================================================================


