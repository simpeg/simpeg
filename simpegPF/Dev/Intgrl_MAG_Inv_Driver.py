import os

#home_dir = 'C:\Users\dominiquef.MIRAGEOSCIENCE\Documents\GIT\SimPEG\simpegpf\simpegPF\Dev'
home_dir = 'C:\\Users\\dominiquef.MIRAGEOSCIENCE\\ownCloud\\Research\\Modelling\\Synthetic\\Block_Gaussian_topo'

inpfile = 'PYMAG3D_inv.inp'

dsep = '\\'
os.chdir(home_dir)

#%%
from SimPEG import np, Utils, mkvc
import simpegPF as PF
import pylab as plt

## New scripts to be added to basecode
#from fwr_MAG_data import fwr_MAG_data
#from read_MAGfwr_inp import read_MAGfwr_inp

#%%
# Read input file
[mshfile, obsfile, topofile, mstart, mref, magfile, wgtfile, chi, alphas, bounds, lpnorms] = PF.BaseMag.read_MAGinv_inp(home_dir + dsep + inpfile)

# Load mesh file
mesh = Utils.meshutils.readUBCTensorMesh(mshfile)
    
# Load in observation file
[B,M,dobs] = PF.BaseMag.readUBCmagObs(obsfile)

rxLoc = dobs[:,0:3]
d = dobs[:,3]
wd = dobs[:,4]

ndata = rxLoc.shape[0]


# Load in topofile or create flat surface
if topofile == 'null':
    
    Nx,Ny = np.meshgrid(mesh.vectorNx,mesh.vectorNy)
    Nz = np.ones(Nx.shape) * mesh.vectorNz[-1]
    
    topo = np.c_[mkvc(Nx),mkvc(Ny),mkvc(Nz)]   
    
else: 
    topo = np.genfromtxt(topofile,skip_header=1)

# Work with flat topogrphy for now
actv = PF.Magnetics.getActiveTopo(mesh,topo,'N')

nC = int(sum(actv))

# Load model file
if isinstance(mstart, float):
    mstart = np.ones(nC) * mstart
else:
    mstart = Utils.meshutils.readUBCTensorModel(mstart,mesh)
    mstart = mstart[actv==1]

# Get magnetization vector for MOF
if magfile=='DEFAULT':
    
    M_xyz = PF.Magnetics.dipazm_2_xyz(np.ones(nC) * M[0], np.ones(nC) * M[1])
    
else:
    M_xyz = np.genfromtxt(magfile,delimiter=' \n',dtype=np.str,comments='!')

# Get index of the center
midx = int(mesh.nCx/2)
midy = int(mesh.nCy/2)

# Create forward operator
F = PF.Magnetics.Intrgl_Fwr_Op(mesh,B,M_xyz,rxLoc,actv,'tmi')

# Get distance weighting function
wr = PF.Magnetics.get_dist_wgt(mesh,rxLoc,3.,np.min(mesh.hx)/4)
wrMap = PF.BaseMag.WeightMap(mesh, wr)

Utils.writeUBCTensorModel(home_dir+dsep+'wr.dat',mesh,wr)

# Write out the predicted
pred = F.dot(mstart)
PF.Magnetics.writeUBCobs(home_dir + dsep + 'Pred.dat',B,M,rxLoc,pred,wd)

#%%
plt.figure()
ax = plt.subplot()
mesh.plotSlice(wr, ax = ax, normal = 'Y', ind=midx)
plt.title('Distance weighting')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')


