import os

home_dir = 'C:\\LC\\Private\\dominiquef\\Projects\\4414_Minsim\\Modeling\\MAG\\Lalor'

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
[mshfile, obsfile, modfile, magfile, topofile] = PF.BaseMag.read_MAGfwr_inp(inpfile)

# Load mesh file
mesh = Utils.meshutils.readUBCTensorMesh(mshfile)

# Load in observation file
#[B,M,dobs] = PF.BaseMag.readUBCmagObs(obsfile)

# Read in topo surface
topsurf = "Topography_Local.ts"
geosurf = [['_Top_Bollock_Rhyodacite_SUB.ts',False,True],
['_Top_Bollock_Rhyodacite_SUB_B.ts',False,True],
['_Top_Mafic_Intrusion.ts',True,True],
['_Top_Mafic_Volcaniclastic.ts',False,True],
['UnitA.ts',True,True],
['UnitB.ts',True,True],
['UnitC.ts',True,True],
['UnitD.ts',True,True],
['UnitE.ts',True,True],
['UnitF.ts',True,True],
]

vals = np.asarray([0.005,0.005,0.01,0.0025])

# Background density
bkgr = 0.0001

# Offset data above topo 
zoffset = 2


#%% Script starts here       
# # Create a grid of observations and offset the z from topo
#xr = np.linspace(mesh.vectorCCx[0], mesh.vectorCCx[-1], 99)
#yr = np.linspace(mesh.vectorCCy[0], mesh.vectorCCy[-1], 74)

xr = mesh.vectorCCx[::3]
yr = mesh.vectorCCy[::3]
X, Y = np.meshgrid(xr, yr)

topo = np.genfromtxt(topofile,skip_header=1)
F = interpolation.NearestNDInterpolator(topo[:,0:2],topo[:,2])
Z = F(X,Y) + zoffset
rxLoc = np.c_[Utils.mkvc(X.T), Utils.mkvc(Y.T), Utils.mkvc(Z.T)]
    
ndata = rxLoc.shape[0]

B = np.array(([90.,0.,50000.]))
  
M = np.array(([90.,0.,315.]))     

model= np.ones(mesh.nC) * bkgr
# Load GOCAD surf
#[vrtx, trgl] = PF.BaseMag.read_GOCAD_ts(tsfile)
# Find active cells from surface

for ii in range(len(geosurf)):
    tin = tm.time()
    print "Computing indices with VTK: " + geosurf[ii][0]
    indx = PF.BaseMag.gocad2vtk(geosurf[ii][0],mesh, bcflag = geosurf[ii][1], inflag = geosurf[ii][2])
    print "VTK operation completed in " + str(tm.time() - tin)
    
    model[indx] = vals[ii]
    
indx = PF.BaseMag.gocad2vtk(topsurf,mesh, bcflag = False, inflag = True) 
actv = np.zeros(mesh.nC)
actv[indx] = 1

model[actv==0] = -100

Utils.meshutils.writeUBCTensorModel('VTKout.dat',mesh,model)

Utils.meshutils.writeUBCTensorMesh('Mesh_temp.msh',mesh)

start_time = tm.time()
    
d = PF.Magnetics.Intgrl_Fwr_Data(mesh,B,M,rxLoc,model,actv,'tmi')

timer = (tm.time() - start_time)

#%% Plot data
plt.figure()
ax = plt.subplot()
plt.imshow(np.reshape(d,X.shape), interpolation="bicubic", extent=[xr.min(), xr.max(), yr.min(), yr.max()], origin = 'lower')
plt.clim(0,25)
plt.colorbar(fraction=0.02)
plt.contour(X,Y, np.reshape(d,X.shape),10)
plt.scatter(X,Y, c=np.reshape(d,X.shape), s=20)

ax.set_title('Forward data')