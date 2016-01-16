import os

home_dir = 'C:\Users\dominiquef.MIRAGEOSCIENCE\ownCloud\Research\Modelling\Synthetic\Block_Gaussian_topo'

inpfile = 'PYMAG3C_fwr.inp'

dsep = '\\'

os.chdir(home_dir)

#%%
from SimPEG import np, sp, Utils, mkvc, Maps
import simpegPF as PF
import pylab as plt

## New scripts to be added to basecode
#from fwr_MAG_data import fwr_MAG_data
#from read_MAGfwr_inp import read_MAGfwr_inp

#%%
# Read input file
[mshfile, obsfile, modfile, magfile, topofile] = PF.BaseMag.read_MAGfwr_inp(inpfile)

# Load mesh file
mesh = Utils.meshutils.readUBCTensorMesh(mshfile)

# Load model file
model = Utils.meshutils.readUBCTensorModel(modfile,mesh)
  
# Load in topofile or create flat surface
if topofile == 'null':
 
    actv = np.ones(mesh.nC)   
    
else: 
    topo = np.genfromtxt(topofile,skip_header=1)
    actv = PF.Magnetics.getActiveTopo(mesh,topo,'N')


Utils.writeUBCTensorModel('nullcell.dat',mesh,actv)
         
# Load in observation file
[B,M,dobs] = PF.BaseMag.readUBCmagObs(obsfile)

rxLoc = dobs[:,0:3]
#rxLoc[:,2] += 5 # Temporary change for test
ndata = rxLoc.shape[0]

# Load GOCAD surf
tsfile = 'SphereA.ts'
[vrtx, trgl] = PF.BaseMag.read_GOCAD_ts(tsfile)

#%% Run forward modeling
# Compute forward model using integral equation
d = PF.Magnetics.Intgrl_Fwr_Data(mesh,B,M,rxLoc,model,actv,'tmi')

# Form data object with coordinates and write to file
wd =  np.zeros((ndata,1))

# Save forward data to file
PF.Magnetics.writeUBCobs(home_dir + dsep + 'FWR_data.dat',B,M,rxLoc,d,wd)


