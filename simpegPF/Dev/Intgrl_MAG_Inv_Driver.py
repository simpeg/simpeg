import os

home_dir = 'C:\Users\dominiquef.MIRAGEOSCIENCE\Documents\GIT\SimPEG\simpegpf\simpegPF\Dev'

inpfile = 'MAG3Cinv.inp'

os.chdir(home_dir)

#%%
from SimPEG import np, Utils
import simpegPF as PF

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
    
# Load in observation file
[B,M,dobs] = PF.BaseMag.readUBCmagObs(obsfile)

rxLoc = dobs[:,0:3]
ndata = rxLoc.shape[0]

# Create forward operator
F = PF.Magnetics.Intrgl_Fwr_Op(mesh,B,M,rxLoc,'tmi')



