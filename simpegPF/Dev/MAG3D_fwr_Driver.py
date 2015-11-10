import os

home_dir = 'C:\Users\dominiquef.MIRAGEOSCIENCE\Documents\GIT\Research\SimPeg'

inpfile = 'MAG3Cfwr.inp'

os.chdir(home_dir)

#%%
from SimPEG import np, Utils
from simpegPF import BaseMag

## New scripts to be added to basecode
from fwr_MAG_obs import fwr_MAG_obs
from read_MAGfwr_inp import read_MAGfwr_inp

#%%
# Read input file
[mshfile, obsfile, modfile, magfile, topofile] = read_MAGfwr_inp(inpfile)

# Load mesh file
mesh = Utils.meshutils.readUBCTensorMesh(mshfile)

# Load model file
model = Utils.meshutils.readUBCTensorModel(modfile,mesh)
    
# Load in observation file
[B,M,dobs] = BaseMag.readUBCmagObs(obsfile)

rxLoc = dobs[:,0:3]

# Compute forward model using integral equation
d = fwr_MAG_obs(xn,yn,zn,B,M,rxLoc,model)

# Form data object with coordinates and write to file
data = np.c_[rxLoc , d , np.zeros((ndata,1))]

# Save forward data to file
with file('FWR_data.dat','w') as fid:
    fid.write('%6.2f %6.2f %6.2f\n' %(B[0], B[1], B[2]) )
    fid.write('%6.2f %6.2f %6.2f\n' %(M[0], M[1], 1) )  
    fid.write('%i\n' %(ndata) ) 
    np.savetxt(fid, data, fmt='%e',delimiter=' ',newline='\n')

print "Observation file saved to " + home_dir + '\FWR_data.dat'


