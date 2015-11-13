import os

home_dir = 'C:\Users\dominiquef.MIRAGEOSCIENCE\Documents\GIT\SimPEG\simpegdc\simpegDCIP\Dev'

inpfile = 'MAG3Cfwr.inp'

os.chdir(home_dir)

#%%
from SimPEG import np, Utils, Mesh
import simpegDCIP as DC
import pylab as plt

from readUBC_DC2DMesh import readUBC_DC2DMesh
from readUBC_DC2DModel import readUBC_DC2DModel

# Load UBC mesh 2D
mesh = readUBC_DC2DMesh('mesh2d.txt')

# Load model
model = readUBC_DC2DModel('model2d.con')

# load obs file

#%% Plot model
fig, axs = plt.subplots(1,1, figsize=(10,7))
h1 = mesh.plotImage(model, ax = axs)
plt.ylim([-1000,0])
plt.xlim([0,2000])

