## Script to run 3D forward model
## Forward model a data
import numpy as np, sys, os, time, gzip, cPickle as pickle
sys.path.append('/tera_raid/gudni/gitCodes/simpegmt')
sys.path.append('/tera_raid/gudni/gitCodes/simpegem')
sys.path.append('/tera_raid/gudni/gitCodes/simpeg')
import simpegMT as simpegmt, SimPEG as simpeg
import numpy as np, scipy
## Setup the forward modeling
# Read the model
modelname = "simpegTDmodel.con"
sigma = np.loadtxt(modelname)
# Make the mesh.
mTensor = simpeg.Utils.meshTensor
cSize = [50,20]
# Cells constant size mesh
hx = mTensor([(cSize[0],50)])
hy = mTensor([(cSize[0],50)])
hz = mTensor([(cSize[1],48)])
x0 = np.array([-1250,-1250,- 30*20])
mesh3dCons = simpeg.Mesh.TensorMesh([hx,hy,hz],x0)
# With padding
hPad = mTensor([(cSize[0],5,1.5)])
aPad = mTensor([(cSize[1],13,1.3)])
bPad = mTensor([(cSize[1],5,-1.5)])
hxPad = np.hstack((hPad[::-1],mTensor([(cSize[0],40)]),hPad))
hyPad = np.hstack((hPad[::-1],mTensor([(cSize[0],40)]),hPad))
hzPad = np.hstack((bPad,mTensor([(cSize[1],30)]),aPad))
x0Pad = np.array([-(np.sum(hPad)+1000),-(np.sum(hPad)+1000),-(np.sum(bPad)+600)])
mesh3d = simpeg.Mesh.TensorMesh([hxPad,hyPad,hzPad],x0Pad)

# Load the model to the uniform cell mesh
modelUniCell = simpeg.Utils.meshutils.readUBCTensorModel(modelname,mesh3dCons)
# Save as a vtk file
simpeg.Utils.meshutils.writeVTRFile('modelTDuniMesh.vtr',mesh3dCons,{'S/m':modelUniCell})

# Load the model to the mesh with padding cells
modelTD = simpeg.Utils.meshutils.readUBCTensorModel(modelname,mesh3d)
# Save as a vtk file
simpeg.Utils.meshutils.writeVTRFile('modelTDpaddedMesh.vtr',mesh3d,{'S/m':modelTD})

# Define the data locations
xG,yG = np.meshgrid(np.linspace(-700,700,8),np.linspace(-700,700,8))
zG = np.zeros_like(xG)
locs = np.hstack((simpeg.mkvc(xG.ravel(),2),simpeg.mkvc(yG.ravel(),2),simpeg.mkvc(zG.ravel(),2)))

# Make the receiver list
rxList = []
for rxType in ['zxxr','zxxi','zxyr','zxyi','zyxr','zyxi','zyyr','zyyi']:
    rxList.append(simpegmt.SurveyMT.RxMT(locs,rxType))
# Source list
srcList =[]
freqs = np.logspace(3,0,13)
for freq in freqs:
    srcList.append(simpegmt.SurveyMT.srcMT_polxy_1Dprimary(rxList,freq))
# Survey MT
survey = simpegmt.SurveyMT.SurveyMT(srcList)

# Setup the problem object
sigma1d = mesh3d.r(modelTD,'CC','CC','M')[0,0,:] # Use the edge column as a background model
problem = simpegmt.ProblemMT3D.eForm_ps(mesh3d,sigmaPrimary = sigma1d)
problem.verbose = False
from pymatsolver import MumpsSolver
problem.Solver = MumpsSolver
problem.pair(survey)

# Forward model the data
fields = problem.fields(modelTD)
mtData = survey.projectFields(fields)
# Save the data
np.save('seogiModel_MTdata.npy',simpeg.mkvc(mtData,1))

