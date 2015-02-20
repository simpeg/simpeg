# Test script to use simpegMT platform to forward model synthetic data.

# Import 
import simpegMT as simpegmt, SimPEG as simpeg 
import numpy as np

# Make a mesh
M = simpeg.Mesh.TensorMesh([[(100,5,-1.5),(100.,10),(100,5,1.5)],[(100,5,-1.5),(100.,10),(100,5,1.5)],[(100,5,1.6),(100.,10),(100,3,2)]], x0=['C','C',-3529.5360])
# Setup the model
conds = [1e-2,1]
sig = simpeg.Utils.ModelBuilder.defineBlock(M.gridCC,[-1000,-1000,-400],[1000,1000,-200],conds)
sig[M.gridCC[:,2]>0] = 1e-8
sig[M.gridCC[:,2]<-600] = 1e-1
sigBG = np.zeros(M.nC) + conds[0]
sigBG[M.gridCC[:,2]>0] = 1e-8

## Setup the the survey object
# Receiver locations
rx_x, rx_y = np.meshgrid(np.arange(-500,501,50),np.arange(-500,501,50))
rx_loc = np.hstack((simpeg.Utils.mkvc(rx_x,2),simpeg.Utils.mkvc(rx_y,2),np.zeros((np.prod(rx_x.shape),1))))
# Make a receiver list
rxList = []
for loc in rx_loc:
    # NOTE: loc has to be a (1,3) np.ndarray otherwise errors accure
    for rxType in ['zxxr','zxxi','zxyr','zxyi','zyxr','zyxi','zyyr','zyyi']:
        rxList.append(simpegmt.SurveyMT.RxMT(simpeg.mkvc(loc,2).T,rxType))
# Source list
srcList =[]
for freq in np.logspace(3,-1,5):
    srcList.append(simpegmt.SurveyMT.srcMT(freq,rxList))
# Survey MT 
survey = simpegmt.SurveyMT.SurveyMT(srcList)

## Setup the problem objec
problem = simpegmt.ProblemMT.MTProblem(M)
problem.pair(survey)

fields = problem.fields(sig,sigBG)
mtData = survey.projectFields(fields)

