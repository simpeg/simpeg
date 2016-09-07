# Test script to use SimPEG.MT platform to forward model synthetic data.

# Import
import SimPEG as simpeg
from SimPEG import MT
import numpy as np
try:
    from pymatsolver import PardisoSolver as Solver
except:
    from SimPEG import Solver

def run(plotIt=True, nFreq=1):
    """
        MT: 3D: Forward
        ===============

        Forward model 3D MT data.

    """

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
        for rxType in ['zxxr','zxxi','zxyr','zxyi','zyxr','zyxi','zyyr','zyyi','tzxr','tzxi','tzyr','tzyi']:
            rxList.append(MT.Rx(simpeg.mkvc(loc,2).T,rxType))
    # Source list
    srcList =[]
    for freq in np.logspace(3,-3,nFreq):
        srcList.append(MT.SrcMT.polxy_1Dprimary(rxList,freq))
    # Survey MT
    survey = MT.Survey(srcList)

    ## Setup the problem object
    problem = MT.Problem3D.eForm_ps(M, sigmaPrimary=sigBG, Solver=Solver)
    problem.pair(survey)

    # Calculate the data
    fields = problem.fields(sig)
    dataVec = survey.eval(fields)

    # Make the data
    mtData = MT.Data(survey, dataVec)
    # Add plots
    if plotIt:
        pass

if __name__ == '__main__':
    run()
