# from SimPEG import Problem
from SimPEG import Utils
from SimPEG import Mesh
from SimPEG import Survey
from SimPEG import Regularization
from SimPEG import DataMisfit
from SimPEG import Optimization
from SimPEG import InvProblem
from SimPEG import Directives
from SimPEG import Inversion

from SimPEG.SEIS.StraightRay import *

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt


def run(plotIt=False):

    """
        Seismic Straight Ray Tomography
        ===============================

    """

    O = np.r_[-1.2, -1.]
    D = np.r_[10., 10.]
    x = np.r_[0., 1.]
    y = np.r_[0., 1.]
    print 'length:', lengthInCell(O, D, x, y, plotIt=True)
    O = np.r_[0, -1.]
    D = np.r_[1., 1.]*1.5
    print 'length:', lengthInCell(O, D, x*2, y*2, plotIt=True)

    nC = 20
    M = Mesh.TensorMesh([nC, nC])
    y = np.linspace(0., 1., nC/2)
    rlocs = np.c_[y*0+M.vectorCCx[-1], y]
    rx = Survey.BaseRx(rlocs, None)

    srcList = [
        Survey.BaseSrc(loc=np.r_[M.vectorCCx[0], yi], rxList=[rx])
        for yi in y
    ]

    survey = StraightRaySurvey(srcList)
    problem = StraightRayProblem(M)
    problem.pair(survey)

    s = Utils.mkvc(Utils.ModelBuilder.randomModel(M.vnC)) + 1.
    survey.dobs = survey.dpred(s)
    survey.std = 0.01
    plt.plot(survey.dobs)
    plt.plot(survey.dpred(s*0+1.5))
    ax = plt.subplot(111)
    M.plotImage(s, ax=ax)
    survey.plot(ax=ax)

    # Create an optimization program
    opt = Optimization.InexactGaussNewton(maxIter=10)
    opt.remember('xc')
    # Create a regularization program
    # regModel = Model.BaseModel(M)

    reg = Regularization.Tikhonov(M)
    dmis = DataMisfit.l2_DataMisfit(survey)
    opt = Optimization.InexactGaussNewton(maxIter=40)
    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
    beta = Directives.BetaSchedule()
    betaest = Directives.BetaEstimate_ByEig()
    inv = Inversion.BaseInversion(invProb, directiveList=[beta, betaest])

    # Start the inversion with a model of zeros, and run the inversion
    m0 = np.ones(M.nC)*1.5
    mopt = inv.run(m0)

    plt.colorbar(M.plotImage(mopt)[0])
    plt.colorbar(M.plotImage(s)[0])

if __name__ == '__main__':
    run(plotIt=True)
    plt.show()
