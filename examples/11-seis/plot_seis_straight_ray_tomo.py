"""
Seismic Straight Ray Tomography
===============================

Example of a straight ray tomography inverse problem.

"""

from SimPEG import Utils
from SimPEG import Mesh
from SimPEG import Maps
from SimPEG import Survey
from SimPEG import Regularization
from SimPEG import DataMisfit
from SimPEG import Optimization
from SimPEG import InvProblem
from SimPEG import Directives
from SimPEG import Inversion

from SimPEG.SEIS import StraightRay

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt


def run(plotIt=False):

    O = np.r_[-1.2, -1.]
    D = np.r_[10., 10.]
    x = np.r_[0., 1.]
    y = np.r_[0., 1.]
    print('length:', StraightRay.lengthInCell(O, D, x, y, plotIt=plotIt))
    O = np.r_[0, -1.]
    D = np.r_[1., 1.]*1.5
    print('length:', StraightRay.lengthInCell(O, D, x*2, y*2, plotIt=plotIt))

    nC = 20
    M = Mesh.TensorMesh([nC, nC])
    y = np.linspace(0., 1., nC/2)
    rlocs = np.c_[y*0+M.vectorCCx[-1], y]
    rx = StraightRay.Rx(rlocs, None)

    srcList = [
        StraightRay.Src(loc=np.r_[M.vectorCCx[0], yi], rxList=[rx])
        for yi in y
    ]

    survey = StraightRay.Survey(srcList)
    problem = StraightRay.Problem(M, slownessMap=Maps.IdentityMap(M))
    problem.pair(survey)

    s = Utils.mkvc(Utils.ModelBuilder.randomModel(M.vnC)) + 1.
    survey.dobs = survey.dpred(s)
    survey.std = 0.01

    # Create an optimization program

    reg = Regularization.Tikhonov(M)
    dmis = DataMisfit.l2_DataMisfit(survey)
    opt = Optimization.InexactGaussNewton(maxIter=40)
    opt.remember('xc')
    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
    beta = Directives.BetaSchedule()
    betaest = Directives.BetaEstimate_ByEig()
    inv = Inversion.BaseInversion(invProb, directiveList=[beta, betaest])

    # Start the inversion with a model of zeros, and run the inversion
    m0 = np.ones(M.nC)*1.5
    mopt = inv.run(m0)

    if plotIt is True:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[1].plot(survey.dobs)
        ax[1].plot(survey.dpred(m0), 's')
        ax[1].plot(survey.dpred(mopt), 'o')
        ax[1].legend(['dobs', 'starting dpred', 'dpred'])
        M.plotImage(s, ax=ax[0])
        survey.plot(ax=ax[0])
        ax[0].set_title('survey')

        plt.tight_layout()

    if plotIt is True:
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        plt.colorbar(M.plotImage(m0, ax=ax[0])[0], ax=ax[0])
        plt.colorbar(M.plotImage(mopt, ax=ax[1])[0], ax=ax[1])
        plt.colorbar(M.plotImage(s, ax=ax[2])[0], ax=ax[2])

        ax[0].set_title('Starting Model')
        ax[1].set_title('Recovered Model')
        ax[2].set_title('True Model')

        plt.tight_layout()

if __name__ == '__main__':
    run(plotIt=True)
    plt.show()
