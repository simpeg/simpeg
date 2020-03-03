"""
Seismic Straight Ray Tomography
===============================

Example of a straight ray tomography inverse problem.

"""

from SimPEG import utils
import discretize
from SimPEG import maps
from SimPEG import regularization
from SimPEG import data_misfit
from SimPEG import optimization
from SimPEG import inverse_problem
from SimPEG import directives
from SimPEG import inversion

from SimPEG.seismic import straight_ray_tomography as tomo

import numpy as np
import matplotlib.pyplot as plt


def run(plotIt=False):

    O = np.r_[-1.2, -1.]
    D = np.r_[10., 10.]
    x = np.r_[0., 1.]
    y = np.r_[0., 1.]
    print('length:', tomo.lengthInCell(O, D, x, y, plotIt=plotIt))
    O = np.r_[0, -1.]
    D = np.r_[1., 1.]*1.5
    print('length:', tomo.lengthInCell(O, D, x*2, y*2, plotIt=plotIt))

    nC = 20
    M = discretize.TensorMesh([nC, nC])
    y = np.linspace(0., 1., nC//2)
    rlocs = np.c_[y*0+M.vectorCCx[-1], y]
    rx = tomo.Rx(rlocs)

    srcList = [
        tomo.Src(location=np.r_[M.vectorCCx[0], yi], receiver_list=[rx])
        for yi in y
    ]

    survey = tomo.Survey(srcList)
    problem = tomo.Simulation(
        M, survey=survey, slownessMap=maps.IdentityMap(M))

    s = utils.mkvc(utils.model_builder.randomModel(M.vnC)) + 1.
    data = problem.make_synthetic_data(s, standard_deviation=0.01)

    # Create an optimization program

    reg = regularization.Tikhonov(M)
    dmis = data_misfit.L2DataMisfit(simulation=problem, data=data)
    opt = optimization.InexactGaussNewton(maxIter=40)
    opt.remember('xc')
    invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)
    beta = directives.BetaSchedule()
    betaest = directives.BetaEstimate_ByEig()
    inv = inversion.BaseInversion(invProb, directiveList=[beta, betaest])

    # Start the inversion with a model of zeros, and run the inversion
    m0 = np.ones(M.nC)*1.5
    mopt = inv.run(m0)

    if plotIt is True:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[1].plot(data.dobs)
        ax[1].plot(problem.dpred(m0), 's')
        ax[1].plot(problem.dpred(mopt), 'o')
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
