"""
Inversion: Linear: IRLS
=======================

Here we go over the basics of creating a linear problem and inversion.
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from SimPEG import Mesh
from SimPEG import Problem
from SimPEG import Survey
from SimPEG import DataMisfit
from SimPEG import Directives
from SimPEG import Optimization
from SimPEG import Regularization
from SimPEG import InvProblem
from SimPEG import Inversion
from SimPEG import Maps


def run(N=100, plotIt=True):

    np.random.seed(1)

    std_noise = 1e-2

    mesh = Mesh.TensorMesh([N])

    m0 = np.ones(mesh.nC) * 1e-4
    mref = np.zeros(mesh.nC)

    nk = 20
    jk = np.linspace(1., 60., nk)
    p = -0.25
    q = 0.25

    def g(k):
        return (
            np.exp(p*jk[k]*mesh.vectorCCx) *
            np.cos(np.pi*q*jk[k]*mesh.vectorCCx)
        )

    G = np.empty((nk, mesh.nC))

    for i in range(nk):
        G[i, :] = g(i)

    mtrue = np.zeros(mesh.nC)
    mtrue[mesh.vectorCCx > 0.3] = 1.
    mtrue[mesh.vectorCCx > 0.45] = -0.5
    mtrue[mesh.vectorCCx > 0.6] = 0

    prob = Problem.LinearProblem(mesh, G=G)
    survey = Survey.LinearSurvey()
    survey.pair(prob)
    survey.dobs = prob.fields(mtrue) + std_noise * np.random.randn(nk)

    wd = np.ones(nk) * std_noise

    # Distance weighting
    wr = np.sum(prob.getJ(m0)**2., axis=0)**0.5
    wr = wr/np.max(wr)

    dmis = DataMisfit.l2_DataMisfit(survey)
    dmis.W = 1./wd

    betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1e0)

    # Creat reduced identity map
    idenMap = Maps.IdentityMap(nP=mesh.nC)

    reg = Regularization.Sparse(mesh, mapping=idenMap)
    reg.mref = mref
    reg.cell_weights = wr
    reg.norms = np.c_[0., 0., 2., 2.]
    reg.mref = np.zeros(mesh.nC)

    opt = Optimization.ProjectedGNCG(
        maxIter=100, lower=-2., upper=2.,
        maxIterLS=20, maxIterCG=10, tolCG=1e-3
    )
    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
    update_Jacobi = Directives.UpdatePreconditioner()

    # Set the IRLS directive, penalize the lowest 25 percentile of model values
    # Start with an l2-l2, then switch to lp-norms

    IRLS = Directives.Update_IRLS(
        maxIRLSiter=40, minGNiter=1, f_min_change=1e-4)
    saveDict = Directives.SaveOutputEveryIteration(save_txt=False)
    inv = Inversion.BaseInversion(
        invProb,
        directiveList=[IRLS, betaest, update_Jacobi, saveDict]
    )

    # Run inversion
    mrec = inv.run(m0)

    print("Final misfit:" + str(invProb.dmisfit(mrec)))

    if plotIt:
        fig, axes = plt.subplots(2, 2, figsize=(12*1.2, 8*1.2))
        for i in range(prob.G.shape[0]):
            axes[0, 0].plot(prob.G[i, :])
        axes[0, 0].set_title('Columns of matrix G')

        axes[0, 1].plot(mesh.vectorCCx, mtrue, 'b-')
        axes[0, 1].plot(mesh.vectorCCx, invProb.l2model, 'r-')
        # axes[0, 1].legend(('True Model', 'Recovered Model'))
        axes[0, 1].set_ylim(-1.0, 1.25)

        axes[0, 1].plot(mesh.vectorCCx, mrec, 'k-', lw=2)
        axes[0, 1].legend(
            (
                'True Model',
                'Smooth l2-l2',
                'Sparse norms: {0}'.format(*reg.norms)
            ),
            fontsize=12
        )

        axes[1, 1].plot(saveDict.phi_d, 'k', lw=2)

        twin = axes[1, 1].twinx()
        twin.plot(saveDict.phi_m, 'k--', lw=2)
        axes[1, 1].plot(
            np.r_[IRLS.iterStart, IRLS.iterStart],
            np.r_[0, np.max(saveDict.phi_d)], 'k:'
        )
        axes[1, 1].text(
            IRLS.iterStart, 0.,
            'IRLS Start', va='bottom', ha='center',
            rotation='vertical', size=12,
            bbox={'facecolor': 'white'}
        )

        axes[1, 1].set_ylabel('$\phi_d$', size=16, rotation=0)
        axes[1, 1].set_xlabel('Iterations', size=14)
        axes[1, 0].axis('off')
        twin.set_ylabel('$\phi_m$', size=16, rotation=0)

    return prob, survey, mesh, mrec


if __name__ == '__main__':
    run()
    plt.show()
