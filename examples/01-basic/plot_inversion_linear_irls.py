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

    F = np.empty((nk, mesh.nC))

    for i in range(nk):
        F[i, :] = g(i)

    mtrue = np.zeros(mesh.nC)
    mtrue[mesh.vectorCCx > 0.3] = 1.
    mtrue[mesh.vectorCCx > 0.45] = -0.5
    mtrue[mesh.vectorCCx > 0.6] = 0

    prob = Problem.LinearProblem(mesh, F=F)
    survey = Survey.LinearSurvey()
    survey.pair(prob)
    survey.dobs = prob.fields(mtrue) + std_noise * np.random.randn(nk)

    wd = np.ones(nk) * std_noise

    # Distance weighting
    wr = np.sum(prob.F**2., axis=0)**0.5
    wr = wr/np.max(wr)

    dmis = DataMisfit.l2_DataMisfit(survey)
    dmis.W = 1./wd

    betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1e-2)

    # Creat reduced identity map
    idenMap = Maps.IdentityMap(nP=mesh.nC)

    reg = Regularization.Sparse(mesh, mapping=idenMap)
    reg.mref = mref
    reg.cell_weights = wr
    reg.norms = [0., 0., 2., 2.]

    # The treshold for lq-norm is hardwired here, the
    # autopicker needs to be improved.
    reg.eps_q = 2e-2
    reg.mref = np.zeros(mesh.nC)

    opt = Optimization.ProjectedGNCG(
        maxIter=100, lower=-2., upper=2.,
        maxIterLS=20, maxIterCG=10, tolCG=1e-3
    )
    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
    update_Jacobi = Directives.Update_lin_PreCond()

    # Set the IRLS directive, penalize the lowest 25 percentile of model values
    # Start with an l2-l2, then switch to lp-norms

    IRLS = Directives.Update_IRLS(prctile=25, maxIRLSiter=15, minGNiter=3)

    inv = Inversion.BaseInversion(
        invProb,
        directiveList=[betaest, IRLS, update_Jacobi]
    )

    # Run inversion
    mrec = inv.run(m0)

    print("Final misfit:" + str(invProb.dmisfit(mrec)))

    if plotIt:
        fig, axes = plt.subplots(1, 2, figsize=(12*1.2, 4*1.2))
        for i in range(prob.F.shape[0]):
            axes[0].plot(prob.F[i, :])
        axes[0].set_title('Columns of matrix F')

        axes[1].plot(mesh.vectorCCx, mtrue, 'b-')
        axes[1].plot(mesh.vectorCCx, reg.l2model, 'r-')
        # axes[1].legend(('True Model', 'Recovered Model'))
        axes[1].set_ylim(-1.0, 1.25)

        axes[1].plot(mesh.vectorCCx, mrec, 'k-', lw=2)
        axes[1].legend(
            (
                'True Model',
                'Smooth l2-l2',
                'Sparse lp: {0}, lqx: {1}'.format(*reg.norms)
            ),
            fontsize=12
        )

    return prob, survey, mesh, mrec

if __name__ == '__main__':
    run()
    plt.show()
