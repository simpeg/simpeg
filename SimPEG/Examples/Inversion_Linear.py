from __future__ import print_function
import numpy as np
from SimPEG import Mesh
from SimPEG import Problem
from SimPEG import Survey
from SimPEG import DataMisfit
from SimPEG import Directives
from SimPEG import Optimization
from SimPEG import Regularization
from SimPEG import InvProblem
from SimPEG import Inversion
import matplotlib.pyplot as plt


def run(N=100, plotIt=True):
    """
        Inversion: Linear Problem
        =========================

        Here we go over the basics of creating a linear problem and inversion.

    """

    np.random.seed(1)

    mesh = Mesh.TensorMesh([N])

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
    survey.makeSyntheticData(mtrue, std=0.01)

    M = prob.mesh

    reg = Regularization.Tikhonov(mesh, alpha_s=1., alpha_x=1.)
    dmis = DataMisfit.l2_DataMisfit(survey)
    opt = Optimization.InexactGaussNewton(maxIter=60)
    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
    directives = [
        Directives.BetaEstimate_ByEig(beta0_ratio=1e-2),
        Directives.TargetMisfit()
    ]
    inv = Inversion.BaseInversion(invProb, directiveList=directives)
    m0 = np.zeros_like(survey.mtrue)

    mrec = inv.run(m0)

    if plotIt:
        fig, axes = plt.subplots(1, 2, figsize=(12*1.2, 4*1.2))
        for i in range(prob.G.shape[0]):
            axes[0].plot(prob.G[i, :])
        axes[0].set_title('Columns of matrix G')

        axes[1].plot(M.vectorCCx, survey.mtrue, 'b-')
        axes[1].plot(M.vectorCCx, mrec, 'r-')
        axes[1].legend(('True Model', 'Recovered Model'))
        axes[1].set_ylim([-2, 2])

    return prob, survey, mesh, mrec

if __name__ == '__main__':
    run()
    plt.show()
