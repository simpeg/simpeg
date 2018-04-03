"""
Inversion: Linear Problem
=========================

Here we go over the basics of creating a linear problem and inversion.

"""
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
from SimPEG import Maps
import matplotlib.pyplot as plt


def run(N=100, plotIt=True):

    N = 100
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
    m = np.r_[mtrue, mtrue]
    wires = Maps.Wires(('m1', mesh.nC), ('m2', mesh.nC))
    prob1 = Problem.LinearProblem(mesh, G=G, modelMap=wires.m1)
    survey1 = Survey.LinearSurvey()
    survey1.pair(prob1)
    survey1.makeSyntheticData(m, std=0.01)

    prob2 = Problem.LinearProblem(mesh, G=G, modelMap=wires.m2)
    survey2 = Survey.LinearSurvey()
    survey2.pair(prob2)
    survey2.makeSyntheticData(m, std=0.01)

    reg1 = Regularization.Tikhonov(
        mesh, alpha_s=1., alpha_x=1., mapping=wires.m1
        )
    reg2 = Regularization.Tikhonov(
        mesh, alpha_s=1., alpha_x=1., mapping=wires.m2
        )
    reg = reg1 + reg2

    dmis1 = DataMisfit.l2_DataMisfit(survey1)
    dmis2 = DataMisfit.l2_DataMisfit(survey2)
    dmis = 0.75*dmis1 + 0.25*dmis2

    opt = Optimization.InexactGaussNewton(maxIter=100)
    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
    directives = [
        Directives.BetaEstimate_ByEig(beta0_ratio=1e-2),
        Directives.BetaSchedule(),
        Directives.PetroTargetMisfit(TriggerSmall=False, verbose=True),
        # Directives.JointScalingSchedule(verbose=True),
        # Directives.AlphasSmoothEstimate_ByEig(ninit=10)
        Directives.ScalingEstimate_ByEig(verbose=True),
        Directives.AddMrefInSmooth(verbose=True)
        ]
    inv = Inversion.BaseInversion(invProb, directiveList=directives)
    m0 = np.zeros(mesh.nC*2)
    mrec = inv.run(m0)

    print(reg.multipliers)

    if plotIt:
        fig, axes = plt.subplots(1, 2, figsize=(12*1.2, 4*1.2))
        for i in range(prob1.G.shape[0]):
            axes[0].plot(prob1.G[i, :])
        axes[0].set_title('Columns of matrix F')

        axes[1].plot(mesh.vectorCCx, mtrue, 'b-')
        axes[1].plot(mesh.vectorCCx, wires.m1*mrec, 'r-')
        axes[1].plot(mesh.vectorCCx, wires.m2*mrec, 'r.')
        axes[1].legend(('True Model', 'Recovered Model1', 'Recovered Model2'))
        axes[1].set_ylim([-2, 2])

    return prob1, prob2, survey1, survey2, mesh, mrec

if __name__ == '__main__':
    run()
    plt.show()
