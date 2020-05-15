"""
Inversion: Linear Problem
=========================

Here we go over the basics of creating a linear problem and inversion.

"""
from __future__ import print_function
import numpy as np
import discretize
from SimPEG import simulation
from SimPEG import maps
from SimPEG import data_misfit
from SimPEG import directives
from SimPEG import optimization
from SimPEG import regularization
from SimPEG import inverse_problem
from SimPEG import inversion
import matplotlib.pyplot as plt


def run(N=100, plotIt=True):

    np.random.seed(1)

    mesh = discretize.TensorMesh([N])

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

    prob = simulation.LinearSimulation(mesh, G=G, model_map=maps.IdentityMap(mesh))
    data = prob.make_synthetic_data(mtrue, relative_error=0.01, add_noise=True)

    M = prob.mesh

    reg = regularization.Tikhonov(mesh, alpha_s=1., alpha_x=1.)
    dmis = data_misfit.L2DataMisfit(simulation=prob, data=data)
    opt = optimization.InexactGaussNewton(maxIter=60)
    invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)
    directive_list = [
        directives.BetaEstimate_ByEig(beta0_ratio=1e-2),
        directives.TargetMisfit()
    ]
    inv = inversion.BaseInversion(invProb, directiveList=directive_list)
    m0 = np.zeros(M.nC)

    mrec = inv.run(m0)

    if plotIt:
        fig, axes = plt.subplots(1, 2, figsize=(12*1.2, 4*1.2))
        for i in range(prob.G.shape[0]):
            axes[0].plot(prob.G[i, :])
        axes[0].set_title('Columns of matrix G')

        axes[1].plot(M.vectorCCx, mtrue, 'b-')
        axes[1].plot(M.vectorCCx, mrec, 'r-')
        axes[1].legend(('True Model', 'Recovered Model'))
        axes[1].set_ylim([-2, 2])

    return prob, data, mesh, mrec

if __name__ == '__main__':
    run()
    plt.show()
