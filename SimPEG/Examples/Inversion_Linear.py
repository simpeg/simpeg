from SimPEG import *


def run(N=100, plotIt=True):
    """
        Inversion: Linear Problem
        =========================

        Here we go over the basics of creating a linear problem and inversion.

    """

    np.random.seed(1)

    mesh = Mesh.TensorMesh([N])

    nk = 20
    jk = np.linspace(1.,20.,nk)
    p = -0.25
    q = 0.25

    g = lambda k: np.exp(p*jk[k]*mesh.vectorCCx)*np.cos(2*np.pi*q*jk[k]*mesh.vectorCCx)

    G = np.empty((nk, mesh.nC))

    for i in range(nk):
        G[i,:] = g(i)

    mtrue = np.zeros(mesh.nC)
    mtrue[mesh.vectorCCx > 0.3] = 1.
    mtrue[mesh.vectorCCx > 0.45] = -0.5
    mtrue[mesh.vectorCCx > 0.6] = 0

    prob = Problem.LinearProblem(mesh, G)
    survey = Survey.LinearSurvey()
    survey.pair(prob)
    survey.makeSyntheticData(mtrue, std=0.01)

    M = prob.mesh

    reg = Regularization.Tikhonov(mesh)
    dmis = DataMisfit.l2_DataMisfit(survey)
    opt = Optimization.InexactGaussNewton(maxIter=35)
    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
    beta = Directives.BetaSchedule()
    betaest = Directives.BetaEstimate_ByEig()
    inv = Inversion.BaseInversion(invProb, directiveList=[beta, betaest])
    m0 = np.zeros_like(survey.mtrue)

    mrec = inv.run(m0)

    if plotIt:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1,2,figsize=(12*1.2,4*1.2))
        for i in range(prob.G.shape[0]):
            axes[0].plot(prob.G[i,:])
        axes[0].set_title('Columns of matrix G')

        axes[1].plot(M.vectorCCx, survey.mtrue, 'b-')
        axes[1].plot(M.vectorCCx, mrec, 'r-')
        axes[1].legend(('True Model', 'Recovered Model'))
        plt.show()

    return prob, survey, mesh, mrec

if __name__ == '__main__':
    run()
