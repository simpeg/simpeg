from __future__ import print_function
from SimPEG import *


def run(N=100, plotIt=True):
    """
        Inversion for compact models (IRLS)
        ===================================

        Here we go over the basics of creating a linear problem and inversion.

    """


    np.random.seed(1)

    std_noise = 1e-2

    mesh = Mesh.TensorMesh([N])

    m0 = np.ones(mesh.nC) * 1e-4
    mref = np.zeros(mesh.nC)

    nk = 10
    jk = np.linspace(1.,nk,nk)
    p = -2.
    q = 1.

    g = lambda k: np.exp(p*jk[k]*mesh.vectorCCx)*np.cos(np.pi*q*jk[k]*mesh.vectorCCx)

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
    survey.dobs = prob.fields(mtrue) + std_noise * np.random.randn(nk)

    wd = np.ones(nk) * std_noise

    # Distance weighting
    wr = np.sum(prob.G**2.,axis=0)**0.5
    wr = ( wr/np.max(wr) )

    dmis = DataMisfit.l2_DataMisfit(survey)
    dmis.Wd = 1./wd

    betaest = Directives.BetaEstimate_ByEig()

    reg = Regularization.Sparse(mesh)
    reg.mref = mref
    reg.cell_weights = wr

    reg.mref = np.zeros(mesh.nC)
    

    opt = Optimization.ProjectedGNCG(maxIter=100 ,lower=-2.,upper=2., maxIterLS = 20, maxIterCG= 10, tolCG = 1e-3)
    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
    update_Jacobi = Directives.Update_lin_PreCond()
    
    # Set the IRLS directive, penalize the lowest 25 percentile of model values
    # Start with an l2-l2, then switch to lp-norms
    norms   = [0., 0., 2., 2.]    
    IRLS = Directives.Update_IRLS( norms=norms, prctile = 25, maxIRLSiter = 15, minGNiter=3)

    inv = Inversion.BaseInversion(invProb, directiveList=[IRLS,betaest,update_Jacobi])

    # Run inversion
    mrec = inv.run(m0)

    print("Final misfit:" + str(invProb.dmisfit.eval(mrec)))


    if plotIt:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1,2,figsize=(12*1.2,4*1.2))
        for i in range(prob.G.shape[0]):
            axes[0].plot(prob.G[i,:])
        axes[0].set_title('Columns of matrix G')

        axes[1].plot(mesh.vectorCCx, mtrue, 'b-')
        axes[1].plot(mesh.vectorCCx, reg.l2model, 'r-')
        #axes[1].legend(('True Model', 'Recovered Model'))
        axes[1].set_ylim(-1.0,1.25)

        axes[1].plot(mesh.vectorCCx, mrec, 'k-',lw = 2)
        axes[1].legend(('True Model', 'Smooth l2-l2',
        'Sparse lp:' + str(reg.norms[0]) + ', lqx:' + str(reg.norms[1]) ), fontsize = 12)
        plt.show()

    return prob, survey, mesh, mrec

if __name__ == '__main__':
    run()
