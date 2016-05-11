from SimPEG import *


def run(N=200, plotIt=True):
    """
        Inversion: Linear Problem
        =========================

        Here we go over the basics of creating a linear problem and inversion.

    """


    np.random.seed(1)

    std_noise = 1e-2

    mesh = Mesh.TensorMesh([N])

    m0 = np.ones(mesh.nC) * 1e-4
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
    #survey.makeSyntheticData(mtrue, std=std_noise)

    wd = np.ones(nk) * std_noise

    #print survey.std[0]
    #M = prob.mesh
    # Distance weighting
    wr = np.sum(prob.G**2.,axis=0)**0.5
    wr = ( wr/np.max(wr) )

    reg = Regularization.Simple(mesh)
    reg.wght = wr

    dmis = DataMisfit.l2_DataMisfit(survey)
    dmis.Wd = 1./wd

    opt = Optimization.ProjectedGNCG(maxIter=30,lower=-2.,upper=2., maxIterCG= 20, tolCG = 1e-4)
    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
    invProb.curModel = m0

    beta = Directives.BetaSchedule(coolingFactor=2, coolingRate=1)
    target = Directives.TargetMisfit()

    betaest = Directives.BetaEstimate_ByEig()
    inv = Inversion.BaseInversion(invProb, directiveList=[beta, betaest, target])


    mrec = inv.run(m0)
    ml2 = mrec
    print "Final misfit:" + str(invProb.dmisfit.eval(mrec))

    # Switch regularization to sparse
    phim = invProb.phi_m_last
    phid =  invProb.phi_d

    reg = Regularization.Sparse(mesh)

#==============================================================================
#     fig, axes = plt.subplots(1,2,figsize=(12*1.2,4*1.2))
#     dmdx = reg.mesh.cellDiffxStencil * mrec
#     plt.plot(np.sort(dmdx))
#==============================================================================

    #reg.recModel = mrec
    # reg.cell_weight = np.ones(mesh.nC)
    reg.mref = np.zeros(mesh.nC)
    reg.eps_p = 5e-2
    reg.eps_q = 1e-2
    reg.norms   = [0., 0., 2., 2.]
    reg.cell_weight = wr

    opt = Optimization.ProjectedGNCG(maxIter=10 ,lower=-2.,upper=2., maxIterLS = 20, maxIterCG= 20, tolCG = 1e-3)
    invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta = invProb.beta*2.)
    beta = Directives.BetaSchedule(coolingFactor=1, coolingRate=1)
    #betaest = Directives.BetaEstimate_ByEig()
    target = Directives.TargetMisfit()
    IRLS =Directives.Update_IRLS( phi_m_last = phim, phi_d_last = phid )

    inv = Inversion.BaseInversion(invProb, directiveList=[beta,IRLS])

    m0 = mrec

    # Run inversion
    mrec = inv.run(m0)

    print "Final misfit:" + str(invProb.dmisfit.eval(mrec))


    if plotIt:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1,2,figsize=(12*1.2,4*1.2))
        for i in range(prob.G.shape[0]):
            axes[0].plot(prob.G[i,:])
        axes[0].set_title('Columns of matrix G')

        axes[1].plot(mesh.vectorCCx, mtrue, 'b-')
        axes[1].plot(mesh.vectorCCx, ml2, 'r-')
        #axes[1].legend(('True Model', 'Recovered Model'))
        axes[1].set_ylim(-1.0,1.25)

        axes[1].plot(mesh.vectorCCx, mrec, 'k-',lw = 2)
        axes[1].legend(('True Model', 'Smooth l2-l2',
        'Sparse lp:' + str(reg.norms[0]) + ', lqx:' + str(reg.norms[1]) ), fontsize = 12)
        plt.show()

    return prob, survey, mesh, mrec

if __name__ == '__main__':
    run()
