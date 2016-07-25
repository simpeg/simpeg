import SimPEG as simpeg
import numpy as np
import SimPEG.MT as MT
from scipy.constants import mu_0
import matplotlib.pyplot as plt

def run(plotIt=True):
    """
        MT: 1D: Inversion
        =================

        Forward model 1D MT data.
        Setup and run a MT 1D inversion.

    """

    ## Setup the forward modeling
    # Setting up 1D mesh and conductivity models to forward model data.
    # Frequency
    nFreq = 31
    freqs = np.logspace(3,-3,nFreq)
    # Set mesh parameters
    ct = 20
    air = simpeg.Utils.meshTensor([(ct,16,1.4)])
    core = np.concatenate( (  np.kron(simpeg.Utils.meshTensor([(ct,10,-1.3)]),np.ones((5,))) , simpeg.Utils.meshTensor([(ct,5)]) ) )
    bot = simpeg.Utils.meshTensor([(core[0],10,-1.4)])
    x0 = -np.array([np.sum(np.concatenate((core,bot)))])
    # Make the model
    m1d = simpeg.Mesh.TensorMesh([np.concatenate((bot,core,air))], x0=x0)

    # Setup model varibles
    active = m1d.vectorCCx<0.
    layer1 = (m1d.vectorCCx<-500.) & (m1d.vectorCCx>=-800.)
    layer2 = (m1d.vectorCCx<-3500.) & (m1d.vectorCCx>=-5000.)
    # Set the conductivity values
    sig_half = 2e-3
    sig_air = 1e-8
    sig_layer1 = .2
    sig_layer2 = .2
    # Make the true model
    sigma_true = np.ones(m1d.nCx)*sig_air
    sigma_true[active] = sig_half
    sigma_true[layer1] = sig_layer1
    sigma_true[layer2] = sig_layer2
    # Extract the model
    m_true = np.log(sigma_true[active])
    # Make the background model
    sigma_0 = np.ones(m1d.nCx)*sig_air
    sigma_0[active] = sig_half
    m_0 = np.log(sigma_0[active])

    # Set the mapping
    actMap = simpeg.Maps.InjectActiveCells(m1d, active, np.log(1e-8), nC=m1d.nCx)
    mappingExpAct = simpeg.Maps.ExpMap(m1d) * actMap

    ## Setup the layout of the survey, set the sources and the connected receivers
    # Receivers
    rxList = []
    for rxType in ['z1dr','z1di']:
        rxList.append(MT.Rx(simpeg.mkvc(np.array([0.0]),2).T,rxType))
    # Source list
    srcList =[]
    for freq in freqs:
            srcList.append(MT.SrcMT.polxy_1Dprimary(rxList,freq))
    # Make the survey
    survey = MT.Survey(srcList)
    survey.mtrue = m_true

    ## Set the problem
    problem = MT.Problem1D.eForm_psField(m1d,sigmaPrimary=sigma_0,mapping=mappingExpAct)
    problem.pair(survey)

    ## Forward model data
    # Project the data
    survey.dtrue = survey.dpred(m_true)
    survey.dobs = survey.dtrue + 0.025*abs(survey.dtrue)*np.random.randn(*survey.dtrue.shape)

    if plotIt:
        fig = MT.Utils.dataUtils.plotMT1DModelData(problem, [m_0])
        fig.suptitle('Target - smooth true')


    # Assign uncertainties
    std = 0.05 # 5% std
    survey.std = np.abs(survey.dobs*std)
    # Assign the data weight
    Wd = 1./survey.std

    ## Setup the inversion proceedure
    # Define a counter
    C =  simpeg.Utils.Counter()
    # Set the optimization
    opt = simpeg.Optimization.InexactGaussNewton(maxIter = 30)
    opt.counter = C
    opt.LSshorten = 0.5
    opt.remember('xc')
    # Data misfit
    dmis = simpeg.DataMisfit.l2_DataMisfit(survey)
    dmis.Wd = Wd
    # Regularization - with a regularization mesh
    regMesh = simpeg.Mesh.TensorMesh([m1d.hx[problem.mapping.sigmaMap.maps[-1].indActive]],m1d.x0)
    reg = simpeg.Regularization.Tikhonov(regMesh)
    reg.mrefInSmooth = True
    reg.alpha_s = 1e-7
    reg.alpha_x = 1.
    # Inversion problem
    invProb = simpeg.InvProblem.BaseInvProblem(dmis, reg, opt)
    invProb.counter = C
    # Beta cooling
    beta = simpeg.Directives.BetaSchedule()
    beta.coolingRate = 4
    betaest = simpeg.Directives.BetaEstimate_ByEig(beta0_ratio=0.75)
    targmis = simpeg.Directives.TargetMisfit()
    targmis.target = survey.nD
    saveModel = simpeg.Directives.SaveModelEveryIteration()
    saveModel.fileName = 'Inversion_TargMisEqnD_smoothTrue'
    # Create an inversion object
    inv = simpeg.Inversion.BaseInversion(invProb, directiveList=[beta,betaest,targmis])

    ## Run the inversion
    mopt = inv.run(m_0)

    if plotIt:
        fig = MT.Utils.dataUtils.plotMT1DModelData(problem,[mopt])
        fig.suptitle('Target - smooth true')
        plt.show()

if __name__ == '__main__':
    run()
