import matplotlib.pyplot as plt


import SimPEG as simpeg
from SimPEG import np
from SimPEG.EM import NSEM

np.random.seed(1983)

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
    nFreq = 26
    freqs = np.logspace(2,-3,nFreq)
    # Set mesh parameters
    ct = 10
    air = simpeg.Utils.meshTensor([(ct,25,1.4)])
    core = np.concatenate( (  np.kron(simpeg.Utils.meshTensor([(ct,10,-1.3)]),np.ones((5,))) , simpeg.Utils.meshTensor([(ct,5)]) ) )
    bot = simpeg.Utils.meshTensor([(core[0],25,-1.4)])
    x0 = -np.array([np.sum(np.concatenate((core,bot)))])
    # Make the model
    m1d = simpeg.Mesh.TensorMesh([np.concatenate((bot,core,air))], x0=x0)

    # Setup model varibles
    active = m1d.vectorCCx<0.
    layer1 = (m1d.vectorCCx<-500.) & (m1d.vectorCCx>=-800.)
    layer2 = (m1d.vectorCCx<-3500.) & (m1d.vectorCCx>=-5000.)
    # Set the conductivity values
    sig_half = 1e-2
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
    rxList.append(NSEM.Rx.Point_impedance1D(simpeg.mkvc(np.array([-0.5]),2).T,'real'))
    rxList.append(NSEM.Rx.Point_impedance1D(simpeg.mkvc(np.array([-0.5]),2).T,'imag'))
    # Source list
    srcList =[]
    for freq in freqs:
            srcList.append(NSEM.Src.Planewave_xy_1Dprimary(rxList,freq))
    # Make the survey
    survey = NSEM.Survey(srcList)
    survey.mtrue = m_true

    ## Set the problem
    problem = NSEM.Problem1D_ePrimSec(m1d,sigmaPrimary=sigma_0,mapping=mappingExpAct)
    problem.pair(survey)

    ## Forward model data
    # Project the data
    survey.dtrue = survey.dpred(m_true)
    survey.dobs = survey.dtrue + 0.01*abs(survey.dtrue)*np.random.randn(*survey.dtrue.shape)

    if plotIt:
        fig = NSEM.Utils.dataUtils.plotMT1DModelData(problem,[])
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
    opt = simpeg.Optimization.ProjectedGNCG(maxIter = 25)
    opt.counter = C
    opt.lower = np.log(1e-4)
    opt.upper = np.log(5)
    opt.LSshorten = 0.1
    opt.remember('xc')
    # Data misfit
    dmis = simpeg.DataMisfit.l2_DataMisfit(survey)
    dmis.Wd = Wd
    # Regularization - with a regularization mesh
    regMesh = simpeg.Mesh.TensorMesh([m1d.hx[active]],m1d.x0)
    reg = simpeg.Regularization.Tikhonov(regMesh)
    reg.mrefInSmooth = True
    reg.alpha_s = 1e-1
    reg.alpha_x = 1.

    # Inversion problem
    invProb = simpeg.InvProblem.BaseInvProblem(dmis, reg, opt)
    invProb.counter = C
    # Beta cooling
    beta = simpeg.Directives.BetaSchedule()
    beta.coolingRate = 4.
    beta.coolingFactor = 4.
    betaest = simpeg.Directives.BetaEstimate_ByEig(beta0_ratio=100.)
    betaest.beta0 = 1.
    targmis = simpeg.Directives.TargetMisfit()
    targmis.target = survey.nD
    # Create an inversion object
    inv = simpeg.Inversion.BaseInversion(invProb, directiveList=[beta,betaest,targmis])

    ## Run the inversion
    mopt = inv.run(m_0)

    if plotIt:
        fig = NSEM.Utils.dataUtils.plotMT1DModelData(problem,[mopt])
        fig.suptitle('Target - smooth true')
        fig.axes[0].set_ylim([-10000,500])
        plt.show()

if __name__ == '__main__':
    run()
