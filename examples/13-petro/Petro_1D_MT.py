from scipy.constants import mu_0
from SimPEG import (
    Mesh, Problem, Survey, Maps, Utils, EM, DataMisfit,
    Regularization, Optimization, InvProblem,
    Directives, Inversion)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from pymatsolver import PardisoSolver as Solver
import sys
sys.path.append('/Volumes/MacintoshHD/Users/thibautastic/PhD_UBC/GITHUB/tle-magnetotelluric_inversion/')
from MT1D import MT1DProblem, MT1DSurvey, MT1DSrc, ZxyRx, Survey, AppResPhaRx

# Better rendering
import seaborn
seaborn.set()

# Reproducible science
np.random.seed(12345)

layer_tops = np.r_[0., -600., -1991., -5786., -9786.]  # in m
rho_layers = np.r_[250., 25, 100., 10., 25.]

rxloc = np.r_[0.]
frequency = np.logspace(-3, 2, 25)

# Create a receiver object
rx = ZxyRx(
    rxloc, # location of the receiver
    component="both", # measure both the real and imaginary components of the impedance (alternatively "real" / "imag")
    frequency=frequency
)
# create a plane wave source
src = MT1DSrc([rx])

# define a survey
survey = MT1DSurvey([src])

max_depth_core = 15000.
mesh = survey.setMesh(
    sigma=0.01, # approximate conductivity of the background
    max_depth_core=max_depth_core,  # extent of the core region of the mesh
    ncell_per_skind=10,  # number of cells per the smallest skin depth
    n_skind=2,  # number of skin depths that the mesh should extend to ensure the lowest-frequency fields have decayed
    core_meshType = "log",   # cell spacings in the core region of the mesh ("linear" or "log")
    max_hz_core=1000.  # If using a logarithmic core mesh, what is the maximum cell size?
 )

M = mesh

prob = MT1DProblem(
    mesh,  # The mesh contains the geometry, grids, etc necessary for constructing the discrete PDE system
    sigmaMap=Maps.ExpMap(mesh),  # in the inversion, we want to invert for log-conductivity (enforces postivity, electrical conductivity tends to vary logarithmically)
    verbose=False,  # print information as we are setting up and solving
    Solver=Solver  # solver to employ for solving Ax = b
)

# tell the problem and survey about each other so we can construct our matrix system
# and right hand-side
prob.pair(survey)

# start with nans so we can do a check to make sure all
# layer conductivities have been properly assigned
rho = np.ones(mesh.nC) * np.nan

# loop over each layer in the model and assign to mesh
for layer_top, rho_layer in zip(layer_tops, rho_layers):
    inds = mesh.vectorCCx < layer_top
    rho[inds] = rho_layer

sigma = 1./rho
mtrue = np.log(sigma)  # since our "model" is log conductivity, we take the log
dtrue = survey.dpred(mtrue)  # these are clean data (no noise yet.)

np.random.seed(1)  # set a seed to the results are reproducable
std = 0.1  # standard deviation of the noise (10%)

# add noise
uncert = std * np.abs(dtrue)
noise = uncert * np.random.randn(survey.nFreq*2)
survey.dobs = dtrue + noise

def omega(frequency):
    """
    angular frequency
    """
    return 2*np.pi*frequency

def appres_phase_from_data(data, frequency):
    """
    Compute apparent resistivity and phase given impedances (real and imaginary components)
    and the frequency.
    """

    # data are arranged (Zxy_real, Zxy_imag) for each frequency
    Zxy_real = data.reshape((survey.nFreq, 2))[:,0]
    Zxy_imag = data.reshape((survey.nFreq, 2))[:,1]
    Zxy = Zxy_real+1j*Zxy_imag

    # compute apparent resistivity and phase from complex impedance
    app_res = abs(Zxy)**2 / (mu_0*omega(frequency))
    phase = np.rad2deg(np.arctan(Zxy_imag / Zxy_real))

    return app_res, phase

clf = GaussianMixture(n_components=4, covariance_type='full',
                      max_iter=1000, n_init=20, reg_covar=1e-3)
clf.fit(mtrue.reshape(-1,1))
Utils.order_clusters_GM_weight(clf)

sigma_ref = 1e-2  # reference conductivity
sigma_0 = 1e-2  # starting conductivity

# translate the starting and reference model to log-conductivity
mref = np.log(sigma_ref)*np.ones(mesh.nC)
m0 = np.log(sigma_0)*np.ones(mesh.nC)

alpha_s=1e-2  # smallness contribution
alpha_z=1.  # smoothness contribution to the regularization
use_betaest=True  # estimate the initial beta
beta0_ratio=10.   # starting contribution of regularization 10x larger than the data misfit
coolingFactor=1.5
coolingRate=1

# Data misfit
dmisfit = DataMisfit.l2_DataMisfit(survey)
dmisfit.W = 1./uncert

# Regularization
reg = Regularization.Simple(
        prob.mesh, alpha_s=alpha_s, alpha_x=alpha_z, mref=mref
    ) # since we are in 1D, we work with the first dimension

    # Optimization
opt = Optimization.InexactGaussNewton(maxIter=35, LSshorten=0.05)

    # Statement of the inverse problem
invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)

    # Inversion Directives
beta = Directives.BetaSchedule(
        coolingFactor=coolingFactor, coolingRate=coolingRate
    )
#invProb.beta = beta0
target = Directives.TargetMisfit()

    #Plot = PlotIter()

directives = [beta,target]#Plot,

betaest = Directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio)
directives.append(betaest)

    # assemble in an inversion
inv = Inversion.BaseInversion(invProb, directiveList=directives)
prob.counter = opt.counter = Utils.Counter()
opt.remember('xc')



# run the inversion
l2model = inv.run(m0)

# Petrophysics Inversion
########################

# Data misfit
dmisfit = DataMisfit.l2_DataMisfit(survey)
dmisfit.W = 1./uncert

reg = Regularization.SimplePetroRegularization(GMmref=clf,
                                               mesh=prob.mesh,
                                               mref=m0)
reg.mrefInSmooth = False
#reg.alpha_s = 1.
#reg.alpha_x = 1.

opt = Optimization.InexactGaussNewton(maxIter=20)
opt.remember('xc')

# Statement of the inverse problem
invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)

gamma = np.ones(clf.n_components)*0.75
invProb.reg.gamma = gamma

# Data misfit
dmisfit = DataMisfit.l2_DataMisfit(survey)
dmisfit.W = 1./uncert

reg = Regularization.SimplePetroRegularization(GMmref=clf, mesh=prob.mesh, mref=m0,evaltype='full')
reg.mrefInSmooth = False
reg.alpha_s = 1.
reg.alpha_x = 1.

opt = Optimization.InexactGaussNewton(maxIter=20)
opt.remember('xc')

# Statement of the inverse problem
invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)

gamma = np.ones(clf.n_components)#*0.75
invProb.reg.gamma = gamma

# Directives
Alphas = Directives.AlphasSmoothEstimate_ByEig(alpha0_ratio=1e-2,ninit=10)
targets = Directives.PetroTargetMisfit(TriggerTheta=False,
                                       ToleranceTheta=0.25,
                                       verbose=True,
                                       chifact=1.1,
                                       chiSmall=1.1)
MrefInSmooth = Directives.AddMrefInSmooth()
petrodir = Directives.GaussianMixtureUpdateModel(verbose=False)
beta = Directives.BetaEstimate_ByEig(beta0_ratio=1.,ninit=20)
betaIt = Directives.PetroBetaReWeighting(verbose=True, tolerance=0.,
                                         rateCooling=5., rateWarming=5.,
                                         UpdateRate=1)
#invProb.beta = 2e-2
directives = [Alphas, beta, betaIt, petrodir, targets, MrefInSmooth]

# assemble in an inversion
inv = Inversion.BaseInversion(invProb, directiveList=directives)

# run the inversion
mcluster = inv.run(m0)

print('All stopping Criteria: ', targets.AllStop)
print('Final Data Misfit: ', dmisfit(mcluster))
print('Final Cluster Scorce: ', invProb.reg.objfcts[0](mcluster, externalW=False))
print('Final DP misfit: ', targets.ThetaTarget())

fig0 = plt.figure()
ax1 = fig0.add_subplot(111)

M = prob.mesh
modelref = invProb.reg.mref
plt.semilogx(-M.vectorCCx, mtrue, color='black')
plt.semilogx(-M.vectorCCx, invProb.reg.mref, color='black', linestyle='dashed')
plt.semilogx(-M.vectorCCx, mcluster, color='red')
plt.semilogx(-M.vectorCCx, l2model, color='blue', linestyle='dashed')
plt.legend(['True Model',
            'Petro Learned Reference model',
            'Petro-constrained model',
            'L2 model'])

plt.show()
