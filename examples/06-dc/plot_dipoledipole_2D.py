from SimPEG import DC
from SimPEG import (Mesh, Maps, Utils, DataMisfit, Regularization,
                    Optimization, Inversion, InvProblem, Directives)
from pymatsolver import PardisoSolver
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors
import numpy as np
from pylab import hist
try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

showIt = True

IO = DC.IO()
A, B, M, N = IO.genLocs_2D('dipole-dipole', 0., 200., 10., 10)
survey = IO.fromABMN_to_survey(
    A, B, M, N, 'dipole-dipole', dataType='volt'
)
mesh, actind = IO.setMesh()

inds = Utils.ModelBuilder.getIndicesSphere(np.r_[100., -25.], 15., mesh.gridCC)
sigma = np.ones(mesh.nC)*1e-2
sigma[inds] = 0.1
rho = 1./sigma

if showIt:
    fig = plt.figure(figsize = (12, 3))
    ax = plt.subplot(111)
    mesh.plotImage(1./sigma, grid=True, ax=ax, gridOpts={'alpha':0.2}, pcolorOpts={"cmap":"jet"})
    plt.plot(IO.uniqElecLocs[:,0], IO.uniqElecLocs[:,1], 'k.')
    plt.show()

mapping = Maps.ExpMap(mesh)
mtrue = np.ones(mesh.nC)*np.log(sigma)

prb = DC.Problem2D_N(
    mesh, sigmaMap=mapping, storeJ=True,
    Solver = PardisoSolver
)
try:
    prb.pair(survey)
except:
    survey.unpair()
    prb.pair(survey)

dtrue = survey.makeSyntheticData(mtrue, std=0.05)
if showIt:
    IO.plotPseudoSection(dobs=survey.dobs)

m0 = np.ones(mesh.nC)*np.log(1./10**2.05)

if showIt:
    fig = plt.figure()
    out = hist(survey.dobs/IO.G, bins=20)
    plt.show()

m0 = np.ones(mesh.nC)*np.log(1./110.)

eps = 10**-3.2
std = 0.05
dmisfit = DataMisfit.l2_DataMisfit(survey)
uncert = abs(survey.dobs) * std + eps
dmisfit.W = 1./uncert
regmap = Maps.IdentityMap(nP=int(actind.sum()))
reg = Regularization.Simple(mesh, indActive=actind, mapping=regmap)
opt = Optimization.InexactGaussNewton(maxIter=15)
invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)
# Create an inversion object
beta = Directives.BetaSchedule(coolingFactor=5, coolingRate=2)
betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1e0)
target = Directives.TargetMisfit()
inv = Inversion.BaseInversion(invProb, directiveList=[beta, betaest, target])
prb.counter = opt.counter = Utils.Counter()
opt.LSshorten = 0.5
opt.remember('xc')
mopt = inv.run(m0)

rho_est = 1./(mapping*mopt)
rho_est[~actind] = np.nan

if showIt:
    vmin, vmax = rho.min(), rho.max()
    fig, ax = plt.subplots(figsize=(20, 3))
    out = mesh.plotImage(rho_est, clim=(vmin, vmax), pcolorOpts={"cmap":"jet", "norm":colors.LogNorm()}, ax=ax)
    ax.set_xlim(IO.grids[:,0].min(), IO.grids[:,0].max())
    ax.set_ylim(-IO.grids[:,1].max(), IO.grids[:,1].min())
    cb = plt.colorbar(out[0])
    cb.set_label("Resistivity ($\Omega$m)")
    ax.set_xlabel("Northing (m)")
    ax.set_ylabel("Elevation (m)")
    plt.show()

if showIt:
    IO.plotPseudoSection(dobs=invProb.dpred)
