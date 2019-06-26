"""
1D DC inversion of Schlumberger array
=====================================

This is an example for 1D DC Sounding inversion. This 1D inversion usually use
analytic foward modeling, which is efficient. However, we choose different
approach to show flexibility in geophysical inversion through mapping.
Here mapping, :math:`\\mathcal{M}`, indicates transformation of our model to a
different space:

.. math::
    \\sigma = \\mathcal{M}(\\mathbf{m})

Now we consider a transformation, which maps 3D conductivity model to 1D layer
model. That is, 3D distribution of conducitivity can be parameterized as 1D
model. Once we can compute derivative of this transformation, we can change our
model space, based on the transformation.

Following example will show you how user can implement this set up with 1D DC
inversion example. Note that we have 3D forward modeling mesh.
"""

from SimPEG import (
    Mesh, Maps, DataMisfit, Regularization, Optimization,
    InvProblem, Directives, Inversion, Versions
)
from SimPEG.Utils import plotLayer
try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver
from SimPEG.EM.Static import DC
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Step 1
# ------
#
# Generate mesh

cs = 25.
npad = 11
hx = [(cs, npad, -1.3), (cs, 41), (cs, npad, 1.3)]
hy = [(cs, npad, -1.3), (cs, 17), (cs, npad, 1.3)]
hz = [(cs, npad, -1.3), (cs, 20)]
mesh = Mesh.TensorMesh([hx, hy, hz], 'CCN')

###############################################################################
# Step 2
# ------
#
# Generating model and mapping (1D to 3D)

mapping = Maps.ExpMap(mesh)*Maps.SurjectVertical1D(mesh)
siglay1 = 1./(100.)
siglay2 = 1./(500.)
sighalf = 1./(100.)
sigma = np.ones(mesh.nCz)*siglay1
sigma[mesh.vectorCCz <= -100.] = siglay2
sigma[mesh.vectorCCz < -150.] = sighalf
mtrue = np.log(sigma)


fig, ax = plt.subplots(1, 2, figsize=(18*0.8, 7*0.8))
plotLayer(np.log(sigma), mesh.vectorCCz, 'linear', showlayers=True, ax=ax[0])
ax[0].invert_xaxis()
ax[0].set_ylim(-500, 0)
ax[0].set_xlim(-7, -4)
ax[0].set_xlabel('$log(\sigma)$', fontsize=25)
ax[0].set_ylabel('Depth (m)', fontsize=25)
dat = mesh.plotSlice((mapping*mtrue), normal='Y', ind=9, ax=ax[1])
cb = plt.colorbar(dat[0], ax=ax[1])
ax[0].set_title("(a) Conductivity with depth", fontsize=25)
ax[1].set_title("(b) Vertical section", fontsize=25)
cb.set_label("Conductivity (S/m)", fontsize=25)
ax[1].set_xlabel('Easting (m)', fontsize=25)
ax[1].set_ylabel(' ', fontsize=25)
ax[1].set_xlim(-1000., 1000.)
ax[1].set_ylim(-500., 0.)

###############################################################################
# Step 3
# ------
#
# Design survey: Schulumberger array
#
# .. image:: http://www.landrinstruments.com/_/rsrc/1271695892678/home/ultra-minires/additional-information-1/schlumberger-soundings/schlum%20array.JPG
#    :alt: Schulumberger array
#
# .. math::
#   \rho_a = \frac{V}{I}\pi\frac{b(b+a)}{a}
#
# Let :math:`b=na`, then we rewrite above equation as:
#
# .. math::
#   \rho_a = \frac{V}{I}\pi na(n+1)
#
# Since AB/2 can be a good measure for depth of investigation, we express
#
# .. math::
#   AB/2 = \frac{(2n+1)a}{2}


ntx = 16
xtemp_txP = np.arange(ntx)*(25.)-500.
xtemp_txN = -xtemp_txP
ytemp_tx = np.zeros(ntx)
xtemp_rxP = -50.
xtemp_rxN = 50.
ytemp_rx = 0.
abhalf = abs(xtemp_txP-xtemp_txN)*0.5
a = xtemp_rxN-xtemp_rxP
b = ((xtemp_txN-xtemp_txP)-a)*0.5

fig, ax = plt.subplots(1, 1, figsize=(12, 3))
for i in range(ntx):
    ax.plot(
        np.r_[xtemp_txP[i], xtemp_txP[i]],
        np.r_[0., 0.4-0.01*(i-1)], 'k-', lw=1
    )
    ax.plot(
        np.r_[xtemp_txN[i], xtemp_txN[i]],
        np.r_[0., 0.4-0.01*(i-1)], 'k-', lw=1
    )
    ax.plot(xtemp_txP[i], ytemp_tx[i], 'bo')
    ax.plot(xtemp_txN[i], ytemp_tx[i], 'ro')
    ax.plot(
        np.r_[xtemp_txP[i], xtemp_txN[i]],
        np.r_[0.4-0.01*(i-1), 0.4-0.01*(i-1)], 'k-', lw=1
    )

ax.plot(np.r_[xtemp_rxP, xtemp_rxP], np.r_[0., 0.2], 'k-', lw=1)
ax.plot(np.r_[xtemp_rxN, xtemp_rxN], np.r_[0., 0.2], 'k-', lw=1)
ax.plot(xtemp_rxP, ytemp_rx, 'ko')
ax.plot(xtemp_rxN, ytemp_rx, 'go')
ax.plot(np.r_[xtemp_rxP, xtemp_rxN], np.r_[0.2, 0.2], 'k-', lw=1)

ax.grid(True)
ax.set_ylim(-0.2, 0.6)

###############################################################################
# Look at the survey in map view

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(xtemp_txP, ytemp_tx, 'bo')
ax.plot(xtemp_txN, ytemp_tx, 'ro')
ax.plot(xtemp_rxP, ytemp_rx, 'ko')
ax.plot(xtemp_rxN, ytemp_rx, 'go')
ax.legend(('A (C+)', 'B (C-)', 'M (P+)', 'N (C-)'), fontsize=14)
mesh.plotSlice(
    np.log10(mapping*mtrue), grid=True, ax=ax, pcolorOpts={'cmap': 'binary'}
)
ax.set_xlim(-600, 600)
ax.set_ylim(-200, 200)
ax.set_title('Survey geometry (Plan view)')
ax.set_xlabel('Easting (m)')
ax.set_ylabel('Northing (m)')
ax.text(-600, 210, '(a)', fontsize=16)

# We generate tx and rx lists:
srclist = []
for i in range(ntx):
    rx = DC.Rx.Dipole(np.r_[xtemp_rxP, ytemp_rx, -12.5], np.r_[xtemp_rxN, ytemp_rx, -12.5])
    locA = np.r_[xtemp_txP[i], ytemp_tx[i], -12.5]
    locB = np.r_[xtemp_txN[i], ytemp_tx[i], -12.5]
    src = DC.Src.Dipole([rx], locA, locB)
    srclist.append(src)

###############################################################################
# Step 4
# ------
#
# Set up problem and pair with survey

survey = DC.Survey(srclist)
problem = DC.Problem3D_CC(mesh, sigmaMap=mapping)
problem.pair(survey)
problem.Solver = Solver

###############################################################################
# Step 5
# ------
#
# Run :code:`survey.dpred` to comnpute syntetic data
#
# .. math::
#   \rho_a = \frac{V}{I}\pi\frac{b(b+a)}{a}
#
# To make synthetic example you can use survey.makeSyntheticData, which
# generates related setups.

data = survey.dpred(mtrue)
survey.makeSyntheticData(mtrue, std=0.01, force=True)

appres = data*np.pi*b*(b+a)/a
appres_obs = survey.dobs*np.pi*b*(b+a)/a
fig, ax = plt.subplots(1, 2, figsize=(16, 4))
ax[1].semilogx(abhalf, appres, 'k.-')
ax[1].set_xscale('log')
ax[1].set_ylim(100., 180.)
ax[1].set_xlabel('AB/2')
ax[1].set_ylabel('Apparent resistivity ($\Omega m$)')
ax[1].grid(True)

dat = mesh.plotSlice((mapping*mtrue), normal='Y', ind=9, ax=ax[0])
cb = plt.colorbar(dat[0], ax=ax[0])
ax[0].set_title("Vertical section")
cb.set_label("Conductivity (S/m)")
ax[0].set_xlabel('Easting (m)')
ax[0].set_ylabel('Depth (m)')
ax[0].set_xlim(-1000., 1000.)
ax[0].set_ylim(-500., 0.)

###############################################################################
# Step 6
# ------
#
# Run inversion

regmesh = Mesh.TensorMesh([31])
dmis = DataMisfit.l2_DataMisfit(survey)
reg = Regularization.Tikhonov(regmesh)
opt = Optimization.InexactGaussNewton(maxIter=7, tolX=1e-15)
opt.remember('xc')
invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
beta = Directives.BetaEstimate_ByEig(beta0_ratio=1e1)
betaSched = Directives.BetaSchedule(coolingFactor=5, coolingRate=2)
inv = Inversion.BaseInversion(invProb, directiveList=[beta, betaSched])

# Choose an initial starting model of the background conductivity
m0 = np.log(np.ones(mapping.nP)*sighalf)
mopt = inv.run(m0)


###############################################################################
# Step 7
# ------
#
# Plot the results
appres = data*np.pi*b*(b+a)/a
appres_obs = survey.dobs*np.pi*b*(b+a)/a
appres_pred = invProb.dpred*np.pi*b*(b+a)/a
fig, ax = plt.subplots(1, 2, figsize=(17, 6))
ax[0].plot(abhalf, appres_obs, 'k.-')
ax[0].plot(abhalf, appres_pred, 'r.-')
ax[0].set_xscale('log')
ax[0].set_ylim(100., 180.)
ax[0].set_xlabel('AB/2', fontsize=25)
ax[0].set_ylabel('Apparent resistivity ($\Omega m$)', fontsize=25)
ax[0].set_title('(a)', fontsize=25)
ax[0].grid(True)
ax[0].legend(('Observed', 'Predicted'), loc=1, fontsize=20)
ax[1].plot(1., 1., 'k', lw=2)
ax[1].plot(1., 1., 'r', lw=2)
ax[1].legend(('True', 'Predicted'), loc=3, fontsize=20)
plotLayer(
    (np.exp(mopt)), mesh.vectorCCz, 'log',
    ax=ax[1], **{'lw': 2, 'color': 'r'}
)
plotLayer(
    (np.exp(mtrue)), mesh.vectorCCz, 'log', showlayers=True,
    ax=ax[1], **{'lw': 2}
)
ax[1].set_ylim(-500, 0)
ax[1].set_xlabel('Conductivity (S/m)', fontsize=25)
ax[1].set_ylabel('Depth (m)', fontsize=25)
ax[1].set_title('(b)', fontsize=25)

###############################################################################
# Print the version of SimPEG and dependencies
# --------------------------------------------
#

Versions()
