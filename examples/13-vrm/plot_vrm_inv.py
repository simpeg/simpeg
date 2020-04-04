"""
Method of Equivalent Sources for Removing VRM Responses
=======================================================

Here, we use an equivalent source inversion to remove the VRM response from TEM
data collected by a small coincident loop system. The data being inverted are
the same as in the forward modeling example. To remove the VRM signal we:

    1. invert the late time data to recover an equivalent source surface layer of cells.
    2. use the recovered model to predict the VRM response at all times
    3. subtract the predicted VRM response from the observed data
"""

#########################################################################
# Import modules
# --------------
#

from SimPEG.electromagnetics import viscous_remanent_magnetization as VRM
import numpy as np
import discretize
from SimPEG import (
    utils, maps, data_misfit, directives, optimization, regularization,
    inverse_problem, inversion, data
    )
import matplotlib.pyplot as plt
import matplotlib as mpl


##########################################################################
# Defining the mesh
# -----------------
#

cs, ncx, ncy, ncz, npad = 2., 35, 35, 20, 5
hx = [(cs, npad, -1.3), (cs, ncx), (cs, npad, 1.3)]
hy = [(cs, npad, -1.3), (cs, ncy), (cs, npad, 1.3)]
hz = [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
mesh = discretize.TensorMesh([hx, hy, hz], 'CCC')

##########################################################################
# Defining the true model
# -----------------------
#
# Create xi model (amalgamated magnetic property). Here the model is made by
# summing a set of 3D Gaussian distributions. And only active cells have a
# model value.
#

topoCells = mesh.gridCC[:, 2] < 0.  # define topography

xyzc = mesh.gridCC[topoCells, :]
c = 2*np.pi*8**2
pc = np.r_[4e-4, 4e-4, 4e-4, 6e-4, 8e-4, 6e-4, 8e-4, 8e-4]
x_0 = np.r_[50., -50., -40., -20., -15., 20., -10., 25.]
y_0 = np.r_[0., 0., 40., 10., -20., 15., 0., 0.]
z_0 = np.r_[0., 0., 0., 0., 0., 0., 0., 0.]
var_x = c*np.r_[3., 3., 3., 1., 3., 0.5, 0.1, 0.1]
var_y = c*np.r_[20., 20., 1., 1., 0.4, 0.5, 0.1, 0.4]
var_z = c*np.r_[1., 1., 1., 1., 1., 1., 1., 1.]

xi_true = np.zeros(np.shape(xyzc[:, 0]))

for ii in range(0, 8):
    xi_true += (
        pc[ii]*np.exp(-(xyzc[:, 0]-x_0[ii])**2/var_x[ii]) *
        np.exp(-(xyzc[:, 1]-y_0[ii])**2/var_y[ii]) *
        np.exp(-(xyzc[:, 2]-z_0[ii])**2/var_z[ii])
    )

xi_true += 1e-5

##########################################################################
# Survey
# ------
#
# Here we must set the transmitter waveform, which defines the off-time decay
# of the VRM response. Next we define the sources, receivers and time channels
# for the survey. Our example is similar to an EM-63 survey.
#

waveform = VRM.waveforms.StepOff()

times = np.logspace(-5, -2, 31)  # Observation times
x, y = np.meshgrid(np.linspace(-30, 30, 21), np.linspace(-30, 30, 21))
z = 0.5*np.ones(x.shape)
loc = np.c_[utils.mkvc(x), utils.mkvc(y), utils.mkvc(z)]  # Src and Rx Locations

srcListVRM = []

for pp in range(0, loc.shape[0]):

    loc_pp = np.reshape(loc[pp, :], (1, 3))
    rxListVRM = [VRM.Rx.Point(loc_pp, times=times, fieldType='dbdt', orientation='z')]

    srcListVRM.append(
        VRM.Src.MagDipole(rxListVRM, utils.mkvc(loc[pp, :]), [0., 0., 0.01], waveform)
    )

survey_vrm = VRM.Survey(srcListVRM)

##########################################################################
# Forward Problem
# ---------------
#
# Here we predict data by solving the forward problem. For the VRM problem,
# we use a sensitivity refinement strategy for cells # that are proximal to
# transmitters. This is controlled through the *refinement_factor* and *refinement_distance*
# properties.
#

# Defining the problem
problem_vrm = VRM.Simulation3DLinear(
    mesh, survey=survey_vrm, indActive=topoCells,
    refinement_factor=3, refinement_distance=[1.25, 2.5, 3.75]
)

# Predict VRM response
fields_vrm = problem_vrm.dpred(xi_true)

# Add an artificial TEM response. An analytic solution for the response near
# the surface of a conductive half-space (Nabighian, 1979) is scaled at each
# location to provide lateral variability in the TEM response.
n_times = len(times)
n_loc = loc.shape[0]

sig = 1e-1
mu0 = 4*np.pi*1e-7
fields_tem = -sig**1.5*mu0**2.5*times**-2.5/(20*np.pi**1.5)
fields_tem = np.kron(np.ones(n_loc), fields_tem)
c = (
   np.exp(-(loc[:, 0]-10)**2/(25**2))*np.exp(-(loc[:, 1]-20)**2/(35**2)) +
   np.exp(-(loc[:, 0]+20)**2/(20**2))*np.exp(-(loc[:, 1]+20)**2/(40**2)) +
   1.5*np.exp(-(loc[:, 0]-25)**2/(10**2))*np.exp(-(loc[:, 1]+25)**2/(10**2)) +
   0.25
)

c = np.kron(c, np.ones(n_times))
fields_tem = c*fields_tem

fields_tot = fields_tem + fields_vrm
fields_tot = fields_tot + 0.05*np.abs(fields_tot)*np.random.normal(size=fields_tot.shape)

##########################################################################
# Inverse Problem
# ---------------
#
# Here, we invert late-time data to recover an equivalent source model. To
# recover the equivalent source model, only cells at the surface are set
# as active in the inversion.
#

# Define problem
#survey_inv = VRM.Survey(srcListVRM)
actCells = (mesh.gridCC[:, 2] < 0.) & (mesh.gridCC[:, 2] > -2.)
problem_inv = VRM.Simulation3DLinear(
    mesh, survey=survey_vrm, indActive=actCells,
    refinement_factor=3, refinement_distance=[1.25, 2.5, 3.75]
)
survey_vrm.set_active_interval(1e-3, 1e-2)

dobs = fields_tot[survey_vrm.t_active]
std = 0.05*np.abs(fields_tot[survey_vrm.t_active])
eps = 1e-11
data_vrm = data.Data(dobs=dobs, survey=survey_vrm, standard_deviation=std, noise_floor=eps)

# Setup and run inversion
dmis = data_misfit.L2DataMisfit(simulation=problem_inv, data=data_vrm)

w = utils.mkvc((np.sum(np.array(problem_inv.A)**2, axis=0)))**0.5
w = w/np.max(w)
w = w

reg = regularization.SimpleSmall(mesh=mesh, indActive=actCells,  cell_weights=w)
opt = optimization.ProjectedGNCG(maxIter=20, lower=0., upper=1e-2, maxIterLS=20, tolCG=1e-4)
invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)
directives = [
    directives.BetaSchedule(coolingFactor=2, coolingRate=1),
    directives.TargetMisfit()
]
inv = inversion.BaseInversion(invProb, directiveList=directives)

xi_0 = 1e-3*np.ones(actCells.sum())
xi_rec = inv.run(xi_0)

# Predict VRM response at all times for recovered model
survey_vrm.set_active_interval(0., 1.)
fields_pre = problem_inv.dpred(xi_rec)

################################
# Plotting
# --------
#

fields_tot = np.reshape(fields_tot, (n_loc, n_times))
fields_vrm = np.reshape(fields_vrm, (n_loc, n_times))
fields_tem = np.reshape(fields_tem, (n_loc, n_times))
fields_pre = np.reshape(fields_pre, (n_loc, n_times))


Fig = plt.figure(figsize=(10, 10))
font_size = 12

# Plot models
invMap = maps.InjectActiveCells(mesh, actCells, 0.)  # Maps to mesh
topoMap = maps.InjectActiveCells(mesh, topoCells, 0.)
max_val = np.max(np.r_[xi_true, xi_rec])
ax1 = 3*[None]
cplot1 = 2*[None]
xi_mod = [xi_true, xi_rec]
map_mod = [topoMap, invMap]
titlestr1 = [
    "True Model (z = 0 m)",
    "Equivalent Source Model"
]

for qq in range(0, 2):
    ax1[qq] = Fig.add_axes([0.15+0.35*qq, 0.7, 0.25, 0.25])
    cplot1[qq] = mesh.plotSlice(
        map_mod[qq]*xi_mod[qq], ind=int((ncz+2*npad)/2-1),
        ax=ax1[qq], grid=True, pcolorOpts={'cmap': 'gist_heat_r'})
    cplot1[qq][0].set_clim((0., max_val))
    ax1[qq].set_xlabel('X [m]', fontsize=font_size)
    ax1[qq].set_ylabel('Y [m]', fontsize=font_size, labelpad=-5)
    ax1[qq].tick_params(labelsize=font_size-2)
    ax1[qq].set_title(titlestr1[qq], fontsize=font_size+2)

ax1[2] = Fig.add_axes([0.78, 0.7, 0.01, 0.25])
norm = mpl.colors.Normalize(vmin=0., vmax=max_val)
cbar14 = mpl.colorbar.ColorbarBase(
    ax1[2], cmap=mpl.cm.gist_heat_r, norm=norm, orientation='vertical'
)
cbar14.set_label(
    '$\Delta \chi /$ln$(\lambda_2 / \lambda_1 )$ [SI]',
    rotation=270, labelpad=15, size=font_size
)

# Plot decays
N = x.shape[0]
ax2 = 2*[None]
for qq in range(0, 2):
    ax2[qq] = Fig.add_axes([0.1+0.45*qq, 0.36, 0.35, 0.26])
    k = int((N**2-1)/2 - 3*N*(-1)**qq)
    di_tot = utils.mkvc(np.abs(fields_tot[k, :]))
    di_pre = utils.mkvc(np.abs(fields_vrm[k, :]))
    di_tem = utils.mkvc(np.abs(fields_tem[k, :]))
    ax2[qq].loglog(times, di_tot, 'k.-')
    ax2[qq].loglog(times, di_tem, 'r.-')
    ax2[qq].loglog(times, di_pre, 'b.-')
    ax2[qq].loglog(times, np.abs(di_tot-di_pre), 'g.-')
    ax2[qq].set_xlabel('t [s]', fontsize=font_size, labelpad=-10)
    if qq == 0:
        ax2[qq].set_ylabel('|dBz/dt| [T/s]', fontsize=font_size)
    else:
        ax2[qq].axes.get_yaxis().set_visible(False)
    ax2[qq].tick_params(labelsize=font_size-2)
    ax2[qq].set_xbound(np.min(times), np.max(times))
    ax2[qq].set_ybound(1.2*np.max(di_tot), 1e-5*np.max(di_tot))
    titlestr2 = (
        "Decay at X = " + '{:.2f}'.format(loc[k, 0]) +
        " m and Y = " + '{:.2f}'.format(loc[k, 1]) + " m"
    )
    ax2[qq].set_title(titlestr2, fontsize=font_size+2)
    if qq == 0:
        ax2[qq].text(1.2e-5, 54*np.max(di_tot)/1e5, "Observed", fontsize=font_size, color='k')
        ax2[qq].text(1.2e-5, 18*np.max(di_tot)/1e5, "True TEM", fontsize=font_size, color='r')
        ax2[qq].text(1.2e-5, 6*np.max(di_tot)/1e5, "Predicted VRM", fontsize=font_size, color='b')
        ax2[qq].text(1.2e-5, 2*np.max(di_tot)/1e5, "Recovered TEM", fontsize=font_size, color='g')

# Plot anomalies
d = [
    np.reshape(np.abs(fields_tot[:, 10]), (N, N)),
    np.reshape(np.abs(fields_tem[:, 10]), (N, N)),
    np.reshape(np.abs(fields_tot[:, 10]-fields_pre[:, 10]), (N, N))
]

min_val = np.min(np.r_[d[0], d[1], d[2]])
max_val = np.max(np.r_[d[0], d[1], d[2]])

ax3 = 4*[None]
cplot3 = 3*[None]
title_str = [
    "Observed at t=",
    "True TEM at t=",
    "Recov. TEM at t="
]

for qq in range(0, 3):
    ax3[qq] = Fig.add_axes([0.07+0.28*qq, 0.05, 0.24, 0.24])
    cplot3[qq] = ax3[qq].contourf(x, y, d[qq].T, 40, cmap='magma_r')
    ax3[qq].set_xticks(np.linspace(-30, 30, 7))
    ax3[qq].set_xlabel('X [m]', fontsize=font_size)
    if qq == 0:
        ax3[qq].scatter(x, y, color=(0, 0, 0), s=4)
        ax3[qq].set_ylabel('Y [m]', fontsize=font_size, labelpad=-12)
    else:
        ax3[qq].axes.get_yaxis().set_visible(False)
    ax3[qq].tick_params(labelsize=font_size-2)
    ax3[qq].set_xbound(np.min(x), np.max(x))
    ax3[qq].set_ybound(np.min(y), np.max(y))
    titlestr3 = title_str[qq] + '{:.1e}'.format(times[10]) + " s"
    ax3[qq].set_title(titlestr3, fontsize=font_size+2)

ax3[3] = Fig.add_axes([0.88, 0.05, 0.01, 0.24])
norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
cbar34 = mpl.colorbar.ColorbarBase(
    ax3[3], cmap=mpl.cm.magma_r, norm=norm, orientation='vertical', format='%.1e'
)
cbar34.set_label('dBz/dt [T/s]', rotation=270, size=font_size, labelpad=15)
plt.show()
