
"""
DC Resistivity and IP Inversion
===============================

Here we invert total magnetic intensity (TMI) data to recover a magnetic
susceptibility model. We formulate the inverse problem as an iteratively
re-weighted least-squares (IRLS) optimization problem. For this tutorial, we
focus on the following:

    - Defining the survey from xyz formatted data
    - Generating a mesh based on survey geometry
    - Including surface topography
    - Defining the inverse problem (data misfit, regularization, directives)
    - Applying sensitivity weighting
    - Setting sparse and blocky norms
    - Plotting the recovered model and data misfit

Although we consider TMI data in this tutorial, the same approach
can be used to invert other types of geophysical data.
    

"""

#########################################################################
# Import modules
# --------------
#

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz

from SimPEG.Utils import ModelBuilder, surface2ind_topo
from SimPEG import (
    Maps, DC, IP, InvProblem, DataMisfit, Regularization, Optimization,
    Directives, Inversion, Utils
    )
from SimPEG.EM.Static.Utils import plot_pseudoSection

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

# sphinx_gallery_thumbnail_number = 4

#############################################
# Load Data, Define Survey and Plot
# ---------------------------------
#
# Here we load and plot synthetic DCIP data.
#

# Load data
topo_name = os.path.dirname(DC.__file__) + '\\..\\..\\tutorials\\assets\\dcip_topo.txt'
dc_data_name = os.path.dirname(DC.__file__) + '\\..\\..\\tutorials\\assets\\dc_data.txt'
ip_data_name = os.path.dirname(DC.__file__) + '\\..\\..\\tutorials\\assets\\ip_data.txt'
topo_xyz = np.loadtxt(str(topo_name))
topo_xyz = topo_xyz[:, [0, 2]]
dobs_dc = np.loadtxt(str(dc_data_name))
dobs_ip = np.loadtxt(str(ip_data_name))

a_locs = dobs_dc[:, [0, 2]]
b_locs = dobs_dc[:, [3, 5]]
m_locs = dobs_dc[:, [6, 8]]
n_locs = dobs_dc[:, [9, 11]]
dobs_dc = dobs_dc[:, -1]
dobs_ip = dobs_ip[:, -1]

# Define survey
unique_tx, k = np.unique(np.c_[a_locs, b_locs], axis=0, return_index=True)
n_tx = len(k)
k = np.r_[k, len(a_locs)+1]

src_list = []
for ii in range(0, n_tx):
    
    rx_locs_m = m_locs[k[ii]:k[ii+1], :]
    rx_locs_n = n_locs[k[ii]:k[ii+1], :]
    rx_list = [DC.RxDC.Dipole(rx_locs_m, rx_locs_n, rxType='phi')]
    
    src_locs_a = a_locs[k[ii], :]
    src_locs_b = b_locs[k[ii], :]
    src_list.append(DC.SrcDC.Dipole(rx_list, src_locs_a, src_locs_b))

# Define survey
survey_dc = DC.Survey(src_list)
survey_ip = IP.from_dc_to_ip_survey(survey_dc, dim="2D")

# Plot apparent conductivity pseudo-section
fig = plt.figure(figsize=(11, 5))

ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9])
plot_pseudoSection(
    survey_dc, dobs=dobs_dc, ax=ax1, survey_type='dipole-dipole',
    data_type='appConductivity', space_type='half-space',
    scale='log', sameratio=True, pcolorOpts={}
)
ax1.set_title('Apparent Conductivity [S/m]')

plt.show()

# Plot
fig = plt.figure(figsize=(11, 5))

ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9])
plot_pseudoSection(
    survey_ip, dobs=dobs_ip, V0=dobs_dc, ax=ax1, survey_type='dipole-dipole',
    data_type='appChargeability', space_type='half-space',
    scale='linear', clim=None,
)
ax1.set_title('Apparent Chargeability (mV/V)')

plt.show()

#############################################
# Assign Uncertainty
# ------------------

dunc_dc = 0.05*np.abs(dobs_dc)
dunc_ip = 0.05*np.abs(dobs_ip)

########################################################
# Create OcTree Mesh
# ------------------
#
# Here, we create the OcTree mesh that will be used to predict both DC
# resistivity and IP data.
#

dh = 10.                                                    # base cell width
dom_width_x = 3000.                                         # domain width x                                        # domain width y
dom_width_z = 1500.                                         # domain width z
nbcx = 2**int(np.round(np.log(dom_width_x/dh)/np.log(2.)))  # num. base cells x
nbcz = 2**int(np.round(np.log(dom_width_z/dh)/np.log(2.)))  # num. base cells z

# Define the base mesh
hx = [(dh, nbcx)]
hz = [(dh, nbcz)]
mesh = TreeMesh([hx, hz], x0='CN')

# Mesh refinement based on topography
mesh = refine_tree_xyz(
    mesh, topo_xyz, octree_levels=[0, 0, 0, 0, 1], method='surface', finalize=False
)

# Mesh refinement near transmitters and receivers
mesh = refine_tree_xyz(
    mesh, np.r_[a_locs, b_locs], octree_levels=[2, 4], method='radial', finalize=False
)

# Refine core mesh region
xp, zp = np.meshgrid([-600., 600.], [-500., 0.])
xyz = np.c_[mkvc(xp), mkvc(zp)]
mesh = refine_tree_xyz(
    mesh, xyz, octree_levels=[0, 2, 2], method='box', finalize=False
    )

mesh.finalize()

########################################################
# Starting/Reference Model and Mapping on OcTree Mesh
# ---------------------------------------------------
#
# Here, we would create starting and/or reference models for the inversion as
# well as the mapping from the model space to the active cells. Starting and
# reference models can be a constant background value or contain a-priori
# structures. Here, the background is 1e-4 SI.
#

# Define conductivity model in S/m (or resistivity model in Ohm m)
air_val = -8
background_val = -2

ind_active = surface2ind_topo(mesh, topo_xyz)





active_map = Maps.InjectActiveCells(mesh, ind_active, 10**air_val)
nC = int(ind_active.sum())

exp_map = Maps.ExpMap()
fwd_dc_map = active_map*exp_map

# Define model
m0 = background_val*np.ones(nC)

# Plot Conductivity Model
fig = plt.figure(figsize=(8.5, 4))

plotting_map = Maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9])
mesh.plotSlice(
    plotting_map*m0, normal='Y', ax=ax1, ind=int(mesh.hy.size/2),
    grid=True, clim=(-4, -1)
)
ax1.set_title('Starting Model at Y = 0 m')

ax2 = fig.add_axes([0.87, 0.05, 0.05, 0.9])
norm = mpl.colors.Normalize(vmin=-4, vmax=-1)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation='vertical', format="$10^{%.1f}$"
)
cbar.set_label(
    'Conductivity [S/m]', rotation=270, labelpad=15, size=12
)

plt.show()

##############################################
# Define the Physics
# ------------------
#
# Here, we define the physics of the gravity problem.
# 

# Define the problem. Define the cells below topography and the mapping
prob_dc = DC.Problem3D_N(mesh, sigmaMap=fwd_dc_map, Solver=Solver)

# Pair the survey and problem
survey_dc.pair(prob_dc)

# Define the observed data and uncertainties
survey_dc.dobs = dobs_dc
survey_dc.std = dunc_dc

#####################################################
# Define Inverse Problem
# ----------------------
#
# Here we define the inverse problem.
#

# Define the data misfit (Here we use weighted L2-norm)
dmis = DataMisfit.l2_DataMisfit(survey_dc)
dmis.W = Utils.sdiag(1/dunc_dc)

# Define the regularization (model objective function)
reg = Regularization.Simple(
    mesh, indActive=ind_active, mref=m0,# mapping=exp_map,
    alpha_s=1, alpha_x=1, alpha_y=1, alpha_z=1
)

# Define how the optimization problem is solved.
opt = Optimization.ProjectedGNCG(
    maxIter=5, lower=-6., upper=1.,
    maxIterLS=20, maxIterCG=10, tolCG=1e-3
)

# Here we define the inverse problem that is to be solved
inv_prob = InvProblem.BaseInvProblem(dmis, reg, opt)

# Here we define any directive that are carried out during the inversion
betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1e-1)
saveDict = Directives.SaveOutputEveryIteration(save_txt=False)
update_Jacobi = Directives.UpdatePreconditioner()
betaSched = Directives.BetaSchedule(coolingFactor=5., coolingRate=2)
updateSensW = Directives.UpdateSensitivityWeights(threshold=1e-3)

# Here we combine the inverse problem and the set of directives
inv = Inversion.BaseInversion(
    inv_prob, directiveList=[betaest, update_Jacobi, betaSched, updateSensW, saveDict]
)

# Run inversion
mrec = inv.run(m0)


############################################################
# Plotting True Model and Recovered Model
# ---------------------------------------
#

# Construct True Model

# Define conductivity model in S/m (or resistivity model in Ohm m)
air_val = -8
background_val = -2
conductor_val = -1
resistor_val = -4

# Define model
mtrue = background_val*np.ones(nC)

ind_conductor = ModelBuilder.getIndicesSphere(
    np.r_[-120., 0., -180.], 60., mesh.gridCC
)
ind_conductor = ind_conductor[ind_active]
mtrue[ind_conductor] = conductor_val

ind_resistor = ModelBuilder.getIndicesSphere(
    np.r_[120., 0., -180.], 60., mesh.gridCC
)
ind_resistor = ind_resistor[ind_active]
mtrue[ind_resistor] = resistor_val

# Plot True Model
fig = plt.figure(figsize=(9, 4))

ax1 = fig.add_axes([0.05, 0.05, 0.78, 0.9])
mesh.plotSlice(
    plotting_map*mtrue, normal='Y', ax=ax1, ind=int(mesh.hy.size/2), grid=True,
    clim=(np.min(mtrue), np.max(mtrue)), pcolorOpts={'cmap': 'jet'}
    )
ax1.set_title('True model at y = 0 m')

ax2 = fig.add_axes([0.85, 0.05, 0.05, 0.9])
norm = mpl.colors.Normalize(vmin=np.min(mtrue), vmax=np.max(mtrue))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation='vertical', cmap='jet', format='10^%.1f'
)
cbar.set_label(
    '$S/m$',
    rotation=270, labelpad=15, size=12
)

plt.show()

# Plot Recovered Model
fig = plt.figure(figsize=(9, 4))

ax1 = fig.add_axes([0.05, 0.05, 0.78, 0.9])
mesh.plotSlice(
    plotting_map*mrec, normal='Y', ax=ax1, ind=int(mesh.hy.size/2), grid=True,
    clim=(np.min(mrec), np.max(mrec)), pcolorOpts={'cmap': 'jet'}
)
ax1.set_title('Recovered model at y = 0 m')

ax2 = fig.add_axes([0.85, 0.05, 0.05, 0.9])
norm = mpl.colors.Normalize(vmin=np.min(mrec), vmax=np.max(mrec))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation='vertical', cmap='jet', format='10^%.1f'
    )
cbar.set_label('$S/m$',rotation=270, labelpad=15, size=12)

plt.show()

###################################################################
# Plotting Predicted Data and Misfit
# ----------------------------------
#

dpred_dc = inv_prob.dpred

data_array = np.c_[dobs_dc, dpred_dc, (dobs_dc-dpred_dc)/dunc_dc]

fig = plt.figure(figsize=(17, 4))
plot_title=['Observed', 'Predicted', 'Normalized Misfit']
plot_units=['S/m', 'S/m', '']
plot_type=['appConductivity', 'appConductivity', 'appConductivity']

ax1 = 3*[None]
ax2 = 3*[None]
norm = 3*[None]
cbar = 3*[None]
cplot = 3*[None]
v_lim = [np.max(np.abs(dobs_dc)), np.max(np.abs(dobs_dc)), 2]
for ii in range(0, 3):
    
    ax1[ii] = fig.add_axes([0.33*ii+0.03, 0.05, 0.25, 0.9])
    cplot[ii] = plot_pseudoSection(
            survey_dc, dobs=data_array[:, ii], ax=ax1[ii], survey_type='dipole-dipole',
            data_type='appConductivity', space_type='half-space',
            scale='log', sameratio=True, pcolorOpts={}
            )
    ax1[ii].set_title(plot_title[ii])

plt.show()


