"""
2.5D DC inversion of with Topography
====================================

This is an example for 2.5D DC Inversion. Earth includes a topography,
and below the topography conductive and resistive cylinders are embedded.
Sensitivity weighting is used for the inversion.
Approximate depth of investigation is computed by selecting
1 percent of max(sqrt(diag(JtJ))), and regions having smaller sensitivity
than this is blanked.
User is promoted to try different suvey_type such as 'pole-dipole',
'dipole-pole', and 'pole-pole'.
"""

from SimPEG import DC
from SimPEG import (Maps, Utils, DataMisfit, Regularization,
                    Optimization, Inversion, InvProblem, Directives)
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from pylab import hist
try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver


def run(plotIt=True, survey_type="dipole-dipole"):
    np.random.seed(1)
    # Initiate I/O class for DC
    IO = DC.IO()
    # Obtain ABMN locations

    xmin, xmax = 0., 200.
    ymin, ymax = 0., 0.
    zmin, zmax = 0, 0
    endl = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
    # Generate DC survey object
    survey = DC.Utils.gen_DCIPsurvey(endl, survey_type=survey_type, dim=2,
                                     a=10, b=10, n=10)
    survey.getABMN_locations()
    survey = IO.from_ambn_locations_to_survey(
        survey.a_locations, survey.b_locations,
        survey.m_locations, survey.n_locations,
        survey_type, data_dc_type='volt'
    )

    # Obtain 2D TensorMesh
    mesh, actind = IO.set_mesh()
    topo, mesh1D = DC.Utils.genTopography(mesh, -10, 0, its=100)
    actind = Utils.surface2ind_topo(mesh, np.c_[mesh1D.vectorCCx, topo])
    survey.drapeTopo(mesh, actind, option="top")

    # Build a conductivity model
    blk_inds_c = Utils.ModelBuilder.getIndicesSphere(
        np.r_[60., -25.], 12.5, mesh.gridCC
    )
    blk_inds_r = Utils.ModelBuilder.getIndicesSphere(
        np.r_[140., -25.], 12.5, mesh.gridCC
    )
    layer_inds = mesh.gridCC[:, 1] > -5.
    sigma = np.ones(mesh.nC)*1./100.
    sigma[blk_inds_c] = 1./10.
    sigma[blk_inds_r] = 1./1000.
    sigma[~actind] = 1./1e8
    rho = 1./sigma

    # Show the true conductivity model
    if plotIt:
        fig = plt.figure(figsize=(12, 3))
        ax = plt.subplot(111)
        temp = rho.copy()
        temp[~actind] = np.nan
        out = mesh.plotImage(
            temp, grid=True, ax=ax, gridOpts={'alpha': 0.2},
            clim=(10, 1000),
            pcolorOpts={"cmap": "viridis", "norm": colors.LogNorm()}
        )
        ax.plot(
            survey.electrode_locations[:, 0],
            survey.electrode_locations[:, 1], 'k.'
        )
        ax.set_xlim(IO.grids[:, 0].min(), IO.grids[:, 0].max())
        ax.set_ylim(-IO.grids[:, 1].max(), IO.grids[:, 1].min())
        cb = plt.colorbar(out[0])
        cb.set_label("Resistivity (ohm-m)")
        ax.set_aspect('equal')
        plt.show()

    # Use Exponential Map: m = log(rho)
    actmap = Maps.InjectActiveCells(
        mesh, indActive=actind, valInactive=np.log(1e8)
    )
    mapping = Maps.ExpMap(mesh) * actmap

    # Generate mtrue
    mtrue = np.log(rho[actind])

    # Generate 2.5D DC problem
    # "N" means potential is defined at nodes
    prb = DC.Problem2D_N(
        mesh, rhoMap=mapping, storeJ=True,
        Solver=Solver
    )
    # Pair problem with survey
    try:
        prb.pair(survey)
    except:
        survey.unpair()
        prb.pair(survey)

    geometric_factor = survey.set_geometric_factor(
        data_type="apparent_resistivity",
        survey_type='dipole-dipole',
        space_type='half-space'
    )

    # Make synthetic DC data with 5% Gaussian noise
    dtrue = survey.makeSyntheticData(mtrue, std=0.05, force=True)

    IO.data_dc = dtrue
    # Show apparent resisitivty pseudo-section
    if plotIt:
        IO.plotPseudoSection(
            data=survey.dobs, data_type='apparent_resistivity'
        )

    # Show apparent resisitivty histogram
    if plotIt:
        fig = plt.figure()
        out = hist(survey.dobs, bins=20)
        plt.xlabel("Apparent Resisitivty ($\Omega$m)")
        plt.show()

    # Set initial model based upon histogram
    m0 = np.ones(actmap.nP)*np.log(100.)

    # Set uncertainty
    # floor (10 ohm-m)
    eps = 1.
    # percentage
    std = 0.05
    dmisfit = DataMisfit.l2_DataMisfit(survey)
    uncert = abs(survey.dobs) * std + eps
    dmisfit.W = 1./uncert

    # Map for a regularization
    regmap = Maps.IdentityMap(nP=int(actind.sum()))

    # Related to inversion
    reg = Regularization.Sparse(mesh, indActive=actind, mapping=regmap)
    opt = Optimization.InexactGaussNewton(maxIter=15)
    invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)
    beta = Directives.BetaSchedule(coolingFactor=5, coolingRate=2)
    betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1e0)
    target = Directives.TargetMisfit()
    updateSensW = Directives.UpdateSensitivityWeights()
    update_Jacobi = Directives.UpdatePreconditioner()
    inv = Inversion.BaseInversion(
        invProb, directiveList=[
            beta, betaest, target, updateSensW, update_Jacobi
        ]
        )
    prb.counter = opt.counter = Utils.Counter()
    opt.LSshorten = 0.5
    opt.remember('xc')

    # Run inversion
    mopt = inv.run(m0)

    # Get diag(JtJ)
    mask_inds = np.ones(mesh.nC, dtype=bool)
    jtj = np.sqrt(updateSensW.JtJdiag[0])
    jtj /= jtj.max()
    temp = np.ones_like(jtj, dtype=bool)
    temp[jtj > 0.005] = False
    mask_inds[actind] = temp
    actind_final = np.logical_and(actind, ~mask_inds)
    jtj_cc = np.ones(mesh.nC)*np.nan
    jtj_cc[actind] = jtj

    # Show the sensitivity
    if plotIt:
        fig = plt.figure(figsize=(12, 3))
        ax = plt.subplot(111)
        temp = rho.copy()
        temp[~actind] = np.nan
        out = mesh.plotImage(
            jtj_cc, grid=True, ax=ax,
            gridOpts={'alpha': 0.2}, clim=(0.005, 0.5),
            pcolorOpts={"cmap": "viridis", "norm": colors.LogNorm()}
        )
        ax.plot(
            survey.electrode_locations[:, 0],
            survey.electrode_locations[:, 1], 'k.'
        )
        ax.set_xlim(IO.grids[:, 0].min(), IO.grids[:, 0].max())
        ax.set_ylim(-IO.grids[:, 1].max(), IO.grids[:, 1].min())
        cb = plt.colorbar(out[0])
        cb.set_label("Sensitivity")
        ax.set_aspect('equal')
        plt.show()

    # Convert obtained inversion model to resistivity
    # rho = M(m), where M(.) is a mapping

    rho_est = mapping*mopt
    rho_est[~actind_final] = np.nan
    rho_true = rho.copy()
    rho_true[~actind_final] = np.nan

    # show recovered conductivity
    if plotIt:
        vmin, vmax = rho.min(), rho.max()
        fig, ax = plt.subplots(2, 1, figsize=(20, 6))
        out1 = mesh.plotImage(
                rho_true, clim=(10, 1000),
                pcolorOpts={"cmap": "viridis", "norm": colors.LogNorm()},
                ax=ax[0]
        )
        out2 = mesh.plotImage(
            rho_est, clim=(10, 1000),
            pcolorOpts={"cmap": "viridis", "norm": colors.LogNorm()},
            ax=ax[1]
        )
        out = [out1, out2]
        for i in range(2):
            ax[i].plot(
                survey.electrode_locations[:, 0],
                survey.electrode_locations[:, 1], 'kv'
            )
            ax[i].set_xlim(IO.grids[:, 0].min(), IO.grids[:, 0].max())
            ax[i].set_ylim(-IO.grids[:, 1].max(), IO.grids[:, 1].min())
            cb = plt.colorbar(out[i][0], ax=ax[i])
            cb.set_label("Resistivity ($\Omega$m)")
            ax[i].set_xlabel("Northing (m)")
            ax[i].set_ylabel("Elevation (m)")
            ax[i].set_aspect('equal')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    run()
