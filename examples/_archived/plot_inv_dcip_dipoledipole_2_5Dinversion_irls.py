"""
2.5D DC inversion of with Iterative Reweighted Least Squares
============================================================

This is an example for 2.5D DC Inversion with Iterative Reweighted
Least Squares (IRLS). Earth includes a topography,
and below the topography conductive and resistive cylinders are embedded.
User is promoted to try different p, qx, qz.
For instance a set of paraemters (default):

* p=0 (sparse model, m)
* qx=2 (smooth model, m in x-direction)
* qz=2 (smooth model, m in z-direction)

But if you want share edges of the model, you can try:

* p=0 (sparse model, m)
* qx=0 (smooth model, m in x-direction)
* qz=2 (smooth model, m in z-direction)

"""

from SimPEG.electromagnetics.static import resistivity as DC
from SimPEG.electromagnetics.static.utils import gen_DCIPsurvey, genTopography
from SimPEG import (
    maps,
    utils,
    data_misfit,
    regularization,
    optimization,
    inversion,
    inverse_problem,
    directives,
)
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from pylab import hist

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver


def run(plotIt=True, survey_type="dipole-dipole", p=0.0, qx=2.0, qz=2.0):
    np.random.seed(1)
    # Initiate I/O class for DC
    IO = DC.IO()
    # Obtain ABMN locations

    xmin, xmax = 0.0, 200.0
    ymin, ymax = 0.0, 0.0
    zmin, zmax = 0, 0
    endl = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
    # Generate DC survey object
    survey = gen_DCIPsurvey(endl, survey_type=survey_type, dim=2, a=10, b=10, n=10)
    survey = IO.from_abmn_locations_to_survey(
        survey.locations_a,
        survey.locations_b,
        survey.locations_m,
        survey.locations_n,
        survey_type,
        data_dc_type="volt",
    )

    # Obtain 2D TensorMesh
    mesh, actind = IO.set_mesh()
    topo, mesh1D = genTopography(mesh, -10, 0, its=100)
    actind = utils.surface2ind_topo(mesh, np.c_[mesh1D.vectorCCx, topo])
    survey.drape_electrodes_on_topography(mesh, actind, option="top")

    # Build a conductivity model
    blk_inds_c = utils.model_builder.getIndicesSphere(
        np.r_[60.0, -25.0], 12.5, mesh.gridCC
    )
    blk_inds_r = utils.model_builder.getIndicesSphere(
        np.r_[140.0, -25.0], 12.5, mesh.gridCC
    )
    layer_inds = mesh.gridCC[:, 1] > -5.0
    sigma = np.ones(mesh.nC) * 1.0 / 100.0
    sigma[blk_inds_c] = 1.0 / 10.0
    sigma[blk_inds_r] = 1.0 / 1000.0
    sigma[~actind] = 1.0 / 1e8
    rho = 1.0 / sigma

    # Show the true conductivity model
    if plotIt:
        fig = plt.figure(figsize=(12, 3))
        ax = plt.subplot(111)
        temp = rho.copy()
        temp[~actind] = np.nan
        out = mesh.plotImage(
            temp,
            grid=True,
            ax=ax,
            gridOpts={"alpha": 0.2},
            clim=(10, 1000),
            pcolorOpts={"cmap": "viridis", "norm": colors.LogNorm()},
        )
        ax.plot(
            survey.electrode_locations[:, 0], survey.electrode_locations[:, 1], "k."
        )
        ax.set_xlim(IO.grids[:, 0].min(), IO.grids[:, 0].max())
        ax.set_ylim(-IO.grids[:, 1].max(), IO.grids[:, 1].min())
        cb = plt.colorbar(out[0])
        cb.set_label("Resistivity (ohm-m)")
        ax.set_aspect("equal")
        plt.show()

    # Use Exponential Map: m = log(rho)
    actmap = maps.InjectActiveCells(mesh, indActive=actind, valInactive=np.log(1e8))
    mapping = maps.ExpMap(mesh) * actmap

    # Generate mtrue
    mtrue = np.log(rho[actind])

    # Generate 2.5D DC problem
    # "N" means potential is defined at nodes
    prb = DC.Simulation2DNodal(
        mesh, survey=survey, rhoMap=mapping, storeJ=True, Solver=Solver, verbose=True
    )

    # Make synthetic DC data with 5% Gaussian noise
    data = prb.make_synthetic_data(mtrue, relative_error=0.05, add_noise=True)

    IO.data_dc = data.dobs
    # Show apparent resisitivty pseudo-section
    if plotIt:
        IO.plotPseudoSection(data=data.dobs / IO.G, data_type="apparent_resistivity")

    # Show apparent resisitivty histogram
    if plotIt:
        fig = plt.figure()
        out = hist(data.dobs / IO.G, bins=20)
        plt.xlabel("Apparent Resisitivty ($\Omega$m)")
        plt.show()

    # Set initial model based upon histogram
    m0 = np.ones(actmap.nP) * np.log(100.0)

    # Set standard_deviation
    # floor
    eps = 10 ** (-3.2)
    # percentage
    relative = 0.05
    dmisfit = data_misfit.L2DataMisfit(simulation=prb, data=data)
    uncert = abs(data.dobs) * relative + eps
    dmisfit.standard_deviation = uncert

    # Map for a regularization
    regmap = maps.IdentityMap(nP=int(actind.sum()))

    # Related to inversion
    reg = regularization.Sparse(
        mesh, indActive=actind, mapping=regmap, gradientType="components"
    )
    #     gradientType = 'components'
    reg.norms = [p, qx, qz, 0.0]
    IRLS = directives.Update_IRLS(
        max_irls_iterations=20, minGNiter=1, beta_search=False, fix_Jmatrix=True
    )

    opt = optimization.InexactGaussNewton(maxIter=40)
    invProb = inverse_problem.BaseInvProblem(dmisfit, reg, opt)
    beta = directives.BetaSchedule(coolingFactor=5, coolingRate=2)
    betaest = directives.BetaEstimate_ByEig(beta0_ratio=1e0)
    target = directives.TargetMisfit()
    update_Jacobi = directives.UpdatePreconditioner()
    inv = inversion.BaseInversion(invProb, directiveList=[betaest, IRLS])
    prb.counter = opt.counter = utils.Counter()
    opt.LSshorten = 0.5
    opt.remember("xc")

    # Run inversion
    mopt = inv.run(m0)

    rho_est = mapping * mopt
    rho_est_l2 = mapping * invProb.l2model
    rho_est[~actind] = np.nan
    rho_est_l2[~actind] = np.nan
    rho_true = rho.copy()
    rho_true[~actind] = np.nan

    # show recovered conductivity
    if plotIt:
        vmin, vmax = rho.min(), rho.max()
        fig, ax = plt.subplots(3, 1, figsize=(20, 9))
        out1 = mesh.plotImage(
            rho_true,
            clim=(10, 1000),
            pcolorOpts={"cmap": "viridis", "norm": colors.LogNorm()},
            ax=ax[0],
        )
        out2 = mesh.plotImage(
            rho_est_l2,
            clim=(10, 1000),
            pcolorOpts={"cmap": "viridis", "norm": colors.LogNorm()},
            ax=ax[1],
        )
        out3 = mesh.plotImage(
            rho_est,
            clim=(10, 1000),
            pcolorOpts={"cmap": "viridis", "norm": colors.LogNorm()},
            ax=ax[2],
        )

        out = [out1, out2, out3]
        titles = ["True", "L2", ("L%d, Lx%d, Lz%d") % (p, qx, qz)]
        for i in range(3):
            ax[i].plot(
                survey.electrode_locations[:, 0], survey.electrode_locations[:, 1], "kv"
            )
            ax[i].set_xlim(IO.grids[:, 0].min(), IO.grids[:, 0].max())
            ax[i].set_ylim(-IO.grids[:, 1].max(), IO.grids[:, 1].min())
            cb = plt.colorbar(out[i][0], ax=ax[i])
            cb.set_label("Resistivity ($\Omega$m)")
            ax[i].set_xlabel("Northing (m)")
            ax[i].set_ylabel("Elevation (m)")
            ax[i].set_aspect("equal")
            ax[i].set_title(titles[i])
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    run()
