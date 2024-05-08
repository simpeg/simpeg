"""
PF: Magnetic: Inversion Linear
===============================

Create a synthetic block model and invert
with a compact norm

"""

import matplotlib.pyplot as plt
import numpy as np
from discretize import TensorMesh
from discretize.utils import active_from_xyz
from SimPEG.potential_fields import magnetics
from SimPEG import utils
from SimPEG import (
    data,
    data_misfit,
    maps,
    regularization,
    optimization,
    inverse_problem,
    directives,
    inversion,
)


def run(plotIt=True):
    # Define the inducing field parameter
    h0_amplitude, h0_inclination, h0_declination = (50000, 90, 0)

    # Create a mesh
    dx = 5.0

    hxind = [(dx, 5, -1.3), (dx, 10), (dx, 5, 1.3)]
    hyind = [(dx, 5, -1.3), (dx, 10), (dx, 5, 1.3)]
    hzind = [(dx, 5, -1.3), (dx, 10)]

    mesh = TensorMesh([hxind, hyind, hzind], "CCC")

    # Get index of the center
    midx = int(mesh.shape_cells[0] / 2)
    midy = int(mesh.shape_cells[1] / 2)

    # Lets create a simple Gaussian topo and set the active cells
    [xx, yy] = np.meshgrid(mesh.nodes_x, mesh.nodes_y)
    zz = -np.exp((xx**2 + yy**2) / 75**2) + mesh.nodes_z[-1]

    # We would usually load a topofile
    topo = np.c_[utils.mkvc(xx), utils.mkvc(yy), utils.mkvc(zz)]

    # Go from topo to array of indices of active cells
    actv = active_from_xyz(mesh, topo, "N")
    actv = np.where(actv)[0]
    nC = len(actv)

    # Create and array of observation points
    xr = np.linspace(-20.0, 20.0, 20)
    yr = np.linspace(-20.0, 20.0, 20)
    X, Y = np.meshgrid(xr, yr)

    # Move the observation points 5m above the topo
    Z = -np.exp((X**2 + Y**2) / 75**2) + mesh.nodes_z[-1] + 5.0

    # Create a MAGsurvey
    rxLoc = np.c_[utils.mkvc(X.T), utils.mkvc(Y.T), utils.mkvc(Z.T)]
    rxLoc = magnetics.receivers.Point(rxLoc, components=["tmi"])
    srcField = magnetics.sources.UniformBackgroundField(
        receiver_list=[rxLoc],
        amplitude=h0_amplitude,
        inclination=h0_inclination,
        declination=h0_declination,
    )
    survey = magnetics.survey.Survey(srcField)

    # We can now create a susceptibility model and generate data
    # Here a simple block in half-space
    model = np.zeros(mesh.shape_cells)
    model[(midx - 2) : (midx + 2), (midy - 2) : (midy + 2), -6:-2] = 0.02
    model = utils.mkvc(model)
    model = model[actv]

    # Create active map to go from reduce set to full
    actvMap = maps.InjectActiveCells(mesh, actv, -100)

    # Create reduced identity map
    idenMap = maps.IdentityMap(nP=nC)

    # Create the forward model operator
    simulation = magnetics.simulation.Simulation3DIntegral(
        survey=survey,
        mesh=mesh,
        chiMap=idenMap,
        ind_active=actv,
    )

    # Compute linear forward operator and compute some data
    d = simulation.dpred(model)

    # Add noise and uncertainties
    # We add some random Gaussian noise (1nT)
    synthetic_data = d + np.random.randn(len(d))
    wd = np.ones(len(synthetic_data)) * 1.0  # Assign flat uncertainties

    data_object = data.Data(survey, dobs=synthetic_data, noise_floor=wd)

    # Create a regularization
    reg = regularization.Sparse(mesh, active_cells=actv, mapping=idenMap)
    reg.mref = np.zeros(nC)
    reg.norms = [0, 0, 0, 0]
    # reg.eps_p, reg.eps_q = 1e-0, 1e-0

    # Create sensitivity weights from our linear forward operator
    rxLoc = survey.source_field.receiver_list[0].locations
    m0 = np.ones(nC) * 1e-4  # Starting model

    # Data misfit function
    dmis = data_misfit.L2DataMisfit(simulation=simulation, data=data_object)
    dmis.W = 1 / wd

    # Add directives to the inversion
    opt = optimization.ProjectedGNCG(
        maxIter=20, lower=0.0, upper=1.0, maxIterLS=20, maxIterCG=20, tolCG=1e-3
    )
    invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)
    betaest = directives.BetaEstimate_ByEig(beta0_ratio=1e-1)

    # Here is where the norms are applied
    # Use pick a threshold parameter empirically based on the distribution of
    #  model parameters
    IRLS = directives.Update_IRLS(f_min_change=1e-3, max_irls_iterations=40)
    saveDict = directives.SaveOutputEveryIteration(save_txt=False)
    update_Jacobi = directives.UpdatePreconditioner()
    # Add sensitivity weights
    sensitivity_weights = directives.UpdateSensitivityWeights(every_iteration=False)

    inv = inversion.BaseInversion(
        invProb,
        directiveList=[sensitivity_weights, IRLS, betaest, update_Jacobi, saveDict],
    )

    # Run the inversion
    mrec = inv.run(m0)

    if plotIt:
        # Here is the recovered susceptibility model
        ypanel = midx
        zpanel = -5
        m_l2 = actvMap * invProb.l2model
        m_l2[m_l2 == -100] = np.nan

        m_lp = actvMap * mrec
        m_lp[m_lp == -100] = np.nan

        m_true = actvMap * model
        m_true[m_true == -100] = np.nan

        # Plot the data
        utils.plot_utils.plot2Ddata(rxLoc, d)

        plt.figure()

        # Plot L2 model
        ax = plt.subplot(321)
        mesh.plot_slice(
            m_l2,
            ax=ax,
            normal="Z",
            ind=zpanel,
            grid=True,
            clim=(model.min(), model.max()),
        )
        plt.plot(
            ([mesh.cell_centers_x[0], mesh.cell_centers_x[-1]]),
            ([mesh.cell_centers_y[ypanel], mesh.cell_centers_y[ypanel]]),
            color="w",
        )
        plt.title("Plan l2-model.")
        plt.gca().set_aspect("equal")
        plt.ylabel("y")
        ax.xaxis.set_visible(False)
        plt.gca().set_aspect("equal", adjustable="box")

        # Vertica section
        ax = plt.subplot(322)
        mesh.plot_slice(
            m_l2,
            ax=ax,
            normal="Y",
            ind=midx,
            grid=True,
            clim=(model.min(), model.max()),
        )
        plt.plot(
            ([mesh.cell_centers_x[0], mesh.cell_centers_x[-1]]),
            ([mesh.cell_centers_z[zpanel], mesh.cell_centers_z[zpanel]]),
            color="w",
        )
        plt.title("E-W l2-model.")
        plt.gca().set_aspect("equal")
        ax.xaxis.set_visible(False)
        plt.ylabel("z")
        plt.gca().set_aspect("equal", adjustable="box")

        # Plot Lp model
        ax = plt.subplot(323)
        mesh.plot_slice(
            m_lp,
            ax=ax,
            normal="Z",
            ind=zpanel,
            grid=True,
            clim=(model.min(), model.max()),
        )
        plt.plot(
            ([mesh.cell_centers_x[0], mesh.cell_centers_x[-1]]),
            ([mesh.cell_centers_y[ypanel], mesh.cell_centers_y[ypanel]]),
            color="w",
        )
        plt.title("Plan lp-model.")
        plt.gca().set_aspect("equal")
        ax.xaxis.set_visible(False)
        plt.ylabel("y")
        plt.gca().set_aspect("equal", adjustable="box")

        # Vertical section
        ax = plt.subplot(324)
        mesh.plot_slice(
            m_lp,
            ax=ax,
            normal="Y",
            ind=midx,
            grid=True,
            clim=(model.min(), model.max()),
        )
        plt.plot(
            ([mesh.cell_centers_x[0], mesh.cell_centers_x[-1]]),
            ([mesh.cell_centers_z[zpanel], mesh.cell_centers_z[zpanel]]),
            color="w",
        )
        plt.title("E-W lp-model.")
        plt.gca().set_aspect("equal")
        ax.xaxis.set_visible(False)
        plt.ylabel("z")
        plt.gca().set_aspect("equal", adjustable="box")

        # Plot True model
        ax = plt.subplot(325)
        mesh.plot_slice(
            m_true,
            ax=ax,
            normal="Z",
            ind=zpanel,
            grid=True,
            clim=(model.min(), model.max()),
        )
        plt.plot(
            ([mesh.cell_centers_x[0], mesh.cell_centers_x[-1]]),
            ([mesh.cell_centers_y[ypanel], mesh.cell_centers_y[ypanel]]),
            color="w",
        )
        plt.title("Plan true model.")
        plt.gca().set_aspect("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.gca().set_aspect("equal", adjustable="box")

        # Vertical section
        ax = plt.subplot(326)
        mesh.plot_slice(
            m_true,
            ax=ax,
            normal="Y",
            ind=midx,
            grid=True,
            clim=(model.min(), model.max()),
        )
        plt.plot(
            ([mesh.cell_centers_x[0], mesh.cell_centers_x[-1]]),
            ([mesh.cell_centers_z[zpanel], mesh.cell_centers_z[zpanel]]),
            color="w",
        )
        plt.title("E-W true model.")
        plt.gca().set_aspect("equal")
        plt.xlabel("x")
        plt.ylabel("z")
        plt.gca().set_aspect("equal", adjustable="box")

        # Plot convergence curves
        plt.figure()
        axs = plt.subplot()
        axs.plot(saveDict.phi_d, "k", lw=2)
        axs.plot(
            np.r_[IRLS.iterStart, IRLS.iterStart],
            np.r_[0, np.max(saveDict.phi_d)],
            "k:",
        )

        twin = axs.twinx()
        twin.plot(saveDict.phi_m, "k--", lw=2)
        axs.text(
            IRLS.iterStart,
            0,
            "IRLS Steps",
            va="bottom",
            ha="center",
            rotation="vertical",
            size=12,
            bbox={"facecolor": "white"},
        )

        axs.set_ylabel(r"$\phi_d$", size=16, rotation=0)
        axs.set_xlabel("Iterations", size=14)
        twin.set_ylabel(r"$\phi_m$", size=16, rotation=0)


if __name__ == "__main__":
    run()
    plt.show()
