"""
PF: Gravity: Inversion Linear
=============================

Create a synthetic block model and invert
with a compact norm

"""
import numpy as np
import matplotlib.pyplot as plt

from discretize import TensorMesh
from SimPEG.potential_fields import gravity
from SimPEG import (
    maps,
    data,
    data_misfit,
    regularization,
    optimization,
    inverse_problem,
    directives,
    inversion,
)

from SimPEG import utils
from SimPEG.utils import plot2Ddata


def run(plotIt=True):

    # Create a mesh
    dx = 5.0

    hxind = [(dx, 5, -1.3), (dx, 15), (dx, 5, 1.3)]
    hyind = [(dx, 5, -1.3), (dx, 15), (dx, 5, 1.3)]
    hzind = [(dx, 5, -1.3), (dx, 7), (3.5, 1), (2, 5)]

    mesh = TensorMesh([hxind, hyind, hzind], "CCC")

    # Get index of the center
    midx = int(mesh.nCx / 2)
    midy = int(mesh.nCy / 2)

    # Lets create a simple Gaussian topo and set the active cells
    [xx, yy] = np.meshgrid(mesh.vectorNx, mesh.vectorNy)
    zz = -np.exp((xx ** 2 + yy ** 2) / 75 ** 2) + mesh.vectorNz[-1]

    # We would usually load a topofile
    topo = np.c_[utils.mkvc(xx), utils.mkvc(yy), utils.mkvc(zz)]

    # Go from topo to array of indices of active cells
    actv = utils.surface2ind_topo(mesh, topo, "N")
    actv = np.where(actv)[0]
    nC = len(actv)

    # Create and array of observation points
    xr = np.linspace(-30.0, 30.0, 20)
    yr = np.linspace(-30.0, 30.0, 20)
    X, Y = np.meshgrid(xr, yr)

    # Move the observation points 5m above the topo
    Z = -np.exp((X ** 2 + Y ** 2) / 75 ** 2) + mesh.vectorNz[-1] + 0.1

    # Create a GRAVsurvey
    rxLoc = np.c_[utils.mkvc(X.T), utils.mkvc(Y.T), utils.mkvc(Z.T)]
    rxLoc = gravity.receivers.Point(rxLoc)
    srcField = gravity.sources.SourceField([rxLoc])
    survey = gravity.survey.Survey(srcField)

    # We can now create a density model and generate data
    # Here a simple block in half-space
    model = np.zeros((mesh.nCx, mesh.nCy, mesh.nCz))
    model[(midx - 5) : (midx - 1), (midy - 2) : (midy + 2), -10:-6] = 0.75
    model[(midx + 1) : (midx + 5), (midy - 2) : (midy + 2), -10:-6] = -0.75
    model = utils.mkvc(model)
    model = model[actv]

    # Create active map to go from reduce set to full
    actvMap = maps.InjectActiveCells(mesh, actv, -100)

    # Create reduced identity map
    idenMap = maps.IdentityMap(nP=nC)

    # Create the forward simulation
    simulation = gravity.simulation.Simulation3DIntegral(
        survey=survey, mesh=mesh, rhoMap=idenMap, actInd=actv
    )

    # Compute linear forward operator and compute some data
    d = simulation.fields(model)

    # Add noise and uncertainties
    # We add some random Gaussian noise (1nT)
    synthetic_data = d + np.random.randn(len(d)) * 1e-3
    wd = np.ones(len(synthetic_data)) * 1e-3  # Assign flat uncertainties

    data_object = data.Data(survey, dobs=synthetic_data, noise_floor=wd)

    m0 = np.ones(nC) * 1e-4  # Starting model

    # Create sensitivity weights from our linear forward operator
    rxLoc = survey.source_field.receiver_list[0].locations

    # Create a regularization
    reg = regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
    reg.norms = [0, 0, 0, 0]

    # Data misfit function
    dmis = data_misfit.L2DataMisfit(data=data_object, simulation=simulation)
    dmis.W = utils.sdiag(1 / wd)

    # Add directives to the inversion
    opt = optimization.ProjectedGNCG(
        maxIter=100, lower=-1.0, upper=1.0, maxIterLS=20, maxIterCG=10, tolCG=1e-3
    )
    invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)
    betaest = directives.BetaEstimate_ByEig(beta0_ratio=1e-1)

    # Here is where the norms are applied
    # Use pick a threshold parameter empirically based on the distribution of
    # model parameters
    update_IRLS = directives.Update_IRLS(
        f_min_change=1e-4,
        max_irls_iterations=30,
        coolEpsFact=1.5,
        beta_tol=1e-2,
    )
    saveDict = directives.SaveOutputEveryIteration(save_txt=False)
    update_Jacobi = directives.UpdatePreconditioner()
    sensitivity_weights = directives.UpdateSensitivityWeights(everyIter=False)
    inv = inversion.BaseInversion(
        invProb,
        directiveList=[
            update_IRLS,
            sensitivity_weights,
            betaest,
            update_Jacobi,
            saveDict,
        ],
    )

    # Run the inversion

    mrec = inv.run(m0)

    if plotIt:
        # Here is the recovered susceptibility model
        ypanel = midx
        zpanel = -7
        m_l2 = actvMap * invProb.l2model
        m_l2[m_l2 == -100] = np.nan

        m_lp = actvMap * mrec
        m_lp[m_lp == -100] = np.nan

        m_true = actvMap * model
        m_true[m_true == -100] = np.nan

        vmin, vmax = mrec.min(), mrec.max()

        # Plot the data
        plot2Ddata(rxLoc, data_object.dobs)

        plt.figure()

        # Plot L2 model
        ax = plt.subplot(321)
        mesh.plotSlice(
            m_l2, ax=ax, normal="Z", ind=zpanel, grid=True, clim=(vmin, vmax)
        )
        plt.plot(
            ([mesh.vectorCCx[0], mesh.vectorCCx[-1]]),
            ([mesh.vectorCCy[ypanel], mesh.vectorCCy[ypanel]]),
            color="w",
        )
        plt.title("Plan l2-model.")
        plt.gca().set_aspect("equal")
        plt.ylabel("y")
        ax.xaxis.set_visible(False)
        plt.gca().set_aspect("equal", adjustable="box")

        # Vertical section
        ax = plt.subplot(322)
        mesh.plotSlice(m_l2, ax=ax, normal="Y", ind=midx, grid=True, clim=(vmin, vmax))
        plt.plot(
            ([mesh.vectorCCx[0], mesh.vectorCCx[-1]]),
            ([mesh.vectorCCz[zpanel], mesh.vectorCCz[zpanel]]),
            color="w",
        )
        plt.title("E-W l2-model.")
        plt.gca().set_aspect("equal")
        ax.xaxis.set_visible(False)
        plt.ylabel("z")
        plt.gca().set_aspect("equal", adjustable="box")

        # Plot Lp model
        ax = plt.subplot(323)
        mesh.plotSlice(
            m_lp, ax=ax, normal="Z", ind=zpanel, grid=True, clim=(vmin, vmax)
        )
        plt.plot(
            ([mesh.vectorCCx[0], mesh.vectorCCx[-1]]),
            ([mesh.vectorCCy[ypanel], mesh.vectorCCy[ypanel]]),
            color="w",
        )
        plt.title("Plan lp-model.")
        plt.gca().set_aspect("equal")
        ax.xaxis.set_visible(False)
        plt.ylabel("y")
        plt.gca().set_aspect("equal", adjustable="box")

        # Vertical section
        ax = plt.subplot(324)
        mesh.plotSlice(m_lp, ax=ax, normal="Y", ind=midx, grid=True, clim=(vmin, vmax))
        plt.plot(
            ([mesh.vectorCCx[0], mesh.vectorCCx[-1]]),
            ([mesh.vectorCCz[zpanel], mesh.vectorCCz[zpanel]]),
            color="w",
        )
        plt.title("E-W lp-model.")
        plt.gca().set_aspect("equal")
        ax.xaxis.set_visible(False)
        plt.ylabel("z")
        plt.gca().set_aspect("equal", adjustable="box")

        # Plot True model
        ax = plt.subplot(325)
        mesh.plotSlice(
            m_true, ax=ax, normal="Z", ind=zpanel, grid=True, clim=(vmin, vmax)
        )
        plt.plot(
            ([mesh.vectorCCx[0], mesh.vectorCCx[-1]]),
            ([mesh.vectorCCy[ypanel], mesh.vectorCCy[ypanel]]),
            color="w",
        )
        plt.title("Plan true model.")
        plt.gca().set_aspect("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.gca().set_aspect("equal", adjustable="box")

        # Vertical section
        ax = plt.subplot(326)
        mesh.plotSlice(
            m_true, ax=ax, normal="Y", ind=midx, grid=True, clim=(vmin, vmax)
        )
        plt.plot(
            ([mesh.vectorCCx[0], mesh.vectorCCx[-1]]),
            ([mesh.vectorCCz[zpanel], mesh.vectorCCz[zpanel]]),
            color="w",
        )
        plt.title("E-W true model.")
        plt.gca().set_aspect("equal")
        plt.xlabel("x")
        plt.ylabel("z")
        plt.gca().set_aspect("equal", adjustable="box")

        # Plot convergence curves
        fig, axs = plt.figure(), plt.subplot()
        axs.plot(saveDict.phi_d, "k", lw=2)
        axs.plot(
            np.r_[update_IRLS.iterStart, update_IRLS.iterStart],
            np.r_[0, np.max(saveDict.phi_d)],
            "k:",
        )

        twin = axs.twinx()
        twin.plot(saveDict.phi_m, "k--", lw=2)
        axs.text(
            update_IRLS.iterStart,
            np.max(saveDict.phi_d) / 2.0,
            "IRLS Steps",
            va="bottom",
            ha="center",
            rotation="vertical",
            size=12,
            bbox={"facecolor": "white"},
        )

        axs.set_ylabel("$\phi_d$", size=16, rotation=0)
        axs.set_xlabel("Iterations", size=14)
        twin.set_ylabel("$\phi_m$", size=16, rotation=0)


if __name__ == "__main__":
    run()
    plt.show()
