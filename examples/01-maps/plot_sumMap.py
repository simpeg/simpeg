"""
Maps: ComboMaps
===============

Invert synthetic magnetic data with variable background values
and a single block anomaly buried at depth. We will use the Sum Map
to invert for both the background values and an heterogeneous susceptibiilty
model.

.. code-block:: python
    :linenos:


"""
import discretize
from SimPEG import (
    utils,
    maps,
    regularization,
    data_misfit,
    optimization,
    inverse_problem,
    directives,
    inversion,
)
from SimPEG.potential_fields import magnetics
import numpy as np
import matplotlib.pyplot as plt


def run(plotIt=True):

    H0 = (50000.0, 90.0, 0.0)

    # Create a mesh
    dx = 5.0

    hxind = [(dx, 5, -1.3), (dx, 10), (dx, 5, 1.3)]
    hyind = [(dx, 5, -1.3), (dx, 10), (dx, 5, 1.3)]
    hzind = [(dx, 5, -1.3), (dx, 10)]

    mesh = discretize.TensorMesh([hxind, hyind, hzind], "CCC")

    # Lets create a simple Gaussian topo and set the active cells
    [xx, yy] = np.meshgrid(mesh.vectorNx, mesh.vectorNy)
    zz = -np.exp((xx ** 2 + yy ** 2) / 75 ** 2) + mesh.vectorNz[-1]

    # We would usually load a topofile
    topo = np.c_[utils.mkvc(xx), utils.mkvc(yy), utils.mkvc(zz)]

    # Go from topo to array of indices of active cells
    actv = utils.surface2ind_topo(mesh, topo, "N")
    nC = int(actv.sum())
    # Create and array of observation points
    xr = np.linspace(-20.0, 20.0, 20)
    yr = np.linspace(-20.0, 20.0, 20)
    X, Y = np.meshgrid(xr, yr)

    # Move the observation points 5m above the topo
    Z = -np.exp((X ** 2 + Y ** 2) / 75 ** 2) + mesh.vectorNz[-1] + 5.0

    # Create a MAGsurvey
    rxLoc = np.c_[utils.mkvc(X.T), utils.mkvc(Y.T), utils.mkvc(Z.T)]
    rxLoc = magnetics.Point(rxLoc)
    srcField = magnetics.SourceField([rxLoc], parameters=H0)
    survey = magnetics.Survey(srcField)

    # We can now create a susceptibility model and generate data
    model = np.zeros(mesh.nC)

    # Change values in half the domain
    model[mesh.gridCC[:, 0] < 0] = 0.01

    # Add a block in half-space
    model = utils.model_builder.addBlock(
        mesh.gridCC, model, np.r_[-10, -10, 20], np.r_[10, 10, 40], 0.05
    )

    model = utils.mkvc(model)
    model = model[actv]

    # Create active map to go from reduce set to full
    actvMap = maps.InjectActiveCells(mesh, actv, np.nan)

    # Create reduced identity map
    idenMap = maps.IdentityMap(nP=nC)

    # Create the forward model operator
    prob = magnetics.Simulation3DIntegral(
        mesh,
        survey=survey,
        chiMap=idenMap,
        actInd=actv,
        store_sensitivities="forward_only",
    )

    # Compute linear forward operator and compute some data
    data = prob.make_synthetic_data(
        model, relative_error=0.0, noise_floor=1, add_noise=True
    )

    # Create a homogenous maps for the two domains
    domains = [mesh.gridCC[actv, 0] < 0, mesh.gridCC[actv, 0] >= 0]
    homogMap = maps.SurjectUnits(domains)

    # Create a wire map for a second model space, voxel based
    wires = maps.Wires(("homo", len(domains)), ("hetero", nC))

    # Create Sum map
    sumMap = maps.SumMap([homogMap * wires.homo, wires.hetero])

    # Create the forward model operator
    prob = magnetics.Simulation3DIntegral(
        mesh, survey=survey, chiMap=sumMap, actInd=actv, store_sensitivities="ram"
    )

    # Make depth weighting
    wr = np.zeros(sumMap.shape[1])

    # print(prob.M.shape) # why does this reset nC
    G = prob.G

    # Take the cell number out of the scaling.
    # Want to keep high sens for large volumes
    scale = utils.sdiag(np.r_[utils.mkvc(1.0 / homogMap.P.sum(axis=0)), np.ones(nC)])

    # for ii in range(survey.nD):
    #     wr += (
    #         (prob.G[ii, :] * prob.chiMap.deriv(np.ones(sumMap.shape[1]) * 1e-4) * scale)
    #         / data.standard_deviation[ii]
    #     ) ** 2.0 / np.r_[homogMap.P.T * mesh.cell_volumes[actv], mesh.cell_volumes[actv]] **2.

    wr = (
        prob.getJtJdiag(np.ones(sumMap.shape[1]))
        / np.r_[homogMap.P.T * mesh.cell_volumes[actv], mesh.cell_volumes[actv]] ** 2.0
    )
    # Scale the model spaces independently
    wr[wires.homo.index] /= np.max((wires.homo * wr)) * utils.mkvc(
        homogMap.P.sum(axis=0).flatten()
    )
    wr[wires.hetero.index] /= np.max(wires.hetero * wr)
    wr = wr ** 0.5

    ## Create a regularization
    # For the homogeneous model
    regMesh = discretize.TensorMesh([len(domains)])

    reg_m1 = regularization.Sparse(regMesh, mapping=wires.homo)
    reg_m1.cell_weights = wires.homo * wr
    reg_m1.norms = [0, 2]
    reg_m1.mref = np.zeros(sumMap.shape[1])

    # Regularization for the voxel model
    reg_m2 = regularization.Sparse(
        mesh, active_cells=actv, mapping=wires.hetero, gradient_type="components"
    )
    reg_m2.cell_weights = wires.hetero * wr
    reg_m2.norms = [0, 0, 0, 0]
    reg_m2.mref = np.zeros(sumMap.shape[1])

    reg = reg_m1 + reg_m2

    # Data misfit function
    dmis = data_misfit.L2DataMisfit(simulation=prob, data=data)

    # Add directives to the inversion
    opt = optimization.ProjectedGNCG(
        maxIter=100,
        lower=0.0,
        upper=1.0,
        maxIterLS=20,
        maxIterCG=10,
        tolCG=1e-3,
        tolG=1e-3,
        eps=1e-6,
    )
    invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)
    betaest = directives.BetaEstimate_ByEig(beta0_ratio=1e-2)

    # Here is where the norms are applied
    # Use pick a threshold parameter empirically based on the distribution of
    #  model parameters
    IRLS = directives.Update_IRLS(f_min_change=1e-3, minGNiter=1)
    update_Jacobi = directives.UpdatePreconditioner()
    inv = inversion.BaseInversion(invProb, directiveList=[IRLS, betaest, update_Jacobi])

    # Run the inversion
    m0 = np.ones(sumMap.shape[1]) * 1e-4  # Starting model
    prob.model = m0
    mrecSum = inv.run(m0)
    if plotIt:

        mesh.plot_3d_slicer(
            actvMap * model,
            aspect="equal",
            zslice=30,
            pcolorOpts={"cmap": "inferno_r"},
            transparent="slider",
        )

        mesh.plot_3d_slicer(
            actvMap * sumMap * mrecSum,
            aspect="equal",
            zslice=30,
            pcolorOpts={"cmap": "inferno_r"},
            transparent="slider",
        )


if __name__ == "__main__":
    run()
    plt.show()
