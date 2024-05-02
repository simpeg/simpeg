import pytest
import numpy as np

import discretize
from discretize.utils import active_from_xyz
from simpeg import (
    utils,
    maps,
    regularization,
    data_misfit,
    optimization,
    inverse_problem,
    directives,
    inversion,
)
from simpeg.potential_fields import gravity


@pytest.mark.parametrize("engine", ("geoana", "choclo"))
def test_gravity_inversion_linear(engine):
    """Test gravity inversion."""
    # Create a mesh
    dx = 5.0
    hxind = [(dx, 5, -1.3), (dx, 5), (dx, 5, 1.3)]
    hyind = [(dx, 5, -1.3), (dx, 5), (dx, 5, 1.3)]
    hzind = [(dx, 5, -1.3), (dx, 6)]
    mesh = discretize.TensorMesh([hxind, hyind, hzind], "CCC")

    # Get index of the center
    midx = int(mesh.shape_cells[0] / 2)
    midy = int(mesh.shape_cells[1] / 2)

    # Lets create a simple Gaussian topo and set the active cells
    [xx, yy] = np.meshgrid(mesh.nodes_x, mesh.nodes_y)
    zz = -np.exp((xx**2 + yy**2) / 75**2) + mesh.nodes_z[-1]

    # Go from topo to actv cells
    topo = np.c_[utils.mkvc(xx), utils.mkvc(yy), utils.mkvc(zz)]
    actv = active_from_xyz(mesh, topo, "N")
    nC = int(actv.sum())

    # Create and array of observation points
    xr = np.linspace(-20.0, 20.0, 20)
    yr = np.linspace(-20.0, 20.0, 20)
    X, Y = np.meshgrid(xr, yr)

    # Move the observation points 5m above the topo
    Z = -np.exp((X**2 + Y**2) / 75**2) + mesh.nodes_z[-1] + 5.0

    # Create a MAGsurvey
    locXYZ = np.c_[utils.mkvc(X.T), utils.mkvc(Y.T), utils.mkvc(Z.T)]
    rxLoc = gravity.Point(locXYZ)
    srcField = gravity.SourceField([rxLoc])
    survey = gravity.Survey(srcField)

    # We can now create a density model and generate data
    # Here a simple block in half-space
    model = np.zeros(
        (
            mesh.shape_cells[0],
            mesh.shape_cells[1],
            mesh.shape_cells[2],
        )
    )
    model[(midx - 2) : (midx + 2), (midy - 2) : (midy + 2), -6:-2] = 0.5
    model = utils.mkvc(model)
    model = model[actv]

    # Create reduced identity map
    idenMap = maps.IdentityMap(nP=nC)

    # Create the forward model operator
    kwargs = dict()
    if engine == "geoana":
        kwargs["n_processes"] = None
    sim = gravity.Simulation3DIntegral(
        mesh,
        survey=survey,
        rhoMap=idenMap,
        ind_active=actv,
        store_sensitivities="ram",
        engine=engine,
        **kwargs,
    )

    # Compute linear forward operator and compute some data
    data = sim.make_synthetic_data(
        model, relative_error=0.0, noise_floor=0.0005, add_noise=True, random_seed=2
    )

    # Create a regularization
    reg = regularization.Sparse(mesh, active_cells=actv, mapping=idenMap)
    reg.norms = [0, 0, 0, 0]
    reg.gradient_type = "components"

    # Data misfit function
    dmis = data_misfit.L2DataMisfit(simulation=sim, data=data)

    # Add directives to the inversion
    opt = optimization.ProjectedGNCG(
        maxIter=100, lower=-1.0, upper=1.0, maxIterLS=20, maxIterCG=10, tolCG=1e-3
    )
    invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)

    # Here is where the norms are applied
    starting_beta = directives.BetaEstimateMaxDerivative(10.0)
    IRLS = directives.Update_IRLS()
    update_Jacobi = directives.UpdatePreconditioner()
    sensitivity_weights = directives.UpdateSensitivityWeights(every_iteration=False)
    inv = inversion.BaseInversion(
        invProb,
        directiveList=[IRLS, sensitivity_weights, starting_beta, update_Jacobi],
    )

    # Run the inversion
    mrec = inv.run(model)
    residual = np.linalg.norm(mrec - model) / np.linalg.norm(model)

    # Assert result
    assert np.all(residual < 0.05)
