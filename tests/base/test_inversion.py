import discretize
import pytest
import numpy as np

import simpeg.optimization as smp_opt
import simpeg.inversion as smp_inv
import simpeg.directives as smp_drcs
import simpeg.simulation

from simpeg.inverse_problem import BaseInvProblem

SIMPEG_OPTIMIZERS = [
    smp_opt.ProjectedGradient,
    smp_opt.BFGS,
    smp_opt.InexactGaussNewton,
    smp_opt.SteepestDescent,
    smp_opt.ProjectedGNCG,
]


@pytest.fixture(params=SIMPEG_OPTIMIZERS)
def inversion(request):
    opt = request.param(maxIter=0)

    mesh = discretize.TensorMesh([10])
    n_d = 5
    sim = simpeg.simulation.ExponentialSinusoidSimulation(
        mesh=mesh,
        n_kernels=n_d,
        model_map=simpeg.maps.IdentityMap(mesh),
    )
    m0 = np.zeros(mesh.n_cells)
    data = sim.make_synthetic_data(
        m0, add_noise=True, relative_error=0, noise_floor=0.1, seed=0
    )
    dmis = simpeg.data_misfit.L2DataMisfit(data, sim)
    reg = simpeg.regularization.Smallness(mesh)

    prob = BaseInvProblem(dmis, reg, opt)
    return smp_inv.BaseInversion(prob)


@pytest.mark.parametrize("dlist", [[], [smp_drcs.UpdatePreconditioner()]])
def test_bfgs_init_logic(inversion, dlist, capsys):
    dlist = smp_drcs.DirectiveList(*dlist, inversion=inversion)
    inversion.directiveList = dlist

    inv_prb = inversion.invProb

    # Always defaults to trying to initialize bfgs with reg.deriv2
    assert inv_prb.init_bfgs

    m0 = np.zeros(10)
    inversion.run(m0)
    captured = capsys.readouterr()

    if isinstance(inv_prb.opt, smp_opt.InexactGaussNewton) and any(
        isinstance(dr, smp_drcs.UpdatePreconditioner) for dr in dlist
    ):
        assert not inv_prb.init_bfgs
        assert "bfgsH0" not in captured.out
    elif isinstance(inv_prb.opt, (smp_opt.BFGS, smp_opt.InexactGaussNewton)):
        assert inv_prb.init_bfgs
        assert "bfgsH0" in captured.out
    else:
        assert inv_prb.init_bfgs  # defaults to True even if opt would not use it.
        assert (
            "bfgsH0" not in captured.out
        )  # But shouldn't say anything if it doesn't use it.
