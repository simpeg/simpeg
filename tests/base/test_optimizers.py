import re
import pytest

from simpeg.optimization import ProjectedGNCG
from simpeg.utils import sdiag
import numpy as np
import numpy.testing as npt
import scipy.sparse as sp
from simpeg import optimization
from discretize.tests import get_quadratic, rosenbrock

TOL = 1e-2

OPTIMIZERS = [
    optimization.GaussNewton,
    optimization.InexactGaussNewton,
    optimization.BFGS,
    optimization.ProjectedGradient,
    optimization.SteepestDescent,
    optimization.ProjectedGNCG,
]

OPT_KWARGS = {
    optimization.GaussNewton: {},
    optimization.InexactGaussNewton: dict(cg_rtol=1e-6, cg_maxiter=100),
    optimization.BFGS: dict(maxIter=100, tolG=1e-2, maxIterLS=20),
    optimization.ProjectedGradient: dict(maxIter=100, cg_rtol=1e-6, cg_maxiter=100),
    optimization.SteepestDescent: dict(maxIter=10000, tolG=1e-5, tolX=1e-8, eps=1e-8),
    optimization.ProjectedGNCG: dict(cg_rtol=1e-6, cg_maxiter=100),
}


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize(
    ("func", "x_true", "x0"),
    [
        (rosenbrock, np.array([1.0, 1.0]), np.array([0, 0])),
        (
            get_quadratic(sp.identity(2).tocsr(), np.array([-5, 5])),
            np.array([5, -5]),
            np.zeros(2),
        ),
    ],
    ids=["rosenbrock", "quadratic"],
)
class TestUnboundOptimizers:

    def test_minimizer(self, optimizer, func, x_true, x0):
        opt = optimizer(**OPT_KWARGS[optimizer])
        xopt = opt.minimize(func, x0)
        npt.assert_allclose(xopt, x_true, rtol=TOL)


def test_NewtonRoot():
    def fun(x, return_g=True):
        if return_g:
            return np.sin(x), sdiag(np.cos(x))
        return np.sin(x)

    x = np.array([np.pi - 0.3, np.pi + 0.1, 0])
    xopt = optimization.NewtonRoot(comments=False).root(fun, x)
    x_true = np.array([np.pi, np.pi, 0])
    npt.assert_allclose(xopt, x_true, rtol=0, atol=TOL)


@pytest.mark.parametrize(
    "optimizer", filter(lambda x: issubclass(x, optimization.Bounded), OPTIMIZERS)
)
@pytest.mark.parametrize(
    ("lower", "upper", "x_true", "x0"),
    [
        (-2, 2, np.array([2.0, -2.0]), np.zeros(2)),
        (-2, 8, np.array([5, -2]), np.zeros(2)),
        (-8, 2, np.array([2, -5]), np.zeros(2)),
    ],
    ids=["both active", "lower active", "upper active"],
)
class TestBoundedOptimizers:
    def test_minimizer(self, optimizer, lower, upper, x_true, x0):
        func = get_quadratic(sp.identity(2).tocsr(), np.array([-5, 5]))
        opt = optimizer(lower=lower, upper=upper)
        xopt = opt.minimize(func, x0)
        npt.assert_allclose(xopt, x_true, rtol=TOL)


@pytest.mark.parametrize(
    ("x0", "bounded"),
    [(np.array([8, 2]), False), (np.array([4, 0]), True)],
    ids=["active not bound", "active and bound"],
)
def test_projected_gncg_active_not_bound_branch(x0, bounded):
    # tests designed to test the branches of the
    # projected gncg when a point is in the active set but not in the binding set.
    func = get_quadratic(sp.identity(2).tocsr(), np.array([-5, 5]))
    opt = ProjectedGNCG(upper=8, lower=0)
    _, g = func(x0, return_g=True, return_H=False)

    opt.g = g
    active = opt.activeSet(x0)
    bound = opt.bindingSet(x0)

    # assert that the initial point is what we intend to hit the correct branch
    # in the minimizer.
    assert not np.any(active & ~bound) is bounded

    xopt = opt.minimize(func, x0)
    x_true = np.array([5, 0])
    npt.assert_allclose(xopt, x_true, rtol=TOL)


@pytest.mark.parametrize("lower", [None, 0.0, np.zeros(10)])
@pytest.mark.parametrize("upper", [None, 1.0, np.ones(10)])
class TestBounded:

    def test_project(self, lower, upper):
        x = np.linspace(-9.5, 8.2, 10)
        bnd = optimization.Bounded(lower=lower, upper=upper)

        x_proj = bnd.projection(x)
        if lower is not None:
            assert x_proj.min() == 0.0
        else:
            assert x_proj.min() == x.min()

        if upper is not None:
            assert x_proj.max() == 1.0
        else:
            assert x_proj.max() == x.max()

    def test_active_set(self, lower, upper):
        x = np.linspace(-9.5, 8.2, 10)
        bnd = optimization.Bounded(lower=lower, upper=upper)

        active_set = bnd.activeSet(x)

        if lower is not None:
            assert all(active_set[x <= lower])
        else:
            assert not any(active_set[x <= 0])

        if upper is not None:
            assert all(active_set[x >= upper])
        else:
            assert not any(active_set[x >= 1])

    def test_inactive_set(self, lower, upper):
        x = np.linspace(-9.5, 8.2, 10)
        bnd = optimization.Bounded(lower=lower, upper=upper)

        inactive_set = bnd.inactiveSet(x)

        if lower is not None:
            assert not any(inactive_set[x <= lower])
        else:
            assert all(inactive_set[x <= 0])

        if upper is not None:
            assert not any(inactive_set[x >= upper])
        else:
            assert all(inactive_set[x >= 1])

    def test_binding_set(self, lower, upper):
        x = np.linspace(-9.5, 8.2, 10)
        g = (np.ones(5)[:, None] * np.array([-1, 1])).reshape(-1)
        assert len(x) == len(g)
        assert g[0] == -1 and g[1] == 1 and g[2] == -1  # and so on
        bnd = optimization.Bounded(lower=lower, upper=upper)
        bnd.g = g

        bnd_set = bnd.bindingSet(x)

        if lower is not None:
            assert all(bnd_set[(x <= lower) & (g >= 0)])
        else:
            assert not any(bnd_set[(x <= 0) & (g >= 0)])

        if upper is not None:
            assert all(bnd_set[(x >= upper) & (g <= 0)])
        else:
            assert not any(bnd_set[(x >= 1) & (g <= 0)])


def test_bounded_kwargs_only():
    with pytest.raises(
        TypeError,
        match=re.escape(
            "Bounded.__init__() takes 1 positional argument but 2 were given"
        ),
    ):
        optimization.Bounded(None)


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (np.zeros(11), None),
        (None, np.ones(11)),
        (np.zeros(11), np.ones(10)),
        (np.zeros(10), np.ones(11)),
        (np.zeros(11), np.ones(11)),
    ],
    ids=["only_lower", "only_upper", "bad_lower", "bad_upper", "both_bad"],
)
@pytest.mark.parametrize(
    "opt_class", [optimization.ProjectedGradient, optimization.ProjectedGNCG]
)
def test_bad_bounds(lower, upper, opt_class):
    m = np.linspace(-9.5, 8.2, 10)
    opt = opt_class(lower=lower, upper=upper)
    with pytest.raises(RuntimeError, match="Initial model is not projectable"):
        opt.startup(m)


class TestInexactCGParams:

    def test_defaults(self):
        cg_pars = optimization.InexactCG()
        assert cg_pars.cg_atol == 0.0
        assert cg_pars.cg_rtol == 1e-1
        assert cg_pars.cg_maxiter == 5

    def test_init(self):
        cg_pars = optimization.InexactCG(cg_rtol=1e-3, cg_atol=1e-5, cg_maxiter=10)
        assert cg_pars.cg_atol == 1e-5
        assert cg_pars.cg_rtol == 1e-3
        assert cg_pars.cg_maxiter == 10

    def test_kwargs_only(self):
        with pytest.raises(
            TypeError,
            match=re.escape(
                "InexactCG.__init__() takes 1 positional argument but 2 were given"
            ),
        ):
            optimization.InexactCG(1e-3)

    def test_deprecated(self):
        with pytest.warns(FutureWarning, match=".*tolCG has been deprecated.*"):
            cg_pars = optimization.InexactCG(tolCG=1e-3)
        assert cg_pars.cg_atol == 0.0
        assert cg_pars.cg_rtol == 1e-3

        with pytest.warns(FutureWarning, match=".*maxIterCG has been deprecated.*"):
            cg_pars = optimization.InexactCG(maxIterCG=3)
        assert cg_pars.cg_atol == 0.0
        assert cg_pars.cg_rtol == 1e-1
        assert cg_pars.cg_maxiter == 3


class TestProjectedGradient:

    def test_defaults(self):
        opt = optimization.ProjectedGradient()
        assert opt.cg_rtol == 1e-1
        assert opt.cg_atol == 0.0
        assert opt.cg_maxiter == 5
        assert np.isneginf(opt.lower)
        assert np.isposinf(opt.upper)

    def test_init(self):
        opt = optimization.ProjectedGradient(
            cg_rtol=1e-3, cg_atol=1e-5, cg_maxiter=10, lower=0.0, upper=1.0
        )
        assert opt.cg_rtol == 1e-3
        assert opt.cg_atol == 1e-5
        assert opt.cg_maxiter == 10
        assert opt.lower == 0.0
        assert opt.upper == 1.0

    def test_kwargs_only(self):
        with pytest.raises(
            TypeError,
            match=re.escape(
                "ProjectedGradient.__init__() takes 1 positional argument but 2 were given"
            ),
        ):
            optimization.ProjectedGradient(10)

    @pytest.mark.parametrize("on_init", [True, False], ids=["init", "attribute setter"])
    def test_deprecated_tolCG(self, on_init):
        match = ".*tolCG has been deprecated.*cg_rtol.*"
        if on_init:
            with pytest.warns(FutureWarning, match=match):
                opt = optimization.ProjectedGradient(tolCG=1e-3)
        else:
            opt = optimization.ProjectedGradient()
            with pytest.warns(FutureWarning, match=match):
                opt.tolCG = 1e-3

        with pytest.warns(FutureWarning, match=match):
            assert opt.tolCG == 1e-3
        assert opt.cg_atol == 0.0
        assert opt.cg_rtol == 1e-3

        # test setting new changes old
        opt.cg_rtol = 1e-4

        with pytest.warns(FutureWarning, match=match):
            assert opt.tolCG == 1e-4

    @pytest.mark.parametrize("on_init", [True, False], ids=["init", "attribute setter"])
    def test_deprecated_maxIterCG(self, on_init):

        match = ".*maxIterCG has been deprecated.*"
        if on_init:
            with pytest.warns(FutureWarning, match=match):
                opt = optimization.ProjectedGradient(maxIterCG=3)
        else:
            opt = optimization.ProjectedGradient()
            with pytest.warns(FutureWarning, match=match):
                opt.maxIterCG = 3

        with pytest.warns(FutureWarning, match=match):
            assert opt.maxIterCG == 3

        assert opt.cg_maxiter == 3

        # test setting new changes old
        opt.cg_maxiter = 8
        with pytest.warns(FutureWarning, match=match):
            assert opt.maxIterCG == 8


class TestInexactGaussNewton:

    def test_defaults(self):
        opt = optimization.InexactGaussNewton()
        assert opt.cg_rtol == 1e-1
        assert opt.cg_atol == 0.0
        assert opt.cg_maxiter == 5

    def test_init(self):
        opt = optimization.InexactGaussNewton(cg_rtol=1e-3, cg_atol=1e-5, cg_maxiter=10)
        assert opt.cg_rtol == 1e-3
        assert opt.cg_atol == 1e-5
        assert opt.cg_maxiter == 10

    def test_kwargs_only(self):
        with pytest.raises(
            TypeError,
            match=re.escape(
                "InexactGaussNewton.__init__() takes 1 positional argument but 2 were given"
            ),
        ):
            optimization.InexactGaussNewton(10)

    @pytest.mark.parametrize("on_init", [True, False], ids=["init", "attribute setter"])
    def test_deprecated_tolCG(self, on_init):
        match = ".*tolCG has been deprecated.*cg_rtol.*"
        if on_init:
            with pytest.warns(FutureWarning, match=match):
                opt = optimization.InexactGaussNewton(tolCG=1e-3)
        else:
            opt = optimization.InexactGaussNewton()
            with pytest.warns(FutureWarning, match=match):
                opt.tolCG = 1e-3

        with pytest.warns(FutureWarning, match=match):
            assert opt.tolCG == 1e-3
        assert opt.cg_atol == 0.0
        assert opt.cg_rtol == 1e-3

        # test setting new changes old
        opt.cg_rtol = 1e-4

        with pytest.warns(FutureWarning, match=match):
            assert opt.tolCG == 1e-4

    @pytest.mark.parametrize("on_init", [True, False], ids=["init", "attribute setter"])
    def test_deprecated_maxIterCG(self, on_init):

        match = ".*maxIterCG has been deprecated.*"
        if on_init:
            with pytest.warns(FutureWarning, match=match):
                opt = optimization.InexactGaussNewton(maxIterCG=3)
        else:
            opt = optimization.InexactGaussNewton()
            with pytest.warns(FutureWarning, match=match):
                opt.maxIterCG = 3

        with pytest.warns(FutureWarning, match=match):
            assert opt.maxIterCG == 3

        assert opt.cg_maxiter == 3

        # test setting new changes old
        opt.cg_maxiter = 8
        with pytest.warns(FutureWarning, match=match):
            assert opt.maxIterCG == 8


class TestProjectedGNCG:

    @pytest.mark.parametrize("cg_tol_defaults", ["atol", "rtol", "both"])
    def test_defaults(self, cg_tol_defaults):
        # testing setting the new default value of rtol if only atol is passed
        if cg_tol_defaults == "rtol":
            opt = optimization.ProjectedGNCG(cg_atol=1e-5)
            assert opt.cg_atol == 1e-5
            assert opt.cg_rtol == 1e-3
        # testing setting the new default value of atol if only rtol is passed
        elif cg_tol_defaults == "atol":
            opt = optimization.ProjectedGNCG(cg_rtol=1e-4)
            assert opt.cg_atol == 0.0
            assert opt.cg_rtol == 1e-4
        # test the old defaults
        else:
            with pytest.warns(
                FutureWarning, match="The defaults for ProjectedGNCG will change.*"
            ):
                opt = optimization.ProjectedGNCG()
            assert opt.cg_rtol == 0.0
            assert opt.cg_atol == 1e-3
        assert opt.cg_maxiter == 5
        assert np.isneginf(opt.lower)
        assert np.isposinf(opt.upper)

    def test_init(self):
        opt = optimization.ProjectedGNCG(
            cg_rtol=1e-3, cg_atol=1e-5, cg_maxiter=10, lower=0.0, upper=1.0
        )
        assert opt.cg_rtol == 1e-3
        assert opt.cg_atol == 1e-5
        assert opt.cg_maxiter == 10
        assert opt.lower == 0.0
        assert opt.upper == 1.0

    def test_kwargs_only(self):
        with pytest.raises(
            TypeError,
            match=re.escape(
                "ProjectedGNCG.__init__() takes 1 positional argument but 2 were given"
            ),
        ):
            optimization.ProjectedGNCG(10)

    @pytest.mark.parametrize("on_init", [True, False], ids=["init", "attribute setter"])
    def test_deprecated_tolCG(self, on_init):
        if on_init:
            with pytest.warns(
                FutureWarning, match=".*tolCG has been deprecated.*cg_atol.*"
            ):
                opt = optimization.ProjectedGNCG(tolCG=1e-5)
        else:
            opt = optimization.ProjectedGNCG()
            with pytest.warns(
                FutureWarning, match=".*tolCG has been deprecated.*cg_atol.*"
            ):
                opt.tolCG = 1e-5

        with pytest.warns(FutureWarning, match=".*tolCG has been deprecated.*"):
            assert opt.tolCG == 1e-5

        assert opt.cg_atol == 1e-5
        assert opt.cg_rtol == 0.0

        # test setting new changes old
        opt.cg_atol = 1e-4

        with pytest.warns(FutureWarning, match=".*tolCG has been deprecated.*"):
            assert opt.tolCG == 1e-4

    @pytest.mark.parametrize("on_init", [True, False], ids=["init", "attribute setter"])
    @pytest.mark.parametrize(
        ("old_name", "new_name", "val1", "val2"),
        [
            ("maxIterCG", "cg_maxiter", 3, 8),
            ("stepActiveSet", "step_active_set", True, False),
            ("stepOffBoundsFact", "active_set_grad_scale", 1.2, 1.4),
        ],
        ids=["maxIterCG", "stepActiveSet", "stepOffBoundsFact"],
    )
    def test_deprecated_maxIterCG(self, on_init, old_name, new_name, val1, val2):

        match = f".*{old_name} has been deprecated.*"
        if on_init:
            with pytest.warns(FutureWarning, match=match):
                opt = optimization.ProjectedGNCG(**{old_name: val1})
        else:
            opt = optimization.ProjectedGNCG()
            with pytest.warns(FutureWarning, match=match):
                setattr(opt, old_name, val1)
                opt.maxIterCG = 3

        with pytest.warns(FutureWarning, match=match):
            assert getattr(opt, old_name) == val1

        assert getattr(opt, old_name) == val1

        setattr(opt, new_name, val2)

        with pytest.warns(FutureWarning, match=match):
            assert getattr(opt, old_name) == val2
