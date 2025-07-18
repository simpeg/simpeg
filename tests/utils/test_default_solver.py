import warnings
import pytest
from pymatsolver import SolverCG

from simpeg.utils import get_default_solver, set_default_solver
from simpeg.utils.solver_utils import DefaultSolverWarning


@pytest.fixture(autouse=True)
def reset_default_solver():
    # This should get automatically used
    initial_default = get_default_solver()
    yield
    set_default_solver(initial_default)


def test_default_setting():
    set_default_solver(SolverCG)
    new_default = get_default_solver()
    assert new_default == SolverCG


def test_default_error():
    class Temp:
        pass

    with pytest.warns(DefaultSolverWarning):
        initial_default = get_default_solver(warn=True)

    with pytest.raises(
        TypeError,
        match="Default solver must be a subclass of pymatsolver.solvers.Base.",
    ):
        set_default_solver(Temp)

    with pytest.warns(DefaultSolverWarning):
        after_default = get_default_solver(warn=True)

    # make sure we didn't accidentally set the default.
    assert initial_default == after_default


def test_warning():
    """Test if warning is raised when warn=True."""
    with pytest.warns(DefaultSolverWarning, match="Using the default solver"):
        get_default_solver(warn=True)


def test_no_warning():
    """Test if no warning is issued with default parameters."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # raise error if warning was raised
        get_default_solver()
