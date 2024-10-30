import pytest
from pymatsolver import SolverCG

from simpeg.utils.solver_utils import (
    get_default_solver,
    set_default_solver,
    DefaultSolverWarning,
)


@pytest.fixture(autouse=True)
def reset_default_solver():
    # This should get automatically used
    with pytest.warns(DefaultSolverWarning):
        initial_default = get_default_solver()
    yield
    set_default_solver(initial_default)


def test_default_setting():
    set_default_solver(SolverCG)

    with pytest.warns(DefaultSolverWarning, match="Using the default solver: SolverCG"):
        new_default = get_default_solver()

    assert new_default == SolverCG


def test_default_error():
    class Temp:
        pass

    with pytest.warns(DefaultSolverWarning):
        initial_default = get_default_solver()

    with pytest.raises(
        TypeError,
        match="Default solver must be a subclass of pymatsolver.solvers.Base.",
    ):
        set_default_solver(Temp)

    with pytest.warns(DefaultSolverWarning):
        after_default = get_default_solver()

    # make sure we didn't accidentally set the default.
    assert initial_default == after_default
