import re
import pytest
from pymatsolver import SolverCG

from simpeg.utils import get_default_solver, set_default_solver


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

    initial_default = get_default_solver()

    regex = re.escape("Default solver must be a subclass of pymatsolver.solvers.Base.")
    with pytest.raises(TypeError, match=regex):
        set_default_solver(Temp)

    after_default = get_default_solver()

    # make sure we didn't accidentally set the default.
    assert initial_default is after_default
