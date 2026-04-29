import numpy as np
from scipy.constants import mu_0
from discretize.tests import check_derivative
from SimPEG.electromagnetics.utils.testing_utils import (
    getFDEMProblem,
    get_FDEM_hierarchical_problem,
)
import pytest

TOL = 1e-5
FLR = 1e-20  # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0
freq = 3.16
addrandoms = True


# Previously this only tested "electric fields
@pytest.mark.parametrize("receiver_comp", ["r", "i"])
@pytest.mark.parametrize("receiver_dir", ["x", "y", "z"])
@pytest.mark.parametrize(
    "receiver_type",
    ["ElectricField", "MagneticFluxDensity", "CurrentDensity", "MagneticField"],
)
@pytest.mark.parametrize(
    "src_type",
    [
        "MagDipole",
        "CircularLoop",
        "MagDipole_Bfield",
        "RawVec",
        "LineCurrent",
    ],
)
@pytest.mark.parametrize("sim_type", ["e", "b", "e_hier", "b_hier"])
def test_deriv(sim_type, src_type, receiver_type, receiver_dir, receiver_comp):
    rx_type = (receiver_type, receiver_dir, receiver_comp)

    if "hier" in sim_type:
        prb = get_FDEM_hierarchical_problem(sim_type[0], rx_type, [src_type], freq)
    else:
        prb = getFDEMProblem(sim_type, rx_type, [src_type], freq)

    x0 = np.log(np.ones(prb.sigmaMap.nP) * CONDUCTIVITY)  # should work
    # mu = np.log(np.ones(prb.mesh.nC)*MU)

    if addrandoms:
        x0 = x0 + np.random.randn(prb.sigmaMap.nP) * np.log(CONDUCTIVITY) * 1e-1
        # mu = mu + np.random.randn(prb.sigmaMap.nP)*MU*1e-1

    def fun(x):
        return prb.dpred(x), lambda x: prb.Jvec(x0, x)

    check_derivative(fun, x0, num=2, plotIt=False, eps=FLR)
