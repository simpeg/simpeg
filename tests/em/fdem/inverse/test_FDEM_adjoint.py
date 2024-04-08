import numpy as np
from scipy.constants import mu_0
from SimPEG.electromagnetics.utils.testing_utils import (
    getFDEMProblem,
    get_FDEM_hierarchical_problem,
)
import pytest

testE = True
testB = True

verbose = False

TOL = 1e-6
FLR = 1e-20  # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0
freq = 1e-1
addrandoms = True


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
@pytest.mark.parametrize("sim_type", ["e", "b", "e_hier", "b_hier", "h", "j"])
def test_adjoint(sim_type, src_type, receiver_type, receiver_dir, receiver_comp):
    rx_type = (receiver_type, receiver_dir, receiver_comp)
    if "hier" in sim_type:
        prb = get_FDEM_hierarchical_problem(sim_type[0], rx_type, [src_type], freq)
    else:
        prb = getFDEMProblem(sim_type, rx_type, [src_type], freq)

    m = np.log(
        np.ones(prb.sigmaMap.nP) * CONDUCTIVITY
    )  # works for sigma_only and sigma, tau, kappa
    mu = np.ones(prb.mesh.nC) * MU

    if addrandoms is True:
        m = m + np.random.randn(prb.sigmaMap.nP) * np.log(CONDUCTIVITY) * 1e-1
        mu = mu + np.random.randn(prb.mesh.nC) * MU * 1e-1

    survey = prb.survey
    # prb.PropMap.PropModel.mu = mu
    # prb.PropMap.PropModel.mui = 1./mu
    u = prb.fields(m)

    v = np.random.rand(survey.nD)
    w = np.random.rand(prb.sigmaMap.nP)  # works for sigma_only and sigma, tau, kappa

    vJw = v.dot(prb.Jvec(m, w, u))
    wJtv = w.dot(prb.Jtvec(m, v, u))

    print(vJw, wJtv, vJw - wJtv, np.abs(vJw - wJtv) < FLR + TOL * np.abs(wJtv))
    np.testing.assert_allclose(vJw, wJtv, rtol=TOL, atol=FLR)
