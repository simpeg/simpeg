from __future__ import division, print_function
import unittest
import discretize

# import properties

from SimPEG import maps
from SimPEG.electromagnetics import time_domain as tdem
from SimPEG.electromagnetics import utils
import numpy as np

import warnings
from pymatsolver import Pardiso as Solver

TOL = 1e-4
FLR = 1e-20

# set a seed so that the same conductivity model is used for all runs
np.random.seed(25)


def setUp_TDEM(
    prbtype="MagneticFluxDensity", rxcomp="bz", waveform="stepoff", src_type=None
):
    cs = 5.0
    ncx = 8
    ncy = 8
    ncz = 8
    npad = 4
    # hx = [(cs, ncx), (cs, npad, 1.3)]
    # hz = [(cs, npad, -1.3), (cs, ncy), (cs, npad, 1.3)]
    mesh = discretize.TensorMesh(
        [
            [(cs, npad, -1.3), (cs, ncx), (cs, npad, 1.3)],
            [(cs, npad, -1.3), (cs, ncy), (cs, npad, 1.3)],
            [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)],
        ],
        "CCC",
    )

    active = mesh.cell_centers_z < 0.0
    activeMap = maps.InjectActiveCells(
        mesh, active, np.log(1e-8), nC=mesh.shape_cells[2]
    )
    mapping = maps.ExpMap(mesh) * maps.SurjectVertical1D(mesh) * activeMap

    rxtimes = np.logspace(-4, -3, 20)

    if waveform.upper() == "RAW":
        out = utils.VTEMFun(prb.times, 0.00595, 0.006, 100)
        wavefun = interp1d(prb.times, out)
        t0 = 0.006
        waveform = tdem.Src.RawWaveform(offTime=t0, waveFct=wavefun)
        time_steps = [(1e-3, 5), (1e-4, 5), (5e-5, 10), (5e-5, 10), (1e-4, 10)]
        rxtimes = t0 + rxtimes

    else:
        waveform = tdem.Src.StepOffWaveform()
        time_steps = [(1e-05, 10), (5e-05, 10), (2.5e-4, 10)]

    rxOffset = 10.0
    rx = getattr(tdem.Rx, "Point{}".format(rxcomp[:-1]))(
        np.r_[rxOffset, 0.0, -1e-2], rxtimes, rxcomp[-1]
    )
    if src_type == "LineCurrent":
        src = tdem.sources.LineCurrent(
            [rx],
            location=np.vstack(
                [np.r_[-cs / 2, -cs / 2, -cs / 2], np.r_[cs / 2, -cs / 2, -cs / 2]]
            ),
        )
    else:
        src = tdem.Src.MagDipole(
            [rx], location=np.array([0.0, 0.0, 0.0]), waveform=waveform
        )

    survey = tdem.Survey([src])

    prb = getattr(tdem, "Simulation3D{}".format(prbtype))(
        mesh, survey=survey, time_steps=time_steps, sigmaMap=mapping
    )
    prb.solver = Solver

    m = np.log(1e-1) * np.ones(prb.sigmaMap.nP) + 1e-2 * np.random.rand(prb.sigmaMap.nP)

    return prb, m, mesh


def CrossCheck(
    prbtype1="MagneticFluxDensity",
    prbtype2="ElectricField",
    rxcomp="bz",
    waveform="stepoff",
    src_type=None,
):

    prb1, m1, mesh1 = setUp_TDEM(prbtype1, rxcomp, waveform, src_type)
    prb2, _, mesh2 = setUp_TDEM(prbtype2, rxcomp, waveform, src_type)

    d1 = prb1.dpred(m1)
    d2 = prb2.dpred(m1)

    check = np.linalg.norm(d1 - d2)
    tol = 0.5 * (np.linalg.norm(d1) + np.linalg.norm(d2)) * TOL
    passed = check < tol

    print(
        "Checking {}, {} for {} data, {} waveform, {}".format(
            prbtype1, prbtype2, rxcomp, waveform, src_type
        )
    )
    print(
        "    {}, {}, {} < {} ? {}".format(
            np.linalg.norm(d1), np.linalg.norm(d2), np.linalg.norm(check), tol, passed
        )
    )

    assert passed


class TDEM_cross_check_EB(unittest.TestCase):
    def test_EB_ey_stepoff(self):
        CrossCheck(
            prbtype1="MagneticFluxDensity",
            prbtype2="ElectricField",
            rxcomp="ElectricFieldy",
            waveform="stepoff",
        )

    def test_EB_dbdtx_stepoff(self):
        CrossCheck(
            prbtype1="MagneticFluxDensity",
            prbtype2="ElectricField",
            rxcomp="MagneticFluxTimeDerivativex",
            waveform="stepoff",
        )

    def test_EB_dbdtz_stepoff(self):
        CrossCheck(
            prbtype1="MagneticFluxDensity",
            prbtype2="ElectricField",
            rxcomp="MagneticFluxTimeDerivativez",
            waveform="stepoff",
        )

    def test_HJ_j_stepoff(self):
        CrossCheck(
            prbtype1="MagneticField",
            prbtype2="CurrentDensity",
            rxcomp="CurrentDensityy",
            waveform="stepoff",
        )

    def test_HJ_j_stepoff(self):
        CrossCheck(
            prbtype1="MagneticField",
            prbtype2="CurrentDensity",
            rxcomp="CurrentDensityy",
            waveform="stepoff",
        )

    def test_HJ_dhdtx_stepoff(self):
        CrossCheck(
            prbtype1="MagneticField",
            prbtype2="CurrentDensity",
            rxcomp="MagneticFieldTimeDerivativex",
            waveform="stepoff",
        )

    def test_HJ_dhdtz_stepoff(self):
        CrossCheck(
            prbtype1="MagneticField",
            prbtype2="CurrentDensity",
            rxcomp="MagneticFieldTimeDerivativex",
            waveform="stepoff",
        )

    def test_HJ_jx_stepoff_linecurrent(self):
        CrossCheck(
            prbtype1="MagneticField",
            prbtype2="CurrentDensity",
            rxcomp="CurrentDensityx",
            waveform="stepoff",
            src_type="LineCurrent",
        )

    def test_HJ_jy_stepoff_linecurrent(self):
        CrossCheck(
            prbtype1="MagneticField",
            prbtype2="CurrentDensity",
            rxcomp="CurrentDensityy",
            waveform="stepoff",
            src_type="LineCurrent",
        )

    def test_EB_ey_vtem(self):
        CrossCheck(
            prbtype1="MagneticFluxDensity",
            prbtype2="ElectricField",
            rxcomp="ElectricFieldy",
            waveform="vtem",
        )

    def test_EB_dbdtx_vtem(self):
        CrossCheck(
            prbtype1="MagneticFluxDensity",
            prbtype2="ElectricField",
            rxcomp="MagneticFluxTimeDerivativex",
            waveform="vtem",
        )

    def test_EB_dbdtz_vtem(self):
        CrossCheck(
            prbtype1="MagneticFluxDensity",
            prbtype2="ElectricField",
            rxcomp="MagneticFluxTimeDerivativez",
            waveform="vtem",
        )

    def test_HJ_j_vtem(self):
        CrossCheck(
            prbtype1="MagneticField",
            prbtype2="CurrentDensity",
            rxcomp="CurrentDensityy",
            waveform="vtem",
        )

    def test_HJ_j_vtem(self):
        CrossCheck(
            prbtype1="MagneticField",
            prbtype2="CurrentDensity",
            rxcomp="CurrentDensityy",
            waveform="vtem",
        )

    def test_HJ_dhdtx_vtem(self):
        CrossCheck(
            prbtype1="MagneticField",
            prbtype2="CurrentDensity",
            rxcomp="MagneticFieldTimeDerivativex",
            waveform="vtem",
        )

    def test_HJ_dhdtz_vtem(self):
        CrossCheck(
            prbtype1="MagneticField",
            prbtype2="CurrentDensity",
            rxcomp="MagneticFieldTimeDerivativex",
            waveform="vtem",
        )

    def test_MagDipoleSimpleFail(self):
        print("\ntesting MagDipole error handling")

        with self.assertRaises(TypeError):
            tdem.Src.MagDipole(
                ["a", 0, {"s": 1}],
                location=np.r_[0.0, 0.0, 0.0],
                orientation=np.r_[1.0, 1.0, 0.0],
            )

    def test_waveform_instantiation(self):

        rx_list = [
            tdem.receivers.PointMagneticFluxDensity(
                locations=[np.r_[0.0, 0.0, 0.0]],
                times=np.r_[1e-3, 2e-3, 5e-3],
                orientation="z",
            )
        ]
        src_loop = tdem.sources.CircularLoop(
            receiver_list=rx_list,
            location=np.r_[0.0, 0.0, 0.0],
            orientation="z",
            radius=300,
            current=1000.0,
        )

        offTime = 1e-3
        src_loop.waveform = tdem.sources.QuarterSineRampOnWaveform(
            ramp_on=np.r_[0.0, 5.0e-4], ramp_off=np.r_[offTime, offTime + 1e-4]
        )

        self.assertIsInstance(src_loop.waveform, tdem.sources.QuarterSineRampOnWaveform)

        with self.assertRaises(TypeError):
            src_loop.waveform = (1, 5)
