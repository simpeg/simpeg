from __future__ import division, print_function

import unittest

import discretize
import matplotlib.pyplot as plt
import numpy as np
from pymatsolver import Pardiso as Solver
from scipy.constants import mu_0
from scipy.interpolate import interp1d
from SimPEG import maps
from SimPEG.electromagnetics import analytics
from SimPEG.electromagnetics import time_domain as tdem
from SimPEG.electromagnetics import utils


def halfSpaceProblemAnaDiff(
    meshType,
    srctype="MagDipole",
    sig_half=1e-2,
    rxOffset=50.0,
    bounds=None,
    plotIt=False,
    rxType="MagneticFluxDensityz",
):

    if bounds is None:
        bounds = [1e-5, 1e-3]
    if meshType == "CYL":
        cs, ncx, ncz, npad = 15.0, 30, 10, 15
        hx = [(cs, ncx), (cs, npad, 1.3)]
        hz = [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
        mesh = discretize.CylMesh([hx, 1, hz], "00C")

    elif meshType == "TENSOR":
        cs, nc, npad = 20.0, 20, 7
        hx = [(cs, npad, 1.5), (cs, nc), (cs, npad, 1.5)]
        hy = [(cs, npad, 1.5), (cs, nc), (cs, npad, 1.5)]
        hz = [(cs, npad, 1.5), (cs, nc), (cs, npad, 1.5)]
        mesh = discretize.TensorMesh([hx, hy, hz], "CCC")

    active = mesh.vectorCCz < 0.0
    actMap = maps.InjectActiveCells(mesh, active, np.log(1e-8), nC=mesh.nCz)
    mapping = maps.ExpMap(mesh) * maps.SurjectVertical1D(mesh) * actMap

    time_steps = [(1e-3, 5), (1e-4, 5), (5e-5, 10), (5e-5, 10), (1e-4, 10)]
    time_mesh = discretize.TensorMesh(
        [
            time_steps,
        ]
    )
    times = time_mesh.nodes_x
    out = utils.VTEMFun(times, 0.00595, 0.006, 100)
    wavefun = interp1d(times, out)
    t0 = 0.006
    waveform = tdem.Src.RawWaveform(offTime=t0, waveFct=wavefun)

    rx = getattr(tdem.Rx, "Point{}".format(rxType[:-1]))(
        np.array([[rxOffset, 0.0, 0.0]]), np.logspace(-4, -3, 31) + t0, rxType[-1]
    )

    if srctype == "MagDipole":
        src = tdem.Src.MagDipole(
            [rx], waveform=waveform, location=np.array([0, 0.0, 0.0])
        )
    elif srctype == "CircularLoop":
        src = tdem.Src.CircularLoop(
            [rx], waveform=waveform, location=np.array([0.0, 0.0, 0.0]), radius=13.0
        )
    elif srctype == "LineCurrent":
        side = 100.0 * np.sqrt(np.pi)
        loop_path = np.c_[
            [-side / 2, -side / 2, 0],
            [side / 2, -side / 2, 0],
            [side / 2, side / 2, 0],
            [-side / 2, side / 2, 0],
            [-side / 2, -side / 2, 0],
        ].T
        src = tdem.sources.LineCurrent([rx], waveform=waveform, location=loop_path)

    survey = tdem.Survey([src])
    prb = tdem.Simulation3DMagneticFluxDensity(
        mesh, survey=survey, sigmaMap=mapping, time_steps=time_steps
    )
    prb.solver = Solver

    sigma = np.ones(mesh.nCz) * 1e-8
    sigma[active] = sig_half
    sigma = np.log(sigma[active])

    if srctype == "MagDipole":
        bz_ana = mu_0 * analytics.hzAnalyticDipoleT(
            rx.locations[0][0] + 1e-3, rx.times - t0, sig_half
        )
    elif srctype == "CircularLoop":
        bz_ana = mu_0 * analytics.hzAnalyticCentLoopT(13, rx.times - t0, sig_half)
    elif srctype == "LineCurrent":
        bz_ana = mu_0 * analytics.hzAnalyticCentLoopT(100.0, rx.times - t0, sig_half)

    bz_calc = prb.dpred(sigma)
    ind = np.logical_and(rx.times - t0 > bounds[0], rx.times - t0 < bounds[1])
    log10diff = np.linalg.norm(
        np.log10(np.abs(bz_calc[ind])) - np.log10(np.abs(bz_ana[ind]))
    ) / np.linalg.norm(np.log10(np.abs(bz_ana[ind])))

    print(
        " |bz_ana| = {ana} |bz_num| = {num} |bz_ana-bz_num| = {diff}".format(
            ana=np.linalg.norm(bz_ana),
            num=np.linalg.norm(bz_calc),
            diff=np.linalg.norm(bz_ana - bz_calc),
        )
    )
    print("Difference: {}".format(log10diff))

    if plotIt is True:
        plt.loglog(
            rx.times[bz_calc > 0] - t0,
            bz_calc[bz_calc > 0],
            "r",
            rx.times[bz_calc < 0] - t0,
            -bz_calc[bz_calc < 0],
            "r--",
        )
        plt.loglog(rx.times - t0, abs(bz_ana), "b*")
        plt.title("sig_half = {:e}".format(sig_half))
        plt.show()

    return log10diff


class TDEM_SimpleSrcTests(unittest.TestCase):
    def test_source(self):
        waveform = tdem.Src.StepOffWaveform()
        assert waveform.eval(0.0) == 1.0


class TDEM_bTests(unittest.TestCase):
    def test_analytic_p0_CYL_1m_MagDipole(self):
        self.assertTrue(
            halfSpaceProblemAnaDiff("CYL", rxOffset=1.0, sig_half=1e0) < 0.01
        )

    def test_analytic_m1_CYL_1m_MagDipole(self):
        self.assertTrue(
            halfSpaceProblemAnaDiff("CYL", rxOffset=1.0, sig_half=1e-1) < 0.01
        )

    def test_analytic_m2_CYL_1m_MagDipole(self):
        self.assertTrue(
            halfSpaceProblemAnaDiff("CYL", rxOffset=1.0, sig_half=1e-2) < 0.01
        )

    def test_analytic_m3_CYL_1m_MagDipole(self):
        self.assertTrue(
            halfSpaceProblemAnaDiff("CYL", rxOffset=1.0, sig_half=1e-3) < 0.01
        )

    def test_analytic_p0_CYL_0m_CircularLoop(self):
        self.assertTrue(
            halfSpaceProblemAnaDiff(
                "CYL", srctype="CircularLoop", rxOffset=0.0, sig_half=1e0
            )
            < 0.01
        )

    def test_analytic_m1_CYL_0m_CircularLoop(self):
        self.assertTrue(
            halfSpaceProblemAnaDiff(
                "CYL", srctype="CircularLoop", rxOffset=0.0, sig_half=1e-1
            )
            < 0.01
        )

    def test_analytic_m2_CYL_0m_CircularLoop(self):
        self.assertTrue(
            halfSpaceProblemAnaDiff(
                "CYL", srctype="CircularLoop", rxOffset=0.0, sig_half=1e-2
            )
            < 0.01
        )

    def test_analytic_m3_CYL_0m_CircularLoop(self):
        self.assertTrue(
            halfSpaceProblemAnaDiff(
                "CYL",
                srctype="CircularLoop",
                rxOffset=0.0,
                sig_half=1e-3,
            )
            < 0.01
        )

    def test_analytic_m1_TENSOR_0m_LineCurrent(self):
        self.assertTrue(
            halfSpaceProblemAnaDiff(
                "TENSOR",
                srctype="LineCurrent",
                rxOffset=0.0,
                sig_half=1e-1,
            )
            < 0.01
        )

    def test_analytic_m2_TENSOR_0m_LineCurrent(self):
        self.assertTrue(
            halfSpaceProblemAnaDiff(
                "TENSOR",
                srctype="LineCurrent",
                rxOffset=0.0,
                sig_half=1e-2,
            )
            < 0.01
        )

    def test_analytic_m3_TENSOR_0m_LineCurrent(self):
        self.assertTrue(
            halfSpaceProblemAnaDiff(
                "TENSOR",
                srctype="LineCurrent",
                rxOffset=0.0,
                sig_half=1e-3,
            )
            < 0.01
        )
