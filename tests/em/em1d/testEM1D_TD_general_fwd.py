import unittest
from SimPEG import maps
import matplotlib.pyplot as plt
import SimPEG.electromagnetics.time_domain as tdem
from SimPEG.electromagnetics.utils import convolve_with_waveform
from geoana.em.tdem import (
    vertical_magnetic_flux_horizontal_loop as b_loop,
    vertical_magnetic_flux_time_deriv_horizontal_loop as dbdt_loop,
)
import numpy as np


class EM1D_TD_CircularLoop_FwdProblemTests(unittest.TestCase):
    def setUp(self):
        nearthick = np.logspace(-1, 1, 5)
        deepthick = np.logspace(1, 2, 10)
        thicknesses = np.r_[nearthick, deepthick]
        topo = np.r_[0.0, 0.0, 100.0]

        source_location = np.array([0.0, 0.0, 100.0 + 1e-5])
        receiver_locations = np.array([[0.0, 0.0, 100.0 + 1e-5]])
        receiver_orientation = "z"  # "x", "y" or "z"
        times = np.logspace(-5, -2, 31)
        radius = 20.0

        # Waveform
        waveform = tdem.sources.TriangularWaveform(
            startTime=-0.01, peakTime=-0.005, offTime=0.0
        )

        # Receiver list

        # Define receivers at each location.
        b_receiver = tdem.receivers.PointMagneticFluxDensity(
            receiver_locations, times, receiver_orientation
        )
        dbzdt_receiver = tdem.receivers.PointMagneticFluxTimeDerivative(
            receiver_locations, times, receiver_orientation
        )
        receivers_list = [
            b_receiver,
            dbzdt_receiver,
        ]  # Make a list containing all receivers even if just one

        # Must define the transmitter properties and associated receivers
        source_list = [
            tdem.sources.CircularLoop(
                receivers_list,
                location=source_location,
                waveform=waveform,
                radius=radius,
            )
        ]

        survey = tdem.Survey(source_list)

        sigma = 1e-2

        self.topo = topo
        self.survey = survey
        self.showIt = False
        self.sigma = sigma
        self.times = times
        self.thicknesses = thicknesses
        self.nlayers = len(thicknesses) + 1
        self.a = radius
        self.waveform = waveform

    def test_em1dtd_circular_loop_single_pulse(self):
        sigma_map = maps.ExpMap(nP=self.nlayers)
        sim = tdem.Simulation1DLayered(
            survey=self.survey,
            thicknesses=self.thicknesses,
            sigmaMap=sigma_map,
            topo=self.topo,
        )

        m_1D = np.log(np.ones(self.nlayers) * self.sigma)
        d = sim.dpred(m_1D)
        bz = d[0 : len(self.times)]
        dbdt = d[len(self.times) :]

        bz_analytic = convolve_with_waveform(
            b_loop,
            self.waveform,
            self.times,
            fkwargs={"sigma": self.sigma, "radius": self.a},
        )

        np.testing.assert_allclose(bz, bz_analytic, atol=0.0, rtol=1e-5)

        dbdt_analytic = convolve_with_waveform(
            dbdt_loop,
            self.waveform,
            self.times,
            fkwargs={"sigma": self.sigma, "radius": self.a},
        )

        np.testing.assert_allclose(dbdt, dbdt_analytic, atol=0.0, rtol=1e-2)


class EM1D_TD_MagDipole_FwdProblemTests(unittest.TestCase):
    def setUp(self):

        nearthick = np.logspace(-1, 1, 5)
        deepthick = np.logspace(1, 2, 10)
        thicknesses = np.r_[nearthick, deepthick]
        topo = np.r_[0.0, 0.0, 100.0]

        source_location = np.array([0.0, 0.0, 100.0 + 1e-5])
        receiver_locations = np.array([[10.0, 0.0, 100.0 + 1e-5]])
        receiver_orientation = "z"  # "x", "y" or "z"
        times = np.logspace(-5, -2, 31)
        radius = 20.0
        # Waveform
        waveform = tdem.sources.TriangularWaveform(
            startTime=-0.01, peakTime=-0.005, offTime=0.0
        )

        # Receiver list

        # Define receivers at each location.
        b_receiver = tdem.receivers.PointMagneticFluxDensity(
            receiver_locations, times, receiver_orientation
        )
        dbzdt_receiver = tdem.receivers.PointMagneticFluxTimeDerivative(
            receiver_locations, times, receiver_orientation
        )
        receivers_list = [
            b_receiver,
            dbzdt_receiver,
        ]  # Make a list containing all receivers even if just one

        # Must define the transmitter properties and associated receivers
        source_list = [
            tdem.sources.MagDipole(
                receivers_list, location=source_location, waveform=waveform
            )
        ]

        survey = tdem.Survey(source_list)

        sigma = 1e-2

        self.topo = topo
        self.survey = survey
        self.showIt = False
        self.sigma = sigma
        self.times = times
        self.thicknesses = thicknesses
        self.nlayers = len(thicknesses) + 1
        self.a = radius

    def test_em1dtd_mag_dipole_single_pulse(self):
        sigma_map = maps.ExpMap(nP=self.nlayers)
        sim = tdem.Simulation1DLayered(
            survey=self.survey,
            thicknesses=self.thicknesses,
            sigmaMap=sigma_map,
            topo=self.topo,
        )

        m_1D = np.log(np.ones(self.nlayers) * self.sigma)
        d = sim.dpred(m_1D)
        bz = d[0 : len(self.times)]
        dbdt = d[len(self.times) :]

        def step_func_Bzt(times):
            return Bz_vertical_magnetic_dipole(10.0, times, self.sigma)

        bz_analytic = piecewise_ramp(
            step_func_Bzt,
            self.times,
            sim.survey.source_list[0].waveform.waveform_times,
            sim.survey.source_list[0].waveform.waveform_current,
        )

        if self.showIt:
            plt.subplot(121)
            plt.loglog(self.times, bz, "b*")
            plt.loglog(self.times, bz_analytic, "b")
            plt.subplot(122)
            plt.loglog(self.times, abs((bz - bz_analytic) / bz_analytic), "r:")
            plt.show()

        err = np.linalg.norm(bz - bz_analytic) / np.linalg.norm(bz_analytic)
        print("Bz error = ", err)
        self.assertTrue(err < 6e-2)

        print("EM1DTD-CirculurLoop-general for real conductivity works")


if __name__ == "__main__":
    unittest.main()
