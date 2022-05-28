import unittest
from SimPEG import maps
import SimPEG.electromagnetics.time_domain as tdem
from SimPEG.electromagnetics.utils import convolve_with_waveform
from geoana.em.tdem import (
    vertical_magnetic_flux_horizontal_loop as b_loop,
    vertical_magnetic_flux_time_deriv_horizontal_loop as dbdt_loop,
    magnetic_flux_vertical_magnetic_dipole as b_dipole,
    magnetic_flux_time_deriv_magnetic_dipole as dbdt_dipole,
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
            start_time=-0.01, peak_time=-0.005, off_time=0.0
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

        source_location = np.array([0.0, 0.0, 100.0])
        receiver_locations = np.array([[100.0, 0.0, 100.0]])
        times = np.logspace(-5, -2, 31)
        # Waveform
        waveform = tdem.sources.TriangularWaveform(
            start_time=-0.01, peak_time=-0.005, off_time=0.0
        )

        # Receiver list
        receivers_list = [
            tdem.receivers.PointMagneticFluxDensity(receiver_locations, times, "z"),
            tdem.receivers.PointMagneticFluxTimeDerivative(
                receiver_locations, times, "z"
            ),
            tdem.receivers.PointMagneticFluxDensity(receiver_locations, times, "x"),
            tdem.receivers.PointMagneticFluxTimeDerivative(
                receiver_locations, times, "x"
            ),
        ]

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
        self.sigma = sigma
        self.times = times
        self.thicknesses = thicknesses
        self.nlayers = len(thicknesses) + 1
        self.waveform = waveform
        self.rx_locations = receiver_locations

        sigma_map = maps.ExpMap(nP=self.nlayers)
        sim = tdem.Simulation1DLayered(
            survey=self.survey,
            thicknesses=self.thicknesses,
            sigmaMap=sigma_map,
            topo=self.topo,
        )

        m_1D = np.log(np.ones(self.nlayers) * self.sigma)
        d = sim.dpred(m_1D)
        self.bz, self.bzdt, self.bx, self.bxdt = d.reshape(4, -1)

    def test_em1dtd_mag_dipole_bz(self):
        def func(t, *fargs, **fkwargs):
            return b_dipole(t, *fargs, **fkwargs)[:, 0, 2]

        analytic = convolve_with_waveform(
            func,
            self.waveform,
            self.times,
            fargs=(self.rx_locations,),
            fkwargs={"sigma": self.sigma},
        )

        np.testing.assert_allclose(self.bz, analytic, rtol=1e-3)

    def test_em1dtd_mag_dipole_bzdt(self):
        def func(t, *fargs, **fkwargs):
            return dbdt_dipole(t, *fargs, **fkwargs)[:, 0, 2]

        analytic = convolve_with_waveform(
            func,
            self.waveform,
            self.times,
            fargs=(self.rx_locations,),
            fkwargs={"sigma": self.sigma},
        )

        np.testing.assert_allclose(self.bzdt, analytic, rtol=1e-2)

    def test_em1dtd_mag_dipole_bx(self):
        def func(t, *fargs, **fkwargs):
            return b_dipole(t, *fargs, **fkwargs)[:, 0, 0]

        analytic = convolve_with_waveform(
            func,
            self.waveform,
            self.times,
            fargs=(self.rx_locations,),
            fkwargs={"sigma": self.sigma},
        )

        np.testing.assert_allclose(self.bx, analytic, rtol=1e-4)

    def test_em1dtd_mag_dipole_bxdt(self):
        def func(t, *fargs, **fkwargs):
            return dbdt_dipole(t, *fargs, **fkwargs)[:, 0, 0]

        analytic = convolve_with_waveform(
            func,
            self.waveform,
            self.times,
            fargs=(self.rx_locations,),
            fkwargs={"sigma": self.sigma},
        )

        np.testing.assert_allclose(self.bxdt, analytic, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
