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


class EM1D_TD_LineCurrent1D_FwdProblemTests(unittest.TestCase):
    def setUp(self):
        # WalkTEM waveform
        # Low moment
        lm_waveform_times = np.r_[-1.041e-03, -9.850e-04, 0.000e00, 4.000e-06]
        lm_waveform_current = np.r_[0.0, 1.0, 1.0, 0.0]

        # High moment
        hm_waveform_times = np.r_[-8.333e-03, -8.033e-03, 0.000e00, 5.600e-06]
        hm_waveform_current = np.r_[0.0, 1.0, 1.0, 0.0]

        # Low moment
        lm_off_time = np.array(
            [
                1.149e-05,
                1.350e-05,
                1.549e-05,
                1.750e-05,
                2.000e-05,
                2.299e-05,
                2.649e-05,
                3.099e-05,
                3.700e-05,
                4.450e-05,
                5.350e-05,
                6.499e-05,
                7.949e-05,
                9.799e-05,
                1.215e-04,
                1.505e-04,
                1.875e-04,
                2.340e-04,
                2.920e-04,
                3.655e-04,
                4.580e-04,
                5.745e-04,
                7.210e-04,
            ]
        )

        # High moment
        hm_off_time = np.array(
            [
                9.810e-05,
                1.216e-04,
                1.506e-04,
                1.876e-04,
                2.341e-04,
                2.921e-04,
                3.656e-04,
                4.581e-04,
                5.746e-04,
                7.211e-04,
                9.056e-04,
                1.138e-03,
                1.431e-03,
                1.799e-03,
                2.262e-03,
                2.846e-03,
                3.580e-03,
                4.505e-03,
                5.670e-03,
                7.135e-03,
            ]
        )

        # WalkTEM geometry
        x_path = np.array([-20, -20, 20, 20, -20])
        y_path = np.array([-20, 20, 20, -20, -20])

        wire_paths = np.c_[x_path, y_path, np.zeros(5)]
        source_list = []
        receiver_list_lm = []
        receiver_list_hm = []
        receiver_location = np.array([[0, 0, 0]])
        receiver_orientation = "z"

        receiver_list_lm.append(
            tdem.receivers.PointMagneticFluxTimeDerivative(
                receiver_location, times=lm_off_time, orientation=receiver_orientation
            )
        )

        receiver_list_hm.append(
            tdem.receivers.PointMagneticFluxTimeDerivative(
                receiver_location, times=hm_off_time, orientation=receiver_orientation
            )
        )

        lm_wave = tdem.sources.PiecewiseLinearWaveform(
            lm_waveform_times, lm_waveform_current
        )
        hm_wave = tdem.sources.PiecewiseLinearWaveform(
            hm_waveform_times, hm_waveform_current
        )

        source_lm = tdem.sources.LineCurrent1D(
            receiver_list_lm, wire_paths, waveform=lm_wave
        )
        source_hm = tdem.sources.LineCurrent1D(
            receiver_list_hm, wire_paths, waveform=hm_wave
        )
        source_list.append(source_lm)
        source_list.append(source_hm)

        # Define a 1D TDEM survey
        survey = tdem.survey.Survey(source_list)

        # Physical properties
        model = np.array([1.0 / 10, 1.0 / 1])

        # Layer thicknesses
        thicknesses = np.array([30.0])
        n_layer = len(thicknesses) + 1

        # Define a mapping from model parameters to conductivities
        model_mapping = maps.IdentityMap(nP=n_layer)

        simulation = tdem.Simulation1DLayered(
            survey=survey,
            thicknesses=thicknesses,
            sigmaMap=model_mapping,
        )

        self.bzdt = simulation.dpred(model)

    def test_em1dtd_mag_dipole_bzdt(self):

        empymod_solution = np.array(
            [
                9.34490123e-04,
                6.94483184e-04,
                5.30780675e-04,
                4.13985491e-04,
                3.12288448e-04,
                2.30231119e-04,
                1.66949789e-04,
                1.15353241e-04,
                7.46550309e-05,
                4.66347045e-05,
                2.88132824e-05,
                1.72912356e-05,
                1.03331515e-05,
                6.25627742e-06,
                3.92023957e-06,
                2.58557463e-06,
                1.75333907e-06,
                1.21267186e-06,
                8.45204615e-07,
                5.84105319e-07,
                3.98577218e-07,
                2.67232863e-07,
                1.75618980e-07,
                6.47837095e-06,
                4.06848723e-06,
                2.70385819e-06,
                1.85598003e-06,
                1.30331551e-06,
                9.26701266e-07,
                6.56612546e-07,
                4.62303042e-07,
                3.22156332e-07,
                2.21869133e-07,
                1.50726507e-07,
                1.00915787e-07,
                6.65266826e-08,
                4.32260520e-08,
                2.76450055e-08,
                1.73721794e-08,
                1.07354465e-08,
                6.50886256e-09,
                3.86887941e-09,
                2.25589672e-09,
            ]
        )

        np.testing.assert_allclose(self.bzdt, empymod_solution, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
