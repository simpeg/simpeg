import unittest
from SimPEG import maps
import matplotlib.pyplot as plt
import SimPEG.electromagnetics.time_domain as tdem
import numpy as np
from SimPEG.electromagnetics.utils import convolve_with_waveform
from geoana.em.tdem import (
    vertical_magnetic_flux_horizontal_loop as b_loop,
    vertical_magnetic_flux_time_deriv_horizontal_loop as dbdt_loop,
)
import numpy as np


class EM1D_TD_FwdProblemTests(unittest.TestCase):
    def setUp(self):

        source_location = np.array([0.0, 0.0, 0.0])
        source_orientation = "z"  # "x", "y" or "z"
        source_current = 1.0
        source_radius = 10.0
        moment_amplitude = 1.0

        receiver_locations = np.array([[0.0, 0.0, 0.0]])
        receiver_orientation = "z"  # "x", "y" or "z"

        times_hm = np.logspace(-6, -3, 31)
        times_lm = np.logspace(-5, -2, 31)

        # Waveforms
        waveform_hm = tdem.sources.TriangularWaveform(
            start_time=-0.01, peak_time=-0.005, off_time=0.0
        )
        waveform_lm = tdem.sources.TriangularWaveform(
            start_time=-0.01, peak_time=-0.0001, off_time=0.0
        )

        # Receiver list

        # Define receivers at each location.
        dbzdt_receiver_hm = tdem.receivers.PointMagneticFluxTimeDerivative(
            receiver_locations, times_hm, receiver_orientation
        )
        dbzdt_receiver_lm = tdem.receivers.PointMagneticFluxTimeDerivative(
            receiver_locations, times_lm, receiver_orientation
        )
        # Make a list containing all receivers even if just one

        # Must define the transmitter properties and associated receivers
        source_list = [
            tdem.sources.CircularLoop(
                [dbzdt_receiver_hm],
                location=source_location,
                waveform=waveform_hm,
                radius=source_radius,
            ),
            tdem.sources.CircularLoop(
                [dbzdt_receiver_lm],
                location=source_location,
                waveform=waveform_lm,
                radius=source_radius,
            ),
        ]

        survey = tdem.Survey(source_list)

        thicknesses = np.ones(3)
        sigma = 1e-2
        n_layer = thicknesses.size + 1

        sigma_model = sigma * np.ones(n_layer)

        model_mapping = maps.IdentityMap(nP=n_layer)
        simulation = tdem.Simulation1DLayered(
            survey=survey,
            thicknesses=thicknesses,
            sigmaMap=model_mapping,
        )

        self.survey = survey
        self.simulation = simulation
        self.showIt = False
        self.sigma_model = sigma_model
        self.sigma_halfspace = sigma
        self.source_radius = source_radius
        self.waveform_hm = waveform_hm
        self.waveform_lm = waveform_lm
        self.times_lm = times_lm
        self.times_hm = times_hm

    def test_em1dtd_circular_loop_single_pulse(self):

        src = self.survey.source_list[0]
        rx = src.receiver_list[0]
        dbzdt = self.simulation.dpred(self.sigma_model)
        dbzdt_hm = dbzdt[: rx.times.size]
        dbzdt_lm = dbzdt[rx.times.size :]

        dbzdt_lm_analytic = convolve_with_waveform(
            dbdt_loop,
            self.waveform_lm,
            self.times_lm,
            fkwargs={"sigma": self.sigma_halfspace, "radius": self.source_radius},
        )

        dbzdt_hm_analytic = convolve_with_waveform(
            dbdt_loop,
            self.waveform_hm,
            self.times_hm,
            fkwargs={"sigma": self.sigma_halfspace, "radius": self.source_radius},
        )

        err = np.linalg.norm(dbzdt_hm - dbzdt_hm_analytic) / np.linalg.norm(
            dbzdt_hm_analytic
        )

        print("dBzdt error (hm) = ", err)

        self.assertTrue(err < 5e-2)
        err = np.linalg.norm(dbzdt_lm - dbzdt_lm_analytic) / np.linalg.norm(
            dbzdt_lm_analytic
        )

        print("dBzdt error (lm) = ", err)
        self.assertTrue(err < 5e-2)

        print("EM1DTD-CirculurLoop-general for real conductivity works")


if __name__ == "__main__":
    unittest.main()
