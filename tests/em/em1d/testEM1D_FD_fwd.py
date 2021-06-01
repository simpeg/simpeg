import unittest
from SimPEG import *
from discretize import TensorMesh
import matplotlib.pyplot as plt
import SimPEG.electromagnetics.frequency_domain as fdem
from SimPEG.electromagnetics import frequency_domain_1d as em1d
from SimPEG.electromagnetics.analytics.em1d_analytics import *
import numpy as np
from scipy.constants import mu_0


class EM1D_FD_FwdProblemTests(unittest.TestCase):
    def setUp(self):

        nearthick = np.logspace(-1, 1, 5)
        deepthick = np.logspace(1, 2, 10)
        thicknesses = np.r_[nearthick, deepthick]
        topo = np.r_[0.0, 0.0, 100.0]

        offset = 10.0
        src_location = np.array([[0.0, 0.0, 100.0 + 1e-5]])
        rx_location = np.array([[offset, 0.0, 100.0 + 1e-5]])
        frequencies = np.logspace(-1, 5, 61)

        # Receiver list
        receiver_list = []
        receiver_list.append(
            fdem.receivers.PointMagneticFieldSecondary(
                rx_location, orientation="z", component="real"
            )
        )
        receiver_list.append(
            fdem.receivers.PointMagneticFieldSecondary(
                rx_location, orientation="z", component="imag"
            )
        )
        receiver_list.append(
            fdem.receivers.PointMagneticFieldSecondary(
                rx_location, orientation="x", component="real"
            )
        )
        receiver_list.append(
            fdem.receivers.PointMagneticFieldSecondary(
                rx_location, orientation="x", component="imag"
            )
        )

        source_list = []
        for ii, frequency in enumerate(frequencies):
            src = fdem.sources.MagDipole(
                receiver_list, frequency, src_location, orientation="z"
            )
            source_list.append(src)

        # Survey
        # survey = em1d.survey.EM1DSurveyFD(source_list)
        survey = fdem.Survey(source_list)

        sigma = 1e-2
        chi = 0.0
        tau = 1e-3
        eta = 2e-1
        c = 1.0

        self.topo = topo
        self.survey = survey
        self.showIt = False
        self.sigma = sigma
        self.tau = tau
        self.eta = eta
        self.c = c
        self.chi = chi
        self.offset = offset
        self.frequencies = frequencies
        self.thicknesses = thicknesses
        self.nlayers = len(thicknesses) + 1

    def test_EM1DFDfwd_VMD_Halfspace(self):

        sigma_map = maps.ExpMap(nP=1)
        sim = em1d.simulation.EM1DFMSimulation(
            survey=self.survey, sigmaMap=sigma_map, topo=self.topo
        )

        m_1D = np.array([np.log(self.sigma)])
        H = sim.dpred(m_1D)

        H_analytic = []
        for frequency in self.frequencies:
            soln_analytic_z = Hz_vertical_magnetic_dipole(
                frequency, self.offset, self.sigma, "secondary"
            )
            soln_analytic_r = Hr_vertical_magnetic_dipole(
                frequency, self.offset, self.sigma
            )
            soln_analytic = np.r_[
                np.real(soln_analytic_z),
                np.imag(soln_analytic_z),
                np.real(soln_analytic_r),
                np.imag(soln_analytic_r),
            ]
            H_analytic.append(soln_analytic)
        H_analytic = np.hstack(H_analytic)
        err = np.linalg.norm(H - H_analytic) / np.linalg.norm(H_analytic)
        self.assertTrue(err < 1e-5)
        print("EM1DFD-VMD for halfspace works")

    def test_EM1DFDfwd_VMD_RealCond(self):

        sigma_map = maps.ExpMap(nP=self.nlayers)
        sim = em1d.simulation.EM1DFMSimulation(
            survey=self.survey,
            thicknesses=self.thicknesses,
            sigmaMap=sigma_map,
            topo=self.topo,
        )

        m_1D = np.log(np.ones(self.nlayers) * self.sigma)
        H = sim.dpred(m_1D)

        H_analytic = []
        for frequency in self.frequencies:
            soln_analytic_z = Hz_vertical_magnetic_dipole(
                frequency, self.offset, self.sigma, "secondary"
            )
            soln_analytic_r = Hr_vertical_magnetic_dipole(
                frequency, self.offset, self.sigma
            )
            soln_analytic = np.r_[
                np.real(soln_analytic_z),
                np.imag(soln_analytic_z),
                np.real(soln_analytic_r),
                np.imag(soln_analytic_r),
            ]
            H_analytic.append(soln_analytic)

        H_analytic = np.hstack(H_analytic)
        err = np.linalg.norm(H - H_analytic) / np.linalg.norm(H_analytic)
        self.assertTrue(err < 1e-5)
        print("EM1DFD-VMD for real conductivity works")

    def test_EM1DFDfwd_VMD_ComplexCond(self):

        sigma_map = maps.IdentityMap(nP=self.nlayers)
        chi = np.zeros(self.nlayers)
        tau = self.tau * np.ones(self.nlayers)
        c = self.c * np.ones(self.nlayers)
        eta = self.eta * np.ones(self.nlayers)

        sim = em1d.simulation.EM1DFMSimulation(
            survey=self.survey,
            thicknesses=self.thicknesses,
            topo=self.topo,
            sigmaMap=sigma_map,
            eta=eta,
            tau=tau,
            c=c,
            chi=chi,
        )

        m_1D = self.sigma * np.ones(self.nlayers)
        H = sim.dpred(m_1D)

        sigma_colecole = ColeCole(
            self.frequencies, self.sigma, self.eta, self.tau, self.c
        )

        H_analytic = []

        for i_freq, frequency in enumerate(self.frequencies):
            soln_analytic_z = Hz_vertical_magnetic_dipole(
                frequency, self.offset, sigma_colecole[i_freq], "secondary"
            )
            soln_analytic_r = Hr_vertical_magnetic_dipole(
                frequency, self.offset, sigma_colecole[i_freq]
            )
            soln_analytic = np.r_[
                np.real(soln_analytic_z),
                np.imag(soln_analytic_z),
                np.real(soln_analytic_r),
                np.imag(soln_analytic_r),
            ]

            H_analytic.append(soln_analytic)
        H_analytic = np.hstack(H_analytic)
        err = np.linalg.norm(H - H_analytic) / np.linalg.norm(H_analytic)
        self.assertTrue(err < 1e-5)
        print("EM1DFD-VMD for complex conductivity works")

    def test_EM1DFDfwd_HMD_RealCond(self):

        src_location = np.array([0.0, 0.0, 100.0 + 1e-5])
        rx_location = np.array([self.offset, 0.0, 100.0 + 1e-5])
        frequencies = np.logspace(-1, 5, 61)

        # Receiver list
        receiver_list = []
        receiver_list.append(
            fdem.receivers.PointMagneticFieldSecondary(
                rx_location, orientation="z", component="real"
            )
        )
        receiver_list.append(
            fdem.receivers.PointMagneticFieldSecondary(
                rx_location, orientation="z", component="imag"
            )
        )

        source_list = []
        for ii, frequency in enumerate(frequencies):
            src = fdem.sources.MagDipole(
                receiver_list, frequency, src_location, orientation="x"
            )
            source_list.append(src)

        survey = fdem.Survey(source_list)

        sigma_map = maps.ExpMap(nP=self.nlayers)
        sim = em1d.simulation.EM1DFMSimulation(
            survey=survey,
            thicknesses=self.thicknesses,
            sigmaMap=sigma_map,
            topo=self.topo,
        )

        m_1D = np.log(np.ones(self.nlayers) * self.sigma)
        Hz = sim.dpred(m_1D)

        H_analytic = []
        for frequency in self.frequencies:
            soln_analytic_complex = Hz_horizontal_magnetic_dipole(
                frequency, self.offset, self.offset, self.sigma
            )
            soln_analytic = np.r_[
                np.real(soln_analytic_complex), np.imag(soln_analytic_complex)
            ]
            H_analytic.append(soln_analytic)
        H_analytic = np.hstack(H_analytic)
        err = np.linalg.norm(Hz - H_analytic) / np.linalg.norm(H_analytic)
        self.assertTrue(err < 1e-5)
        print("EM1DFD-HMD for real conductivity works")

    def test_EM1DFDfwd_CircularLoop_RealCond(self):

        src_location = np.array([0.0, 0.0, 100.0 + 1e-5])
        rx_location = np.array([0.0, 0.0, 100.0 + 1e-5])
        frequencies = np.logspace(-1, 5, 61)

        # Receiver list
        receiver_list = []
        receiver_list.append(
            fdem.receivers.PointMagneticFieldSecondary(
                rx_location, orientation="z", component="real"
            )
        )
        receiver_list.append(
            fdem.receivers.PointMagneticFieldSecondary(
                rx_location, orientation="z", component="imag"
            )
        )

        source_list = []
        for ii, frequency in enumerate(frequencies):
            src = fdem.sources.CircularLoop(
                receiver_list, frequency, src_location, radius=5.0
            )
            source_list.append(src)

        survey = fdem.Survey(source_list)

        sigma_map = maps.ExpMap(nP=self.nlayers)
        sim = em1d.simulation.EM1DFMSimulation(
            survey=survey,
            thicknesses=self.thicknesses,
            sigmaMap=sigma_map,
            topo=self.topo,
        )

        m_1D = np.log(np.ones(self.nlayers) * self.sigma)
        Hz = sim.dpred(m_1D)

        H_analytic = []
        for frequency in self.frequencies:
            soln_analytic_complex = Hz_horizontal_circular_loop(
                frequency, 1.0, 5.0, self.sigma, "secondary"
            )
            soln_analytic = np.r_[
                np.real(soln_analytic_complex), np.imag(soln_analytic_complex)
            ]
            H_analytic.append(soln_analytic)
        H_analytic = np.hstack(H_analytic)
        err = np.linalg.norm(Hz - H_analytic) / np.linalg.norm(H_analytic)
        self.assertTrue(err < 1e-5)
        print("EM1DFD-CircularLoop for real conductivity works")

    def test_EM1DFDfwd_CircularLoop_ComplexCond(self):

        src_location = np.array([0.0, 0.0, 100.0 + 1e-5])
        rx_location = np.array([0.0, 0.0, 100.0 + 1e-5])
        frequencies = np.logspace(-1, 5, 61)

        # Receiver list
        receiver_list = []
        receiver_list.append(
            fdem.receivers.PointMagneticFieldSecondary(
                rx_location, orientation="z", component="real"
            )
        )
        receiver_list.append(
            fdem.receivers.PointMagneticFieldSecondary(
                rx_location, orientation="z", component="imag"
            )
        )

        source_list = []
        for ii, frequency in enumerate(frequencies):
            src = fdem.sources.CircularLoop(
                receiver_list, frequency, src_location, radius=5.0
            )
            source_list.append(src)

        survey = fdem.Survey(source_list)

        sigma_map = maps.IdentityMap(nP=self.nlayers)
        chi = np.zeros(self.nlayers)
        tau = self.tau * np.ones(self.nlayers)
        c = self.c * np.ones(self.nlayers)
        eta = self.eta * np.ones(self.nlayers)

        sim = em1d.simulation.EM1DFMSimulation(
            survey=survey,
            thicknesses=self.thicknesses,
            topo=self.topo,
            sigmaMap=sigma_map,
            eta=eta,
            tau=tau,
            c=c,
            chi=chi,
        )

        m_1D = self.sigma * np.ones(self.nlayers)
        Hz = sim.dpred(m_1D)

        sigma_colecole = ColeCole(
            self.frequencies, self.sigma, self.eta, self.tau, self.c
        )
        H_analytic = []
        for i_freq, frequency in enumerate(self.frequencies):
            soln_analytic_complex = Hz_horizontal_circular_loop(
                frequency, 1.0, 5.0, sigma_colecole[i_freq], "secondary"
            )
            soln_analytic = np.r_[
                np.real(soln_analytic_complex), np.imag(soln_analytic_complex)
            ]
            H_analytic.append(soln_analytic)
        H_analytic = np.hstack(H_analytic)
        err = np.linalg.norm(Hz - H_analytic) / np.linalg.norm(H_analytic)

        self.assertTrue(err < 1e-5)
        print("EM1DFD-CircularLoop for complex conductivity works")


if __name__ == "__main__":
    unittest.main()
