import unittest
import SimPEG.electromagnetics.frequency_domain as fdem
from SimPEG import maps
import numpy as np
from scipy.constants import mu_0
from geoana.em.fdem import MagneticDipoleHalfSpace
from SimPEG.electromagnetics.analytics.FDEM import hz_horizontal_circular_loop


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

        sigma = 1.0
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
        sim = fdem.Simulation1DLayered(
            survey=self.survey, sigmaMap=sigma_map, topo=self.topo
        )

        m_1D = np.array([np.log(self.sigma)])
        H = sim.dpred(m_1D)

        dip = MagneticDipoleHalfSpace(
            location=np.r_[0.0, 0.0, 0.0],
            orientation="z",
            frequency=self.frequencies,
            sigma=self.sigma,
            quasistatic=True,
        )
        H_analytic = np.squeeze(dip.magnetic_field(np.array([[self.offset, 0.0]])))
        Hx = H_analytic[:, 0]
        Hz = H_analytic[:, 2]
        H_analytic = np.c_[Hz.real, Hz.imag, Hx.real, Hx.imag].reshape(-1)

        err = np.linalg.norm(H - H_analytic) / np.linalg.norm(H_analytic)
        self.assertLess(err, 1e-5)
        print("EM1DFD-VMD for halfspace works")

    def test_EM1DFDfwd_VMD_RealCond(self):

        sigma_map = maps.ExpMap(nP=self.nlayers)
        sim = fdem.Simulation1DLayered(
            survey=self.survey,
            thicknesses=self.thicknesses,
            sigmaMap=sigma_map,
            topo=self.topo,
        )

        m_1D = np.log(np.ones(self.nlayers) * self.sigma)
        H = sim.dpred(m_1D)

        dip = MagneticDipoleHalfSpace(
            location=np.r_[0.0, 0.0, 0.0],
            orientation="z",
            frequency=self.frequencies,
            sigma=self.sigma,
            quasistatic=True,
        )
        H_analytic = np.squeeze(dip.magnetic_field(np.array([[self.offset, 0.0]])))
        Hx = H_analytic[:, 0]
        Hz = H_analytic[:, 2]
        H_analytic = np.c_[Hz.real, Hz.imag, Hx.real, Hx.imag].reshape(-1)

        err = np.linalg.norm(H - H_analytic) / np.linalg.norm(H_analytic)
        self.assertLess(err, 1e-5)
        print("EM1DFD-VMD for real conductivity works")

    # TODO add supplied complex conductivity support to geoana
    """
    def test_EM1DFDfwd_VMD_ComplexCond(self):

        sigma_map = maps.IdentityMap(nP=self.nlayers)
        mu = mu_0*np.ones(self.nlayers)
        tau = self.tau * np.ones(self.nlayers)
        c = self.c * np.ones(self.nlayers)
        eta = self.eta * np.ones(self.nlayers)

        sim = fdem.Simulation1DLayered(
            survey=self.survey,
            thicknesses=self.thicknesses,
            topo=self.topo,
            sigmaMap=sigma_map,
            eta=eta,
            tau=tau,
            c=c,
            mu=mu,
        )

        m_1D = self.sigma * np.ones(self.nlayers)
        H = sim.dpred(m_1D)

        sigmas = sim.compute_complex_sigma(self.frequencies)

        H_analytic = []
        for sigma, frequency in zip(sigmas, self.frequencies):
            dip = MagneticDipoleHalfSpace(
                location=np.r_[0.0, 0.0, 0.0],
                orientation='z',
                frequency=frequency,
                sigma=sigma,
                quasistatic=True
            )
            H_analytic = np.squeeze(dip.magnetic_field(np.array([[self.offset, 0.0]])))

            H_analytic.append(soln_analytic)
        H_analytic = np.hstack(H_analytic)
        err = np.linalg.norm(H - H_analytic) / np.linalg.norm(H_analytic)
        self.assertTrue(err < 1e-5)
        print("EM1DFD-VMD for complex conductivity works")
    """

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
        sim = fdem.Simulation1DLayered(
            survey=survey,
            thicknesses=self.thicknesses,
            sigmaMap=sigma_map,
            topo=self.topo,
        )

        m_1D = np.log(np.ones(self.nlayers) * self.sigma)
        Hz = sim.dpred(m_1D)

        dip = MagneticDipoleHalfSpace(
            location=np.r_[0.0, 0.0, 0.0],
            orientation="x",
            frequency=self.frequencies,
            sigma=self.sigma,
            quasistatic=True,
        )
        H_analytic = np.squeeze(dip.magnetic_field(np.array([[self.offset, 0.0]])))[
            :, 2
        ]
        H_analytic = np.c_[H_analytic.real, H_analytic.imag].reshape(-1)

        err = np.linalg.norm(Hz - H_analytic) / np.linalg.norm(H_analytic)
        self.assertLess(err, 1e-5)
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
        sim = fdem.Simulation1DLayered(
            survey=survey,
            thicknesses=self.thicknesses,
            sigmaMap=sigma_map,
            topo=self.topo,
        )

        m_1D = np.log(np.ones(self.nlayers) * self.sigma)
        Hz = sim.dpred(m_1D)

        H_analytic = []
        for frequency in self.frequencies:
            soln_analytic_complex = hz_horizontal_circular_loop(
                frequency, 1.0, 5.0, self.sigma,
            )
            soln_analytic = np.r_[
                np.real(soln_analytic_complex), np.imag(soln_analytic_complex)
            ]
            H_analytic.append(soln_analytic)
        H_analytic = np.hstack(H_analytic)
        err = np.linalg.norm(Hz - H_analytic) / np.linalg.norm(H_analytic)
        self.assertLess(err, 1e-5)
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
        mu = mu_0 * np.ones(self.nlayers)
        tau = self.tau * np.ones(self.nlayers)
        c = self.c * np.ones(self.nlayers)
        eta = self.eta * np.ones(self.nlayers)

        sim = fdem.Simulation1DLayered(
            survey=survey,
            thicknesses=self.thicknesses,
            topo=self.topo,
            sigmaMap=sigma_map,
            eta=eta,
            tau=tau,
            c=c,
            mu=mu,
        )

        m_1D = self.sigma * np.ones(self.nlayers)
        Hz = sim.dpred(m_1D)

        sigma_colecole = sim.compute_complex_sigma(self.frequencies)[0, :]
        H_analytic = []
        for i_freq, frequency in enumerate(self.frequencies):
            soln_analytic_complex = hz_horizontal_circular_loop(
                frequency, 1.0, 5.0, sigma_colecole[i_freq], "secondary"
            )
            soln_analytic = np.r_[
                np.real(soln_analytic_complex), np.imag(soln_analytic_complex)
            ]
            H_analytic.append(soln_analytic)
        H_analytic = np.hstack(H_analytic)
        err = np.linalg.norm(Hz - H_analytic) / np.linalg.norm(H_analytic)

        self.assertLess(err, 1e-5)
        print("EM1DFD-CircularLoop for complex conductivity works")


if __name__ == "__main__":
    unittest.main()
