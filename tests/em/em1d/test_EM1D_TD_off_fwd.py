import unittest
import numpy as np
from SimPEG import maps
import matplotlib.pyplot as plt
import SimPEG.electromagnetics.time_domain as tdem
from geoana.em.tdem import (
    vertical_magnetic_flux_horizontal_loop as b_loop,
    vertical_magnetic_flux_time_deriv_horizontal_loop as dbdt_loop,
    magnetic_flux_vertical_magnetic_dipole as b_dipole,
    magnetic_flux_time_deriv_magnetic_dipole as dbdt_dipole,
)


class EM1D_TD_FwdProblemTests(unittest.TestCase):
    def setUp(self):

        nearthick = np.logspace(-1, 1, 5)
        deepthick = np.logspace(1, 2, 10)
        thicknesses = np.r_[nearthick, deepthick]
        topo = np.r_[0.0, 0.0, 100.0]

        source_location = np.array([0.0, 0.0, 100.0 + 1e-5])
        receiver_locations = np.array([[0.0, 0.0, 100.0 + 1e-5]])
        dip_receiver_locations = np.array([[100.0, 0.0, 100.0 + 1e-5]])
        times = np.logspace(-5, -2, 31)
        radius = 20.0
        waveform = tdem.sources.StepOffWaveform(offTime=0.0)

        # Receiver list
        # Define receivers at each location.
        loop_receivers_list = [
            tdem.receivers.PointMagneticFluxDensity(receiver_locations, times, "z"),
            tdem.receivers.PointMagneticFluxTimeDerivative(
                receiver_locations, times, "z"
            ),
        ]  # Make a list containing all receivers even if just one

        dip_receiver_list = [
            tdem.receivers.PointMagneticFluxDensity(dip_receiver_locations, times, "x"),
            tdem.receivers.PointMagneticFluxTimeDerivative(
                dip_receiver_locations, times, "x"
            ),
            tdem.receivers.PointMagneticFluxDensity(dip_receiver_locations, times, "z"),
            tdem.receivers.PointMagneticFluxTimeDerivative(
                dip_receiver_locations, times, "z"
            ),
        ]
        # Must define the transmitter properties and associated receivers
        source_list = [
            tdem.sources.CircularLoop(
                loop_receivers_list,
                location=source_location,
                waveform=waveform,
                radius=radius,
            ),
            tdem.sources.MagDipole(
                dip_receiver_list, location=source_location, waveform=waveform
            ),
        ]

        survey = tdem.Survey(source_list)

        sigma = 1e-2
        # chi = 0.
        # tau = 1e-3
        # eta = 2e-1
        # c = 1.
        # dchi = 0.05
        # tau1 = 1e-10
        # tau2 = 1e2

        self.topo = topo
        self.survey = survey
        self.showIt = False
        self.sigma = sigma
        # self.tau = tau
        # self.eta = eta
        # self.c = c
        # self.chi = chi
        # self.dchi = dchi
        # self.tau1 = tau1
        # self.tau2 = tau2
        self.times = times
        self.thicknesses = thicknesses
        self.nlayers = len(thicknesses) + 1
        self.a = radius
        self.dipole_rx_locations = dip_receiver_locations

        sigma_map = maps.ExpMap(nP=self.nlayers)
        sim = tdem.Simulation1DLayered(
            survey=self.survey,
            thicknesses=self.thicknesses,
            sigmaMap=sigma_map,
            topo=self.topo,
        )

        m_1D = np.log(np.ones(self.nlayers) * self.sigma)
        d = sim.dpred(m_1D).reshape(6, -1)
        self.bz_loop = d[0]
        self.bzdt_loop = d[1]
        self.bx_dip = d[2]
        self.bxdt_dip = d[3]
        self.bz_dip = d[4]
        self.bzdt_dip = d[5]

    def test_loop_bz(self):

        analytic = b_loop(self.times, sigma=self.sigma, radius=self.a)

        np.testing.assert_allclose(self.bz_loop, analytic, rtol=1e-3)

    def test_loop_bzdt(self):

        analytic = dbdt_loop(self.times, sigma=self.sigma, radius=self.a)

        np.testing.assert_allclose(self.bzdt_loop, analytic, rtol=1e-2)

    def test_magdipole_bz(self):

        analytic = b_dipole(self.times, self.dipole_rx_locations, sigma=self.sigma)[
            :, 0, 2
        ]

        np.testing.assert_allclose(self.bz_dip, analytic, rtol=1e-4)

    def test_magdipole_bzdt(self):

        analytic = dbdt_dipole(self.times, self.dipole_rx_locations, sigma=self.sigma)[
            :, 0, 2
        ]

        np.testing.assert_allclose(self.bzdt_dip, analytic, rtol=1e-2)

    def test_magdipole_bx(self):

        analytic = b_dipole(self.times, self.dipole_rx_locations, sigma=self.sigma)[
            :, 0, 0
        ]

        np.testing.assert_allclose(self.bx_dip, analytic, rtol=1e-4)

    def test_magdipole_bxdt(self):

        analytic = dbdt_dipole(self.times, self.dipole_rx_locations, sigma=self.sigma)[
            :, 0, 0
        ]

        np.testing.assert_allclose(self.bxdt_dip, analytic, rtol=1e-3)


"""
    def test_EM1DTDfwd_CirLoop_ComplexCond(self):

        sigma_map = maps.IdentityMap(nP=self.nlayers)
        chi = np.zeros(self.nlayers)
        tau = self.tau*np.ones(self.nlayers)
        c = self.c*np.ones(self.nlayers)
        eta = self.eta*np.ones(self.nlayers)

        sim = em1d.simulation.EM1DTMSimulation(
            survey=self.survey, thicknesses=self.thicknesses,
            sigmaMap=sigma_map, topo=self.topo,
            eta=eta, tau=tau, c=c, chi=chi
        )

        m_1D = np.ones(self.nlayers)*self.sigma
        d = sim.dpred(m_1D)
        bz = d[0:len(self.times)]
        dbdt = d[len(self.times):]

        w_, _, omega_int = setFrequency(self.times)
        sigCole = ColeCole(
            omega_int/(2*np.pi), self.sigma,
            self.eta, self.tau, self.c
        )

        bzanal = Bz_horizontal_circular_loop_ColeCole(
            self.a, self.times, sigCole
        )

        if self.showIt is True:

            plt.loglog(self.times, (bz), 'b')
            plt.loglog(self.times, (bzanal), 'b*')
            plt.show()

        err = np.linalg.norm(bz-bzanal)/np.linalg.norm(bzanal)
        print ('Bz error = ', err)
        self.assertTrue(err < 1e-2)

        dbdtanal = dBzdt_horizontal_circular_loop_ColeCole(
            self.a, self.times, sigCole
        )

        if self.showIt is True:

            plt.loglog(self.times, - dbdt, 'b')
            plt.loglog(self.times, - dbdtanal, 'b*')
            plt.show()

        err = np.linalg.norm(dbdt-dbdtanal)/np.linalg.norm(dbdtanal)
        print ('dBzdt error = ', err)
        self.assertTrue(err < 5e-2)
        print ("EM1DTD-CirculurLoop for Complex conductivity works")


    def test_EM1DTDfwd_CirLoop_VRM(self):

        sigma_map = maps.IdentityMap(nP=self.nlayers)
        chi = np.zeros(self.nlayers)
        dchi = self.dchi*np.ones(self.nlayers)
        tau1 = self.tau1*np.ones(self.nlayers)
        tau2 = self.tau2*np.ones(self.nlayers)

        sim = em1d.simulation.EM1DTMSimulation(
            survey=self.survey, thicknesses=self.thicknesses,
            sigmaMap=sigma_map, topo=self.topo,
            chi=chi, dchi=dchi, tau1=tau1, tau2=tau2,
            time_filter='key_201_CosSin_2012'
        )

        m_1D = 1e-8 * np.ones(self.nlayers)
        d = sim.dpred(m_1D)
        bz = d[0:len(self.times)]
        dbdt = d[len(self.times):]

        bzanal = Bz_horizontal_circular_loop_VRM(
            self.a, 1e-5, 1e-5, self.times, self.dchi, self.tau1, self.tau2
        )

        if self.showIt is True:

            plt.loglog(self.times, (bz), 'b')
            plt.loglog(self.times, (bzanal), 'b*')
            plt.show()

        # Not sure why, but this does not work
        # err = np.linalg.norm(bz-bzanal)/np.linalg.norm(bzanal)
        # print ('Bz error = ', err)
        # self.assertTrue(err < 5e-2)

        dbdtanal = dBzdt_horizontal_circular_loop_VRM(
            self.a, 1e-5, 1e-5, self.times, self.dchi, self.tau1, self.tau2
        )

        if self.showIt is True:

            plt.loglog(self.times, - dbdt, 'b')
            plt.loglog(self.times, - dbdtanal, 'b*')
            plt.show()

        err = np.linalg.norm(dbdt-dbdtanal)/np.linalg.norm(dbdtanal)
        print ('dBzdt error = ', err)
        self.assertTrue(err < 1e-2)
        print ("EM1DTD-CirculurLoop for viscous remanent magnetization works")
"""

if __name__ == "__main__":
    unittest.main()
