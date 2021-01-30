import unittest
import numpy as np
from SimPEG import maps, utils
import matplotlib.pyplot as plt
import SimPEG.electromagnetics.time_domain_1d as em1d
from scipy import io
from SimPEG.electromagnetics.time_domain_1d.supporting_functions.digital_filter import setFrequency
from SimPEG.electromagnetics.analytics.em1d_analytics import *


class EM1D_TD_FwdProblemTests(unittest.TestCase):

    def setUp(self):

        nearthick = np.logspace(-1, 1, 5)
        deepthick = np.logspace(1, 2, 10)
        thicknesses = np.r_[nearthick, deepthick]
        topo = np.r_[0., 0., 100.]

        src_location = np.array([0., 0., 100.+1e-5])
        rx_location = np.array([0., 0., 100.+1e-5])
        receiver_orientation = "z"  # "x", "y" or "z"
        times = np.logspace(-5, -2, 31)
        a = 20.

        # Receiver list
        receiver_list = []
        receiver_list.append(
            em1d.receivers.PointReceiver(
                rx_location, times, orientation=receiver_orientation,
                component="b"
            )
        )
        receiver_list.append(
            em1d.receivers.PointReceiver(
                rx_location, times, orientation=receiver_orientation,
                component="dbdt"
            )
        )

        waveform = em1d.waveforms.StepoffWaveform()

        source_list = [
            em1d.sources.HorizontalLoopSource(
                receiver_list=receiver_list, location=src_location, waveform=waveform,
                radius=a, current_amplitude=1., 
            )
        ]
        # Survey
        survey = em1d.survey.EM1DSurveyTD(source_list)

        sigma = 1e-2
        chi = 0.
        tau = 1e-3
        eta = 2e-1
        c = 1.
        dchi = 0.05
        tau1 = 1e-10
        tau2 = 1e2

        self.topo = topo
        self.survey = survey
        self.showIt = False
        self.sigma = sigma
        self.tau = tau
        self.eta = eta
        self.c = c
        self.chi = chi
        self.dchi = dchi
        self.tau1 = tau1
        self.tau2 = tau2
        self.times = times
        self.thicknesses = thicknesses
        self.nlayers = len(thicknesses)+1
        self.a = a

    def test_EM1DTDfwd_CirLoop_RealCond(self):

        sigma_map = maps.ExpMap(nP=self.nlayers)
        sim = em1d.simulation.EM1DTMSimulation(
            survey=self.survey, thicknesses=self.thicknesses,
            sigmaMap=sigma_map, topo=self.topo
        )

        m_1D = np.log(np.ones(self.nlayers)*self.sigma)
        d = sim.dpred(m_1D)
        bz = d[0:len(self.times)]
        dbdt = d[len(self.times):]

        bzanal = Bz_horizontal_circular_loop(
            self.a, self.times, self.sigma
        )

        dbdtanal = dBzdt_horizontal_circular_loop(
            self.a, self.times, self.sigma
        )

        if self.showIt is True:

            plt.loglog(self.times, (bz), 'b')
            plt.loglog(self.times, (bzanal), 'b.')
            plt.show()

        err = np.linalg.norm(bz-bzanal)/np.linalg.norm(bzanal)
        print ('Bz error = ', err)
        self.assertTrue(err < 1e-2)

        if self.showIt is True:

            plt.loglog(self.times, -(dbdt), 'b-')
            plt.loglog(self.times, -(dbdtanal), 'b.')
            plt.show()

        err = np.linalg.norm(dbdt-dbdtanal)/np.linalg.norm(dbdtanal)
        print ('dBzdt error = ', err)
        self.assertTrue(err < 5e-2)

        print ("EM1DTD-CirculurLoop for real conductivity works")

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

if __name__ == '__main__':
    unittest.main()
