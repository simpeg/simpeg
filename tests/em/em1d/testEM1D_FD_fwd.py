import unittest
from SimPEG import *
from discretize import TensorMesh
import matplotlib.pyplot as plt
import simpegEM1D as em1d
from simpegEM1D.analytics import *
#from simpegEM1D import EM1D, EM1DAnalytics, EM1DSurveyFD
import numpy as np
from scipy.constants import mu_0


class EM1D_FD_FwdProblemTests(unittest.TestCase):

    def setUp(self):

        nearthick = np.logspace(-1, 1, 5)
        deepthick = np.logspace(1, 2, 10)
        thicknesses = np.r_[nearthick, deepthick]
        topo = np.r_[0., 0., 100.]
        
        offset = 10.
        src_location = np.array([0., 0., 100.+1e-5])  
        rx_location = np.array([offset, 0., 100.+1e-5])
        field_type = "secondary"  # "secondary", "total" or "ppm"
        frequencies = np.logspace(-1, 5, 61)
        
        # Receiver list
        receiver_list = []
        receiver_list.append(
            em1d.receivers.HarmonicPointReceiver(
                rx_location, frequencies, orientation="z",
                field_type=field_type, component="real"
            )
        )
        receiver_list.append(
            em1d.receivers.HarmonicPointReceiver(
                rx_location, frequencies, orientation="z",
                field_type=field_type, component="imag"
            )
        )
        receiver_list.append(
            em1d.receivers.HarmonicPointReceiver(
                rx_location, frequencies, orientation="x",
                field_type=field_type, component="real"
            )
        )
        receiver_list.append(
            em1d.receivers.HarmonicPointReceiver(
                rx_location, frequencies, orientation="x",
                field_type=field_type, component="imag"
            )
        )
            
        source_list = [
            em1d.sources.HarmonicMagneticDipoleSource(
                receiver_list=receiver_list, location=src_location, orientation="z"
            )
        ]

        # Survey
        survey = em1d.survey.EM1DSurveyFD(source_list)
        
        sigma = 1e-2
        chi = 0.
        tau = 1e-3
        eta = 2e-1
        c = 1.

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
        self.nlayers = len(thicknesses)+1


    def test_EM1DFDfwd_VMD_Halfspace(self):
        
        sigma_map = maps.ExpMap(nP=1)
        sim = em1d.simulation.EM1DFMSimulation(
            survey=self.survey, sigmaMap=sigma_map, topo=self.topo
        )
        
        m_1D = np.array([np.log(self.sigma)])
        H = sim.dpred(m_1D)
        
        soln_anal_z = Hz_vertical_magnetic_dipole(
            self.frequencies, self.offset, self.sigma, 'secondary'
        )
        soln_anal_r = Hr_vertical_magnetic_dipole(
            self.frequencies, self.offset, self.sigma 
        )
        
        if self.showIt is True:
            N=int(len(H)/4)
            plt.loglog(self.frequencies, abs(Hz[0:N]), 'b')
            plt.loglog(self.frequencies, abs(soln_anal_z.real), 'b*')
            plt.loglog(self.frequencies, abs(Hz[N:2*N]), 'r')
            plt.loglog(self.frequencies, abs(soln_anal_z.imag), 'r*')
            plt.show()
        
        soln_anal = np.r_[
            np.real(soln_anal_z), np.imag(soln_anal_z),
            np.real(soln_anal_r), np.imag(soln_anal_r)
        ]
        
        err = np.linalg.norm(H-soln_anal)/np.linalg.norm(soln_anal)
        self.assertTrue(err < 1e-5)
        print ("EM1DFD-VMD for halfspace works")

    def test_EM1DFDfwd_VMD_RealCond(self):
        
        sigma_map = maps.ExpMap(nP=self.nlayers)
        sim = em1d.simulation.EM1DFMSimulation(
            survey=self.survey, thicknesses=self.thicknesses,
            sigmaMap=sigma_map, topo=self.topo
        )
        
        m_1D = np.log(np.ones(self.nlayers)*self.sigma)
        H = sim.dpred(m_1D)
        
        soln_anal_z = Hz_vertical_magnetic_dipole(
            self.frequencies, self.offset, self.sigma, 'secondary'
        )
        soln_anal_r = Hr_vertical_magnetic_dipole(
            self.frequencies, self.offset, self.sigma
        )
        
        if self.showIt is True:
            N=int(len(Hz)/2)
            plt.loglog(self.frequencies, abs(Hz[0:N]), 'b')
            plt.loglog(self.frequencies, abs(soln_anal.real), 'b*')
            plt.loglog(self.frequencies, abs(Hz[N:]), 'r')
            plt.loglog(self.frequencies, abs(soln_anal.imag), 'r*')
            plt.show()
        
        soln_anal = np.r_[
            np.real(soln_anal_z), np.imag(soln_anal_z),
            np.real(soln_anal_r), np.imag(soln_anal_r)
        ]
        
        err = np.linalg.norm(H-soln_anal)/np.linalg.norm(soln_anal)
        self.assertTrue(err < 1e-5)
        print ("EM1DFD-VMD for real conductivity works")

    def test_EM1DFDfwd_VMD_ComplexCond(self):

        sigma_map = maps.IdentityMap(nP=self.nlayers)
        chi = np.zeros(self.nlayers)
        tau = self.tau*np.ones(self.nlayers)
        c = self.c*np.ones(self.nlayers)
        eta = self.eta*np.ones(self.nlayers)
        
        sim = em1d.simulation.EM1DFMSimulation(
            survey=self.survey, thicknesses=self.thicknesses, topo=self.topo,
            sigmaMap=sigma_map, eta=eta, tau=tau, c=c, chi=chi
        )
        
        m_1D = self.sigma*np.ones(self.nlayers)
        H = sim.dpred(m_1D)
        
        sigma_colecole = ColeCole(
            self.frequencies, self.sigma, self.eta, self.tau, self.c
        )
        
        soln_anal_z = Hz_vertical_magnetic_dipole(
            self.frequencies, self.offset, sigma_colecole, 'secondary'
        )
        soln_anal_r = Hr_vertical_magnetic_dipole(
            self.frequencies, self.offset, sigma_colecole
        )

        if self.showIt is True:
            N=int(len(Hz)/2)
            plt.loglog(self.frequencies, abs(Hz[0:N]), 'b')
            plt.loglog(self.frequencies, abs(soln_anal.real), 'b*')
            plt.loglog(self.frequencies, abs(Hz[N:]), 'r')
            plt.loglog(self.frequencies, abs(soln_anal.imag), 'r*')
            plt.show()
        
        soln_anal = np.r_[
            np.real(soln_anal_z), np.imag(soln_anal_z),
            np.real(soln_anal_r), np.imag(soln_anal_r)
        ]
        
        err = np.linalg.norm(H-soln_anal)/np.linalg.norm(soln_anal)
        self.assertTrue(err < 1e-5)
        print ("EM1DFD-VMD for complex conductivity works")

    def test_EM1DFDfwd_HMD_RealCond(self):
        
        src_location = np.array([0., 0., 100.+1e-5])  
        rx_location = np.array([self.offset, 0., 100.+1e-5])
        receiver_orientation = "z"  # "x", "y" or "z"
        field_type = "secondary"  # "secondary", "total" or "ppm"
        frequencies = np.logspace(-1, 5, 61)
        
        # Receiver list
        receiver_list = []
        receiver_list.append(
            em1d.receivers.HarmonicPointReceiver(
                rx_location, frequencies, orientation="z",
                field_type=field_type, component="real"
            )
        )
        receiver_list.append(
            em1d.receivers.HarmonicPointReceiver(
                rx_location, frequencies, orientation="z",
                field_type=field_type, component="imag"
            )
        )
            
        source_list = [
            em1d.sources.HarmonicMagneticDipoleSource(
                receiver_list=receiver_list, location=src_location, orientation="x"
            )
        ]
        
        survey = em1d.survey.EM1DSurveyFD(source_list)
        
        sigma_map = maps.ExpMap(nP=self.nlayers)
        sim = em1d.simulation.EM1DFMSimulation(
            survey=survey, thicknesses=self.thicknesses,
            sigmaMap=sigma_map, topo=self.topo
        )
        
        m_1D = np.log(np.ones(self.nlayers)*self.sigma)
        Hz = sim.dpred(m_1D)
        
        soln_anal = Hz_horizontal_magnetic_dipole(
            self.frequencies, self.offset, self.offset, self.sigma 
        )

        if self.showIt is True:
            N=int(len(Hz)/2)
            plt.loglog(self.frequencies, abs(Hz[0:N]), 'b')
            plt.loglog(self.frequencies, abs(soln_anal.real), 'b*')
            plt.loglog(self.frequencies, abs(Hz[N:]), 'r')
            plt.loglog(self.frequencies, abs(soln_anal.imag), 'r*')
            plt.show()

        soln_anal = np.r_[np.real(soln_anal), np.imag(soln_anal)]
        
        err = np.linalg.norm(Hz-soln_anal)/np.linalg.norm(soln_anal)
        self.assertTrue(err < 1e-5)
        print ("EM1DFD-HMD for real conductivity works")    


    def test_EM1DFDfwd_CircularLoop_RealCond(self):
        
        src_location = np.array([0., 0., 100.+1e-5])  
        rx_location = np.array([0., 0., 100.+1e-5])
        receiver_orientation = "z"  # "x", "y" or "z"
        field_type = "secondary"  # "secondary", "total" or "ppm"
        frequencies = np.logspace(-1, 5, 61)
        
        # Receiver list
        receiver_list = []
        receiver_list.append(
            em1d.receivers.HarmonicPointReceiver(
                rx_location, frequencies, orientation=receiver_orientation,
                field_type=field_type, component="real"
            )
        )
        receiver_list.append(
            em1d.receivers.HarmonicPointReceiver(
                rx_location, frequencies, orientation=receiver_orientation,
                field_type=field_type, component="imag"
            )
        )
            
        source_list = [
            em1d.sources.HarmonicHorizontalLoopSource(
                receiver_list=receiver_list, location=src_location, a=5.
            )
        ]
        
        survey = em1d.survey.EM1DSurveyFD(source_list)
        
        sigma_map = maps.ExpMap(nP=self.nlayers)
        sim = em1d.simulation.EM1DFMSimulation(
            survey=survey, thicknesses=self.thicknesses,
            sigmaMap=sigma_map, topo=self.topo
        )
        
        m_1D = np.log(np.ones(self.nlayers)*self.sigma)
        Hz = sim.dpred(m_1D)
        
        soln_anal = Hz_horizontal_circular_loop(
            self.frequencies, 1., 5., self.sigma, 'secondary'
        )

        if self.showIt is True:
            N=int(len(Hz)/2)
            plt.loglog(self.frequencies, abs(Hz[0:N]), 'b')
            plt.loglog(self.frequencies, abs(soln_anal.real), 'b*')
            plt.loglog(self.frequencies, abs(Hz[N:]), 'r')
            plt.loglog(self.frequencies, abs(soln_anal.imag), 'r*')
            plt.show()

        soln_anal = np.r_[np.real(soln_anal), np.imag(soln_anal)]
        
        err = np.linalg.norm(Hz-soln_anal)/np.linalg.norm(soln_anal)
        self.assertTrue(err < 1e-5)
        print ("EM1DFD-CircularLoop for real conductivity works")

    def test_EM1DFDfwd_CircularLoop_ComplexCond(self):

        src_location = np.array([0., 0., 100.+1e-5])  
        rx_location = np.array([0., 0., 100.+1e-5])
        receiver_orientation = "z"  # "x", "y" or "z"
        field_type = "secondary"  # "secondary", "total" or "ppm"
        frequencies = np.logspace(-1, 5, 61)
        
        # Receiver list
        receiver_list = []
        receiver_list.append(
            em1d.receivers.HarmonicPointReceiver(
                rx_location, frequencies, orientation=receiver_orientation,
                field_type=field_type, component="real"
            )
        )
        receiver_list.append(
            em1d.receivers.HarmonicPointReceiver(
                rx_location, frequencies, orientation=receiver_orientation,
                field_type=field_type, component="imag"
            )
        )
            
        source_list = [
            em1d.sources.HarmonicHorizontalLoopSource(
                receiver_list=receiver_list, location=src_location, a=5.
            )
        ]

        # Survey
        survey = em1d.survey.EM1DSurveyFD(source_list)
        
        sigma_map = maps.IdentityMap(nP=self.nlayers)
        chi = np.zeros(self.nlayers)
        tau = self.tau*np.ones(self.nlayers)
        c = self.c*np.ones(self.nlayers)
        eta = self.eta*np.ones(self.nlayers)
        
        sim = em1d.simulation.EM1DFMSimulation(
            survey=survey, thicknesses=self.thicknesses, topo=self.topo,
            sigmaMap=sigma_map, eta=eta, tau=tau, c=c, chi=chi
        )
        
        m_1D = self.sigma*np.ones(self.nlayers)
        Hz = sim.dpred(m_1D)
        
        sigma_colecole = ColeCole(
            self.frequencies, self.sigma, self.eta, self.tau, self.c
        )
        
        soln_anal = Hz_horizontal_circular_loop(
            self.frequencies, 1., 5., sigma_colecole, 'secondary'
        )

        if self.showIt is True:
            N=int(len(Hz)/2)
            plt.loglog(self.frequencies, abs(Hz[0:N]), 'b')
            plt.loglog(self.frequencies, abs(soln_anal.real), 'b*')
            plt.loglog(self.frequencies, abs(Hz[N:]), 'r')
            plt.loglog(self.frequencies, abs(soln_anal.imag), 'r*')
            plt.show()

        soln_anal = np.r_[np.real(soln_anal), np.imag(soln_anal)]
        
        err = np.linalg.norm(Hz-soln_anal)/np.linalg.norm(soln_anal)
        self.assertTrue(err < 1e-5)
        print ("EM1DFD-CircularLoop for complex conductivity works")


if __name__ == '__main__':
    unittest.main()
