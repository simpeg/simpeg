import unittest
import numpy as np
from SimPEG import maps
import matplotlib.pyplot as plt
import SimPEG.electromagnetics.time_domain as tdem
from scipy.constants import mu_0
from geoana.em.tdem import (
    vertical_magnetic_flux_horizontal_loop as b_loop,
    vertical_magnetic_flux_time_deriv_horizontal_loop as dbdt_loop,
    magnetic_flux_vertical_magnetic_dipole as b_dipole,
    magnetic_flux_time_deriv_magnetic_dipole as dbdt_dipole,
)


class EM1D_FD_test_failures(unittest.TestCase):
    def setUp(self):
        
        nearthick = np.logspace(-1, 1, 5)
        deepthick = np.logspace(1, 2, 10)
        thicknesses = np.r_[nearthick, deepthick]
        topo = np.r_[0.0, 0.0, 100.0]
        
        self.topo = topo
        self.thicknesses = thicknesses
        self.nlayers = len(thicknesses) + 1
    
    def test_instantiation_failures(self):
        
        times = np.logspace(-5, -2, 31)
        waveform = tdem.sources.StepOffWaveform(offTime=0.0)
        x_offset = 10.
        z_tx = [-10., 1., 1., 1.]
        z_rx = [1., -10., -10., 1.]
        use_source_receiver_offset = [False, False, True, False]
        error_type = [ValueError, ValueError, ValueError, Exception]
        fftfilt_type = [
            "key_81_CosSin_2009",
            "key_201_CosSin_2012",
            "key_601_CosSin_2009",
            "non_existent_filter"
        ]
        test_type_string = [
            'NO SOURCE BELOW SURFACE',
            'NO RX BELOW SURFACE (STANDARD)',
            'NO RX BELOW SURFACE (OFFSET)',
            'FFTFILT NOT RECOGNIZED'
        ]

        for ii in range(0, len(error_type)):
            if use_source_receiver_offset[ii]:
                rx_location = np.array([[x_offset, 0.0, z_rx[ii]]])
            else:
                rx_location = np.array([[x_offset, 0.0, z_rx[ii]+self.topo[2]]])
        
            receiver_list = [
                tdem.receivers.PointMagneticFluxDensity(
                    rx_location, times, orientation="z",
                    use_source_receiver_offset=use_source_receiver_offset[ii]
                )
            ]
            
            src_location = np.array([[0.0, 0.0, z_tx[ii]+self.topo[2]]])
        
            source_list = [
                tdem.sources.MagDipole(
                    receiver_list, location=src_location, orientation="z"
                )
            ]

            survey = tdem.Survey(source_list)
            
            self.assertRaises(
                error_type[ii],
                tdem.Simulation1DLayered,
                survey=survey,
                thicknesses=self.thicknesses,
                topo=self.topo,
                time_filter=fftfilt_type[ii]
            )
        
            print(test_type_string[ii] + " TEST PASSED")


class EM1D_TD_MagDipole_Tests(unittest.TestCase):
    # Test magnetic dipole source and receiver on Earth's surface against
    # analytic solutions from Ward and Hohmann.
    # - Tests x,y,z source and receiver locations
    # - Static conductivity
     
    
    def setUp(self):

        nearthick = np.logspace(-1, 1, 5)
        deepthick = np.logspace(1, 2, 10)
        thicknesses = np.r_[nearthick, deepthick]
        topo = np.r_[0.0, 0.0, 100.0]

        src_location = np.array([0.0, 0.0, 100.0 + 1e-5])
        rx_location = np.array([[50.0, 50.0, 100.0 + 1e-5]])
        times = np.logspace(-5, -2, 31)
        waveform = tdem.sources.StepOffWaveform(offTime=0.0)
        orientations = ['x','y','z']
        
        sigma = 0.01
        chi = 0.
        tau = 1e-3
        eta = 2e-1
        c = 1.
        dchi = 0.05
        tau1 = 1e-10
        tau2 = 1e2
        
        self.thicknesses = thicknesses
        self.nlayers = len(thicknesses) + 1
        self.topo = topo
        self.src_location = src_location
        self.rx_location = rx_location
        self.times = times
        self.waveform = waveform
        self.orientations = orientations
        self.sigma = sigma
        self.tau = tau
        self.eta = eta
        self.c = c
        self.chi = chi
        self.dchi = dchi
        self.tau1 = tau1
        self.tau2 = tau2


    def test_dipole_source_static_conductivity_b(self):
        # Test b-field computation for magnetic dipole sources to step-off. Tests:
        # - x,y,z oriented source and receivers
        # - static conductivity only
        
        for tx_orientation in self.orientations:
            
            rx_list = [
                tdem.receivers.PointMagneticFluxDensity(
                    self.rx_location, self.times, ii
                ) for ii in self.orientations
            ]
            src_list = [tdem.sources.MagDipole(rx_list, location=self.src_location, orientation=tx_orientation)]
            survey = tdem.Survey(src_list)
            
            sigma_map = maps.ExpMap(nP=self.nlayers)
            sim = tdem.Simulation1DLayered(
                survey=survey, thicknesses=self.thicknesses, sigmaMap=sigma_map, topo=self.topo
            )

            m_1D = np.log(np.ones(self.nlayers) * self.sigma)
            d_numeric = sim.dpred(m_1D).reshape(3, -1).T
            
            if tx_orientation == 'z':
            
                d_analytic = b_dipole(self.times, self.rx_location, sigma=self.sigma)[:, 0, :]
                np.testing.assert_allclose(d_numeric, d_analytic, rtol=1e-3)
                print(("\n{}-dipole source accuracy test passed".format(tx_orientation)).upper())
                
            else:
                
                print(("\n{}-dipole source analytic solution not available for accuracy test".format(tx_orientation)).upper())
                
    
    def test_dipole_source_static_conductivity_dbdt(self):
        # Test db/dt computation for magnetic dipole sources to step-off. Tests:
        # - x,y,z oriented source and receivers
        # - static conductivity only
        
        for tx_orientation in self.orientations:
            
            rx_list = [
                tdem.receivers.PointMagneticFluxTimeDerivative(
                    self.rx_location, self.times, ii
                ) for ii in self.orientations
            ]
            src_list = [tdem.sources.MagDipole(rx_list, location=self.src_location, orientation=tx_orientation)]
            survey = tdem.Survey(src_list)
            
            sigma_map = maps.ExpMap(nP=self.nlayers)
            sim = tdem.Simulation1DLayered(
                survey=survey, thicknesses=self.thicknesses, sigmaMap=sigma_map, topo=self.topo
            )

            m_1D = np.log(np.ones(self.nlayers) * self.sigma)
            d_numeric = sim.dpred(m_1D).reshape(3, -1).T
            
            if tx_orientation == 'z':
            
                d_analytic = dbdt_dipole(self.times, self.rx_location, sigma=self.sigma)[:, 0, :]
                np.testing.assert_allclose(d_numeric, d_analytic, rtol=1e-2)
                print(("\n{}-dipole source accuracy test passed".format(tx_orientation)).upper())
                
            else:
                
                print(("\n{}-dipole source analytic solution not available for accuracy test".format(tx_orientation)).upper())
        
       
class EM1D_TD_Loop_Center_Tests(unittest.TestCase):
    # Test TEM response at loop's center. Tests
    # - Dispersive magnetic properties
    
    def setUp(self):

        nearthick = np.logspace(-1, 1, 5)
        deepthick = np.logspace(1, 2, 10)
        thicknesses = np.r_[nearthick, deepthick]
        topo = np.r_[0.0, 0.0, 100.0]

        src_location = np.array([0.0, 0.0, 100.0 + 1e-5])
        rx_location = np.array([[0.0, 0.0, 100.0 + 1e-5]])
        times = np.logspace(-5, -2, 31)
        waveform = tdem.sources.StepOffWaveform(offTime=0.0)
        radius = 25.
        
        sigma = 0.01
        chi = 1.0
        tau = 1e-3
        eta = 2e-1
        c = 1.
        dchi = 0.05
        tau1 = 1e-10
        tau2 = 1e2
        
        self.thicknesses = thicknesses
        self.nlayers = len(thicknesses) + 1
        self.topo = topo
        self.src_location = src_location
        self.rx_location = rx_location
        self.times = times
        self.waveform = waveform
        self.radius = radius
        self.sigma = sigma
        self.tau = tau
        self.eta = eta
        self.c = c
        self.chi = chi
        self.dchi = dchi
        self.tau1 = tau1
        self.tau2 = tau2
        
        
    # def test_conductive_and_permeable_dbdt(self):
    # THE ANALYTIC IS WRONG IN GEOANA
            
    #     rx_list = [tdem.receivers.PointMagneticFluxTimeDerivative(self.rx_location, self.times, 'z')]
    #     src_list = [tdem.sources.CircularLoop(rx_list, location=self.src_location, radius=self.radius)]
    #     survey = tdem.Survey(src_list)
        
    #     wire_map = maps.Wires(
    #         ("sigma", self.nlayers), ("mu", self.nlayers)
    #     )
    #     sigma_map = maps.ExpMap(nP=self.nlayers) * wire_map.sigma
    #     mu_map = maps.IdentityMap(nP=self.nlayers) * wire_map.mu

    #     sim = tdem.Simulation1DLayered(
    #         survey=survey, thicknesses=self.thicknesses, topo=self.topo,
    #         sigmaMap=sigma_map, muMap=mu_map
    #     )
        
    #     mu = mu_0 * (1 + self.chi)
    #     m_1D = np.r_[
    #         np.log(self.sigma) * np.ones(self.nlayers),
    #         mu * np.ones(self.nlayers)
    #     ]
        
    #     d_numeric = sim.dpred(m_1D)
    #     d_analytic = (mu_0 / mu) * dbdt_loop(self.times, radius=self.radius, sigma=self.sigma, mu=mu)
        
    #     np.testing.assert_allclose(d_numeric, d_analytic, rtol=1e-2)
        
    #     print("\nCircular loop center accuracy test passed (high. sus.)".upper())
    
    
    def test_viscous_remanent_magnetization_dbdt(self):
        # Test b-field computation for magnetic dipole sources to step-off. Tests:
        # - x,y,z oriented source and receivers
        # - purely viscous Earth. No conductivity
            
        rx_list = [tdem.receivers.PointMagneticFluxTimeDerivative(self.rx_location, self.times, 'z')]
        src_list = [tdem.sources.CircularLoop(rx_list, location=self.src_location, radius=self.radius)]
        survey = tdem.Survey(src_list)
        
        sigma_map = maps.IdentityMap()

        sim = tdem.Simulation1DLayered(
            survey=survey, thicknesses=self.thicknesses, topo=self.topo,
            sigmaMap=sigma_map, dchi=self.dchi, tau1=self.tau1, tau2=self.tau2
        )
        
        m_1D = 1e-10 * np.ones(self.nlayers)
        d_numeric = sim.dpred(m_1D)
        
        # From Cowan (2016)
        a = self.radius
        d_analytic = (mu_0 / (2 * a)) * (self.dchi / (2 + self.dchi)) * -self.times**-1 / np.log(self.tau2/self.tau1)
        
        np.testing.assert_allclose(d_numeric, d_analytic, rtol=1e-2)
        
        print("\nCircular loop center accuracy test passed (VRM)".upper())




if __name__ == "__main__":
    unittest.main()
