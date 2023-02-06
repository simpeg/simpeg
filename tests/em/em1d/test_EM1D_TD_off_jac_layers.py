import unittest
from SimPEG import maps
from discretize import tests, TensorMesh
import SimPEG.electromagnetics.time_domain as tdem
import numpy as np
from scipy.constants import mu_0


class EM1D_TD_Jacobian_Test_MagDipole(unittest.TestCase):
    # Tests 2nd order convergence of Jvec and Jtvec for magnetic dipole sources.
    # - All rx orientations of b and db/dt
    # - Span many time channels
    # - Tests derivatives wrt sigma, mu and thicknesses
    def setUp(self):
        
        # Layers and topography
        nearthick = np.logspace(-1, 1, 5)
        deepthick = np.logspace(1, 2, 10)
        thicknesses = np.r_[nearthick, deepthick]
        topo = np.r_[0.0, 0.0, 100.0]
        
        # Survey Geometry
        height = 1e-5
        src_location = np.array([0.0, 0.0, 100.0 + height])
        rx_location = np.array([5.0, 5.0, 100.0 + height])
        times = np.logspace(-5, -2, 10)
        orientations = ['x','y','z']
        
        # Define sources and receivers
        source_list = []
        for tx_orientation in orientations:
            
            receiver_list = []
            
            for rx_orientation in orientations:
                    
                receiver_list.append(
                    tdem.receivers.PointMagneticFluxDensity(rx_location, times, rx_orientation)
                )
                receiver_list.append(
                    tdem.receivers.PointMagneticFluxTimeDerivative(rx_location, times, rx_orientation)
                )
            
            source_list.append(
                tdem.sources.MagDipole(
                    receiver_list, location=src_location, orientation=tx_orientation
                )
            )

        # Survey
        survey = tdem.Survey(source_list)

        self.topo = topo
        self.survey = survey
        self.showIt = False
        self.height = height
        self.times = times
        self.thicknesses = thicknesses
        self.nlayers = len(thicknesses) + 1
        
        wire_map = maps.Wires(
            ("sigma", self.nlayers),
            ("mu", self.nlayers),
            ("thicknesses", self.nlayers-1),
            ("h", 1)
        )
        self.sigma_map = maps.ExpMap(nP=self.nlayers) * wire_map.sigma
        self.mu_map = maps.ExpMap(nP=self.nlayers) * wire_map.mu
        self.thicknesses_map = maps.ExpMap(nP=self.nlayers-1) * wire_map.thicknesses
        nP = len(source_list)
        surject_mesh = TensorMesh([np.ones(nP)])
        self.h_map = maps.SurjectFull(surject_mesh) * maps.ExpMap(nP=1) * wire_map.h

        sim = tdem.Simulation1DLayered(
            survey=self.survey,
            sigmaMap=self.sigma_map,
            muMap=self.mu_map,
            thicknessesMap=self.thicknesses_map,
            hMap=self.h_map,
            topo=self.topo,
        )

        self.sim = sim
        
    def test_EM1DFDJvec_Layers(self):

        # Conductivity
        sigma_half = 0.01
        sigma_blk = 0.1
        sig = np.ones(self.nlayers) * sigma_half
        sig[3] = sigma_blk

        # Permeability
        mu_half = mu_0
        mu_blk = 2 * mu_0
        mu = np.ones(self.nlayers) * mu_half
        mu[3] = mu_blk

        # General model
        m_1D = np.r_[
            np.log(sig),
            np.log(mu),
            np.log(self.thicknesses),
            np.log(self.height)
        ]

        def fwdfun(m):
            resp = self.sim.dpred(m)
            return resp
            # return Hz

        def jacfun(m, dm):
            Jvec = self.sim.Jvec(m, dm)
            return Jvec

        dm = m_1D * 0.5
        derChk = lambda m: [fwdfun(m), lambda mx: jacfun(m, mx)]
        passed = tests.checkDerivative(
            derChk, m_1D, num=4, dx=dm, plotIt=False, eps=1e-15
        )
        self.assertTrue(passed)
        if passed:
            print("EM1DTM MagDipole Jvec test works")

    def test_EM1DFDJtvec_Layers(self):

        # Conductivity
        sigma_half = 0.01
        sigma_blk = 0.1
        sig = np.ones(self.nlayers) * sigma_half
        sig[3] = sigma_blk

        # Permeability
        mu_half = mu_0
        mu_blk = 2 * mu_0
        mu = np.ones(self.nlayers) * mu_half
        mu[3] = mu_blk

        # General model
        m_true = np.r_[
            np.log(sig),
            np.log(mu),
            np.log(self.thicknesses),
            np.log(self.height)
        ]

        dobs = self.sim.dpred(m_true)

        m_ini = np.r_[
            np.log(np.ones(self.nlayers) * sigma_half),
            np.log(np.ones(self.nlayers) * 1.5*mu_half),
            np.log(self.thicknesses) * 0.9,
            np.log(self.height) * 0.5
        ]
        resp_ini = self.sim.dpred(m_ini)
        dr = resp_ini - dobs

        def misfit(m, dobs):
            dpred = self.sim.dpred(m)
            misfit = 0.5 * np.linalg.norm(dpred - dobs) ** 2
            dmisfit = self.sim.Jtvec(m, dr)
            return misfit, dmisfit

        derChk = lambda m: misfit(m, dobs)
        passed = tests.checkDerivative(derChk, m_ini, num=4, plotIt=False, eps=1e-27)
        self.assertTrue(passed)
        if passed:
            print("EM1DTM MagDipole Jtvec test works")
            
        
class EM1D_TD_Jacobian_Test_CircularLoop(unittest.TestCase):
    # Tests 2nd order convergence of Jvec and Jtvec for circular loop sources.
    # - All rx orientations of h and dh/dt
    # - Span many time channels
    # - Tests derivatives wrt sigma, mu and thicknesses
    def setUp(self):

        nearthick = np.logspace(-1, 1, 5)
        deepthick = np.logspace(1, 2, 10)
        thicknesses = np.r_[nearthick, deepthick]
        topo = np.r_[0.0, 0.0, 100.0]
        height = 1e-5

        source_location = np.array([0.0, 0.0, 100.0 + 1e-5])
        rx_location = np.array([[0.0, 0.0, 100.0 + 1e-5]])
        radius = 20.0
        waveform = tdem.sources.StepOffWaveform(offTime=0.0)
        times = np.logspace(-5, -2, 10)
        orientations = ['x','y','z']
        
        # Define sources and receivers
        receiver_list = []
        
        for rx_orientation in orientations:
                
            receiver_list.append(
                tdem.receivers.PointMagneticField(rx_location, times, rx_orientation)
            )
            receiver_list.append(
                tdem.receivers.PointMagneticFieldTimeDerivative(rx_location, times, rx_orientation)
            )
        
        source_list = [
            tdem.sources.CircularLoop(
                receiver_list,
                location=source_location,
                waveform=waveform,
                radius=radius,
                current=1.
            )
        ]

        survey = tdem.Survey(source_list)

        self.topo = topo
        self.survey = survey
        self.showIt = False
        self.height = height
        self.times = times
        self.thicknesses = thicknesses
        self.nlayers = len(thicknesses) + 1

        nP = len(source_list)
        
        wire_map = maps.Wires(
            ("sigma", self.nlayers),
            ("mu", self.nlayers),
            ("thicknesses", self.nlayers-1),
            ("h", 1)
        )
        self.sigma_map = maps.ExpMap(nP=self.nlayers) * wire_map.sigma
        self.mu_map = maps.ExpMap(nP=self.nlayers) * wire_map.mu
        self.thicknesses_map = maps.ExpMap(nP=self.nlayers-1) * wire_map.thicknesses
        surject_mesh = TensorMesh([np.ones(nP)])
        self.h_map = maps.SurjectFull(surject_mesh) * maps.ExpMap(nP=1) * wire_map.h

        sim = tdem.Simulation1DLayered(
            survey=self.survey,
            sigmaMap=self.sigma_map,
            muMap=self.mu_map,
            thicknessesMap=self.thicknesses_map,
            hMap=self.h_map,
            topo=self.topo,
        )

        self.sim = sim
        
    def test_EM1DFDJvec_Layers(self):

        # Conductivity
        sigma_half = 0.01
        sigma_blk = 0.1
        sig = np.ones(self.nlayers) * sigma_half
        sig[3] = sigma_blk

        # Permeability
        mu_half = mu_0
        mu_blk = 2 * mu_0
        mu = np.ones(self.nlayers) * mu_half
        mu[3] = mu_blk

        # General model
        m_1D = np.r_[
            np.log(sig),
            np.log(mu),
            np.log(self.thicknesses),
            np.log(self.height)
        ]

        def fwdfun(m):
            resp = self.sim.dpred(m)
            return resp
            # return Hz

        def jacfun(m, dm):
            Jvec = self.sim.Jvec(m, dm)
            return Jvec

        dm = m_1D * 0.5
        derChk = lambda m: [fwdfun(m), lambda mx: jacfun(m, mx)]
        passed = tests.checkDerivative(
            derChk, m_1D, num=4, dx=dm, plotIt=False, eps=1e-15
        )
        self.assertTrue(passed)
        if passed:
            print("EM1DTM Circular Loop Jvec test works")

    def test_EM1DFDJtvec_Layers(self):

        # Conductivity
        sigma_half = 0.01
        sigma_blk = 0.1
        sig = np.ones(self.nlayers) * sigma_half
        sig[3] = sigma_blk

        # Permeability
        mu_half = mu_0
        mu_blk = 2 * mu_0
        mu = np.ones(self.nlayers) * mu_half
        mu[3] = mu_blk

        # General model
        m_true = np.r_[
            np.log(sig),
            np.log(mu),
            np.log(self.thicknesses),
            np.log(self.height)
        ]

        dobs = self.sim.dpred(m_true)

        m_ini = np.r_[
            np.log(np.ones(self.nlayers) * sigma_half),
            np.log(np.ones(self.nlayers) * 1.5*mu_half),
            np.log(self.thicknesses) * 0.9,
            np.log(0.5 * self.height)
        ]
        resp_ini = self.sim.dpred(m_ini)
        dr = resp_ini - dobs

        def misfit(m, dobs):
            dpred = self.sim.dpred(m)
            misfit = 0.5 * np.linalg.norm(dpred - dobs) ** 2
            dmisfit = self.sim.Jtvec(m, dr)
            return misfit, dmisfit

        derChk = lambda m: misfit(m, dobs)
        passed = tests.checkDerivative(derChk, m_ini, num=4, plotIt=False, eps=1e-27)
        self.assertTrue(passed)
        if passed:
            print("EM1DTM Circular Loop Jtvec test works")



if __name__ == "__main__":
    unittest.main()
