import unittest
from SimPEG import *
import matplotlib.pyplot as plt
import SimPEG.electromagnetics.time_domain_1d as em1d
import numpy as np


class EM1D_TD_Jac_layers_ProblemTests(unittest.TestCase):

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
            
        source_list = [
            em1d.sources.HorizontalLoopSource(
                receiver_list=receiver_list, location=src_location,
                a=a, I=1., wave_type="stepoff"
            )
        ]
        # Survey
        survey = em1d.survey.EM1DSurveyTD(source_list)

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
        self.times = times
        self.thicknesses = thicknesses
        self.nlayers = len(thicknesses)+1
        self.a = a


    def test_EM1DTDJvec_Layers(self):
        
        sigma_map = maps.ExpMap(nP=self.nlayers)
        sim = em1d.simulation.EM1DTMSimulation(
            survey=self.survey, thicknesses=self.thicknesses,
            sigmaMap=sigma_map, topo=self.topo
        )
        
        m_1D = np.log(np.ones(self.nlayers)*self.sigma)
        
        def fwdfun(m):
            resp = sim.dpred(m)
            return resp

        def jacfun(m, dm):
            Jvec = sim.Jvec(m, dm)
            return Jvec

        dm = m_1D*0.5
        derChk = lambda m: [fwdfun(m), lambda mx: jacfun(m, mx)]
        passed = tests.checkDerivative(
            derChk, m_1D, num=4, dx=dm, plotIt=False, eps=1e-15
        )

        if passed:
            print ("EM1DTD-layers Jvec works")


    def test_EM1DTDJtvec_Layers(self):

        sigma_map = maps.ExpMap(nP=self.nlayers)
        sim = em1d.simulation.EM1DTMSimulation(
            survey=self.survey, thicknesses=self.thicknesses,
            sigmaMap=sigma_map, topo=self.topo
        )        

        sigma_layer = 0.1
        sigma = np.ones(self.nlayers)*self.sigma
        sigma[3] = sigma_layer
        m_true = np.log(sigma)
        
        dobs = sim.dpred(m_true)
        
        m_ini = np.log(np.ones(self.nlayers)*self.sigma)
        resp_ini = sim.dpred(m_ini)
        dr = resp_ini-dobs

        def misfit(m, dobs):
            dpred = sim.dpred(m)
            misfit = 0.5*np.linalg.norm(dpred-dobs)**2
            dmisfit = sim.Jtvec(m, dr)
            return misfit, dmisfit

        derChk = lambda m: misfit(m, dobs)
        passed = tests.checkDerivative(
            derChk, m_ini, num=4, plotIt=False, eps=1e-26
        )
        self.assertTrue(passed)
        if passed:
            print ("EM1DTD-layers Jtvec works")



class EM1D_TD_Jac_layers_ProblemTests_Height(unittest.TestCase):

    def setUp(self):
        
        topo = np.r_[0., 0., 100.]
        
        src_location = np.array([0., 0., 100.+20.])  
        rx_location = np.array([0., 0., 100.+20.])
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
            
        source_list = [
            em1d.sources.HorizontalLoopSource(
                receiver_list=receiver_list, location=src_location,
                a=a, I=1., wave_type="stepoff"
            )
        ]
        # Survey
        survey = em1d.survey.EM1DSurveyTD(source_list)

        wires = maps.Wires(('sigma', 1),('height', 1))
        expmap = maps.ExpMap(nP=1)
        sigma_map = expmap * wires.sigma

        self.topo = topo
        self.survey = survey
        self.showIt = False
        self.times = times
        self.nlayers = 1
        self.a = a
        self.sigma_map = sigma_map
        self.h_map = wires.height


    def test_EM1DTDJvec_Layers(self):
        
        sim = em1d.simulation.EM1DTMSimulation(
            survey=self.survey,
            sigmaMap=self.sigma_map, hMap=self.h_map, topo=self.topo
        )
        
        sigma_half = 0.01
        height = 20.
        
        m_1D = np.r_[np.log(sigma_half), height]
        
        def fwdfun(m):
            resp = sim.dpred(m)
            return resp

        def jacfun(m, dm):
            Jvec = sim.Jvec(m, dm)
            return Jvec

        dm = m_1D*0.5
        derChk = lambda m: [fwdfun(m), lambda mx: jacfun(m, mx)]
        passed = tests.checkDerivative(
            derChk, m_1D, num=4, dx=dm, plotIt=False, eps=1e-15
        )

        if passed:
            print ("EM1DTD-layers Jvec works")


    def test_EM1DTDJtvec_Layers(self):

        sim = em1d.simulation.EM1DTMSimulation(
            survey=self.survey,
            sigmaMap=self.sigma_map, hMap=self.h_map, topo=self.topo
        ) 
        
        sigma_half = 0.01
        height = 20.

        m_true = np.r_[np.log(sigma_half), height]
        
        dobs = sim.dpred(m_true)
        
        m_ini = 1.2 * np.r_[np.log(sigma_half), height]
        resp_ini = sim.dpred(m_ini)
        dr = resp_ini-dobs

        def misfit(m, dobs):
            dpred = sim.dpred(m)
            misfit = 0.5*np.linalg.norm(dpred-dobs)**2
            dmisfit = sim.Jtvec(m, dr)
            return misfit, dmisfit

        derChk = lambda m: misfit(m, dobs)
        passed = tests.checkDerivative(
            derChk, m_ini, num=4, plotIt=False, eps=1e-26
        )
        self.assertTrue(passed)
        if passed:
            print ("EM1DTD-layers Jtvec works")


if __name__ == '__main__':
    unittest.main()
