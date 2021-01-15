import unittest
from SimPEG import *
import numpy as np
import matplotlib.pyplot as plt
import SimPEG.electromagnetics.time_domain_1d as em1d
from SimPEG.electromagnetics.time_domain_1d.waveforms import TriangleFun, TriangleFunDeriv


class EM1D_TD_general_Jac_layers_ProblemTests(unittest.TestCase):

    def setUp(self):    

        nearthick = np.logspace(-1, 1, 5)
        deepthick = np.logspace(1, 2, 10)
        thicknesses = np.r_[nearthick, deepthick]
        topo = np.r_[0., 0., 100.]
        a = 20.
        
        src_location = np.array([0., 0., 100.+1e-5])  
        rx_location = np.array([0., 0., 100.+1e-5])
        receiver_orientation = "z"  # "x", "y" or "z"
        times = np.logspace(-5, -2, 31)
        
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
        
        time_input_currents = np.r_[-np.logspace(-2, -5, 31), 0.]
        input_currents = TriangleFun(time_input_currents+0.01, 5e-3, 0.01)
        source_list = [
            em1d.sources.HorizontalLoopSource(
                receiver_list=receiver_list,
                location=src_location,
                a=a, I=1.,
                wave_type="general",
                time_input_currents=time_input_currents,
                input_currents=input_currents,
                n_pulse = 1,
                base_frequency = 25.,
                use_lowpass_filter=False,
                high_cut_frequency=210*1e3
            )
        ]
            
        # Survey
        survey = em1d.survey.EM1DSurveyTD(source_list)
        
        sigma = 1e-2

        self.topo = topo
        self.survey = survey
        self.showIt = False
        self.sigma = sigma
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

if __name__ == '__main__':
    unittest.main()
