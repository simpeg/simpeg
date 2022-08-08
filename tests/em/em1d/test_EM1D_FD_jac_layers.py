import unittest
from SimPEG import maps
from SimPEG.utils import mkvc
from discretize import tests, TensorMesh
import matplotlib.pyplot as plt
import SimPEG.electromagnetics.frequency_domain as fdem
import numpy as np
from scipy.constants import mu_0


class EM1D_FD_Jac_layers_ProblemTests(unittest.TestCase):
    # TODO update this test to do sigma, mu, and thicknesses at the same time
    def setUp(self):

        nearthick = np.logspace(-1, 1, 5)
        deepthick = np.logspace(1, 2, 10)
        thicknesses = np.r_[nearthick, deepthick]
        topo = np.r_[0.0, 0.0, 100.0]
        height = 1.

        src_location = np.array([0.0, 0.0, 100.0 + height])
        rx_location = np.array([0.0, 0.0, 100.0 + height])
        frequencies = np.logspace(1, 8, 21)

        # Receiver list
        receiver_list = []
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
        receiver_list.append(
            fdem.receivers.PointMagneticFieldSecondary(
                rx_location, orientation="x", component="both"
            )
        )
        receiver_list.append(
            fdem.receivers.PointMagneticFieldSecondary(
                rx_location, orientation="y", component="real"
            )
        )
        receiver_list.append(
            fdem.receivers.PointMagneticFieldSecondary(
                rx_location, orientation="y", component="imag"
            )
        )
        receiver_list.append(
            fdem.receivers.PointMagneticFieldSecondary(
                rx_location, orientation="y", component="both"
            )
        )
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
                rx_location, orientation="z", component="both"
            )
        )
        I = 1.0
        a = 10.0
        source_list = []
        for ii, frequency in enumerate(frequencies):
            src = fdem.sources.CircularLoop(
                receiver_list, frequency, src_location, radius=a, current=I
            )
            source_list.append(src)

        # Survey
        survey = fdem.Survey(source_list)

        self.topo = topo
        self.survey = survey
        self.showIt = False
        self.height = height
        self.frequencies = frequencies
        self.thicknesses = thicknesses
        self.nlayers = len(thicknesses) + 1
        
        wire_map = maps.Wires(
            ("sigma", self.nlayers),
            # ("mu", self.nlayers),
            ("thicknesses", self.nlayers-1),
            # ("h", 1)
        )
        self.sigma_map = maps.ExpMap(nP=self.nlayers) * wire_map.sigma
        # self.mu_map = maps.IdentityMap(nP=self.nlayers) * wire_map.mu
        self.thicknesses_map = maps.ExpMap(nP=self.nlayers-1) * wire_map.thicknesses
        # surject_mesh = TensorMesh([np.ones(len(self.frequencies))])
        # self.h_map = maps.SurjectFull(surject_mesh) * maps.ExpMap(nP=1) * wire_map.h

        sim = fdem.Simulation1DLayered(
            survey=self.survey,
            sigmaMap=self.sigma_map,
            # muMap=self.mu_map,
            thicknessesMap=self.thicknesses_map,
            # hMap=self.h_map,
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
            # mu,
            np.log(self.thicknesses),
            # np.log(self.height)            
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
            print("EM1DFD-layers Jvec works")

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
            # mu,
            np.log(self.thicknesses),
            # np.log(self.height)
        ]

        dobs = self.sim.dpred(m_true)

        m_ini = np.r_[
            np.log(np.ones(self.nlayers) * sigma_half),
            # np.ones(self.nlayers) * 1.5*mu_half,
            np.log(self.thicknesses) * 0.9,
            # np.log(0.5 * self.height)
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
            print("EM1DFD-layers Jtvec works")


class EM1D_FD_Jac_layers_PiecewiseWireLoop(unittest.TestCase):
    def setUp(self):

        x_path = np.array([-2, -2, 2, 2, -2])
        y_path = np.array([-1, 1, 1, -1, -1])
        frequencies = np.logspace(0, 4)

        wire_paths = np.c_[x_path, y_path, np.ones(5) * 0.5]
        source_list = []
        receiver_list = []
        receiver_location = np.array([9.28, 0.0, 0.45])
        receiver_orientation = "z"
        receiver_list.append(
            fdem.receivers.PointMagneticFieldSecondary(
                receiver_location,
                orientation=receiver_orientation,
                data_type="field",
                component="both",
            )
        )

        for freq in frequencies:
            source = fdem.sources.PiecewiseWireLoop(
                receiver_list, wire_paths=wire_paths, frequency=freq
            )
            source_list.append(source)

        # Survey
        survey = fdem.Survey(source_list)
        self.thicknesses = np.array([20.0, 40.0])

        self.nlayers = len(self.thicknesses) + 1
        wire_map = maps.Wires(
            ("sigma", self.nlayers),
            # ("mu", self.nlayers),
            ("thicknesses", self.nlayers-1)
        )
        self.sigma_map = maps.ExpMap(nP=self.nlayers) * wire_map.sigma
        # self.mu_map = maps.IdentityMap(nP=self.nlayers) * wire_map.mu
        self.thicknesses_map = maps.ExpMap(nP=self.nlayers-1) * wire_map.thicknesses

        sim = fdem.Simulation1DLayered(
            survey=survey,
            sigmaMap=self.sigma_map,
            # muMap=self.mu_map,
            thicknessesMap=self.thicknesses_map
        )

        self.sim = sim

    def test_EM1DFDJvec_Layers(self):

        # Conductivity
        sigma_half = 0.01
        sigma_blk = 0.1
        sig = np.ones(self.nlayers) * sigma_half
        sig[1] = sigma_blk
        
        # Permeability
        mu_half = mu_0
        mu_blk = 1.1 * mu_0
        mu = np.ones(self.nlayers) * mu_half
        mu[1] = mu_blk
        
        # General model
        m_1D = np.r_[
            np.log(sig),
            # mu,
            np.log(self.thicknesses)
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

    def test_EM1DFDJtvec_Layers(self):

        # Conductivity
        sigma_half = 0.01
        sigma_blk = 0.1
        sig = np.ones(self.nlayers) * sigma_half
        sig[1] = sigma_blk
        
        # Permeability
        mu_half = mu_0
        mu_blk = 1.1 * mu_0
        mu = np.ones(self.nlayers) * mu_half
        mu[1] = mu_blk
        
        # General model
        m_true = np.r_[
            np.log(sig),
            # mu,
            np.log(self.thicknesses)
        ]

        dobs = self.sim.dpred(m_true)

        m_ini = np.r_[
            np.log(np.ones(self.nlayers) * sigma_half),
            # np.ones(self.nlayers) * mu_half,
            np.log(self.thicknesses) * 0.9
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

#======================================================================
# Revisit this later; should not be any problem in theory
# class EM1D_FD_Jac_layers_ProblemTests_Height(unittest.TestCase):

#     def setUp(self):

#         topo = np.r_[0., 0., 100.]

#         src_location = np.array([0., 0., 100.+20.])
#         rx_location = np.array([10., 0., 100.+20.])
#         field_type = "secondary"  # "secondary", "total" or "ppm"
#         frequencies = np.logspace(1, 8, 21)
#         # Receiver list
#         receiver_list = []
#         receiver_list.append(
#             fdem.receivers.PointMagneticFieldSecondary(
#                 rx_location,
#                 orientation="x",
#                 component="real"
#             )
#         )
#         receiver_list.append(
#             fdem.receivers.PointMagneticFieldSecondary(
#                 rx_location,
#                 orientation="x",
#                 component="imag"
#             )
#         )
#         receiver_list.append(
#             fdem.receivers.PointMagneticFieldSecondary(
#                 rx_location,
#                 orientation="x",
#                 component="both"
#             )
#         )
#         receiver_list.append(
#             fdem.receivers.PointMagneticFieldSecondary(
#                 rx_location,
#                 orientation="y",
#                 component="real"
#             )
#         )
#         receiver_list.append(
#             fdem.receivers.PointMagneticFieldSecondary(
#                 rx_location,
#                 orientation="y",
#                 component="imag"
#             )
#         )
#         receiver_list.append(
#             fdem.receivers.PointMagneticFieldSecondary(
#                 rx_location,
#                 orientation="y",
#                 component="both"
#             )
#         )
#         receiver_list.append(
#             fdem.receivers.PointMagneticFieldSecondary(
#                 rx_location,
#                 orientation="z",
#                 component="real"
#             )
#         )
#         receiver_list.append(
#             fdem.receivers.PointMagneticFieldSecondary(
#                 rx_location,
#                 orientation="z",
#                 component="imag"
#             )
#         )
#         receiver_list.append(
#             fdem.receivers.PointMagneticFieldSecondary(
#                 rx_location,
#                 orientation="z",
#                 component="both"
#             )
#         )
#         I = 1.
#         a = 10.
#         source_list = []
#         for ii, frequency in enumerate(frequencies):
#             src = fdem.sources.CircularLoop(
#                 receiver_list, frequency, src_location, radius=a, current=I
#             )
#             source_list.append(src)

#         # Survey
#         survey = fdem.Survey(source_list)

#         wires = maps.Wires(('sigma', 1),('height', 1))
#         expmap = maps.ExpMap(nP=1)
#         sigma_map = expmap * wires.sigma

#         self.topo = topo
#         self.survey = survey
#         self.showIt = False
#         self.frequencies = frequencies
#         self.nlayers = 1
#         self.sigma_map = sigma_map
#         self.h_map = wires.height

#         sim = em1d.simulation.EM1DFMSimulation(
#             survey=self.survey,
#             sigmaMap=self.sigma_map, hMap=wires.height, topo=self.topo
#         )

#         self.sim = sim

#     def test_EM1DFDJvec_Layers(self):

#         sigma_half = 0.01
#         height = 20.

#         m_1D = np.r_[np.log(sigma_half), height]

#         def fwdfun(m):
#             resp = self.sim.dpred(m)
#             return resp
#             # return Hz

#         def jacfun(m, dm):
#             Jvec = self.sim.Jvec(m, dm)
#             return Jvec

#         dm = m_1D*0.5
#         derChk = lambda m: [fwdfun(m), lambda mx: jacfun(m, mx)]
#         passed = tests.checkDerivative(
#             derChk, m_1D, num=4, dx=dm, plotIt=False, eps=1e-15
#         )

#         if passed:
#             print ("EM1DFD - Jvec with height works")

#     def test_EM1DFDJtvec_Layers(self):

#         sigma_half = 0.01
#         height = 20.

#         m_true = np.r_[np.log(sigma_half), height]

#         dobs = self.sim.dpred(m_true)

#         m_ini = m_true * 1.2
#         resp_ini = self.sim.dpred(m_ini)
#         dr = resp_ini-dobs

#         def misfit(m, dobs):
#             dpred = self.sim.dpred(m)
#             misfit = 0.5*np.linalg.norm(dpred-dobs)**2
#             dmisfit = self.sim.Jtvec(m, dr)
#             return misfit, dmisfit

#         derChk = lambda m: misfit(m, dobs)
#         passed = tests.checkDerivative(
#             derChk, m_ini, num=4, plotIt=False, eps=1e-27
#         )
#         self.assertTrue(passed)
#         if passed:
#             print ("EM1DFD - Jtvec with height works")


if __name__ == "__main__":
    unittest.main()
