from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import unittest
from scipy.constants import mu_0

from SimPEG.electromagnetics import natural_source as nsem
from SimPEG import maps
from discretize import TensorMesh


TOL = 1e-4
FLR = 1e-20  # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0


def JvecAdjointTest_1D(sigmaHalf, formulation="PrimSec"):

    # Frequencies being measured
    frequencies = np.logspace(0, 4, 21)

    # Define a receiver for each data type as a list
    receivers_list = [
        nsem.receivers.AnalyticReceiver1D(component="real"),
        nsem.receivers.AnalyticReceiver1D(component="imag"),
        nsem.receivers.AnalyticReceiver1D(component="app_res"),
        nsem.receivers.AnalyticReceiver1D(component="phase"),
    ]

    # Use a list to define the planewave source at each frequency and assign receivers
    source_list = []
    for ii in range(0, len(frequencies)):
        source_list.append(
            nsem.sources.AnalyticPlanewave1D(receivers_list, frequencies[ii])
        )

    # Define the survey object
    survey = nsem.survey.Survey1D(source_list)

    ###############################################
    # Defining a 1D Layered Earth Model
    # ---------------------------------
    #
    # Here, we define the layer thicknesses and electrical conductivities for our
    # 1D simulation. If we have N layers, we define N electrical conductivity
    # values and N-1 layer thicknesses. The lowest layer is assumed to extend to
    # infinity.
    #

    # Layer thicknesses
    layer_thicknesses = np.array([200, 200])

    # Layer conductivities
    model = np.array([0.001, 0.01, 0.001])

    # Define a mapping for conductivities
    model_mapping = maps.IdentityMap()

    ###############################################################
    # Plot Resistivity Model
    # ----------------------
    #
    # Here we plot the 1D conductivity model.
    #

    # Define a 1D mesh for plotting. Provide a maximum depth for the plot.
    max_depth = 600
    plotting_mesh = TensorMesh(
        [np.r_[layer_thicknesses, max_depth - layer_thicknesses.sum()]]
    )

    #######################################################################
    # Define the Forward Simulation and Predict MT Data
    # -------------------------------------------------
    #
    # Here we predict MT data. If the keyword argument *rhoMap* is
    # defined, the simulation will expect a resistivity model. If the keyword
    # argument *sigmaMap* is defined, the simulation will expect a conductivity model.
    #

    simulation = nsem.simulation_1d.Simulation1DRecursive(
        survey=survey, thicknesses=layer_thicknesses, sigmaMap=model_mapping
    )

    m = model
    u = simulation.fields(m)

    np.random.seed(1983)
    v = np.random.rand(survey.nD,)
    # print problem.PropMap.PropModel.nP
    w = np.random.rand(plotting_mesh.nC,)

    vJw = v.ravel().dot(simulation.Jvec(m, w, u))
    wJtv = w.ravel().dot(simulation.Jtvec(m, v, u))
    tol = np.max([TOL * (10 ** int(np.log10(np.abs(vJw)))), FLR])
    print(" vJw   wJtv  vJw - wJtv     tol    abs(vJw - wJtv) < tol")
    print(vJw, wJtv, vJw - wJtv, tol, np.abs(vJw - wJtv) < tol)
    return np.abs(vJw - wJtv) < tol


def JvecAdjointTest(sigmaHalf, formulation="PrimSec"):
    forType = "PrimSec" not in formulation
    survey, sigma, sigBG, m1d = nsem.utils.test_utils.setup1DSurvey(
        sigmaHalf, tD=forType, structure=False
    )
    print("Adjoint test of e formulation for {:s} comp \n".format(formulation))

    if "PrimSec" in formulation:
        problem = nsem.Simulation1DPrimarySecondary(
            m1d, survey=survey, sigmaPrimary=sigBG, sigmaMap=maps.IdentityMap(m1d)
        )
    else:
        raise NotImplementedError(
            "Only {} formulations are implemented.".format(formulation)
        )
    m = sigma
    u = problem.fields(m)

    np.random.seed(1983)
    v = np.random.rand(survey.nD,)
    # print problem.PropMap.PropModel.nP
    w = np.random.rand(problem.mesh.nC,)

    vJw = v.ravel().dot(problem.Jvec(m, w, u))
    wJtv = w.ravel().dot(problem.Jtvec(m, v, u))
    tol = np.max([TOL * (10 ** int(np.log10(np.abs(vJw)))), FLR])
    print(" vJw   wJtv  vJw - wJtv     tol    abs(vJw - wJtv) < tol")
    print(vJw, wJtv, vJw - wJtv, tol, np.abs(vJw - wJtv) < tol)
    return np.abs(vJw - wJtv) < tol


class NSEM_1D_AdjointTests(unittest.TestCase):
    def setUp(self):
        pass

    # Test the adjoint of Jvec and Jtvec
    # def test_JvecAdjoint_zxxr(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zxxr',.1))
    # def test_JvecAdjoint_zxxi(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zxxi',.1))
    # def test_JvecAdjoint_zxyr(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zxyr',.1))
    # def test_JvecAdjoint_zxyi(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zxyi',.1))
    # def test_JvecAdjoint_zyxr(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zyxr',.1))
    # def test_JvecAdjoint_zyxi(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zyxi',.1))
    # def test_JvecAdjoint_zyyr(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zyyr',.1))
    # def test_JvecAdjoint_zyyi(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zyyi',.1))
    def test_JvecAdjoint_All(self):
        self.assertTrue(JvecAdjointTest(1e-2))

    def test_JvecAdjoint_All_1D(self):
        self.assertTrue(JvecAdjointTest_1D(1e-2))


if __name__ == "__main__":
    unittest.main()
