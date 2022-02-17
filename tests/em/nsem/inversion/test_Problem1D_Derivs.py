from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import unittest
import numpy as np
from scipy.constants import mu_0
from SimPEG import maps, mkvc, tests
from SimPEG.electromagnetics import natural_source as nsem
from pymatsolver import Pardiso
from SimPEG.electromagnetics.static.utils.static_utils import plot_layer
from discretize import TensorMesh

TOL = 1e-4
FLR = 1e-20  # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0


def DerivJvecTest_1D(halfspace_value, freq=False, expMap=True, formulation="e"):

    # survey, sig, sigBG, mesh, freqs = nsem.utils.test_utils.setup1DSurveyElectricMagnetic(
    #     halfspace_value, False, structure=True
    # )

    # if formulation == "e":
    #     simulation = nsem.simulation.Simulation1DElectricField(
    #         mesh=mesh,
    #         solver=Pardiso,
    #         survey=survey,
    #         sigmaMap=maps.IdentityMap(),
    #     )
    # elif formulation == "b":
    #     simulation = nsem.simulation.Simulation1DMagneticFluxDensity(
    #         mesh=mesh,
    #         solver=Pardiso,
    #         survey=survey,
    #         sigmaMap=maps.IdentityMap(),
    #     )

    #####################################################################
    # Create Survey
    # -------------
    #
    # Here we demonstrate a general way to define sources and receivers.
    # For the receivers, you choose one of 4 data type options: 'real', 'imag',
    # 'app_res' or 'phase'. The source is a planewave whose frequency must be
    # defined in Hz.
    #

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

    print("Using {0} solver for the simulation".format(simulation.solver))
    # print(
    #     "Derivative test of Jvec for eForm primary/secondary for 1d comp from {0} to {1} Hz\n".format(
    #         survey.frequencies[0], survey.frequencies[-1]
    #     )
    # )
    # simulation.mapping = maps.ExpMap(simulation.mesh)
    # simulation.sigmaPrimary = np.log(sigBG)
    # Define layer thicknesses

    x0 = model
    # cond = sig[0]
    # x0 = np.log(np.ones(simulation.mesh.nC)*halfspace_value)
    # simulation.sigmaPrimary = x0
    np.random.seed(1983)
    # if True:
    #     x0  = x0 + np.random.randn(simulation.mesh.nC)*halfspace_value*1e-1

    def fun(x):
        return simulation.dpred(x), lambda x: simulation.Jvec(x0, x)

    return tests.checkDerivative(fun, model, num=6, plotIt=False, eps=FLR)


def DerivJvecTest(halfspace_value, freq=False, expMap=True):

    survey, sig, sigBG, mesh = nsem.utils.test_utils.setup1DSurvey(
        halfspace_value, False, structure=True
    )
    simulation = nsem.Simulation1DPrimarySecondary(
        mesh, sigmaPrimary=sigBG, sigmaMap=maps.IdentityMap(mesh), survey=survey
    )
    print("Using {0} solver for the simulation".format(simulation.solver))
    print(
        "Derivative test of Jvec for eForm primary/secondary for 1d comp from {0} to {1} Hz\n".format(
            survey.frequencies[0], survey.frequencies[-1]
        )
    )
    # simulation.mapping = maps.ExpMap(simulation.mesh)
    # simulation.sigmaPrimary = np.log(sigBG)

    x0 = sigBG
    # cond = sig[0]
    # x0 = np.log(np.ones(simulation.mesh.nC)*halfspace_value)
    # simulation.sigmaPrimary = x0
    np.random.seed(1983)
    # if True:
    #     x0  = x0 + np.random.randn(simulation.mesh.nC)*halfspace_value*1e-1
    survey = simulation.survey

    def fun(x):
        return simulation.dpred(x), lambda x: simulation.Jvec(x0, x)

    return tests.checkDerivative(fun, x0, num=4, plotIt=False, eps=FLR)


def DerivProjfieldsTest(inputSetup, comp="All", freq=False):

    survey, simulation = nsem.utils.test_utils.setupSimpegNSEM_ePrimSec(
        inputSetup, comp, freq
    )
    print("Derivative test of data projection for eFormulation primary/secondary\n")
    # simulation.mapping = maps.ExpMap(simulation.mesh)
    # Initate things for the derivs Test
    src = survey.source_list[0]
    np.random.seed(1983)
    u0x = np.random.randn(survey.mesh.nE) + np.random.randn(survey.mesh.nE) * 1j
    u0y = np.random.randn(survey.mesh.nE) + np.random.randn(survey.mesh.nE) * 1j
    u0 = np.vstack((mkvc(u0x, 2), mkvc(u0y, 2)))
    f0 = simulation.fieldsPair(survey.mesh, survey)
    # u0 = np.hstack((mkvc(u0_px,2),mkvc(u0_py,2)))
    f0[src, "e_pxSolution"] = u0[: len(u0) / 2]  # u0x
    f0[src, "e_pySolution"] = u0[len(u0) / 2 : :]  # u0y

    def fun(u):
        f = simulation.fieldsPair(survey.mesh, survey)
        f[src, "e_pxSolution"] = u[: len(u) / 2]
        f[src, "e_pySolution"] = u[len(u) / 2 : :]
        return (
            rx.eval(src, survey.mesh, f),
            lambda t: rx.evalDeriv(src, survey.mesh, f0, mkvc(t, 2)),
        )

    return tests.checkDerivative(fun, u0, num=4, plotIt=False, eps=FLR)


class NSEM_DerivTests(unittest.TestCase):
    def test_derivJvec_Z1dr(self):
        self.assertTrue(DerivJvecTest(1e-2))

    def test_derivJvec_Z1di(self):
        self.assertTrue(DerivJvecTest(1e-2))

    def test_derivJvec_Z1d_e(self):
        self.assertTrue(DerivJvecTest_1D(1e-2, formulation="e"))


if __name__ == "__main__":
    unittest.main()
