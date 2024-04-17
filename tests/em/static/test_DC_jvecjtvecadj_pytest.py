import pytest
import numpy as np
from discretize import TensorMesh, CylindricalMesh
from SimPEG import (
    maps,
    data_misfit,
    tests,
    utils,
)
from SimPEG.electromagnetics import resistivity as dc
from SimPEG.electromagnetics.static.resistivity.simulation import (
    Simulation3DCellCentered,
    Simulation3DNodal,
)
import shutil

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver


np.random.seed(40)

TOL = 1e-5
FLR = 1e-20  # "zero", so if residual below this --> pass regardless of order


# Return mesh for testing
def get_mesh(mesh_type):
    if mesh_type == TensorMesh:
        aSpacing = 2.5
        nElecs = 5
        surveySize = nElecs * aSpacing - aSpacing
        cs = surveySize / nElecs / 4
        mesh = TensorMesh(
            [
                [(cs, 10, -1.3), (cs, surveySize / cs), (cs, 10, 1.3)],
                [(cs, 3, -1.3), (cs, 3, 1.3)],
                # [(cs, 5, -1.3), (cs, 10)]
            ],
            "CN",
        )

    elif mesh_type == CylindricalMesh:
        cs = 10
        nc = 20
        npad = 10
        mesh = CylindricalMesh(
            [
                [(cs, nc), (cs, npad, 1.3)],
                np.r_[2 * np.pi],
                [(cs, npad, -1.3), (cs, nc), (cs, npad, 1.3)],
            ]
        )
        mesh.x0 = np.r_[0.0, 0.0, -mesh.h[2][: npad + nc].sum()]

    return mesh


def tear_down(sim):
    try:
        shutil.rmtree(sim.sensitivity_path)
    except FileNotFoundError:
        pass


CASES_LIST_DATA = [
    (Simulation3DCellCentered, TensorMesh, "isotropic", "Robin", False, 1e-10),
    (Simulation3DCellCentered, TensorMesh, "isotropic", "Robin", True, 1e-10),
    (Simulation3DCellCentered, TensorMesh, "isotropic", "Neumann", False, 1e-10),
    (Simulation3DCellCentered, TensorMesh, "isotropic", "Dirichlet", False, 1e-10),
    (Simulation3DNodal, TensorMesh, "isotropic", "Robin", False, 1e-10),
    (Simulation3DNodal, TensorMesh, "isotropic", "Robin", True, 1e-10),
    (Simulation3DNodal, TensorMesh, "isotropic", "Neumann", False, 1e-10),
    (Simulation3DCellCentered, TensorMesh, "anisotropic", "Neumann", False, 1e-10),
    (Simulation3DCellCentered, TensorMesh, "anisotropic", "Neumann", True, 1e-10),
    (Simulation3DNodal, TensorMesh, "anisotropic", "Neumann", False, 1e-10),
    (Simulation3DNodal, TensorMesh, "anisotropic", "Neumann", True, 1e-10),
]


@pytest.mark.parametrize(
    "formulation, mesh_type, model_type, bc_type, storeJ, adjoint_tol", CASES_LIST_DATA
)
class TestDerivatives:
    def get_setup_objects(self, formulation, model_type, bc_type, storeJ, mesh):
        # Define the survey
        if isinstance(mesh, TensorMesh):
            aSpacing = 2.5
            nElecs = 8
            source_list = dc.utils.WennerSrcList(nElecs, aSpacing, in2D=True)

        else:
            raise TypeError("Tests only written for TensorMesh class")

        survey = dc.survey.Survey(source_list)

        # Define the model
        if model_type == "isotropic":
            m0 = np.ones(mesh.nC)
        elif model_type == "anisotropic":
            m0 = np.ones(2 * mesh.nC)  # YES, 2*nC!!!
        else:
            raise ValueError("model_type not recognized.")

        # Define the simulation
        sim = formulation(
            mesh,
            survey=survey,
            rhoMap=maps.IdentityMap(nP=len(m0)),
            solver=Solver,
            storeJ=storeJ,
            bc_type=bc_type,
        )

        data = sim.make_synthetic_data(m0, add_noise=True)
        dmis = data_misfit.L2DataMisfit(simulation=sim, data=data)

        return m0, dmis

    def test_misfit(
        self, formulation, mesh_type, model_type, bc_type, storeJ, adjoint_tol
    ):
        mesh = get_mesh(mesh_type)
        m0, dmis = self.get_setup_objects(
            formulation, model_type, bc_type, storeJ, mesh
        )
        sim = dmis.simulation

        passed = tests.check_derivative(
            lambda m: (sim.dpred(m), lambda mx: sim.Jvec(m0, mx)),
            m0,
            plotIt=False,
            num=3,
        )

        if storeJ:
            tear_down(sim)

        assert passed

    def test_adjoint(
        self, formulation, mesh_type, model_type, bc_type, storeJ, adjoint_tol
    ):
        mesh = get_mesh(mesh_type)
        m0, dmis = self.get_setup_objects(
            formulation, model_type, bc_type, storeJ, mesh
        )
        sim = dmis.simulation
        n_data = sim.survey.nD

        rng = np.random.default_rng(seed=41)
        v = rng.random(len(m0))
        w = rng.random(n_data)
        wtJv = w.dot(sim.Jvec(m0, v))
        vtJtw = v.dot(sim.Jtvec(m0, w))
        passed = np.abs(wtJv - vtJtw) < adjoint_tol
        print("Adjoint Test", np.abs(wtJv - vtJtw), passed)
        assert passed

        if storeJ:
            tear_down(sim)

    def test_dataObj(
        self, formulation, mesh_type, model_type, bc_type, storeJ, adjoint_tol
    ):
        mesh = get_mesh(mesh_type)
        m0, dmis = self.get_setup_objects(
            formulation, model_type, bc_type, storeJ, mesh
        )
        sim = dmis.simulation

        passed = tests.check_derivative(
            lambda m: [dmis(m), dmis.deriv(m)], m0, plotIt=False, num=3
        )

        if storeJ:
            tear_down(sim)

        assert passed


CASES_LIST_FIELD = [
    (Simulation3DCellCentered, CylindricalMesh, "isotropic", "Dirichlet", False, 1e-10),
]


@pytest.mark.parametrize(
    "formulation, mesh_type, model_type, bc_type, storeJ, adjoint_tol", CASES_LIST_FIELD
)
class TestFieldsDerivatives:
    def get_setup_objects(self, formulation, model_type, bc_type, storeJ, mesh):
        # Define the survey
        if isinstance(mesh, TensorMesh):
            aSpacing = 2.5
            nElecs = 8
            source_list = dc.utils.WennerSrcList(nElecs, aSpacing, in2D=True)

        else:
            rx_x = np.linspace(10, 200, 20)
            rx_z = np.r_[-5]
            rx_locs = utils.ndgrid([rx_x, np.r_[0], rx_z])
            rx_list = [dc.receivers.BaseRx(rx_locs, projField="e", orientation="x")]

            # sources
            src_a = np.r_[0.0, 0.0, -5.0]
            src_b = np.r_[55.0, 0.0, -5.0]

            source_list = [
                dc.sources.Dipole(rx_list, location_a=src_a, location_b=src_b)
            ]

        survey = dc.survey.Survey(source_list)

        # Define the simulation
        if model_type == "isotropic":
            sigma_map = maps.ExpMap(mesh) * maps.InjectActiveCells(
                mesh, mesh.cell_centers[:, 2] <= 0, np.log(1e-8)
            )
        else:
            raise ValueError("model_type not recognized.")

        sim = formulation(
            mesh,
            survey=survey,
            sigmaMap=sigma_map,
            solver=Solver,
            storeJ=storeJ,
            bc_type=bc_type,
        )

        return sim

    def test_e_deriv(
        self, formulation, mesh_type, model_type, bc_type, storeJ, adjoint_tol
    ):
        mesh = get_mesh(mesh_type)
        sim = self.get_setup_objects(formulation, model_type, bc_type, storeJ, mesh)
        x0 = -1 + 1e-1 * np.random.rand(sim.sigmaMap.nP)

        def fun(x):
            return sim.dpred(x), lambda x: sim.Jvec(x0, x)

        assert tests.check_derivative(fun, x0, num=3, plotIt=False)

    def test_e_adjoint(
        self, formulation, mesh_type, model_type, bc_type, storeJ, adjoint_tol
    ):
        print("Adjoint Test for e")

        mesh = get_mesh(mesh_type)
        sim = self.get_setup_objects(formulation, model_type, bc_type, storeJ, mesh)

        m = -1 + 1e-1 * np.random.rand(sim.sigmaMap.nP)
        u = sim.fields(m)
        # u = u[self.survey.source_list,'e']

        v = np.random.rand(sim.survey.nD)
        w = np.random.rand(sim.sigmaMap.nP)

        vJw = v.dot(sim.Jvec(m, w, u))
        wJtv = w.dot(sim.Jtvec(m, v, u))
        tol = np.max([TOL * (10 ** int(np.log10(np.abs(vJw)))), FLR])
        print(
            f"vJw: {vJw:1.2e}, wJTv: {wJtv:1.2e}, tol: {tol:1.0e}, "
            f"passed: {np.abs(vJw - wJtv) < tol}\n"
        )
        assert np.abs(vJw - wJtv) < adjoint_tol
