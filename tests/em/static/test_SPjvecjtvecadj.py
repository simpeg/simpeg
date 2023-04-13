import pytest
import numpy as np
import SimPEG.electromagnetics.static.spontaneous_potential as sp
import SimPEG.electromagnetics.static.resistivity as dc
import discretize
from SimPEG import utils
from SimPEG import maps
from discretize.tests import check_derivative, assert_isadjoint


# setup simulation
mesh = discretize.TensorMesh([10, 11, 12], "CCN")
conductivity = 0.01
base_elec = [-0.4, -0.4, -0.3]
xyz_roving = utils.ndgrid(
    mesh.cell_centers_x[2:-2], mesh.cell_centers_y[2:-2], np.r_[-0.3]
)
xyz_base = np.tile([base_elec], (xyz_roving.shape[0], 1))
rx_dipole = dc.receivers.Dipole(locations_m=xyz_roving, locations_n=xyz_base)
rx_pole = dc.receivers.Pole(locations=np.r_[xyz_roving, [base_elec]])
src = sp.sources.StreamingCurrents([rx_dipole, rx_pole])
survey = sp.Survey([src])
sim = sp.Simulation3DCellCentered(mesh=mesh, survey=survey, sigma=conductivity)


def test_forward():
    # double check qMap is maps.IdentityMap()
    sim.qMap = maps.IdentityMap()
    # We can setup a dc simulation with a dipole source at these
    # two locations to double check everything evaluated correctly.
    q = np.zeros(mesh.nC)
    a_loc = np.r_[-0.5, 0.0, -0.8]
    b_loc = np.r_[0.5, 0.0, -0.8]
    inda = mesh.closest_points_index(a_loc)
    indb = mesh.closest_points_index(b_loc)
    q[inda] = 1.0
    q[indb] = -1.0
    q /= mesh.cell_volumes

    dc_tx = dc.sources.Dipole([rx_dipole, rx_pole], location_a=a_loc, location_b=b_loc)
    dc_survey = dc.Survey([dc_tx])
    sim_dc = dc.Simulation3DCellCentered(
        mesh=mesh, survey=dc_survey, sigma=conductivity
    )

    dc_dpred = sim_dc.make_synthetic_data(None, add_noise=False)
    sp_dpred = sim.make_synthetic_data(q, add_noise=False)

    np.testing.assert_allclose(dc_dpred.dobs, sp_dpred.dobs)


@pytest.mark.parametrize(
    "q_map",
    [
        maps.IdentityMap(mesh),
        sp.CurrentDensityMap(mesh),
        sp.HydraulicHeadMap(mesh, L=1.0),
    ],
)
def test_deriv(q_map):
    sim.model = None
    sim.qMap = q_map

    def func(m):
        f = sim.fields(m)
        d = sim.dpred(m, f=f)

        def Jvec(v):
            return sim.Jvec(m, v, f=f)

        return d, Jvec

    m0 = np.random.randn(q_map.shape[1])
    check_derivative(func, m0, plotIt=False)


@pytest.mark.parametrize(
    "q_map",
    [
        maps.IdentityMap(mesh),
        sp.CurrentDensityMap(mesh),
        sp.HydraulicHeadMap(mesh, L=1.0),
    ],
)
def test_adjoint(q_map):
    sim.model = None
    sim.qMap = q_map

    model = np.random.rand(q_map.shape[1])
    f = sim.fields(model)

    def Jvec(v):
        return sim.Jvec(model, v, f=f)

    def Jtvec(v):
        return sim.Jtvec(model, v, f=f)

    assert_isadjoint(Jvec, Jtvec, shape_u=(q_map.shape[1],), shape_v=(survey.nD))


def test_errors():
    with pytest.raises(ValueError):
        sp.Simulation3DCellCentered(mesh=mesh, survey=survey, sigma=None, rho=None)
    with pytest.raises(ValueError):
        sp.Simulation3DCellCentered(mesh=mesh, survey=survey, sigma=1.0, rho=1.0)


# class SPProblemTestsCC_CurrentSource(unittest.TestCase):
#
#     def setUp(self):
#
#         mesh = discretize.TensorMesh([20, 20, 20], "CCN")
#         sigma = np.ones(mesh.nC)*1./100.
#         actind = mesh.gridCC[:, 2] < -0.2
#         # actMap = maps.InjectActiveCells(mesh, actind, 0.)
#
#         xyzM = utils.ndgrid(np.ones_like(mesh.cell_centers_x[:-1])*-0.4, np.ones_like(mesh.cell_centers_y)*-0.4, np.r_[-0.3])
#         xyzN = utils.ndgrid(mesh.cell_centers_x[1:], mesh.cell_centers_y, np.r_[-0.3])
#
#         rx = sp.receivers.Dipole(xyzN, xyzM)
#         src = sp.sources.StreamingCurrents([rx], L=np.ones(mesh.nC), mesh=mesh,
#                                        modelType="CurrentSource")
#         survey = sp.survey.Survey([src])
#
#         simulation = sp.simulation.Problem_CC(
#                 mesh=mesh, survey=survey, sigma=sigma, qMap=maps.IdentityMap(mesh), solver=PardisoSolver
#                 )
#
#         q = np.zeros(mesh.nC)
#         inda = mesh.closest_points_index(np.r_[-0.5, 0., -0.8])
#         indb = mesh.closest_points_index(np.r_[0.5, 0., -0.8])
#         q[inda] = 1.
#         q[indb] = -1.
#
#         mSynth = q.copy()
#         dpred = simulation.make_synthetic_data(mSynth, add_noise=True)
#
#         # Now set up the problem to do some minimization
#         dmis = data_misfit.L2DataMisfit(data=dpred, simulation=simulation)
#         reg = regularization.Simple(mesh)
#         opt = optimization.InexactGaussNewton(
#             maxIterLS=20, maxIter=10, tolF=1e-6,
#             tolX=1e-6, tolG=1e-6, maxIterCG=6
#         )
#         invProb = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=1e-2)
#         inv = inversion.BaseInversion(invProb)
#
#         self.inv = inv
#         self.reg = reg
#         self.p = simulation
#         self.mesh = mesh
#         self.m0 = mSynth
#         self.survey = survey
#         self.dmis = dmis
#
#     def test_misfit(self):
#         passed = tests.check_derivative(
#             lambda m: [
#                 self.p.dpred(m), lambda mx: self.p.Jvec(self.m0, mx)
#             ],
#             self.m0,
#             plotIt=False,
#             num=3,
#             dx=self.m0*0.1,
#             eps = 1e-8
#         )
#         self.assertTrue(passed)
#
#     def test_adjoint(self):
#         v = self.m0
#         w = self.survey.dobs
#         wtJv = w.dot(self.p.Jvec(self.m0, v))
#         vtJtw = v.dot(self.p.Jtvec(self.m0, w))
#         passed = np.abs(wtJv - vtJtw) < 2e-8
#         print('Adjoint Test', np.abs(wtJv - vtJtw), passed)
#         self.assertTrue(passed)
#
#     def test_dataObj(self):
#         passed = tests.check_derivative(
#             lambda m: [self.dmis(m), self.dmis.deriv(m)],
#             self.m0,
#             plotIt=False,
#             num=3,
#             dx=self.m0*2
#         )
#         self.assertTrue(passed)
