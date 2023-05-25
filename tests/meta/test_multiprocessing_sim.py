import numpy as np

# from SimPEG.potential_fields import gravity
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG import maps
from discretize import TensorMesh
import scipy.sparse as sp
import pytest

from SimPEG.meta import (
    MetaSimulation,
    # SumMetaSimulation,
    # RepeatedSimulation,
    MultiprocessingMetaSimulation,
)

mesh = TensorMesh([16, 16, 16], origin="CCN")

rx_locs = np.mgrid[-0.25:0.25:5j, -0.25:0.25:5j, 0:1:1j]
rx_locs = rx_locs.reshape(3, -1).T
rxs = dc.receivers.Pole(rx_locs)
source_locs = np.mgrid[-0.5:0.5:10j, 0:1:1j, 0:1:1j].reshape(3, -1).T
src_list = [
    dc.sources.Pole(
        [
            rxs,
        ],
        location=loc,
    )
    for loc in source_locs
]

m_test = np.arange(mesh.n_cells) / mesh.n_cells + 0.1

# split by chunks of sources
chunk_size = 3
sims = []
mappings = []
for i in range(0, len(src_list) + 1, chunk_size):
    end = min(i + chunk_size, len(src_list))
    if i == end:
        break
    survey_chunk = dc.Survey(src_list[i:end])
    sims.append(
        dc.Simulation3DNodal(mesh, survey=survey_chunk, sigmaMap=maps.IdentityMap())
    )
    mappings.append(maps.IdentityMap())

serial_sim = MetaSimulation(sims, mappings)


@pytest.fixture
def parallel_sim():
    sim = MultiprocessingMetaSimulation(sims, mappings)
    yield sim
    sim.close()


def test_meta_correctness(parallel_sim):
    # create fields objects
    f_serial = serial_sim.fields(m_test)
    f_parallel = parallel_sim.fields(m_test)

    # test data output
    d_full = serial_sim.dpred(m_test, f=f_serial)
    d_mult = parallel_sim.dpred(m_test, f=f_parallel)
    np.testing.assert_allclose(d_full, d_mult)

    # test Jvec
    u = np.random.rand(mesh.n_cells)
    jvec_full = serial_sim.Jvec(m_test, u, f=f_serial)
    jvec_mult = parallel_sim.Jvec(m_test, u, f=f_parallel)

    np.testing.assert_allclose(jvec_full, jvec_mult)

    # test Jtvec
    v = np.random.rand(serial_sim.survey.nD)
    jtvec_full = serial_sim.Jtvec(m_test, v, f=f_serial)
    jtvec_mult = parallel_sim.Jtvec(m_test, v, f=f_parallel)

    np.testing.assert_allclose(jtvec_full, jtvec_mult)

    # test get diag
    diag_full = serial_sim.getJtJdiag(m_test, f=f_serial)
    diag_mult = parallel_sim.getJtJdiag(m_test, f=f_parallel)

    np.testing.assert_allclose(diag_full, diag_mult)

    # test things also works without passing optional fields
    parallel_sim.model = m_test
    d_mult2 = parallel_sim.dpred()
    np.testing.assert_allclose(d_mult, d_mult2)

    jvec_mult2 = parallel_sim.Jvec(m_test, u)
    np.testing.assert_allclose(jvec_mult, jvec_mult2)

    jtvec_mult2 = parallel_sim.Jtvec(m_test, v)
    np.testing.assert_allclose(jtvec_mult, jtvec_mult2)

    # also pass a diagonal matrix here for testing.
    parallel_sim._jtjdiag = None
    W = sp.eye(parallel_sim.survey.nD)
    diag_mult2 = parallel_sim.getJtJdiag(m_test, W=W)
    np.testing.assert_allclose(diag_mult, diag_mult2)


#
# def test_sum_correctness():
#     mesh = TensorMesh([16, 16, 16], origin="CCN")
#
#     rx_locs = np.mgrid[-0.25:0.25:5j, -0.25:0.25:5j, 0:1:1j].reshape(3, -1).T
#     rx = gravity.Point(rx_locs, components=["gz"])
#     survey = gravity.Survey(gravity.SourceField(rx))
#     serial_sim = gravity.Simulation3DIntegral(
#         mesh, survey=survey, rhoMap=maps.IdentityMap(), n_processes=1
#     )
#
#     mesh_bot = TensorMesh([mesh.h[0], mesh.h[1], mesh.h[2][:8]], origin=mesh.origin)
#     mesh_top = TensorMesh(
#         [mesh.h[0], mesh.h[1], mesh.h[2][8:]], origin=["C", "C", mesh.nodes_z[8]]
#     )
#
#     mappings = [
#         maps.Mesh2Mesh((mesh_bot, mesh)),
#         maps.Mesh2Mesh((mesh_top, mesh)),
#     ]
#     sims = [
#         gravity.Simulation3DIntegral(
#             mesh_bot, survey=survey, rhoMap=maps.IdentityMap(), n_processes=1
#         ),
#         gravity.Simulation3DIntegral(
#             mesh_top, survey=survey, rhoMap=maps.IdentityMap(), n_processes=1
#         ),
#     ]
#
#     sum_sim = SumMetaSimulation(sims, mappings)
#
#     m_test = np.arange(mesh.n_cells) / mesh.n_cells + 0.1
#
#     # test fields objects
#     f_serial = serial_sim.fields(m_test)
#     f_parallel = sum_sim.fields(m_test)
#     np.testing.assert_allclose(f_serial, sum(f_parallel))
#
#     # test data output
#     d_full = serial_sim.dpred(m_test, f=f_serial)
#     d_mult = sum_sim.dpred(m_test, f=f_parallel)
#     np.testing.assert_allclose(d_full, d_mult)
#
#     # test Jvec
#     u = np.random.rand(mesh.n_cells)
#     jvec_full = serial_sim.Jvec(m_test, u, f=f_serial)
#     jvec_mult = sum_sim.Jvec(m_test, u, f=f_parallel)
#
#     np.testing.assert_allclose(jvec_full, jvec_mult)
#
#     # test Jtvec
#     v = np.random.rand(survey.nD)
#     jtvec_full = serial_sim.Jtvec(m_test, v, f=f_serial)
#     jtvec_mult = sum_sim.Jtvec(m_test, v, f=f_parallel)
#
#     np.testing.assert_allclose(jtvec_full, jtvec_mult)
#
#     # test get diag
#     diag_full = serial_sim.getJtJdiag(m_test, f=f_serial)
#     diag_mult = sum_sim.getJtJdiag(m_test, f=f_parallel)
#
#     np.testing.assert_allclose(diag_full, diag_mult)
#
#     # test things also works without passing optional kwargs
#     sum_sim.model = m_test
#     d_mult2 = sum_sim.dpred()
#     np.testing.assert_allclose(d_mult, d_mult2)
#
#     jvec_mult2 = sum_sim.Jvec(m_test, u)
#     np.testing.assert_allclose(jvec_mult, jvec_mult2)
#
#     jtvec_mult2 = sum_sim.Jtvec(m_test, v)
#     np.testing.assert_allclose(jtvec_mult, jtvec_mult2)
#
#     sum_sim._jtjdiag = None
#     diag_mult2 = sum_sim.getJtJdiag(m_test)
#     np.testing.assert_allclose(diag_mult, diag_mult2)
#
#
# def test_repeat_correctness():
#     # meta sim is tested for correctness
#     # so can test the repeat against the meta sim
#     mesh = TensorMesh([8, 8, 8], origin="CCN")
#
#     rx_locs = np.mgrid[-0.25:0.25:5j, -0.25:0.25:5j, 0:1:1j].reshape(3, -1).T
#     rx = gravity.Point(rx_locs, components=["gz"])
#     survey = gravity.Survey(gravity.SourceField(rx))
#     sim = gravity.Simulation3DIntegral(
#         mesh, survey=survey, rhoMap=maps.IdentityMap(), n_processes=1
#     )
#
#     time_mesh = TensorMesh(
#         [
#             8,
#         ],
#         origin=[
#             0,
#         ],
#     )
#     sim_ts = np.linspace(0, 1, 6)
#
#     mappings = []
#     simulations = []
#     eye = sp.eye(mesh.n_cells, mesh.n_cells)
#     for t in sim_ts:
#         ave_time = time_mesh.get_interpolation_matrix(
#             [
#                 t,
#             ]
#         )
#         ave_full = sp.kron(ave_time, eye, format="csr")
#         mappings.append(maps.LinearMap(ave_full))
#         simulations.append(
#             gravity.Simulation3DIntegral(
#                 mesh, survey=survey, rhoMap=maps.IdentityMap(), n_processes=1
#             )
#         )
#
#     parallel_sim = MetaSimulation(simulations, mappings)
#     repeat_sim = RepeatedSimulation(sim, mappings)
#
#     model = np.random.rand(time_mesh.n_cells, mesh.n_cells).reshape(-1)
#
#     # test field things
#     f_serial = parallel_sim.fields(model)
#     f_parallel = repeat_sim.fields(model)
#     np.testing.assert_equal(np.c_[f_serial], np.c_[f_parallel])
#
#     d_full = parallel_sim.dpred(model, f_serial)
#     d_repeat = repeat_sim.dpred(model, f_parallel)
#     np.testing.assert_equal(d_full, d_repeat)
#
#     # test Jvec
#     u = np.random.rand(len(model))
#     jvec_full = parallel_sim.Jvec(model, u, f=f_serial)
#     jvec_mult = repeat_sim.Jvec(model, u, f=f_parallel)
#     np.testing.assert_allclose(jvec_full, jvec_mult)
#
#     # test Jtvec
#     v = np.random.rand(len(sim_ts) * survey.nD)
#     jtvec_full = parallel_sim.Jtvec(model, v, f=f_serial)
#     jtvec_mult = repeat_sim.Jtvec(model, v, f=f_parallel)
#     np.testing.assert_allclose(jtvec_full, jtvec_mult)
#
#     # test get diag
#     diag_full = parallel_sim.getJtJdiag(model, f=f_serial)
#     diag_mult = repeat_sim.getJtJdiag(model, f=f_parallel)
#     np.testing.assert_allclose(diag_full, diag_mult)
#
#
# def test_meta_errors():
#     mesh = TensorMesh([16, 16, 16], origin="CCN")
#
#     rx_locs = np.mgrid[-0.25:0.25:5j, -0.25:0.25:5j, 0:1:1j]
#     rx_locs = rx_locs.reshape(3, -1).T
#     rxs = dc.receivers.Pole(rx_locs)
#     source_locs = np.mgrid[-0.5:0.5:10j, 0:1:1j, 0:1:1j].reshape(3, -1).T
#     src_list = [
#         dc.sources.Pole(
#             [
#                 rxs,
#             ],
#             location=loc,
#         )
#         for loc in source_locs
#     ]
#
#     # split by chunks of sources
#     chunk_size = 3
#     sims = []
#     mappings = []
#     for i in range(0, len(src_list) + 1, chunk_size):
#         end = min(i + chunk_size, len(src_list))
#         if i == end:
#             break
#         survey_chunk = dc.Survey(src_list[i:end])
#         sims.append(
#             dc.Simulation3DNodal(
#                 mesh, survey=survey_chunk, sigmaMap=maps.IdentityMap(mesh)
#             )
#         )
#         mappings.append(maps.IdentityMap(mesh))
#
#     # incompatible length of mappings and simulations lists
#     with pytest.raises(ValueError):
#         MetaSimulation(sims[:-1], mappings)
#
#     # mappings have incompatible input lengths:
#     mappings[0] = maps.Projection(mesh.n_cells + 1, np.arange(mesh.n_cells) + 1)
#     with pytest.raises(ValueError):
#         MetaSimulation(sims, mappings)
#
#     # incompatible mapping and simulation
#     mappings[0] = maps.Projection(mesh.n_cells, [0, 1, 3, 5, 10])
#     with pytest.raises(ValueError):
#         MetaSimulation(sims, mappings)
#
#
# def test_sum_errors():
#     mesh = TensorMesh([16, 16, 16], origin="CCN")
#
#     mesh_bot = TensorMesh([mesh.h[0], mesh.h[1], mesh.h[2][:8]], origin=mesh.origin)
#     mesh_top = TensorMesh(
#         [mesh.h[0], mesh.h[1], mesh.h[2][8:]], origin=["C", "C", mesh.nodes_z[8]]
#     )
#
#     mappings = [
#         maps.Mesh2Mesh((mesh_bot, mesh)),
#         maps.Mesh2Mesh((mesh_top, mesh)),
#     ]
#
#     rx_locs = np.mgrid[-0.25:0.25:5j, -0.25:0.25:5j, 0:1:1j].reshape(3, -1).T
#
#     rx1 = gravity.Point(rx_locs, components=["gz"])
#     survey1 = gravity.Survey(gravity.SourceField(rx1))
#     rx2 = gravity.Point(rx_locs[1:], components=["gz"])
#     survey2 = gravity.Survey(gravity.SourceField(rx2))
#
#     sims = [
#         gravity.Simulation3DIntegral(
#             mesh_bot, survey=survey1, rhoMap=maps.IdentityMap(mesh_bot), n_processes=1
#         ),
#         gravity.Simulation3DIntegral(
#             mesh_top, survey=survey2, rhoMap=maps.IdentityMap(mesh_top), n_processes=1
#         ),
#     ]
#
#     # Test simulations with different numbers of data.
#     with pytest.raises(ValueError):
#         SumMetaSimulation(sims, mappings)
#
#
# def test_repeat_errors():
#     mesh = TensorMesh([16, 16, 16], origin="CCN")
#
#     rx_locs = np.mgrid[-0.25:0.25:5j, -0.25:0.25:5j, 0:1:1j]
#     rx_locs = rx_locs.reshape(3, -1).T
#     rxs = dc.receivers.Pole(rx_locs)
#     source_locs = np.mgrid[-0.5:0.5:10j, 0:1:1j, 0:1:1j].reshape(3, -1).T
#     src_list = [
#         dc.sources.Pole(
#             [
#                 rxs,
#             ],
#             location=loc,
#         )
#         for loc in source_locs
#     ]
#     survey = dc.Survey(src_list)
#     sim = dc.Simulation3DNodal(mesh, survey=survey, sigmaMap=maps.IdentityMap(mesh))
#
#     # split by chunks of sources
#     mappings = []
#     for _i in range(10):
#         mappings.append(maps.IdentityMap(mesh))
#
#     # mappings have incompatible input lengths:
#     mappings[0] = maps.Projection(mesh.n_cells + 1, np.arange(mesh.n_cells) + 1)
#     with pytest.raises(ValueError):
#         RepeatedSimulation(sim, mappings)
#
#     # incompatible mappings and simulations
#     mappings[0] = maps.Projection(mesh.n_cells, [0, 1, 3, 5, 10])
#     with pytest.raises(ValueError):
#         RepeatedSimulation(sim, mappings)
