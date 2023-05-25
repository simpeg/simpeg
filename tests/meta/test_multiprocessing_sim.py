import numpy as np

from SimPEG.potential_fields import gravity
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG import maps
from discretize import TensorMesh
import scipy.sparse as sp
import pytest

from SimPEG.meta import (
    MetaSimulation,
    SumMetaSimulation,
    RepeatedSimulation,
    MultiprocessingMetaSimulation,
    MultiprocessingSumMetaSimulation,
    MultiprocessingRepeatedSimulation,
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
dc_sims = []
dc_mappings = []
for i in range(0, len(src_list) + 1, chunk_size):
    end = min(i + chunk_size, len(src_list))
    if i == end:
        break
    survey_chunk = dc.Survey(src_list[i:end])
    dc_sims.append(
        dc.Simulation3DNodal(mesh, survey=survey_chunk, sigmaMap=maps.IdentityMap())
    )
    dc_mappings.append(maps.IdentityMap())

serial_dc_sim = MetaSimulation(dc_sims, dc_mappings)


@pytest.fixture
def parallel_dc_sim():
    sim = MultiprocessingMetaSimulation(dc_sims, dc_mappings)
    yield sim
    sim.close()


def test_meta_correctness(parallel_dc_sim):
    # create fields objects
    f_serial = serial_dc_sim.fields(m_test)
    f_parallel = parallel_dc_sim.fields(m_test)

    # test data output
    d_full = serial_dc_sim.dpred(m_test, f=f_serial)
    d_mult = parallel_dc_sim.dpred(m_test, f=f_parallel)
    np.testing.assert_allclose(d_full, d_mult)

    # test Jvec
    u = np.random.rand(mesh.n_cells)
    jvec_full = serial_dc_sim.Jvec(m_test, u, f=f_serial)
    jvec_mult = parallel_dc_sim.Jvec(m_test, u, f=f_parallel)

    np.testing.assert_allclose(jvec_full, jvec_mult)

    # test Jtvec
    v = np.random.rand(serial_dc_sim.survey.nD)
    jtvec_full = serial_dc_sim.Jtvec(m_test, v, f=f_serial)
    jtvec_mult = parallel_dc_sim.Jtvec(m_test, v, f=f_parallel)

    np.testing.assert_allclose(jtvec_full, jtvec_mult)

    # test get diag
    diag_full = serial_dc_sim.getJtJdiag(m_test, f=f_serial)
    diag_mult = parallel_dc_sim.getJtJdiag(m_test, f=f_parallel)

    np.testing.assert_allclose(diag_full, diag_mult)

    # test things also works without passing optional fields
    parallel_dc_sim.model = m_test
    d_mult2 = parallel_dc_sim.dpred()
    np.testing.assert_allclose(d_mult, d_mult2)

    jvec_mult2 = parallel_dc_sim.Jvec(m_test, u)
    np.testing.assert_allclose(jvec_mult, jvec_mult2)

    jtvec_mult2 = parallel_dc_sim.Jtvec(m_test, v)
    np.testing.assert_allclose(jtvec_mult, jtvec_mult2)

    # also pass a diagonal matrix here for testing.
    parallel_dc_sim._jtjdiag = None
    W = sp.eye(parallel_dc_sim.survey.nD)
    diag_mult2 = parallel_dc_sim.getJtJdiag(m_test, W=W)
    np.testing.assert_allclose(diag_mult, diag_mult2)


# Create gravity sum sims
rx_locs = np.mgrid[-0.25:0.25:5j, -0.25:0.25:5j, 0:1:1j].reshape(3, -1).T
rx = gravity.Point(rx_locs, components=["gz"])
survey = gravity.Survey(gravity.SourceField(rx))

mesh_bot = TensorMesh([mesh.h[0], mesh.h[1], mesh.h[2][:8]], origin=mesh.origin)
mesh_top = TensorMesh(
    [mesh.h[0], mesh.h[1], mesh.h[2][8:]], origin=["C", "C", mesh.nodes_z[8]]
)

g_mappings = [
    maps.Mesh2Mesh((mesh_bot, mesh)),
    maps.Mesh2Mesh((mesh_top, mesh)),
]
g_sims = [
    gravity.Simulation3DIntegral(
        mesh_bot, survey=survey, rhoMap=maps.IdentityMap(), n_processes=1
    ),
    gravity.Simulation3DIntegral(
        mesh_top, survey=survey, rhoMap=maps.IdentityMap(), n_processes=1
    ),
]

serial_grav_sim = SumMetaSimulation(g_sims, g_mappings)


@pytest.fixture
def parallel_grav_sim():
    sim = MultiprocessingSumMetaSimulation(g_sims, g_mappings)
    yield sim
    sim.close()


def test_sum_correctness(parallel_grav_sim):
    # test fields objects
    f_serial = serial_grav_sim.fields(m_test)
    f_parallel = parallel_grav_sim.fields(m_test)
    # np.testing.assert_allclose(f_serial, sum(f_parallel))

    # test data output
    d_full = serial_grav_sim.dpred(m_test, f=f_serial)
    d_mult = parallel_grav_sim.dpred(m_test, f=f_parallel)
    np.testing.assert_allclose(d_full, d_mult)

    # test Jvec
    u = np.random.rand(mesh.n_cells)
    jvec_full = serial_grav_sim.Jvec(m_test, u, f=f_serial)
    jvec_mult = parallel_grav_sim.Jvec(m_test, u, f=f_parallel)

    np.testing.assert_allclose(jvec_full, jvec_mult)

    # test Jtvec
    v = np.random.rand(survey.nD)
    jtvec_full = serial_grav_sim.Jtvec(m_test, v, f=f_serial)
    jtvec_mult = parallel_grav_sim.Jtvec(m_test, v, f=f_parallel)

    np.testing.assert_allclose(jtvec_full, jtvec_mult)

    # test get diag
    diag_full = serial_grav_sim.getJtJdiag(m_test, f=f_serial)
    diag_mult = parallel_grav_sim.getJtJdiag(m_test, f=f_parallel)

    np.testing.assert_allclose(diag_full, diag_mult)

    # test things also works without passing optional kwargs
    parallel_grav_sim.model = m_test
    d_mult2 = parallel_grav_sim.dpred()
    np.testing.assert_allclose(d_mult, d_mult2)

    jvec_mult2 = parallel_grav_sim.Jvec(m_test, u)
    np.testing.assert_allclose(jvec_mult, jvec_mult2)

    jtvec_mult2 = parallel_grav_sim.Jtvec(m_test, v)
    np.testing.assert_allclose(jtvec_mult, jtvec_mult2)

    parallel_grav_sim._jtjdiag = None
    diag_mult2 = parallel_grav_sim.getJtJdiag(m_test)
    np.testing.assert_allclose(diag_mult, diag_mult2)


############
# Repeat Sim
rx_locs = np.mgrid[-0.25:0.25:5j, -0.25:0.25:5j, 0:1:1j].reshape(3, -1).T
rx = gravity.Point(rx_locs, components=["gz"])
survey = gravity.Survey(gravity.SourceField(rx))
grav_sim = gravity.Simulation3DIntegral(
    mesh, survey=survey, rhoMap=maps.IdentityMap(), n_processes=1
)

time_mesh = TensorMesh(
    [
        8,
    ],
    origin=[
        0,
    ],
)
sim_ts = np.linspace(0, 1, 6)

repeat_mappings = []
repeat_simulations = []
eye = sp.eye(mesh.n_cells, mesh.n_cells)
for t in sim_ts:
    ave_time = time_mesh.get_interpolation_matrix(
        [
            t,
        ]
    )
    ave_full = sp.kron(ave_time, eye, format="csr")
    repeat_mappings.append(maps.LinearMap(ave_full))

r_serial_sim = RepeatedSimulation(grav_sim, repeat_mappings)
t_model = np.random.rand(time_mesh.n_cells, mesh.n_cells).reshape(-1)


@pytest.fixture
def r_parallel_sim():
    sim = MultiprocessingRepeatedSimulation(grav_sim, repeat_mappings)
    yield sim
    sim.close()


def test_repeat_correctness(r_parallel_sim):
    # test field things
    f_serial = r_serial_sim.fields(t_model)
    f_parallel = r_parallel_sim.fields(t_model)
    # np.testing.assert_equal(np.c_[f_serial], np.c_[f_parallel])

    d_full = r_serial_sim.dpred(t_model, f_serial)
    d_repeat = r_parallel_sim.dpred(t_model, f_parallel)
    np.testing.assert_equal(d_full, d_repeat)

    # test Jvec
    u = np.random.rand(len(t_model))
    jvec_full = r_serial_sim.Jvec(t_model, u, f=f_serial)
    jvec_mult = r_parallel_sim.Jvec(t_model, u, f=f_parallel)
    np.testing.assert_allclose(jvec_full, jvec_mult)

    # test Jtvec
    v = np.random.rand(len(sim_ts) * survey.nD)
    jtvec_full = r_serial_sim.Jtvec(t_model, v, f=f_serial)
    jtvec_mult = r_parallel_sim.Jtvec(t_model, v, f=f_parallel)
    np.testing.assert_allclose(jtvec_full, jtvec_mult)

    # test get diag
    diag_full = r_serial_sim.getJtJdiag(t_model, f=f_serial)
    diag_mult = r_parallel_sim.getJtJdiag(t_model, f=f_parallel)
    np.testing.assert_allclose(diag_full, diag_mult)
