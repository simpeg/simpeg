import numpy as np
import multiprocessing as mp
import sys

from SimPEG.potential_fields import gravity
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG import maps
from discretize import TensorMesh
import scipy.sparse as sp

from SimPEG.meta import (
    MetaSimulation,
    SumMetaSimulation,
    RepeatedSimulation,
    MultiprocessingMetaSimulation,
    MultiprocessingSumMetaSimulation,
    MultiprocessingRepeatedSimulation,
)

if sys.version_info[0] == 3 and sys.version_info[1] <= 8:
    mp.set_start_method("spawn")


def test_meta_correctness():
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
    dc_sims2 = []
    dc_mappings = []
    for i in range(0, len(src_list) + 1, chunk_size):
        end = min(i + chunk_size, len(src_list))
        if i == end:
            break
        survey_chunk = dc.Survey(src_list[i:end])
        dc_sims.append(
            dc.Simulation3DNodal(mesh, survey=survey_chunk, sigmaMap=maps.IdentityMap())
        )
        dc_sims2.append(
            dc.Simulation3DNodal(mesh, survey=survey_chunk, sigmaMap=maps.IdentityMap())
        )
        dc_mappings.append(maps.IdentityMap())

    serial_sim = MetaSimulation(dc_sims, dc_mappings)
    parallel_sim = MultiprocessingMetaSimulation(dc_sims2, dc_mappings, n_processes=12)

    rng = np.random.default_rng(seed=0)

    try:
        # create fields objects
        f_serial = serial_sim.fields(m_test)
        f_parallel = parallel_sim.fields(m_test)

        # test data output
        d_full = serial_sim.dpred(m_test, f=f_serial)
        d_mult = parallel_sim.dpred(m_test, f=f_parallel)
        np.testing.assert_allclose(d_full, d_mult)

        # test Jvec
        u = rng.random(mesh.n_cells)
        jvec_full = serial_sim.Jvec(m_test, u, f=f_serial)
        jvec_mult = parallel_sim.Jvec(m_test, u, f=f_parallel)
        np.testing.assert_allclose(jvec_full, jvec_mult)

        # test Jtvec
        v = rng.random(serial_sim.survey.nD)
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
    except Exception as err:
        raise err
    finally:
        parallel_sim.join()


def test_sum_correctness():
    mesh = TensorMesh([16, 16, 16], origin="CCN")
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

    m_test = np.arange(mesh.n_cells) / mesh.n_cells + 0.1

    serial_sim = SumMetaSimulation(g_sims, g_mappings)
    parallel_sim = MultiprocessingSumMetaSimulation(g_sims, g_mappings, n_processes=2)

    rng = np.random.default_rng(0)
    try:
        # test fields objects
        f_serial = serial_sim.fields(m_test)
        f_parallel = parallel_sim.fields(m_test)
        # np.testing.assert_allclose(f_serial, sum(f_parallel))

        # test data output
        d_full = serial_sim.dpred(m_test, f=f_serial)
        d_mult = parallel_sim.dpred(m_test, f=f_parallel)
        np.testing.assert_allclose(d_full, d_mult, rtol=1e-06)

        # test Jvec
        u = rng.random(mesh.n_cells)
        jvec_full = serial_sim.Jvec(m_test, u, f=f_serial)
        jvec_mult = parallel_sim.Jvec(m_test, u, f=f_parallel)

        np.testing.assert_allclose(jvec_full, jvec_mult, rtol=1e-06)

        # test Jtvec
        v = rng.random(survey.nD)
        jtvec_full = serial_sim.Jtvec(m_test, v, f=f_serial)
        jtvec_mult = parallel_sim.Jtvec(m_test, v, f=f_parallel)

        np.testing.assert_allclose(jtvec_full, jtvec_mult, rtol=1e-06)

        # test get diag
        diag_full = serial_sim.getJtJdiag(m_test, f=f_serial)
        diag_mult = parallel_sim.getJtJdiag(m_test, f=f_parallel)

        np.testing.assert_allclose(diag_full, diag_mult, rtol=1e-06)

        # test things also works without passing optional kwargs
        parallel_sim.model = m_test
        d_mult2 = parallel_sim.dpred()
        np.testing.assert_allclose(d_mult, d_mult2, rtol=1e-06)

        jvec_mult2 = parallel_sim.Jvec(m_test, u)
        np.testing.assert_allclose(jvec_mult, jvec_mult2, rtol=1e-06)

        jtvec_mult2 = parallel_sim.Jtvec(m_test, v)
        np.testing.assert_allclose(jtvec_mult, jtvec_mult2, rtol=1e-06)

        parallel_sim._jtjdiag = None
        diag_mult2 = parallel_sim.getJtJdiag(m_test)
        np.testing.assert_allclose(diag_mult, diag_mult2, rtol=1e-06)

    except Exception as err:
        raise err
    finally:
        parallel_sim.join()


def test_repeat_correctness():
    mesh = TensorMesh([16, 16, 16], origin="CCN")
    rx_locs = np.mgrid[-0.25:0.25:5j, -0.25:0.25:5j, 0:1:1j].reshape(3, -1).T
    rx = gravity.Point(rx_locs, components=["gz"])
    survey = gravity.Survey(gravity.SourceField(rx))
    grav_sim = gravity.Simulation3DIntegral(
        mesh, survey=survey, rhoMap=maps.IdentityMap(), n_processes=1
    )

    time_mesh = TensorMesh([8], origin=[0])
    sim_ts = np.linspace(0, 1, 6)

    repeat_mappings = []
    eye = sp.eye(mesh.n_cells, mesh.n_cells)
    for t in sim_ts:
        ave_time = time_mesh.get_interpolation_matrix([t])
        ave_full = sp.kron(ave_time, eye, format="csr")
        repeat_mappings.append(maps.LinearMap(ave_full))

    serial_sim = RepeatedSimulation(grav_sim, repeat_mappings)
    parallel_sim = MultiprocessingRepeatedSimulation(
        grav_sim, repeat_mappings, n_processes=2
    )

    rng = np.random.default_rng(0)

    t_model = rng.random((time_mesh.n_cells, mesh.n_cells)).reshape(-1)

    try:
        # test field things
        f_serial = serial_sim.fields(t_model)
        f_parallel = parallel_sim.fields(t_model)
        # np.testing.assert_equal(np.c_[f_serial], np.c_[f_parallel])

        d_full = serial_sim.dpred(t_model, f_serial)
        d_repeat = parallel_sim.dpred(t_model, f_parallel)
        np.testing.assert_allclose(d_full, d_repeat, rtol=1e-6)

        # test Jvec
        u = rng.random(len(t_model))
        jvec_full = serial_sim.Jvec(t_model, u, f=f_serial)
        jvec_mult = parallel_sim.Jvec(t_model, u, f=f_parallel)
        np.testing.assert_allclose(jvec_full, jvec_mult, rtol=1e-6)

        # test Jtvec
        v = rng.random(len(sim_ts) * survey.nD)
        jtvec_full = serial_sim.Jtvec(t_model, v, f=f_serial)
        jtvec_mult = parallel_sim.Jtvec(t_model, v, f=f_parallel)
        np.testing.assert_allclose(jtvec_full, jtvec_mult, rtol=1e-6)

        # test get diag
        diag_full = serial_sim.getJtJdiag(t_model, f=f_serial)
        diag_mult = parallel_sim.getJtJdiag(t_model, f=f_parallel)
        np.testing.assert_allclose(diag_full, diag_mult, rtol=1e-6)
    except Exception as err:
        raise err
    finally:
        parallel_sim.join()
