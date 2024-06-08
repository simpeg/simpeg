import numpy as np
from simpeg.potential_fields import gravity
from simpeg.electromagnetics.static import resistivity as dc
from simpeg import maps
from discretize import TensorMesh
import scipy.sparse as sp
import pytest
import time

from simpeg.meta import (
    MetaSimulation,
    SumMetaSimulation,
    RepeatedSimulation,
    DaskMetaSimulation,
    DaskSumMetaSimulation,
    DaskRepeatedSimulation,
)

from distributed import Client, LocalCluster


@pytest.fixture(scope="module")
def cluster():
    dask_cluster = LocalCluster(
        n_workers=2, threads_per_worker=2, dashboard_address=None, processes=True
    )
    yield dask_cluster
    dask_cluster.close()


def test_meta_correctness(cluster):
    with Client(cluster) as client:
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
                dc.Simulation3DNodal(
                    mesh, survey=survey_chunk, sigmaMap=maps.IdentityMap()
                )
            )
            mappings.append(maps.IdentityMap())

        serial_sim = MetaSimulation(sims, mappings)
        dask_sim = DaskMetaSimulation(sims, mappings, client)

        # test fields objects
        f_meta = serial_sim.fields(m_test)
        f_dask = dask_sim.fields(m_test)
        # Can't serialize DC nodal fields here, so can't directly test them.
        # sol_meta = np.concatenate([f[:, "phiSolution"] for f in f_meta], axis=1)
        # sol_dask = np.concatenate([f.result()[:, "phiSolution"] for f in f_dask], axis=1)
        # np.testing.assert_allclose(sol_meta, sol_dask)

        # test data output
        d_meta = serial_sim.dpred(m_test, f=f_meta)
        d_dask = dask_sim.dpred(m_test, f=f_dask)
        np.testing.assert_allclose(d_dask, d_meta)

        # test Jvec
        rng = np.random.default_rng(seed=0)
        u = rng.random(mesh.n_cells)
        jvec_meta = serial_sim.Jvec(m_test, u, f=f_meta)
        jvec_dask = dask_sim.Jvec(m_test, u, f=f_dask)

        np.testing.assert_allclose(jvec_dask, jvec_meta)

        # test Jtvec
        v = rng.random(serial_sim.survey.nD)
        jtvec_meta = serial_sim.Jtvec(m_test, v, f=f_meta)
        jtvec_dask = dask_sim.Jtvec(m_test, v, f=f_dask)

        np.testing.assert_allclose(jtvec_dask, jtvec_meta)

        # test get diag
        diag_meta = serial_sim.getJtJdiag(m_test, f=f_meta)
        diag_dask = dask_sim.getJtJdiag(m_test, f=f_dask)

        np.testing.assert_allclose(diag_dask, diag_meta)

        # test things also works without passing optional fields
        dask_sim.model = m_test
        d_dask2 = dask_sim.dpred()
        np.testing.assert_allclose(d_dask, d_dask2)

        jvec_dask2 = dask_sim.Jvec(m_test, u)
        np.testing.assert_allclose(jvec_dask, jvec_dask2)

        jtvec_dask2 = dask_sim.Jtvec(m_test, v)
        np.testing.assert_allclose(jtvec_dask, jtvec_dask2)

        # also pass a diagonal matrix here for testing.
        dask_sim._jtjdiag = None
        W = sp.eye(dask_sim.survey.nD)
        diag_dask2 = dask_sim.getJtJdiag(m_test, W=W)
        np.testing.assert_allclose(diag_dask, diag_dask2)


def test_sum_sim_correctness(cluster):
    with Client(cluster) as client:
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

        serial_sim = SumMetaSimulation(g_sims, g_mappings)
        parallel_sim = DaskSumMetaSimulation(g_sims, g_mappings, client)

        m_test = np.arange(mesh.n_cells) / mesh.n_cells + 0.1

        # test fields objects
        f_full = serial_sim.fields(m_test)
        f_meta = parallel_sim.fields(m_test)
        # Again don't serialize and collect the fields on the main
        # process directly.
        # np.testing.assert_allclose(f_full, sum(f_meta))

        # test data output
        d_full = serial_sim.dpred(m_test, f=f_full)
        d_meta = parallel_sim.dpred(m_test, f=f_meta)
        np.testing.assert_allclose(d_full, d_meta, rtol=1e-6)

        rng = np.random.default_rng(0)

        # test Jvec
        u = rng.random(mesh.n_cells)
        jvec_full = serial_sim.Jvec(m_test, u, f=f_full)
        jvec_meta = parallel_sim.Jvec(m_test, u, f=f_meta)

        np.testing.assert_allclose(jvec_full, jvec_meta, rtol=1e-6)

        # test Jtvec
        v = rng.random(survey.nD)
        jtvec_full = serial_sim.Jtvec(m_test, v, f=f_full)
        jtvec_meta = parallel_sim.Jtvec(m_test, v, f=f_meta)

        np.testing.assert_allclose(jtvec_full, jtvec_meta, rtol=1e-6)

        # test get diag
        diag_full = serial_sim.getJtJdiag(m_test, f=f_full)
        diag_meta = parallel_sim.getJtJdiag(m_test, f=f_meta)

        np.testing.assert_allclose(diag_full, diag_meta, rtol=1e-6)

        # test things also works without passing optional kwargs
        parallel_sim.model = m_test
        d_meta2 = parallel_sim.dpred()
        np.testing.assert_allclose(d_meta, d_meta2)

        jvec_meta2 = parallel_sim.Jvec(m_test, u)
        np.testing.assert_allclose(jvec_meta, jvec_meta2)

        jtvec_meta2 = parallel_sim.Jtvec(m_test, v)
        np.testing.assert_allclose(jtvec_meta, jtvec_meta2)

        parallel_sim._jtjdiag = None
        diag_meta2 = parallel_sim.getJtJdiag(m_test)
        np.testing.assert_allclose(diag_meta, diag_meta2)


def test_repeat_sim_correctness(cluster):
    with Client(cluster) as client:
        # meta sim is tested for correctness
        # so can test the repeat against the meta sim
        mesh = TensorMesh([8, 8, 8], origin="CCN")
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
        parallel_sim = DaskRepeatedSimulation(grav_sim, repeat_mappings, client)

        rng = np.random.default_rng(0)
        model = rng.random((time_mesh.n_cells, mesh.n_cells)).reshape(-1)

        # test field things
        f_full = serial_sim.fields(model)
        f_meta = parallel_sim.fields(model)
        # np.testing.assert_equal(np.c_[f_full], np.c_[f_meta])

        d_full = serial_sim.dpred(model, f_full)
        d_repeat = parallel_sim.dpred(model, f_meta)
        np.testing.assert_allclose(d_full, d_repeat, rtol=1e-6)

        # test Jvec
        u = rng.random(len(model))
        jvec_full = serial_sim.Jvec(model, u, f=f_full)
        jvec_meta = parallel_sim.Jvec(model, u, f=f_meta)
        np.testing.assert_allclose(jvec_full, jvec_meta, rtol=1e-6)

        # test Jtvec
        v = rng.random(len(sim_ts) * survey.nD)
        jtvec_full = serial_sim.Jtvec(model, v, f=f_full)
        jtvec_meta = parallel_sim.Jtvec(model, v, f=f_meta)
        np.testing.assert_allclose(jtvec_full, jtvec_meta, rtol=1e-6)

        # test get diag
        diag_full = serial_sim.getJtJdiag(model, f=f_full)
        diag_meta = parallel_sim.getJtJdiag(model, f=f_meta)
        np.testing.assert_allclose(diag_full, diag_meta, rtol=1e-6)


def test_dask_meta_errors(cluster):
    with Client(cluster) as client:
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
                dc.Simulation3DNodal(
                    mesh, survey=survey_chunk, sigmaMap=maps.IdentityMap(mesh)
                )
            )
            mappings.append(maps.IdentityMap(mesh))

        # incompatible length of mappings and simulations lists
        with pytest.raises(ValueError):
            DaskMetaSimulation(sims[:-1], mappings, client)
        time.sleep(0.1)  # sleep for a bit to let the communicator catch up

        # Bad Simulation type?
        with pytest.raises(TypeError):
            DaskRepeatedSimulation(
                len(sims)
                * [
                    lambda x: x * 2,
                ],
                mappings,
                client,
            )
        time.sleep(0.1)  # sleep for a bit to let the communicator catch up

        # mappings have incompatible input lengths:
        mappings[0] = maps.Projection(mesh.n_cells + 10, np.arange(mesh.n_cells) + 1)
        with pytest.raises(ValueError):
            DaskMetaSimulation(sims, mappings, client)
        time.sleep(0.1)  # sleep for a bit to let the communicator catch up

        # incompatible mapping and simulation
        mappings[0] = maps.Projection(mesh.n_cells, [0, 1, 3, 5, 10])
        with pytest.raises(ValueError):
            DaskMetaSimulation(sims, mappings, client)
        time.sleep(0.1)  # sleep for a bit to let the communicator catch up


def test_sum_errors(cluster):
    with Client(cluster) as client:
        mesh = TensorMesh([16, 16, 16], origin="CCN")

        mesh_bot = TensorMesh([mesh.h[0], mesh.h[1], mesh.h[2][:8]], origin=mesh.origin)
        mesh_top = TensorMesh(
            [mesh.h[0], mesh.h[1], mesh.h[2][8:]], origin=["C", "C", mesh.nodes_z[8]]
        )

        mappings = [
            maps.Mesh2Mesh((mesh_bot, mesh)),
            maps.Mesh2Mesh((mesh_top, mesh)),
        ]

        rx_locs = np.mgrid[-0.25:0.25:5j, -0.25:0.25:5j, 0:1:1j].reshape(3, -1).T

        rx1 = gravity.Point(rx_locs, components=["gz"])
        survey1 = gravity.Survey(gravity.SourceField(rx1))
        rx2 = gravity.Point(rx_locs[1:], components=["gz"])
        survey2 = gravity.Survey(gravity.SourceField(rx2))

        sims = [
            gravity.Simulation3DIntegral(
                mesh_bot,
                survey=survey1,
                rhoMap=maps.IdentityMap(mesh_bot),
                n_processes=1,
            ),
            gravity.Simulation3DIntegral(
                mesh_top,
                survey=survey2,
                rhoMap=maps.IdentityMap(mesh_top),
                n_processes=1,
            ),
        ]

        # Test simulations with different numbers of data.
        with pytest.raises(ValueError):
            DaskSumMetaSimulation(sims, mappings, client)
        time.sleep(0.1)  # sleep for a half second to let the communicator catch up


def test_repeat_errors(cluster):
    with Client(cluster) as client:
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
        survey = dc.Survey(src_list)
        sim = dc.Simulation3DNodal(mesh, survey=survey, sigmaMap=maps.IdentityMap(mesh))

        # split by chunks of sources
        mappings = []
        for _i in range(10):
            mappings.append(maps.IdentityMap(mesh))

        # mappings have incompatible input lengths:
        mappings[0] = maps.Projection(mesh.n_cells + 1, np.arange(mesh.n_cells) + 1)
        with pytest.raises(ValueError):
            DaskRepeatedSimulation(sim, mappings, client)
        time.sleep(0.1)  # sleep for a half second to let the communicator catch up

        # incompatible mappings and simulations
        mappings[0] = maps.Projection(mesh.n_cells, [0, 1, 3, 5, 10])
        with pytest.raises(ValueError):
            DaskRepeatedSimulation(sim, mappings, client)
        time.sleep(0.1)  # sleep for a half second to let the communicator catch up

        # Bad Simulation type?
        with pytest.raises(TypeError):
            DaskRepeatedSimulation(lambda x: x * 2, mappings, client)
        time.sleep(0.1)  # sleep for a half second to let the communicator catch up
