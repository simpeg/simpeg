import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

from simpeg.potential_fields import gravity
from simpeg.electromagnetics.static import resistivity as dc
from simpeg import maps
from discretize import TensorMesh
import scipy.sparse as sp

from simpeg.meta import (
    MetaSimulation,
    SumMetaSimulation,
    RepeatedSimulation,
    MultiprocessingMetaSimulation,
    MultiprocessingSumMetaSimulation,
    MultiprocessingRepeatedSimulation,
)
from simpeg.meta.multiprocessing import _SimulationProcess


class _FlakyFieldsSimulation(dc.Simulation3DNodal):
    """A test double whose `fields` call fails exactly once.

    Used to exercise the worker-process error path: the first call raises
    while its process is mid-request, and a later call on the same worker
    must still succeed cleanly (proving the request/response queue wasn't
    left desynced by the earlier failure).
    """

    _fields_call_count = 0

    def fields(self, m=None, calcJ=True):
        self._fields_call_count += 1
        if self._fields_call_count == 1:
            raise RuntimeError("synthetic fields failure")
        return super().fields(m, calcJ=calcJ)


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


def test_sum_correctness_more_sims_than_processes():
    # Regression test: MultiprocessingSumMetaSimulation must build each
    # per-process chunk as a SumMetaSimulation (summing), not the default
    # MetaSimulation (concatenating). With only 1 sim per process (as in
    # test_sum_correctness) that distinction never showed up: concatenating
    # a single array is a no-op. Here we use twice as many sims as
    # processes so each worker owns >1 sim and the bug would either crash
    # (unequal chunk shapes) or silently sum wrong values (equal chunk
    # shapes, concatenated-then-summed instead of summed).
    mesh = TensorMesh([16, 16, 16], origin="CCN")
    rx_locs = np.mgrid[-0.25:0.25:5j, -0.25:0.25:5j, 0:1:1j].reshape(3, -1).T
    rx = gravity.Point(rx_locs, components=["gz"])
    survey = gravity.Survey(gravity.SourceField(rx))

    n_layers = 4
    edges = np.linspace(0, mesh.shape_cells[2], n_layers + 1, dtype=int)
    layer_meshes = [
        TensorMesh(
            [mesh.h[0], mesh.h[1], mesh.h[2][edges[i] : edges[i + 1]]],
            origin=[mesh.origin[0], mesh.origin[1], mesh.nodes_z[edges[i]]],
        )
        for i in range(n_layers)
    ]

    g_mappings = [maps.Mesh2Mesh((m, mesh)) for m in layer_meshes]
    g_sims = [
        gravity.Simulation3DIntegral(
            m, survey=survey, rhoMap=maps.IdentityMap(), n_processes=1
        )
        for m in layer_meshes
    ]

    m_test = np.arange(mesh.n_cells) / mesh.n_cells + 0.1

    serial_sim = SumMetaSimulation(g_sims, g_mappings)
    # 4 sims over 2 processes -> chunk_sizes = [2, 2]
    parallel_sim = MultiprocessingSumMetaSimulation(g_sims, g_mappings, n_processes=2)

    try:
        d_full = serial_sim.dpred(m_test)
        d_mult = parallel_sim.dpred(m_test)
        np.testing.assert_allclose(d_full, d_mult, rtol=1e-06)
    finally:
        parallel_sim.join()


def test_worker_error_propagation():
    # Regression test: an exception raised inside a worker process while
    # computing fields must (a) be raised in the caller as a real
    # exception, and (b) not leave the worker's request/response queue
    # desynced for later, unrelated calls on the same process.
    mesh = TensorMesh([8, 8, 8], origin="CCN")
    rx_locs = np.mgrid[-0.25:0.25:3j, -0.25:0.25:3j, 0:1:1j].reshape(3, -1).T
    rxs = dc.receivers.Pole(rx_locs)
    source_locs = np.mgrid[-0.5:0.5:4j, 0:1:1j, 0:1:1j].reshape(3, -1).T
    src_list = [dc.sources.Pole([rxs], location=loc) for loc in source_locs]
    survey = dc.Survey(src_list)

    flaky_sim = _FlakyFieldsSimulation(mesh, survey=survey, sigmaMap=maps.IdentityMap())
    reference_sim = dc.Simulation3DNodal(
        mesh, survey=survey, sigmaMap=maps.IdentityMap()
    )

    m_test = np.arange(mesh.n_cells) / mesh.n_cells + 0.1

    parallel_sim = MultiprocessingMetaSimulation(
        [flaky_sim], [maps.IdentityMap()], n_processes=1
    )
    serial_sim = MetaSimulation([reference_sim], [maps.IdentityMap()])

    try:
        with pytest.raises(RuntimeError, match="synthetic fields failure"):
            parallel_sim.dpred(m_test)

        # The same worker process just failed a request; a fresh call must
        # still succeed and match the serial reference, proving no stray
        # item was left behind in the result queue by the earlier failure.
        d_mult = parallel_sim.dpred(m_test)
        d_full = serial_sim.dpred(m_test)
        np.testing.assert_allclose(d_full, d_mult, rtol=1e-06)
    finally:
        parallel_sim.join()


class _BigResultSimulation:
    """Minimal test double (not a real BaseSimulation): `fields()` is a
    no-op and `dpred()` returns a large array. Used to check that a result
    left unread in a worker's result_queue doesn't deadlock `.join()`.
    """

    def __init__(self, size):
        self.size = size
        self.model = None

    def fields(self, m):
        return None

    def dpred(self, m, f):
        return np.zeros(self.size)


def test_join_does_not_hang_on_unread_result():
    # Regression test: if a caller dispatches work to a worker and then
    # never reads its result (e.g. because an earlier collection loop
    # raised before getting to it), the worker's feeder thread can be
    # left blocked flushing that result once the OS pipe buffer fills up.
    # A plain Process.join() would then hang forever. `.join()` must
    # drain in the background instead, so it completes promptly.
    p = _SimulationProcess()
    p.start()
    elapsed = None
    try:
        p.set_sim(_BigResultSimulation(size=2_000_000))  # ~16 MB of float64
        p.store_model(np.zeros(1))
        fields_future = p.get_fields()
        p.start_dpred(fields_future)
        # Deliberately never call p.result() here.

        start = time.monotonic()
        p.join(timeout=10)
        elapsed = time.monotonic() - start
    finally:
        if p.is_alive():
            p.terminate()

    assert elapsed is not None and elapsed < 10
    assert not p.is_alive()


def test_atexit_cleanup_without_join():
    # Regression test: if a script builds a MultiprocessingMetaSimulation
    # and exits (or an exception propagates to the top level) without ever
    # calling .join(), the worker processes are non-daemonic and would
    # otherwise block Python's own atexit machinery from letting the
    # interpreter exit. The atexit fallback registered at construction
    # time must terminate them so the process exits promptly on its own.
    script = textwrap.dedent(
        """
        import numpy as np
        from discretize import TensorMesh
        from simpeg import maps
        from simpeg.electromagnetics.static import resistivity as dc
        from simpeg.meta import MultiprocessingMetaSimulation

        if __name__ == "__main__":
            mesh = TensorMesh([4, 4, 4], origin="CCN")
            rx_locs = np.mgrid[-0.25:0.25:2j, -0.25:0.25:2j, 0:1:1j].reshape(3, -1).T
            rxs = dc.receivers.Pole(rx_locs)
            source_locs = np.mgrid[-0.5:0.5:2j, 0:1:1j, 0:1:1j].reshape(3, -1).T
            src_list = [
                dc.sources.Pole([rxs], location=loc) for loc in source_locs
            ]
            survey = dc.Survey(src_list)
            sim = dc.Simulation3DNodal(
                mesh, survey=survey, sigmaMap=maps.IdentityMap()
            )

            parallel_sim = MultiprocessingMetaSimulation(
                [sim], [maps.IdentityMap()], n_processes=1
            )
            # Intentionally do NOT call parallel_sim.join(): the atexit
            # fallback must still let this process exit promptly.
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        timeout=60,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


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
