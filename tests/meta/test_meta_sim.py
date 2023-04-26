import numpy as np
from SimPEG.potential_fields import gravity
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG import maps
from discretize import TensorMesh
import scipy.sparse as sp
import pytest

from SimPEG.meta import MetaSimulation, SumMetaSimulation, RepeatedSimulation


def test_multi_sim_correctness():
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
    survey_full = dc.Survey(src_list)
    full_sim = dc.Simulation3DNodal(
        mesh, survey=survey_full, sigmaMap=maps.IdentityMap()
    )

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

    multi_sim = MetaSimulation(sims, mappings)

    # test fields objects
    f_full = full_sim.fields(m_test)
    f_mult = multi_sim.fields(m_test)
    sol_full = f_full[:, "phiSolution"]
    sol_mult = np.concatenate([f[:, "phiSolution"] for f in f_mult], axis=1)
    np.testing.assert_allclose(sol_full, sol_mult)

    # test data output
    d_full = full_sim.dpred(m_test, f=f_full)
    d_mult = multi_sim.dpred(m_test, f=f_mult)
    np.testing.assert_allclose(d_full, d_mult)

    # test Jvec
    u = np.random.rand(mesh.n_cells)
    jvec_full = full_sim.Jvec(m_test, u, f=f_full)
    jvec_mult = multi_sim.Jvec(m_test, u, f=f_mult)

    np.testing.assert_allclose(jvec_full, jvec_mult)

    # test Jtvec
    v = np.random.rand(survey_full.nD)
    jtvec_full = full_sim.Jtvec(m_test, v, f=f_full)
    jtvec_mult = multi_sim.Jtvec(m_test, v, f=f_mult)

    np.testing.assert_allclose(jtvec_full, jtvec_mult)

    # test get diag
    diag_full = full_sim.getJtJdiag(m_test, f=f_full)
    diag_mult = multi_sim.getJtJdiag(m_test, f=f_mult)

    np.testing.assert_allclose(diag_full, diag_mult)

    # test things also works without passing optional fields
    multi_sim.model = m_test
    d_mult2 = multi_sim.dpred()
    np.testing.assert_allclose(d_mult, d_mult2)

    jvec_mult2 = multi_sim.Jvec(m_test, u)
    np.testing.assert_allclose(jvec_mult, jvec_mult2)

    jtvec_mult2 = multi_sim.Jtvec(m_test, v)
    np.testing.assert_allclose(jtvec_mult, jtvec_mult2)

    # also pass a diagonal matrix here for testing.
    multi_sim._jtjdiag = None
    W = sp.eye(multi_sim.survey.nD)
    diag_mult2 = multi_sim.getJtJdiag(m_test, W=W)
    np.testing.assert_allclose(diag_mult, diag_mult2)


def test_sum_sim_correctness():
    mesh = TensorMesh([16, 16, 16], origin="CCN")

    rx_locs = np.mgrid[-0.25:0.25:5j, -0.25:0.25:5j, 0:1:1j].reshape(3, -1).T
    rx = gravity.Point(rx_locs, components=["gz"])
    survey = gravity.Survey(gravity.SourceField(rx))
    full_sim = gravity.Simulation3DIntegral(
        mesh, survey=survey, rhoMap=maps.IdentityMap(), n_processes=1
    )

    mesh_bot = TensorMesh([mesh.h[0], mesh.h[1], mesh.h[2][:8]], origin=mesh.origin)
    mesh_top = TensorMesh(
        [mesh.h[0], mesh.h[1], mesh.h[2][8:]], origin=["C", "C", mesh.nodes_z[8]]
    )

    mappings = [
        maps.Mesh2Mesh((mesh_bot, mesh)),
        maps.Mesh2Mesh((mesh_top, mesh)),
    ]
    sims = [
        gravity.Simulation3DIntegral(
            mesh_bot, survey=survey, rhoMap=maps.IdentityMap(), n_processes=1
        ),
        gravity.Simulation3DIntegral(
            mesh_top, survey=survey, rhoMap=maps.IdentityMap(), n_processes=1
        ),
    ]

    sum_sim = SumMetaSimulation(sims, mappings)

    m_test = np.arange(mesh.n_cells) / mesh.n_cells + 0.1

    # test fields objects
    f_full = full_sim.fields(m_test)
    f_mult = sum_sim.fields(m_test)
    np.testing.assert_allclose(f_full, sum(f_mult))

    # test data output
    d_full = full_sim.dpred(m_test, f=f_full)
    d_mult = sum_sim.dpred(m_test, f=f_mult)
    np.testing.assert_allclose(d_full, d_mult)

    # test Jvec
    u = np.random.rand(mesh.n_cells)
    jvec_full = full_sim.Jvec(m_test, u, f=f_full)
    jvec_mult = sum_sim.Jvec(m_test, u, f=f_mult)

    np.testing.assert_allclose(jvec_full, jvec_mult)

    # test Jtvec
    v = np.random.rand(survey.nD)
    jtvec_full = full_sim.Jtvec(m_test, v, f=f_full)
    jtvec_mult = sum_sim.Jtvec(m_test, v, f=f_mult)

    np.testing.assert_allclose(jtvec_full, jtvec_mult)

    # test get diag
    diag_full = full_sim.getJtJdiag(m_test, f=f_full)
    diag_mult = sum_sim.getJtJdiag(m_test, f=f_mult)

    np.testing.assert_allclose(diag_full, diag_mult)

    # test things also works without passing optional kwargs
    sum_sim.model = m_test
    d_mult2 = sum_sim.dpred()
    np.testing.assert_allclose(d_mult, d_mult2)

    jvec_mult2 = sum_sim.Jvec(m_test, u)
    np.testing.assert_allclose(jvec_mult, jvec_mult2)

    jtvec_mult2 = sum_sim.Jtvec(m_test, v)
    np.testing.assert_allclose(jtvec_mult, jtvec_mult2)

    sum_sim._jtjdiag = None
    diag_mult2 = sum_sim.getJtJdiag(m_test)
    np.testing.assert_allclose(diag_mult, diag_mult2)


def test_repeat_sim_correctness():
    # meta sim is tested for correctness
    # so can test the repeat against the meta sim
    mesh = TensorMesh([8, 8, 8], origin="CCN")

    rx_locs = np.mgrid[-0.25:0.25:5j, -0.25:0.25:5j, 0:1:1j].reshape(3, -1).T
    rx = gravity.Point(rx_locs, components=["gz"])
    survey = gravity.Survey(gravity.SourceField(rx))
    sim = gravity.Simulation3DIntegral(
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

    mappings = []
    simulations = []
    eye = sp.eye(mesh.n_cells, mesh.n_cells)
    for t in sim_ts:
        ave_time = time_mesh.get_interpolation_matrix(
            [
                t,
            ]
        )
        ave_full = sp.kron(ave_time, eye, format="csr")
        mappings.append(maps.LinearMap(ave_full))
        simulations.append(
            gravity.Simulation3DIntegral(
                mesh, survey=survey, rhoMap=maps.IdentityMap(), n_processes=1
            )
        )

    multi_sim = MetaSimulation(simulations, mappings)
    repeat_sim = RepeatedSimulation(sim, mappings)

    model = np.random.rand(time_mesh.n_cells, mesh.n_cells).reshape(-1)

    # test field things
    f_full = multi_sim.fields(model)
    f_mult = repeat_sim.fields(model)
    np.testing.assert_equal(np.c_[f_full], np.c_[f_mult])

    d_full = multi_sim.dpred(model, f_full)
    d_repeat = repeat_sim.dpred(model, f_mult)
    np.testing.assert_equal(d_full, d_repeat)

    # test Jvec
    u = np.random.rand(len(model))
    jvec_full = multi_sim.Jvec(model, u, f=f_full)
    jvec_mult = repeat_sim.Jvec(model, u, f=f_mult)
    np.testing.assert_allclose(jvec_full, jvec_mult)

    # test Jtvec
    v = np.random.rand(len(sim_ts) * survey.nD)
    jtvec_full = multi_sim.Jtvec(model, v, f=f_full)
    jtvec_mult = repeat_sim.Jtvec(model, v, f=f_mult)
    np.testing.assert_allclose(jtvec_full, jtvec_mult)

    # test get diag
    diag_full = multi_sim.getJtJdiag(model, f=f_full)
    diag_mult = repeat_sim.getJtJdiag(model, f=f_mult)
    np.testing.assert_allclose(diag_full, diag_mult)


def test_multi_errors():
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
        MetaSimulation(sims[:-1], mappings)

    # mappings have incompatible input lengths:
    mappings[0] = maps.Projection(mesh.n_cells + 1, np.arange(mesh.n_cells) + 1)
    with pytest.raises(ValueError):
        MetaSimulation(sims, mappings)

    # incompatible mapping and simulation
    mappings[0] = maps.Projection(mesh.n_cells, [0, 1, 3, 5, 10])
    with pytest.raises(ValueError):
        MetaSimulation(sims, mappings)


def test_sum_errors():
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
            mesh_bot, survey=survey1, rhoMap=maps.IdentityMap(mesh_bot), n_processes=1
        ),
        gravity.Simulation3DIntegral(
            mesh_top, survey=survey2, rhoMap=maps.IdentityMap(mesh_top), n_processes=1
        ),
    ]

    # Test simulations with different numbers of data.
    with pytest.raises(ValueError):
        SumMetaSimulation(sims, mappings)


def test_repeat_errors():
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
        RepeatedSimulation(sim, mappings)

    # incompatible mappings and simulations
    mappings[0] = maps.Projection(mesh.n_cells, [0, 1, 3, 5, 10])
    with pytest.raises(ValueError):
        RepeatedSimulation(sim, mappings)


def test_cache_clear_on_model_clear():
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

    multi_sim = MetaSimulation(sims, mappings)

    assert multi_sim.model is None
    for sim in multi_sim.simulations:
        assert sim.model is None

    # create fields to do some caching operations
    multi_sim.fields(m_test)
    assert multi_sim.model is not None
    for sim in multi_sim.simulations:
        assert sim._Me_Sigma is not None

    # then set to None to make sure that works (and it clears things)
    multi_sim.model = None
    assert multi_sim.model is None
    for sim in multi_sim.simulations:
        assert sim.model is None
        assert not hasattr(sim, "_Me_Sigma")
