from .....electromagnetics.static.resistivity.simulation_2d import (
    BaseDCSimulation2D as Sim,
)
from .simulation import dask_getJtJdiag, dask_Jvec, dask_Jtvec
import dask.array as da
import numpy as np
import zarr
import numcodecs

numcodecs.blosc.use_threads = False

Sim.sensitivity_path = "./sensitivity/"

Sim.getJtJdiag = dask_getJtJdiag
Sim.Jvec = dask_Jvec
Sim.Jtvec = dask_Jtvec
Sim.clean_on_model_update = ["_Jmatrix", "_jtjdiag"]


def dask_fields(self, m=None, return_Ainv=False):
    if m is not None:
        self.model = m

    kys = self._quad_points
    f = self.fieldsPair(self)
    f._quad_weights = self._quad_weights

    Ainv = {}
    for iky, ky in enumerate(kys):
        A = self.getA(ky)
        Ainv[iky] = self.solver(A, **self.solver_opts)

        RHS = self.getRHS(ky)
        f[:, self._solutionType, iky] = Ainv[iky] * RHS

    if return_Ainv:
        self.Ainv = Ainv

    return f


Sim.fields = dask_fields


def compute_J(self, f=None):
    kys = self._quad_points
    weights = self._quad_weights

    if f is None:
        f = self.fields(self.model, return_Ainv=True)

    m_size = self.model.size
    row_chunks = int(
        np.ceil(
            float(self.survey.nD)
            / np.ceil(
                float(m_size)
                * self.survey.nD
                * len(kys)
                * 8.0
                * 1e-6
                / self.max_chunk_size
            )
        )
    )
    if self.store_sensitivities == "disk":
        Jmatrix = zarr.open(
            self.sensitivity_path + "J.zarr",
            mode="w",
            shape=(self.survey.nD, m_size),
            chunks=(row_chunks, m_size),
        )
    else:
        Jmatrix = np.zeros((self.survey.nD, m_size), dtype=np.float32)

    blocks = []
    count = 0

    for i_src, source in enumerate(self.survey.source_list):
        for rx in source.receiver_list:

            if rx.orientation is not None:
                projected_grid = f._GLoc(rx.projField) + rx.orientation
            else:
                projected_grid = f._GLoc(rx.projField)

            PTv = rx.getP(self.mesh, projected_grid).toarray().T

            for dd in range(int(np.ceil(PTv.shape[1] / row_chunks))):
                start, end = dd * row_chunks, np.min(
                    [(dd + 1) * row_chunks, PTv.shape[1]]
                )
                block = np.zeros((end - start, m_size))
                for iky, ky in enumerate(kys):

                    u_ky = f[:, self._solutionType, iky]
                    u_source = u_ky[:, i_src]
                    ATinvdf_duT = self.Ainv[iky] * PTv[:, start:end]
                    dA_dmT = self.getADeriv(ky, u_source, ATinvdf_duT, adjoint=True)
                    du_dmT = -weights[iky] * dA_dmT
                    block += du_dmT.T.reshape((-1, m_size))

                if len(blocks) == 0:
                    blocks = block
                else:
                    blocks = np.vstack([blocks, block])

                while blocks.shape[0] >= row_chunks:
                    if self.store_sensitivities == "disk":
                        Jmatrix.set_orthogonal_selection(
                            (np.arange(count, count + row_chunks), slice(None)),
                            blocks[:row_chunks, :].astype(np.float32),
                        )
                    else:
                        Jmatrix[count : count + row_chunks, :] = blocks[
                            :row_chunks, :
                        ].astype(np.float32)

                    blocks = blocks[row_chunks:, :].astype(np.float32)
                    count += row_chunks

                del ATinvdf_duT, dA_dmT, block

    if len(blocks) != 0:
        if self.store_sensitivities == "disk":
            Jmatrix.set_orthogonal_selection(
                (np.arange(count, self.survey.nD), slice(None)),
                blocks.astype(np.float32),
            )
        else:
            Jmatrix[count : self.survey.nD, :] = blocks.astype(np.float32)

    for iky, _ in enumerate(kys):
        self.Ainv[iky].clean()

    if self.store_sensitivities == "disk":
        del Jmatrix
        return da.from_zarr(self.sensitivity_path + "J.zarr")
    else:
        return Jmatrix


Sim.compute_J = compute_J


def dask_dpred(self, m=None, f=None, compute_J=False):
    r"""
    dpred(m, f=None)
    Create the projected data from a model.
    The fields, f, (if provided) will be used for the predicted data
    instead of recalculating the fields (which may be expensive!).

    .. math::

        d_\\text{pred} = P(f(m))

    Where P is a projection of the fields onto the data space.
    """
    weights = self._quad_weights
    if self._mini_survey is not None:
        survey = self._mini_survey
    else:
        survey = self.survey

    if survey is None:
        raise AttributeError(
            "The survey has not yet been set and is required to compute "
            "data. Please set the survey for the simulation: "
            "simulation.survey = survey"
        )

    if f is None:
        if m is None:
            m = self.model
        f = self.fields(m, return_Ainv=compute_J)

    temp = np.empty(survey.nD)
    count = 0
    for src in survey.source_list:
        for rx in src.receiver_list:
            d = rx.eval(src, self.mesh, f).dot(weights)
            temp[count : count + len(d)] = d
            count += len(d)

    if compute_J:
        Jmatrix = self.compute_J(f=f)
        return self._mini_survey_data(temp), Jmatrix

    return self._mini_survey_data(temp)


Sim.dpred = dask_dpred


def dask_getSourceTerm(self, _):
    """
    Evaluates the sources, and puts them in matrix form
    :rtype: tuple
    :return: q (nC or nN, nSrc)
    """

    if getattr(self, "_q", None) is None:

        if self._mini_survey is not None:
            Srcs = self._mini_survey.source_list
        else:
            Srcs = self.survey.source_list

        if self._formulation == "EB":
            n = self.mesh.nN
            # return NotImplementedError

        elif self._formulation == "HJ":
            n = self.mesh.nC

        q = np.zeros((n, len(Srcs)), order="F")

        for i, source in enumerate(Srcs):
            q[:, i] = source.eval(self)

        self._q = q

    return self._q


Sim.getSourceTerm = dask_getSourceTerm
