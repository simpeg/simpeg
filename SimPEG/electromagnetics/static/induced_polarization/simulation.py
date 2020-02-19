import numpy as np
import dask
import dask.array as da
import multiprocessing
import scipy.sparse as sp
import sys
import shutil

from .... import props
from ....data import Data
from ....utils import mkvc, sdiag, Zero
from ...base import BaseEMSimulation

from ..resistivity.fields import FieldsDC, Fields_CC, Fields_N
from ..resistivity import Problem3D_CC as BaseProblem3D_CC
from ..resistivity import Problem3D_N as BaseProblem3D_N
import os
import dask
import dask.array as da
from scipy.sparse import csr_matrix as csr
from pyMKL import mkl_set_num_threads
import zarr
import time
import sparse
from dask.delayed import Delayed
# from .survey import Survey


class BaseIPSimulation(BaseEMSimulation):

    sigma = props.PhysicalProperty(
        "Electrical conductivity (S/m)"
    )

    rho = props.PhysicalProperty(
        "Electrical resistivity (Ohm m)"
    )

    props.Reciprocal(sigma, rho)

    eta, etaMap, etaDeriv = props.Invertible(
        "Electrical Chargeability"
    )

    # surveyPair = Survey
    fieldsPair = FieldsDC
    Ainv = None
    _f = None
    storeJ = False
    _Jmatrix = None
    gtgdiag = None
    sign = None
    data_type = 'volt'
    _pred = None
    Jpath = "./sensitivityip/"
    gtgdiag = None
    n_cpu = int(multiprocessing.cpu_count())
    maxRAM = 2
    max_chunk_size = 128

    @dask.delayed(pure=True)
    def fields(self, m=None, calcJ=True):
        if self.verbose is True:
            print(">> Compute fields")

        mkl_set_num_threads(self.n_cpu)

        if m is not None:
            self.model = m
            self._Jmatrix = None

        f = self.fieldsPair(self)
        A = self.getA()
        self.Ainv = self.solver(A, **self.solver_opts)
        RHS = self.getRHS()
        Srcs = self.survey.source_list
        f[Srcs, self._solutionType] = self.Ainv * RHS

        # Compute DC voltage
        if self.data_type == 'apparent_chargeability':
            if self.verbose is True:
                print(">> Data type is apparaent chargeability")
            for src in self.survey.source_list:
                for rx in src.receiver_list:
                    rx._dc_voltage = rx.eval(src, self.mesh, self._f)
                    rx.data_type = self.data_type
                    rx._Ps = {}

        self._pred = self.forward(m, f=f)
        self._f = f

        # if not self.storeJ:
        #     self.Ainv.clean()

        return f

    def dpred(self, m=None, f=None):
        """
            Predicted data.
            .. math::
                d_\\text{pred} = Pf(m)
        """
        if f is None:
            f = self.fields(m)
        if isinstance(f, Delayed):
            f = f.compute()

        return self._pred

    def getJtJdiag(self, m, W=None):
        """
            Return the diagonal of JtJ
        """

        if (self.gtgdiag is None):

            # Need to check if multiplying weights makes sense
            if W is None:
                self.gtgdiag = da.sum((self.getJ(m))**2., 0).compute()
            else:
                self.gtgdiag = da.sum((self.getJ(m))**2., 0).compute()
                # WJ = da.from_delayed(
                #         dask.delayed(csr.dot)(W, self.getJ(m)),
                #         shape=self.getJ(m).shape,
                #         dtype=float
                # )
                # self.gtgdiag = da.sum(WJ**2., 0).compute()

        return self.gtgdiag

    def getJ(self, m, f=None):
        """
            Generate Full sensitivity matrix
        """
        if self._Jmatrix is not None:
            return self._Jmatrix

        self.model = m
        if f is None:
            f = self.fields(m).compute()

        if self.verbose:
            print("Calculating J and storing")

        if os.path.exists(self.Jpath):
            shutil.rmtree(self.Jpath, ignore_errors=True)

            # Wait for the system to clear out the directory
            while os.path.exists(self.Jpath):
                pass
        # start of IP J
        # This is for forming full sensitivity matrix
        nD = self.survey.nD
        nC = m.shape[0]

        # print('DASK: Chunking using parameters')
        nChunks_col = 1
        nChunks_row = 1
        rowChunk = int(np.ceil(nD / nChunks_row))
        colChunk = int(np.ceil(nC / nChunks_col))
        chunk_size = rowChunk * colChunk * 8 * 1e-6  # in Mb

        # Add more chunks until memory falls below target
        while chunk_size >= self.max_chunk_size:

            if rowChunk > colChunk:
                nChunks_row += 1
            else:
                nChunks_col += 1

            rowChunk = int(np.ceil(nD / nChunks_row))
            colChunk = int(np.ceil(nC / nChunks_col))
            chunk_size = rowChunk * colChunk * 8 * 1e-6  # in Mb
        count = 0
        for source in self.survey.source_list:
            u_source = f[source, self._solutionType].copy()
            for rx in source.receiver_list:
                PT = rx.getP(self.mesh, rx.projGLoc(f)).T
                df_duT = PT.toarray()
                # Find a block of receivers
                n_block_col = int(np.ceil(df_duT.size *
                                          8 * 1e-9 / self.maxRAM))

                n_col = int(np.ceil(df_duT.shape[1] / n_block_col))

                nrows = int(self.model.size /
                            np.ceil(self.model.size *
                                    n_col * 8 * 1e-6 /
                                    self.max_chunk_size))
                ind = 0
                for col in range(n_block_col):
                    ATinvdf_duT = da.asarray(self.Ainv * np.asarray(df_duT[:, ind:ind + n_col])).rechunk((nrows, n_col))
                    dA_dmT = self.getADeriv(
                        u_source, ATinvdf_duT, adjoint=True)
                    # du_dmT = -da.from_delayed(dask.delayed(dA_dmT), shape=(self.model.size, n_col), dtype=float)
                    if n_col > 1:
                        du_dmT = da.from_delayed(dask.delayed(-dA_dmT),
                                                 shape=(self.model.size, n_col),
                                                 dtype=float)
                    else:
                        du_dmT = da.from_delayed(dask.delayed(-dA_dmT),
                                                 shape=(self.model.size,),
                                                 dtype=float)
                    blockName = self.Jpath + "J" + str(count) + ".zarr"

                    da.to_zarr((du_dmT.T).rechunk('auto'), blockName)
                    del ATinvdf_duT
                    count += 1
                    ind += n_col

        dask_arrays = []
        for ii in range(count):
            blockName = self.Jpath + "J" + str(ii) + ".zarr"
            J = da.from_zarr(blockName)
            # Stack all the source blocks in one big zarr
            dask_arrays.append(J)

        self._Jmatrix = da.vstack(dask_arrays).rechunk((rowChunk, colChunk))
        self.Ainv.clean()

        return self._Jmatrix

    # @profile
    def Jvec(self, m, v, f=None):

        self.model = m

        if f is None:
            f = self.fields(m)

        if isinstance(f, Delayed):
            f = f.compute()

        # When sensitivity matrix J is stored
        if self.storeJ:
            J = self.getJ(m, f=f).compute()
            Jv = mkvc(np.dot(J, v))
            return self.sign * Jv

        else:
            Jv = []

            for src in self.survey.source_list:
                # solution vector
                u_src = f[src, self._solutionType]
                dA_dm_v = self.getADeriv(u_src.flatten(), v, adjoint=False)
                dRHS_dm_v = self.getRHSDeriv(src, v)
                du_dm_v = self.Ainv * (- dA_dm_v + dRHS_dm_v)

                for rx in src.receiver_list:
                    df_dmFun = getattr(f, '_{0!s}Deriv'.format(rx.projField), None)
                    df_dm_v = df_dmFun(src, du_dm_v, v, adjoint=False)
                    Jv.append(rx.evalDeriv(src, self.mesh, f, df_dm_v))

            # Conductivity (d u / d log sigma) - EB form
            # Resistivity (d u / d log rho) - HJ form
            return self.sign*np.hstack(Jv)

    def forward(self, m, f=None):
        return self.Jvec(m, m, f=f)

    def Jtvec(self, m, v, f=None):
        """
            Compute adjoint sensitivity matrix (J^T) and vector (v) product.

        """
        if f is None:
            f = self.fields(m)

        if isinstance(f, Delayed):
            f = f.compute()

        # When sensitivity matrix J is stored
        if self.storeJ:
            J = self.getJ(m, f=f)
            Jtv = mkvc(da.dot(J.T, v).compute())
            return self.sign * Jtv

        else:
            self.model = m

            if f is None:
                f = self.fields(m).compute()
            return self._Jtvec(m, v=v, f=f)

    def _Jtvec(self, m, v=None, f=None):
        """
            Compute adjoint sensitivity matrix (J^T) and vector (v) product.
            Full J matrix can be computed by inputing v=None
        """

        if v is not None:
            # Ensure v is a data object.
            if not isinstance(v, Data):
                v = Data(self.survey, v)
            Jtv = np.zeros(m.size)
        else:
            # This is for forming full sensitivity matrix
            Jtv = np.zeros((self.model.size, self.survey.nD), order='F')
            istrt = int(0)
            iend = int(0)

        for isrc, src in enumerate(self.survey.source_list):
            u_src = f[src, self._solutionType]
            # if self.storeJ:
            #     # TODO: use logging package
            #     sys.stdout.write(("\r %d / %d") % (isrc+1, self.survey.nSrc))
            #     sys.stdout.flush()

            for rx in src.receiver_list:
                if v is not None:
                    PTv = rx.evalDeriv(
                        src, self.mesh, f, v[src, rx], adjoint=True
                    )  # wrt f, need possibility wrt m
                    df_duTFun = getattr(
                        f, '_{0!s}Deriv'.format(rx.projField), None
                    )
                    df_duT, df_dmT = df_duTFun(src, None, PTv, adjoint=True)
                    ATinvdf_duT = self.Ainv * df_duT
                    dA_dmT = self.getADeriv(
                        u_src.flatten(), ATinvdf_duT, adjoint=True
                    )
                    # dA_dmT = da.from_delayed(dA_dmT, shape=self.model.shape, dtype=float)
                    dRHS_dmT = self.getRHSDeriv(src, ATinvdf_duT, adjoint=True)
                    # dRHS_dmT = da.from_delayed(dRHS_dmT, shape=self.model.shape, dtype=float)
                    du_dmT = (-dA_dmT + dRHS_dmT)
                    Jtv += (df_dmT + du_dmT).astype(float)
                else:
                    P = rx.getP(self.mesh, rx.projGLoc(f)).toarray()
                    ATinvdf_duT = self.Ainv * (P.T)
                    dA_dmT = self.getADeriv(
                        u_src, ATinvdf_duT, adjoint=True
                    )

                    iend = istrt + rx.nD
                    if rx.nD == 1:
                        Jtv[:, istrt] = -dA_dmT
                    else:
                        Jtv[:, istrt:iend] = -dA_dmT
                    istrt += rx.nD

        # Conductivity ((d u / d log sigma).T) - EB form
        # Resistivity ((d u / d log rho).T) - HJ form

        if v is not None:
            return self.sign * mkvc(Jtv)
        else:
            return Jtv
        return

    def getSourceTerm(self):
        """
        takes concept of source and turns it into a matrix
        """
        """
        Evaluates the sources, and puts them in matrix form

        :rtype: (numpy.ndarray, numpy.ndarray)
        :return: q (nC or nN, nSrc)
        """

        Srcs = self.survey.source_list

        if self._formulation == 'EB':
            n = self.mesh.nN
            # return NotImplementedError

        elif self._formulation == 'HJ':
            n = self.mesh.nC

        q = np.zeros((n, len(Srcs)))

        for i, src in enumerate(Srcs):
            q[:, i] = src.eval(self)
        return q

    def delete_these_for_sensitivity(self):
        del self._Jmatrix, self._MfRhoI, self._MeSigma

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        return toDelete

    @property
    def MfRhoDerivMat(self):
        """
        Derivative of MfRho with respect to the model
        """
        if getattr(self, '_MfRhoDerivMat', None) is None:
            drho_dlogrho = sdiag(self.rho)*self.etaDeriv
            self._MfRhoDerivMat = self.mesh.getFaceInnerProductDeriv(
                np.ones(self.mesh.nC)
            )(np.ones(self.mesh.nF)) * drho_dlogrho
        return self._MfRhoDerivMat

    def MfRhoIDeriv(self, u, v, adjoint=False):
        """
            Derivative of :code:`MfRhoI` with respect to the model.
        """
        dMfRhoI_dI = -self.MfRhoI**2
        if self.storeInnerProduct:
            if adjoint:
                return self.MfRhoDerivMat.T * (
                    sdiag(u) * (dMfRhoI_dI.T * v)
                )
            else:
                return dMfRhoI_dI * (sdiag(u) * (self.MfRhoDerivMat*v))
        else:
            dMf_drho = self.mesh.getFaceInnerProductDeriv(self.rho)(u)
            drho_dlogrho = sdiag(self.rho)*self.etaDeriv
            if adjoint:
                return drho_dlogrho.T * (dMf_drho.T * (dMfRhoI_dI.T*v))
            else:
                return dMfRhoI_dI * (dMf_drho * (drho_dlogrho*v))

    MfRhoIDerivDask = MfRhoIDeriv

    @property
    def MeSigmaDerivMat(self):
        """
        Derivative of MeSigma with respect to the model
        """

        if getattr(self, '_MeSigmaDerivMat', None) is None:
            dsigma_dlogsigma = sdiag(self.sigma)*self.etaDeriv
            self._MeSigmaDerivMat = self.mesh.getEdgeInnerProductDeriv(
                np.ones(self.mesh.nC)
            )(np.ones(self.mesh.nE)) * dsigma_dlogsigma
        return self._MeSigmaDerivMat

    # TODO: This should take a vector
    def MeSigmaDeriv(self, u, v, adjoint=False):
        """
        Derivative of MeSigma with respect to the model times a vector (u)
        """
        if self.storeInnerProduct:
            if adjoint:
                return self.MeSigmaDerivMat.T * (sdiag(u)*v)
            else:
                return sdiag(u)*(self.MeSigmaDerivMat * v)
        else:
            dsigma_dlogsigma = sdiag(self.sigma)*self.etaDeriv
            if adjoint:
                return (
                    dsigma_dlogsigma.T * (
                        self.mesh.getEdgeInnerProductDeriv(self.sigma)(u).T * v
                    )
                )
            else:
                return (
                    self.mesh.getEdgeInnerProductDeriv(self.sigma)(u) *
                    (dsigma_dlogsigma * v)
                )


class Problem3D_CC(BaseIPSimulation, BaseProblem3D_CC):

    _solutionType = 'phiSolution'
    _formulation = 'HJ'  # CC potentials means J is on faces
    fieldsPair = Fields_CC
    sign = 1.
    bc_type = 'Dirichlet'

    def __init__(self, mesh, **kwargs):
        super(Problem3D_CC, self).__init__(mesh, **kwargs)
        self.setBC()


class Problem3D_N(BaseIPSimulation, BaseProblem3D_N):

    _solutionType = 'phiSolution'
    _formulation = 'EB'  # N potentials means B is on faces
    fieldsPair = Fields_N
    sign = -1.

    def __init__(self, mesh, **kwargs):
        super(Problem3D_N, self).__init__(mesh, **kwargs)



