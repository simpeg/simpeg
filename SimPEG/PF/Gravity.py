from __future__ import print_function
from SimPEG import Problem, Mesh
from SimPEG import Utils
from SimPEG.Utils import mkvc, matutils, sdiag
from SimPEG import Props
import scipy as sp
import scipy.constants as constants
import os
import time
import numpy as np
import dask
import dask.array as da
from scipy.sparse import csr_matrix as csr
from dask.diagnostics import ProgressBar
import multiprocessing
import shutil

class GravityIntegral(Problem.LinearProblem):

    rho, rhoMap, rhoDeriv = Props.Invertible(
        "Specific density (g/cc)",
        default=1.
    )

    # surveyPair = Survey.LinearSurvey
    forwardOnly = False  # Is TRUE, forward matrix not stored to memory
    actInd = None  #: Active cell indices provided
    rxType = 'z'
    silent = False
    equiSourceLayer = False
    memory_saving_mode = False
    parallelized = "dask"
    n_cpu = None
    n_chunks = 1
    progressIndex = -1
    gtgdiag = None
    Jpath = "./sensitivity.zarr"
    maxRAM = 8  # Maximum memory usage
    verbose = True

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

        if getattr(self, 'actInd', None) is not None:

            if self.actInd.dtype == 'bool':
                inds = np.asarray([inds for inds,
                                  elem in enumerate(self.actInd, 1)
                                  if elem], dtype=int) - 1
            else:
                inds = self.actInd

        else:

            inds = np.asarray(range(self.mesh.nC))

        self.nC = len(inds)

        # Create active cell projector
        P = sp.sparse.csr_matrix(
            (np.ones(self.nC), (inds, range(self.nC))),
            shape=(self.mesh.nC, self.nC)
        )

        # Create vectors of nodal location
        # (lower and upper corners for each cell)
        bsw = (self.mesh.gridCC - self.mesh.h_gridded/2.)
        tne = (self.mesh.gridCC + self.mesh.h_gridded/2.)

        xn1, xn2 = bsw[:, 0], tne[:, 0]
        yn1, yn2 = bsw[:, 1], tne[:, 1]

        self.Yn = P.T*np.c_[mkvc(yn1), mkvc(yn2)]
        self.Xn = P.T*np.c_[mkvc(xn1), mkvc(xn2)]

        if self.mesh.dim > 2:
            zn1, zn2 = bsw[:, 2], tne[:, 2]
            self.Zn = P.T*np.c_[mkvc(zn1), mkvc(zn2)]

    def fields(self, m):
        # self.model = self.rhoMap*m

        if self.forwardOnly:

            # Compute the linear operation without forming the full dense G
            return np.array(self.Intrgl_Fwr_Op(m=m), dtype='float')

        else:
            # fields = da.dot(self.G, m)

            return da.dot(self.G, self.rhoMap*m) #np.array(fields, dtype='float')

    def modelMap(self):
        """
            Call for general mapping of the problem
        """
        return self.rhoMap

    def getJtJdiag(self, m, W=None):
        """
            Return the diagonal of JtJ
        """

        dmudm = self.rhoMap.deriv(m)
        self.model = m

        if self.gtgdiag is None:

            if W is None:
                w = np.ones(self.G.shape[1])
            else:
                w = W.diagonal()

            self.gtgdiag = da.sum(self.G**2., 0).compute()

            # for ii in range(self.G.shape[0]):

            #     self.gtgdiag += (w[ii]*self.G[ii, :]*dmudm)**2.

        return mkvc(np.sum((sdiag(mkvc(self.gtgdiag)**0.5) * dmudm).power(2.), axis=0))

    def getJ(self, m, f=None):
        """
            Sensitivity matrix
        """

        dmudm = self.rhoMap.deriv(m)

        return da.dot(self.G, dmudm)

    def Jvec(self, m, v, f=None):
        dmudm = self.rhoMap.deriv(m)

        # vec = da.dot(self.G, (dmudm*v).astype(np.float32))
        vec = dask.delayed(csr.dot)(dmudm, v)
        dmudm_v = da.from_delayed(vec, dtype=float, shape=[dmudm.shape[0]])

        return da.dot(self.G, dmudm_v)

    def Jtvec(self, m, v, f=None):

        dmudm = self.rhoMap.deriv(m)

        jt_v = da.dot(self.G.T, v)

        dmudm_jt_v = dask.delayed(csr.dot)(dmudm.T, jt_v)

        return da.from_delayed(dmudm_jt_v, dtype=float, shape=[dmudm.shape[1]])

    @property
    def G(self):
        if not self.ispaired:
            raise Exception('Need to pair!')

        if getattr(self, '_G', None) is None:

            self._G = self.Intrgl_Fwr_Op()

        return self._G

    def Intrgl_Fwr_Op(self, m=None, rxType='z'):

        """

        Gravity forward operator in integral form

        flag        = 'z' | 'xyz'

        Return
        _G        = Linear forward modeling operation

        Created on March, 15th 2016

        @author: dominiquef

         """
        if m is not None:
            self.model = self.rhoMap*m
        self.rxLoc = self.survey.srcField.rxList[0].locs
        self.nD = int(self.rxLoc.shape[0])

        # if self.n_cpu is None:
        #     self.n_cpu = multiprocessing.cpu_count()

        # Switch to determine if the process has to be run in parallel
        job = Forward(
                rxLoc=self.rxLoc, Xn=self.Xn, Yn=self.Yn, Zn=self.Zn,
                n_cpu=self.n_cpu, forwardOnly=self.forwardOnly,
                model=self.model, rxType=self.rxType,
                parallelized=self.parallelized, n_chunks=self.n_chunks,
                verbose=self.verbose, Jpath=self.Jpath, maxRAM=self.maxRAM
                )

        G = job.calculate()

        return G


class Forward(object):
    """
        Add docstring once it works
    """

    progressIndex = -1
    parallelized = "dask"
    rxLoc = None
    Xn, Yn, Zn = None, None, None
    n_cpu = None
    n_chunks = None
    forwardOnly = False
    model = None
    rxType = 'z'
    verbose = True
    maxRAM = 1.
    Jpath = "./sensitivity.zarr"

    def __init__(self, **kwargs):
        super(Forward, self).__init__()
        Utils.setKwargs(self, **kwargs)

    def calculate(self):

        self.nD = self.rxLoc.shape[0]
        self.nC = self.Xn.shape[0]

        if self.n_cpu is None:
            self.n_cpu = int(multiprocessing.cpu_count())

        # Set this early so we can get a better memory estimate for dask chunking
        if self.rxType == 'xyz':
            nDataComps = 3
        else:
            nDataComps = 1

        if self.parallelized:

            assert self.parallelized in ["dask", None], (
                "'parallelization' must be 'dask', or None"
                "Value provided -> "
                "{}".format(
                    self.parallelized)

            )

            if self.parallelized == "dask":

                # Chunking only required for dask
                nChunks = self.n_chunks # Number of chunks
                rowChunk, colChunk = int(np.ceil(nDataComps*self.nD/nChunks)), int(np.ceil(self.nC/nChunks)) # Chunk sizes
                totRAM = rowChunk*colChunk*8*self.n_cpu*1e-9
                # Ensure total problem size fits in RAM, and avoid 2GB size limit on dask chunks
                while (totRAM > self.maxRAM) or (totRAM/self.n_cpu) >= 0.128:
#                    print("Dask:", self.n_cpu, nChunks, rowChunk, colChunk, totRAM, self.maxRAM)
                    nChunks += 1
                    rowChunk, colChunk = int(np.ceil(nDataComps*self.nD/nChunks)), int(np.ceil(self.nC/nChunks)) # Chunk sizes
                    totRAM = rowChunk*colChunk*8*self.n_cpu*1e-9

                print("Dask:")
                print("n_cpu: ", self.n_cpu)
                print("n_chunks: ", nChunks)
                print("Chunk sizes: ", rowChunk, colChunk)
                print("RAM/chunk: ", totRAM/self.n_cpu)
                print("Total RAM (x n_cpu): ", totRAM)

                row = dask.delayed(self.calcTrow, pure=True)

                makeRows = [row(self.rxLoc[ii, :]) for ii in range(self.nD)]

                buildMat = [da.from_delayed(makeRow, dtype=float, shape=(nDataComps,  self.nC)) for makeRow in makeRows]

                stack = da.vstack(buildMat)

                if self.forwardOnly:

                    with ProgressBar():
                        pred = da.dot(stack, self.model).compute()

                    return pred

                else:

                    if os.path.exists(self.Jpath):

                        G = da.from_zarr(self.Jpath)

                        if np.all(np.r_[
                                np.any(np.r_[G.chunks[0]] == rowChunk),
                                np.any(np.r_[G.chunks[1]] == colChunk),
                                np.r_[G.shape] == np.r_[nDataComps*self.nD,  self.nC]]):
                            # Check that loaded G matches supplied data and mesh
                            print("Zarr file detected with same shape and chunksize ... re-loading")
                            return G

                        else:
                            del G
                            shutil.rmtree(self.Jpath)
                            print("Zarr file detected with wrong shape and chunksize ... over-writting")
                        # TO-DO: should add

                    # TO-DO: Find a way to create in
                    # chunks instead
                    stack = stack.rechunk((rowChunk, colChunk))

                    with ProgressBar():
                        print("Saving G to zarr: " + self.Jpath)
                        da.to_zarr(stack, self.Jpath)

                    G = da.from_zarr(self.Jpath)

            # elif self.parallelized == "multiprocessing":

            #     totRAM = nDataComps*self.nD*self.nC*8*1e-9
            #     print("Multiprocessing:", self.n_cpu, self.nD, self.nC, totRAM, self.maxRAM)

            #     pool = multiprocessing.Pool(self.n_cpu)

            #     result = pool.map(self.calcTrow, [self.rxLoc[ii, :] for ii in range(self.nD)])
            #     pool.close()
            #     pool.join()

            #     G = np.vstack(result)

        else:

            result = []
            for ii in range(self.nD):
                result += [self.calcTrow(self.rxLoc[ii, :])]
                self.progress(ii, self.nD)

            G = np.vstack(result)

        return G

    def calcTrow(self, xyzLoc):
        """
        Load in the active nodes of a tensor mesh and computes the gravity tensor
        for a given observation location xyzLoc[obsx, obsy, obsz]

        INPUT:
        Xn, Yn, Zn: Node location matrix for the lower and upper most corners of
                    all cells in the mesh shape[nC,2]
        M
        OUTPUT:
        Tx = [Txx Txy Txz]
        Ty = [Tyx Tyy Tyz]
        Tz = [Tzx Tzy Tzz]

        where each elements have dimension 1-by-nC.
        Only the upper half 5 elements have to be computed since symetric.
        Currently done as for-loops but will eventually be changed to vector
        indexing, once the topography has been figured out.

        """

        NewtG = constants.G*1e+8  # Convertion from mGal (1e-5) and g/cc (1e-3)
        eps = 1e-8  # add a small value to the locations to avoid

        # Pre-allocate space for 1D array
        row = np.zeros((1, self.Xn.shape[0]))

        dz = xyzLoc[2] - self.Zn

        dy = self.Yn - xyzLoc[1]

        dx = self.Xn - xyzLoc[0]

        # Compute contribution from each corners
        for aa in range(2):
            for bb in range(2):
                for cc in range(2):

                    r = (
                            mkvc(dx[:, aa]) ** 2 +
                            mkvc(dy[:, bb]) ** 2 +
                            mkvc(dz[:, cc]) ** 2
                        ) ** (0.50)

                    if self.rxType == 'x':
                        row -= NewtG * (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dy[:, bb] * np.log(dz[:, cc] + r + eps) +
                            dz[:, cc] * np.log(dy[:, bb] + r + eps) -
                            dx[:, aa] * np.arctan(dy[:, bb] * dz[:, cc] /
                                                  (dx[:, aa] * r + eps)))

                    elif self.rxType == 'y':
                        row -= NewtG * (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dx[:, aa] * np.log(dz[:, cc] + r + eps) +
                            dz[:, cc] * np.log(dx[:, aa] + r + eps) -
                            dy[:, bb] * np.arctan(dx[:, aa] * dz[:, cc] /
                                                  (dy[:, bb] * r + eps)))

                    else:
                        row -= NewtG * (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dx[:, aa] * np.log(dy[:, bb] + r + eps) +
                            dy[:, bb] * np.log(dx[:, aa] + r + eps) -
                            dz[:, cc] * np.arctan(dx[:, aa] * dy[:, bb] /
                                                  (dz[:, cc] * r + eps)))

        return row

    def progress(self, ind, total):
        """
        progress(ind,prog,final)

        Function measuring the progress of a process and print to screen the %.
        Useful to estimate the remaining runtime of a large problem.

        Created on Dec, 20th 2015

        @author: dominiquef
        """
        arg = np.floor(ind/total*10.)
        if arg > self.progressIndex:
            print("Done " + str(arg*10) + " %")
            self.progressIndex = arg


class Problem3D_Diff(Problem.BaseProblem):
    """
        Gravity in differential equations!
    """

    _depreciate_main_map = 'rhoMap'

    rho, rhoMap, rhoDeriv = Props.Invertible(
        "Specific density (g/cc)",
        default=1.
    )

    solver = None

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

        self.mesh.setCellGradBC('dirichlet')

        self._Div = self.mesh.cellGrad

    @property
    def MfI(self): return self._MfI

    @property
    def Mfi(self): return self._Mfi

    def makeMassMatrices(self, m):
        self.model = m
        self._Mfi = self.mesh.getFaceInnerProduct()
        self._MfI = Utils.sdiag(1. / self._Mfi.diagonal())

    def getRHS(self, m):
        """


        """

        Mc = Utils.sdiag(self.mesh.vol)

        self.model = m
        rho = self.rho

        return Mc * rho

    def getA(self, m):
        """
        GetA creates and returns the A matrix for the Gravity nodal problem

        The A matrix has the form:

        .. math ::

            \mathbf{A} =  \Div(\MfMui)^{-1}\Div^{T}

        """
        return -self._Div.T * self.Mfi * self._Div

    def fields(self, m):
        """
            Return magnetic potential (u) and flux (B)
            u: defined on the cell nodes [nC x 1]
            gField: defined on the cell faces [nF x 1]

            After we compute u, then we update B.

            .. math ::

                \mathbf{B}_s = (\MfMui)^{-1}\mathbf{M}^f_{\mu_0^{-1}}\mathbf{B}_0-\mathbf{B}_0 -(\MfMui)^{-1}\Div^T \mathbf{u}

        """
        from scipy.constants import G as NewtG

        self.makeMassMatrices(m)
        A = self.getA(m)
        RHS = self.getRHS(m)

        if self.solver is None:
            m1 = sp.linalg.interface.aslinearoperator(
                Utils.sdiag(1 / A.diagonal())
            )
            u, info = sp.linalg.bicgstab(A, RHS, tol=1e-6, maxiter=1000, M=m1)

        else:
            print("Solving with Paradiso")
            Ainv = self.solver(A)
            u = Ainv * RHS

        gField = 4. * np.pi * NewtG * 1e+8 * self._Div * u

        return {'G': gField, 'u': u}
