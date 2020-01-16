from __future__ import print_function
from SimPEG import utils
from SimPEG.utils import mkvc, sdiag
from SimPEG import props
from ...simulation import BaseSimulation, LinearSimulation
import scipy as sp
import scipy.constants as constants
import os
import numpy as np
import dask
import dask.array as da
from scipy.sparse import csr_matrix as csr
from dask.diagnostics import ProgressBar
import multiprocessing


class GravityIntegralSimulation(LinearSimulation):
    """
    Gravity simulation in integral form.

    Parameters
    ----------
    store_sensitivity: bool = True
        Forward operator is stored on disk or in memory
    active_indices: numpy.array (mesh.nC, dtype=bool)
        Array of bool defining the active cell where True:Ground, False:Air
    verbose: bool = True
        Flag to display progress
    n_cpu: int = None
      Set the number of processors used by Dask
    parallelized: bool = True
        Use Dask to parallelize the computation and storage of sensitivities
    max_chunk_size: int = None
        Largest chunk size (Mb) used by Dask
    chunk_by_rows: bool = False
        Type of Dask chunking
    sensitivity_path: str = "./sensitivity.zarr"
        Directory used by Dask to store the sensitivity matrix
    max_ram: int = 8
        Target maximum memory usage (Gb)
    """
    rho, rhoMap, rhoDeriv = props.Invertible(
        "Specific density (g/cc)",
        default=1.
    )

    store_sensitivity = True
    active_indices = None
    verbose = True
    n_cpu = None
    parallelized = True
    max_chunk_size = None
    chunk_by_rows = False
    sensitivity_path = "./sensitivity.zarr"
    max_ram = 8

    def __init__(self, mesh, **kwargs):
        BaseSimulation.__init__(self, mesh, **kwargs)

        if getattr(self, 'active_indices', None) is not None:

            # Check if given indices
            if self.active_indices.dtype != 'bool':
                active_indices = np.zeros(self.mesh.nC, dtype='bool')
                active_indices[self.active_indices] = True
                self.active_indices = active_indices
        else:
            self.active_indices = np.ones(self.mesh.nC, dtype='bool')

        self.nC = int(self.active_indices.sum())

        # Create active cell projector
        projection = csr(
            (np.ones(self.nC), (np.where(self.active_indices), range(self.nC))),
            shape=(self.mesh.nC, self.nC)
        )

        # Create vectors of nodal location for the lower and upper corners for each cell
        bsw = (self.mesh.gridCC - self.mesh.h_gridded/2.)
        tne = (self.mesh.gridCC + self.mesh.h_gridded/2.)

        xn1, xn2 = bsw[:, 0], tne[:, 0]
        yn1, yn2 = bsw[:, 1], tne[:, 1]

        self.Yn = projection.T*np.c_[mkvc(yn1), mkvc(yn2)]
        self.Xn = projection.T*np.c_[mkvc(xn1), mkvc(xn2)]

        # Allows for 2D mesh where Zn is defined by user
        if self.mesh.dim > 2:
            zn1, zn2 = bsw[:, 2], tne[:, 2]
            self.Zn = projection.T*np.c_[mkvc(zn1), mkvc(zn2)]

        self.receiver_locations = self.survey.source_field.receiver_list[0].locations
        self.nD = self.receiver_locations.shape[0]

    def fields(self, m):
        # self.model = self.rhoMap*m

        if not self.store_sensitivity:

            # Compute the linear operation without forming the full dense G
            return np.array(self.Intrgl_Fwr_Op(m=m), dtype='float')

        else:

            return da.dot(self.G, (self.rhoMap*m).astype(np.float32)).compute()

    def modelMap(self):
        """
            Call for general mapping of the problem
        """
        return self.rhoMap

    def getJtJdiag(self, m, W=None):
        """
            Return the diagonal of JtJ
        """
        self.model = m

        if self.gtg_diagonal is None:

            if W is None:
                w = np.ones(self.G.shape[1])
            else:
                w = W.diagonal()

            self._gtg_diagonal = da.sum(self.G**2., 0).compute()

        return mkvc(np.sum((sdiag(mkvc(self.gtg_diagonal)**0.5) * self.rhoMap.deriv(m)).power(2.), axis=0))

    def getJ(self, m, f=None):
        """
            Sensitivity matrix
        """
        return da.dot(self.G, self.rhoMap.deriv(m))

    def Jvec(self, m, v, f=None):
        """
        Sensitivity times a vector
        """
        dmu_dm_v = da.from_array(self.rhoMap.deriv(m)*v, chunks=self.G.chunks[1])

        return da.dot(self.G, dmu_dm_v.astype(np.float32))

    def Jtvec(self, m, v, f=None):
        """
        Sensitivity transposed times a vector
        """
        Jtvec = da.dot(v.astype(np.float32), self.G)
        dmudm_v = dask.delayed(csr.dot)(Jtvec, self.rhoMap.deriv(m))

        return da.from_delayed(dmudm_v, dtype=float, shape=[self.rhoMap.deriv(m).shape[1]]).compute()

    @property
    def G(self):
        """
        Gravity forward operator
        """
        if getattr(self, '_G', None) is None:

            self._G = self.Intrgl_Fwr_Op()

        return self._G

    @property
    def gtg_diagonal(self):
        """
        Diagonal of GtG
        """
        if getattr(self, '_gtg_diagonal', None) is None:

            return None

        return self._gtg_diagonal

    def Intrgl_Fwr_Op(self, m=None):

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

        # Switch to determine if the process has to be run in parallel
        job = Forward(
                receiver_locations=self.receiver_locations, Xn=self.Xn, Yn=self.Yn, Zn=self.Zn,
                n_cpu=self.n_cpu, store_sensitivity=self.store_sensitivity,
                model=self.model, components=self.survey.components,
                parallelized=self.parallelized,
                verbose=self.verbose, sensitivity_path=self.sensitivity_path, max_ram=self.max_ram,
                max_chunk_size=self.max_chunk_size, chunk_by_rows=self.chunk_by_rows
                )

        return job.calculate()


class Forward(object):
    """
        Add docstring once it works
    """

    parallelized = True
    receiver_locations = None
    Xn, Yn, Zn = None, None, None
    n_cpu = None
    store_sensitivity = False
    model = None
    components = ['gz']

    verbose = True
    max_ram = 1
    chunk_by_rows = False

    max_chunk_size = None
    sensitivity_path = "./sensitivity.zarr"

    def __init__(self, **kwargs):
        super(Forward, self).__init__()
        utils.setKwargs(self, **kwargs)

    def calculate(self):

        self.nD = self.receiver_locations.shape[0]
        self.nC = self.Xn.shape[0]

        if self.n_cpu is None:
            self.n_cpu = int(multiprocessing.cpu_count())

        # Set this early so we can get a better memory estimate for dask chunking
        nDataComps = len(self.components)

        if self.parallelized:

            row = dask.delayed(self.calculate_g_row, pure=True)

            makeRows = [row(self.receiver_locations[ii, :]) for ii in range(self.nD)]

            buildMat = [da.from_delayed(makeRow, dtype=np.float32, shape=(nDataComps,  self.nC)) for makeRow in makeRows]

            stack = da.vstack(buildMat)

            # Auto rechunk
            # To customise memory use set Dask config in calling scripts: dask.config.set({'array.chunk-size': '128MiB'})
            if not self.store_sensitivity or self.chunk_by_rows:
                # Auto-chunking by rows is faster and more memory efficient for
                # very large problems sensitivity and forward calculations
                target_size = dask.config.get('array.chunk-size').replace('MiB',' MB')
                stack = stack.rechunk({0: 'auto', 1: -1})
            elif self.max_chunk_size:
                # Manual chunking is less sensitive to chunk sizes for some problems
                target_size = "{:.0f} MB".format(self.max_chunk_size)
                n_chunk_col = 1
                n_chunk_row = 1
                row_chunk = int(np.ceil(stack.shape[0]/n_chunk_row))
                col_chunk = int(np.ceil(stack.shape[1]/n_chunk_col))
                chunk_size = row_chunk*col_chunk*8*1e-6  # in Mb

                # Add more chunks until memory falls below target
                while chunk_size >= self.max_chunk_size:

                    if row_chunk > col_chunk:
                        n_chunk_row += 1
                    else:
                        n_chunk_col += 1

                    row_chunk = int(np.ceil(stack.shape[0]/n_chunk_row))
                    col_chunk = int(np.ceil(stack.shape[1]/n_chunk_col))
                    chunk_size = row_chunk*col_chunk*8*1e-6  # in Mb

                stack = stack.rechunk((row_chunk, col_chunk))
            else:
                # Autochunking by columns is faster for Inversions
                target_size = dask.config.get('array.chunk-size').replace('MiB',' MB')
                stack = stack.rechunk({0: -1, 1: 'auto'})

            if self.verbose:
                print('Tile size (nD, nC): ', stack.shape)
                print('Number of chunks: %.0f x %.0f = %.0f' %
                    (len(stack.chunks[0]), len(stack.chunks[1]), len(stack.chunks[0]) * len(stack.chunks[1])))
                print("Target chunk size: %s" % target_size)
                print(
                    'Max chunk size %.0f x %.0f = %.3f MB' % (
                        max(stack.chunks[0]),
                        max(stack.chunks[1]),
                        max(stack.chunks[0]) * max(stack.chunks[1]) * 8*1e-6)
                )
                print('Min chunk size %.0f x %.0f = %.3f MB' % (
                    min(stack.chunks[0]),
                    min(stack.chunks[1]),
                    min(stack.chunks[0]) * min(stack.chunks[1]) * 8*1e-6)
                )
                print('Max RAM (GB x %.0f CPU): %.6f' %
                    (self.n_cpu, max(stack.chunks[0]) * max(stack.chunks[1]) * 8*1e-9 * self.n_cpu))
                print('Tile size (GB): %.3f' % (stack.shape[0] * stack.shape[1] * 8*1e-9))

            if not self.store_sensitivity:

                with ProgressBar():
                    print("Forward calculation: ")
                    pred = da.dot(stack, self.model).compute()

                return pred

            else:

                if os.path.exists(self.sensitivity_path):

                    G = da.from_zarr(self.sensitivity_path)

                    if np.all(np.r_[
                            np.any(np.r_[G.chunks[0]] == stack.chunks[0]),
                            np.any(np.r_[G.chunks[1]] == stack.chunks[1]),
                            np.r_[G.shape] == np.r_[stack.shape]]):
                        # Check that loaded G matches supplied data and mesh
                        print("Zarr file detected with same shape and chunksize ... re-loading")

                        return G
                    else:

                        print("Zarr file detected with wrong shape and chunksize ... over-writing")

                with ProgressBar():
                    print("Saving G to zarr: " + self.sensitivity_path)
                    G = da.to_zarr(stack, self.sensitivity_path, compute=True, return_stored=True, overwrite=True)

        else:

            result = []
            for ii in range(self.nD):
                result += [self.calculate_g_row(self.receiver_locations[ii, :])]
                self.progress(ii, self.nD)

            G = np.vstack(result)

        return G

    def calculate_g_row(self, receiver_location):
        """
            Load in the active nodes of a tensor mesh and computes the magnetic
            forward relation between a cuboid and a given observation
            location outside the Earth [obsx, obsy, obsz]

            INPUT:
            receiver_location:  numpy.ndarray (n_receivers, 3)
                Array of receiver locations as x, y, z columns.

            OUTPUT:
            Tx = [Txx Txy Txz]
            Ty = [Tyx Tyy Tyz]
            Tz = [Tzx Tzy Tzz]

        """
        eps = 1e-8

        dx = self.Xn - receiver_location[0]
        dy = self.Yn - receiver_location[1]
        dz = self.Zn - receiver_location[2]

        components = {key: np.zeros(self.Xn.shape[0]) for key in self.components}

        gxx = np.zeros(self.Xn.shape[0])
        gyy = np.zeros(self.Xn.shape[0])

        for aa in range(2):
            for bb in range(2):
                for cc in range(2):

                    r = (
                            mkvc(dx[:, aa]) ** 2 +
                            mkvc(dy[:, bb]) ** 2 +
                            mkvc(dz[:, cc]) ** 2
                        ) ** (0.50) + eps

                    dz_r = dz[:, cc] + r + eps
                    dy_r = dy[:, bb] + r + eps
                    dx_r = dx[:, aa] + r + eps

                    dxr = dx[:, aa] * r + eps
                    dyr = dy[:, bb] * r + eps
                    dzr = dz[:, cc] * r + eps

                    dydz = dy[:, bb] * dz[:, cc]
                    dxdy = dx[:, aa] * dy[:, bb]
                    dxdz = dx[:, aa] * dz[:, cc]

                    if 'gx' in self.components:
                        components['gx'] += (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dy[:, bb] * np.log(dz_r) +
                            dz[:, cc] * np.log(dy_r) -
                            dx[:, aa] * np.arctan(dydz /
                                                  dxr)
                        )

                    if 'gy' in self.components:
                        components['gy']  += (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dx[:, aa] * np.log(dz_r) +
                            dz[:, cc] * np.log(dx_r) -
                            dy[:, bb] * np.arctan(dxdz /
                                                  dyr)
                        )

                    if 'gz' in self.components:
                        components['gz']  += (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dx[:, aa] * np.log(dy_r) +
                            dy[:, bb] * np.log(dx_r) -
                            dz[:, cc] * np.arctan(dxdy /
                                                  dzr)
                        )

                    arg = dy[:, bb] * dz[:, cc] / dxr

                    if ('gxx' in self.components) or ("gzz" in self.components) or ("guv" in self.components):
                        gxx -= (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dxdy / (r * dz_r + eps) +
                            dxdz / (r * dy_r + eps) -
                            np.arctan(arg+eps) +
                            dx[:, aa] * (1./ (1+arg**2.)) *
                            dydz/dxr**2. *
                            (r + dx[:, aa]**2./r)
                        )

                    if 'gxy' in self.components:
                        components['gxy'] -= (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            np.log(dz_r) + dy[:, bb]**2./ (r*dz_r) +
                            dz[:, cc] / r  -
                            1. / (1+arg**2.+ eps) * (dz[:, cc]/r**2) * (r - dy[:, bb]**2./r)

                        )

                    if 'gxz' in self.components:
                        components['gxz'] -= (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            np.log(dy_r) + dz[:, cc]**2./ (r*dy_r) +
                            dy[:, bb] / r  -
                            1. / (1+arg**2.) * (dy[:, bb]/(r**2)) * (r - dz[:, cc]**2./r)

                        )

                    arg = dx[:, aa]*dz[:, cc]/dyr

                    if ('gyy' in self.components) or ("gzz" in self.components) or ("guv" in self.components):
                        gyy -= (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dxdy / (r*dz_r+ eps) +
                            dydz / (r*dx_r+ eps) -
                            np.arctan(arg+eps) +
                            dy[:, bb] * (1./ (1+arg**2.+ eps)) *
                            dxdz/dyr**2. *
                            (r + dy[:, bb]**2./r)
                        )

                    if 'gyz' in self.components:
                        components['gyz'] -= (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            np.log(dx_r) + dz[:, cc]**2./ (r*(dx_r)) +
                            dx[:, aa] / r  -
                            1. / (1+arg**2.) * (dx[:, aa]/(r**2)) * (r - dz[:, cc]**2./r)

                        )

        if 'gyy' in self.components:
            components['gyy'] = gyy

        if 'gxx' in self.components:
            components['gxx'] = gxx

        if 'gzz' in self.components:
            components['gzz'] = -gxx - gyy

        if 'guv' in self.components:
            components['guv'] = -0.5*(gxx - gyy)

        return np.vstack([constants.G * 1e+8 * components[key] for key in list(components.keys())])


class DifferentialEquationSimulation(BaseSimulation):
    """
        Gravity in differential equations!
    """

    _depreciate_main_map = 'rhoMap'

    rho, rhoMap, rhoDeriv = props.Invertible(
        "Specific density (g/cc)",
        default=1.
    )

    solver = None

    def __init__(self, mesh, **kwargs):
        BaseSimulation.__init__(self, mesh, **kwargs)

        self.mesh.setCellGradBC('dirichlet')

        self._Div = self.mesh.cellGrad

    @property
    def MfI(self): return self._MfI

    @property
    def Mfi(self): return self._Mfi

    def makeMassMatrices(self, m):
        self.model = m
        self._Mfi = self.mesh.getFaceInnerProduct()
        self._MfI = utils.sdiag(1. / self._Mfi.diagonal())

    def getRHS(self, m):
        """


        """

        Mc = utils.sdiag(self.mesh.vol)

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
                utils.sdiag(1 / A.diagonal())
            )
            u, info = sp.linalg.bicgstab(A, RHS, tol=1e-6, maxiter=1000, M=m1)

        else:
            print("Solving with Paradiso")
            Ainv = self.solver(A)
            u = Ainv * RHS

        gField = 4. * np.pi * NewtG * 1e+8 * self._Div * u

        return {'G': gField, 'u': u}
