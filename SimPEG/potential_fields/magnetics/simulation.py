import os
import shutil
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix as csr
from scipy.constants import mu_0
import properties
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import multiprocessing

from ... import props
from ...data import Data
from ...utils import mkvc, matutils, sdiag, setKwargs
from ..base import BasePFSimulation
from .survey import MagneticSurvey
from ...simulation import LinearSimulation
from discretize import TreeMesh


class MagneticIntegralSimulation(BasePFSimulation):

    chi, chiMap, chiDeriv = props.Invertible(
        "Magnetic Susceptibility (SI)",
        default=1.
    )

    coordinate_system = properties.StringChoice(
        "Type of coordinate system we are regularizing in",
        choices=['cartesian', 'spherical'],
        default='cartesian'
    )

    modelType = properties.StringChoice(
        "Type of magnetization model",
        choices=['susceptibility', 'vector', 'amplitude'],
        default='susceptibility'
    )

    survey = properties.Instance(
        "a survey object", MagneticSurvey, required=True
    )

    forwardOnly = False  # If false, matrix is store to memory (watch your RAM)
    actInd = None  #: Active cell indices provided
    M = None  #: Magnetization matrix provided, otherwise all induced
    magType = 'H0'
    equiSourceLayer = False
    silent = False  # Don't display progress on screen
    W = None
    gtgdiag = None
    n_cpu = None
    parallelized = True

    def __init__(self, mesh, **kwargs):

        assert mesh.dim == 3, 'Integral formulation only available for 3D mesh'
        super(LinearSimulation, self).__init__(mesh=mesh, **kwargs)

        if self.modelType == 'vector':
            self.magType = 'full'

        # Find non-zero cells
        if getattr(self, 'actInd', None) is not None:
            if self.actInd.dtype == 'bool':
                inds = np.where(self.actInd)[0]
            else:
                inds = self.actInd

        else:

            inds = np.asarray(range(self.mesh.nC))

        self.nC = len(inds)

        # Create active cell projector
        P = csr((np.ones(self.nC), (inds, range(self.nC))),
                          shape=(self.mesh.nC, self.nC))

        # Create vectors of nodal location
        # (lower and upper coners for each cell)
        if isinstance(self.mesh, TreeMesh):
            # Get upper and lower corners of each cell
            bsw = (self.mesh.gridCC - self.mesh.h_gridded/2.)
            tne = (self.mesh.gridCC + self.mesh.h_gridded/2.)

            xn1, xn2 = bsw[:, 0], tne[:, 0]
            yn1, yn2 = bsw[:, 1], tne[:, 1]
            zn1, zn2 = bsw[:, 2], tne[:, 2]

        else:

            xn = self.mesh.vectorNx
            yn = self.mesh.vectorNy
            zn = self.mesh.vectorNz

            yn2, xn2, zn2 = np.meshgrid(yn[1:], xn[1:], zn[1:])
            yn1, xn1, zn1 = np.meshgrid(yn[:-1], xn[:-1], zn[:-1])

        self.Yn = P.T*np.c_[mkvc(yn1), mkvc(yn2)]
        self.Xn = P.T*np.c_[mkvc(xn1), mkvc(xn2)]
        self.Zn = P.T*np.c_[mkvc(zn1), mkvc(zn2)]

    def fields(self, m):

        if self.coordinate_system == 'cartesian':
            m = self.chiMap*(m)
        else:
            m = self.chiMap*(matutils.atp2xyz(m.reshape((int(len(m)/3), 3), order='F')))

        if self.forwardOnly:
            # Compute the linear operation without forming the full dense F
            return np.array(self.Intrgl_Fwr_Op(m=m, magType=self.magType), dtype='float')

        # else:

        if getattr(self, '_Mxyz', None) is not None:

            vec = dask.delayed(csr.dot)(self.Mxyz, m)
            M = da.from_delayed(vec, dtype=float, shape=[m.shape[0]])
            fields = da.dot(self.G, M).compute()

        else:

            fields = da.dot(self.G, m).compute()

        if self.modelType == 'amplitude':

            fields = self.calcAmpData(fields)

        return fields

    def calcAmpData(self, Bxyz):
        """
            Compute amplitude of the field
        """

        amplitude = da.sum(
            Bxyz.reshape((3, self.nD), order='F')**2., axis=0
        )**0.5

        return amplitude

    @property
    def G(self):

        if getattr(self, '_G', None) is None:

            self._G = self.Intrgl_Fwr_Op(magType=self.magType)

        return self._G

    @property
    def nD(self):
        """
            Number of data
        """
        self._nD = self.survey.receiver_locations.shape[0]

        return self._nD

    @property
    def ProjTMI(self):

        if getattr(self, '_ProjTMI', None) is None:

            # Convert Bdecination from north to cartesian
            self._ProjTMI = matutils.dip_azimuth2cartesian(
                self.survey.source_field.parameters[1],
                self.survey.source_field.parameters[2]
            )

        return self._ProjTMI

    def getJtJdiag(self, m, W=None):
        """
            Return the diagonal of JtJ
        """
        dmudm = self.chiMap.deriv(m)
        self._dSdm = None
        self._dfdm = None
        self.model = m
        if (self.gtgdiag is None) and (self.modelType != 'amplitude'):

            if W is None:
                w = np.ones(self.G.shape[1])
            else:
                w = W.diagonal()

            # self.gtgdiag = np.zeros(dmudm.shape[1])

            # for ii in range(self.G.shape[0]):

            self.gtgdiag = da.sum(self.G**2., 0).compute()

            # self.gtgdiag = np.array(da.sum(da.power(self.G, 2), axis=0))

        if self.coordinate_system == 'cartesian':
            if self.modelType == 'amplitude':
                return np.sum((W * self.dfdm * sdiag(mkvc(self.gtgdiag)**0.5) * dmudm).power(2.), axis=0)
            else:
                return mkvc(np.sum((sdiag(mkvc(self.gtgdiag)**0.5) * dmudm).power(2.), axis=0))

        else:  # spherical
            if self.modelType == 'amplitude':
                return mkvc(np.sum(((W * self.dfdm) * sdiag(mkvc(self.gtgdiag)**0.5) * (self.dSdm * dmudm)).power(2.), axis=0))
            else:

                #Japprox = sdiag(mkvc(self.gtgdiag)**0.5*dmudm) * (self.dSdm * dmudm)
                return mkvc(np.sum((sdiag(mkvc(self.gtgdiag)**0.5) * self.dSdm * dmudm).power(2), axis=0))

    def getJ(self, m, f=None):
        """
            Sensitivity matrix
        """

        if self.coordinate_system == 'cartesian':
            dmudm = self.chiMap.deriv(m)
        else:  # spherical
            dmudm = self.dSdm * self.chiMap.deriv(m)

        if self.modelType == 'amplitude':
            return self.dfdm * da.dot(self.G, dmudm)
        else:

            prod = dask.delayed(
                csr.dot)(
                    self.G, dmudm
                )
            return da.from_delayed(
                prod, dtype=float,
                shape=(self.G.shape[0], dmudm.shape[1])
            )

    def Jvec(self, m, v, f=None):

        if self.coordinate_system == 'cartesian':
            dmudm = self.chiMap.deriv(m)
        else:
            dmudm = dask.delayed(csr.dot)(self.dSdm, self.chiMap.deriv(m))

        if getattr(self, '_Mxyz', None) is not None:

            dmudm_v = dask.delayed(csr.dot)(dmudm, v)
            vec = dask.delayed(csr.dot)(self.Mxyz, dmudm_v)
            M_dmudm_v = da.from_delayed(vec, dtype=float, shape=[self.Mxyz.shape[0]])

            Jvec = da.dot(self.G, M_dmudm_v).compute()

        else:

            vec = dask.delayed(csr.dot)(dmudm, v)
            dmudm_v = da.from_delayed(vec, dtype=float, shape=[self.chiMap.deriv(m).shape[0]])
            Jvec = da.dot(self.G, dmudm_v)

        if self.modelType == 'amplitude':
            dfdm_Jvec = dask.delayed(csr.dot)(self.dfdm, Jvec)

            return da.from_delayed(dfdm_Jvec, dtype=float, shape=[self.dfdm.shape[0]]).compute()
        else:
            return Jvec.compute()

    def Jtvec(self, m, v, f=None):

        if self.coordinate_system == 'cartesian':
            dmudm = self.chiMap.deriv(m)
        else:
            dmudm = dask.delayed(csr.dot)(self.dSdm, self.chiMap.deriv(m))

        if self.modelType == 'amplitude':

            dfdm_v = dask.delayed(csr.dot)(self.dfdm.T, v)
            vec = da.from_delayed(dfdm_v, dtype=float, shape=[self.dfdm.shape[0]])

            if getattr(self, '_Mxyz', None) is not None:

                jtvec = da.dot(self.G.T, vec)

                Jtvec = dask.delayed(csr.dot)(self.Mxyz.T, jtvec)

            else:
                Jtvec = da.dot(self.G.T, vec)

        else:

            Jtvec = da.dot(self.G.T, v)

        dmudm_v = dask.delayed(csr.dot)(dmudm.T, Jtvec)

        return da.from_delayed(dmudm_v, dtype=float, shape=[self.chiMap.deriv(m).shape[1]]).compute()

    @property
    def dSdm(self):

        if getattr(self, '_dSdm', None) is None:

            if self.model is None:
                raise Exception('Requires a chi')

            nC = int(len(self.model)/3)

            m_xyz = self.chiMap * matutils.atp2xyz(self.model.reshape((nC, 3), order='F'))

            nC = int(m_xyz.shape[0]/3.)
            m_atp = matutils.xyz2atp(m_xyz.reshape((nC, 3), order='F'))

            a = m_atp[:nC]
            t = m_atp[nC:2*nC]
            p = m_atp[2*nC:]

            Sx = sp.hstack([sp.diags(np.cos(t)*np.cos(p), 0),
                            sp.diags(-a*np.sin(t)*np.cos(p), 0),
                            sp.diags(-a*np.cos(t)*np.sin(p), 0)])

            Sy = sp.hstack([sp.diags(np.cos(t)*np.sin(p), 0),
                            sp.diags(-a*np.sin(t)*np.sin(p), 0),
                            sp.diags(a*np.cos(t)*np.cos(p), 0)])

            Sz = sp.hstack([sp.diags(np.sin(t), 0),
                            sp.diags(a*np.cos(t), 0),
                            csr((nC, nC))])

            self._dSdm = sp.vstack([Sx, Sy, Sz])

        return self._dSdm

    @property
    def modelMap(self):
        """
            Call for general mapping of the problem
        """
        return self.chiMap

    @property
    def dfdm(self):

        if self.model is None:
            self.model = np.zeros(self.G.shape[1])

        if getattr(self, '_dfdm', None) is None:

            Bxyz = self.Bxyz_a(self.chiMap * self.model)

            # Bx = sp.spdiags(Bxyz[:, 0], 0, self.nD, self.nD)
            # By = sp.spdiags(Bxyz[:, 1], 0, self.nD, self.nD)
            # Bz = sp.spdiags(Bxyz[:, 2], 0, self.nD, self.nD)
            ii = np.kron(np.asarray(range(self.survey.nD), dtype='int'), np.ones(3))
            jj = np.asarray(range(3*self.survey.nD), dtype='int')
            # (data, (row, col)), shape=(3, 3))
            # P = s
            self._dfdm = csr((mkvc(Bxyz), (ii, jj)), shape=(self.survey.nD, 3*self.survey.nD))

        return self._dfdm

    def Bxyz_a(self, m):
        """
            Return the normalized B fields
        """

        # Get field data
        if self.coordinate_system == 'spherical':
            m = matutils.atp2xyz(m)

        if getattr(self, '_Mxyz', None) is not None:
            Bxyz = da.dot(self.G, (self.Mxyz*m))
        else:
            Bxyz = da.dot(self.G, m)

        amp = self.calcAmpData(Bxyz)
        Bamp = sp.spdiags(1./amp, 0, self.nD, self.nD)

        return (Bxyz.reshape((3, self.nD), order='F')*Bamp)

    def Intrgl_Fwr_Op(self, m=None, magType='H0'):
        """

        Magnetic forward operator in integral form

        magType  = 'H0' | 'x' | 'y' | 'z'

        Return
        _G = Linear forward operator | (forwardOnly)=data

         """
        if m is not None:
            self.model = self.chiMap*m

        # survey = self.survey
        self.receiver_locations = self.survey.receiver_locations

        if magType == 'H0':
            if getattr(self, 'M', None) is None:
                self.M = matutils.dip_azimuth2cartesian(np.ones(self.nC) * self.survey.source_field.parameters[1],
                                      np.ones(self.nC) * self.survey.source_field.parameters[2])

            Mx = sdiag(self.M[:, 0] * self.survey.source_field.parameters[0])
            My = sdiag(self.M[:, 1] * self.survey.source_field.parameters[0])
            Mz = sdiag(self.M[:, 2] * self.survey.source_field.parameters[0])

            self.Mxyz = sp.vstack((Mx, My, Mz))

        elif magType == 'full':

            self.Mxyz = sp.identity(3*self.nC) * self.survey.source_field.parameters[0]

        else:
            raise Exception('magType must be: "H0" or "full"')

                # Loop through all observations and create forward operator (nD-by-nC)
        print("Begin forward: M=" + magType + ", components= %s" % list(self.survey.components.keys()))

        # Switch to determine if the process has to be run in parallel
        job = Forward(
                receiver_locations=self.receiver_locations, Xn=self.Xn, Yn=self.Yn, Zn=self.Zn,
                n_cpu=self.n_cpu, forwardOnly=self.forwardOnly,
                model=self.model, components=self.survey.components, Mxyz=self.Mxyz,
                P=self.ProjTMI, parallelized=self.parallelized
                )

        G = job.calculate()

        return G


class Forward(object):

    progressIndex = -1
    parallelized = True
    receiver_locations = None
    Xn, Yn, Zn = None, None, None
    n_cpu = None
    forwardOnly = False
    components = {'tmi': []}
    model = None
    Mxyz = None
    P = None
    verbose = True
    maxRAM = 1
    Jpath = "./sensitivity.zarr"

    def __init__(self, **kwargs):
        super(Forward, self).__init__()
        setKwargs(self, **kwargs)

    def calculate(self):
        self.nD = self.receiver_locations.shape[0]
        self.nC = self.Mxyz.shape[1]

        if self.n_cpu is None:
            self.n_cpu = int(multiprocessing.cpu_count())

        components = list(self.components.keys())
        # Stack all the components 'active' flag
        activeComponents = np.hstack([self.components[component] for component in components])

        if self.parallelized:

            row = dask.delayed(self.calcTrow, pure=True)
            print(components)
            makeRows = [row(self.receiver_locations[ii, :], components[activeComponents[ii,:]]) for ii in range(self.nD)]

            buildMat = [da.from_delayed(makeRow, dtype=float, shape=(int(activeComponents[ind, :].sum()),  self.nC)) for (ind, makeRow) in enumerate(makeRows)]

            stack = da.vstack(buildMat)

            # TO-DO: Find a way to create in
            # chunks instead
            # stack = stack.rechunk('auto')
            nChunks = self.n_cpu # Number of chunks
            rowChunk, colChunk = int(np.ceil(self.nD*nDataComps/nChunks)), int(np.ceil(self.nC/nChunks)) # Chunk sizes
            totRAM = rowChunk*colChunk*8*self.n_cpu*1e-9
            # Ensure total problem size fits in RAM, and avoid 2GB size limit on dask chunks
            while totRAM > self.maxRAM or (totRAM/self.n_cpu) >= 0.125:
#                    print("Dask:", self.n_cpu, nChunks, rowChunk, colChunk, totRAM, self.maxRAM)
                nChunks += 1
                rowChunk, colChunk = int(np.ceil(self.nD*nDataComps/nChunks)), int(np.ceil(self.nC/nChunks)) # Chunk sizes
                totRAM = rowChunk*colChunk*8*self.n_cpu*1e-9

            stack = stack.rechunk((rowChunk, colChunk))

            print('DASK: ')
            print('Tile size (nD, nC): ', stack.shape)
            print('Number of chunks: ', len(stack.chunks[0]), ' x ', len(stack.chunks[1]), ' = ', len(stack.chunks[0]) * len(stack.chunks[1]))
            print("Target chunk size: ", dask.config.get('array.chunk-size'))
            print('Max chunk size (GB): ', max(stack.chunks[0]) * max(stack.chunks[1]) * 8*1e-9)
            print('Max RAM (GB x CPU): ', max(stack.chunks[0]) * max(stack.chunks[1]) * 8*1e-9 * self.n_cpu)
            print('Tile size (GB): ', stack.shape[0] * stack.shape[1] * 8*1e-9)

            if self.forwardOnly:

                with ProgressBar():
                    print("Forward calculation: ")
                    pred = da.dot(stack, self.model).compute()

                return pred

            else:

                if os.path.exists(self.Jpath):

                    G = da.from_zarr(self.Jpath)

                    if np.all(np.r_[
                            np.any(np.r_[G.chunks[0]] == stack.chunks[0]),
                            np.any(np.r_[G.chunks[1]] == stack.chunks[1]),
                            np.r_[G.shape] == np.r_[stack.shape]]):
                        # Check that loaded G matches supplied data and mesh
                        print("Zarr file detected with same shape and chunksize ... re-loading")
                        return G

                    else:
                        del G
                        shutil.rmtree(self.Jpath)
                        print("Zarr file detected with wrong shape and chunksize ... over-writing")

                with ProgressBar():
                    print("Saving G to zarr: " + self.Jpath)
                    da.to_zarr(stack, self.Jpath)

                G = da.from_zarr(self.Jpath)

        else:

            result = []
            for ii in range(self.nD):

                if self.forwardOnly:
                    result += [
                            np.c_[
                                np.dot(
                                    self.calcTrow(self.receiver_locations[ii, :]),
                                    self.model
                                )
                            ]
                        ]
                else:
                    result += [self.calcTrow(self.receiver_locations[ii, :])]
                self.progress(ii, self.nD)

            G = np.vstack(result)

        return G

    def calcTrow(self, xyzLoc, components):
        """
            Load in the active nodes of a tensor mesh and computes the magnetic
            forward relation between a cuboid and a given observation
            location outside the Earth [obsx, obsy, obsz]

            INPUT:
            xyzLoc:  [obsx, obsy, obsz] nC x 3 Array

            OUTPUT:
            Tx = [Txx Txy Txz]
            Ty = [Tyx Tyy Tyz]
            Tz = [Tzx Tzy Tzz]

        """

        rows = calculateIntegralRows(self.Xn, self.Yn, self.Zn, xyzLocation, self.P, components=components)

        return rows * self.Mxyz


def calculateIntegralRows(
    Xn, Yn, Zn, rxlocation, P,
    components=[
        "dbx_dx", "dbx_dy", "dbx_dz", "dby_dy",
        "dby_dz", "dbz_dz", "bx", "by", "bz"
        ]
):
    """
    calcRow
    Takes in the lower SW and upper NE nodes of a tensor mesh,
    observation location receiver_locations[obsx, obsy, obsz] and computes the
    magnetic tensor for the integral of a each prisms

    INPUT:
    Xn, Yn, Zn: Node location matrix for the lower and upper most corners of
                all cells in the mesh shape[nC,2]
    OUTPUT:

    """
    eps = 1e-8  # add a small value to the locations to avoid /0
    # number of cells in mesh
    nC = Xn.shape[0]

    # comp. pos. differences for tne, bsw nodes
    dz2 = Zn[:, 1] - rxlocation[2] + eps
    dz1 = Zn[:, 0] - rxlocation[2] + eps

    dy2 = Yn[:, 1] - rxlocation[1] + eps
    dy1 = Yn[:, 0] - rxlocation[1] + eps

    dx2 = Xn[:, 1] - rxlocation[0] + eps
    dx1 = Xn[:, 0] - rxlocation[0] + eps

    # comp. squared diff
    dx2dx2 = dx2**2.
    dx1dx1 = dx1**2.

    dy2dy2 = dy2**2.
    dy1dy1 = dy1**2.

    dz2dz2 = dz2**2.
    dz1dz1 = dz1**2.

    # 2D radius compent squared of corner nodes
    R1 = (dy2dy2 + dx2dx2)
    R2 = (dy2dy2 + dx1dx1)
    R3 = (dy1dy1 + dx2dx2)
    R4 = (dy1dy1 + dx1dx1)

    # radius to each cell node
    r1 = np.sqrt(dz2dz2 + R2) + eps
    r2 = np.sqrt(dz2dz2 + R1) + eps
    r3 = np.sqrt(dz1dz1 + R1) + eps
    r4 = np.sqrt(dz1dz1 + R2) + eps
    r5 = np.sqrt(dz2dz2 + R3) + eps
    r6 = np.sqrt(dz2dz2 + R4) + eps
    r7 = np.sqrt(dz1dz1 + R4) + eps
    r8 = np.sqrt(dz1dz1 + R3) + eps

    # compactify argument calculations
    arg1_ = dx1 + dy2 + r1
    arg1 = dy2 + dz2 + r1
    arg2 = dx1 + dz2 + r1
    arg3 = dx1 + r1
    arg4 = dy2 + r1
    arg5 = dz2 + r1

    arg6_ = dx2 + dy2 + r2
    arg6 = dy2 + dz2 + r2
    arg7 = dx2 + dz2 + r2
    arg8 = dx2 + r2
    arg9 = dy2 + r2
    arg10 = dz2 + r2

    arg11_ = dx2 + dy2 + r3
    arg11 = dy2 + dz1 + r3
    arg12 = dx2 + dz1 + r3
    arg13 = dx2 + r3
    arg14 = dy2 + r3
    arg15 = dz1 + r3

    arg16_ = dx1 + dy2 + r4
    arg16 = dy2 + dz1 + r4
    arg17 = dx1 + dz1 + r4
    arg18 = dx1 + r4
    arg19 = dy2 + r4
    arg20 = dz1 + r4

    arg21_ = dx2 + dy1 + r5
    arg21 = dy1 + dz2 + r5
    arg22 = dx2 + dz2 + r5
    arg23 = dx2 + r5
    arg24 = dy1 + r5
    arg25 = dz2 + r5

    arg26_ = dx1 + dy1 + r6
    arg26 = dy1 + dz2 + r6
    arg27 = dx1 + dz2 + r6
    arg28 = dx1 + r6
    arg29 = dy1 + r6
    arg30 = dz2 + r6

    arg31_ = dx1 + dy1 + r7
    arg31 = dy1 + dz1 + r7
    arg32 = dx1 + dz1 + r7
    arg33 = dx1 + r7
    arg34 = dy1 + r7
    arg35 = dz1 + r7

    arg36_ = dx2 + dy1 + r8
    arg36 = dy1 + dz1 + r8
    arg37 = dx2 + dz1 + r8
    arg38 = dx2 + r8
    arg39 = dy1 + r8
    arg40 = dz1 + r8

    rows = []
    dbx_dx, dby_dy = [], []
    for comp in components:
        # m_x vector
        if (comp == "dbx_dx") or ("dbz_dz" in components):
            dbx_dx = np.zeros((1, 3 * nC))

            dbx_dx[0, 0:nC] = (
                2 * (
                    (
                        (dx1**2 - r1 * arg1) /
                        (r1 * arg1**2 + dx1**2 * r1 + eps)
                    ) -
                    (
                        (dx2**2 - r2 * arg6) /
                        (r2 * arg6**2 + dx2**2 * r2 + eps)
                    ) +
                    (
                        (dx2**2 - r3 * arg11) /
                        (r3 * arg11**2 + dx2**2 * r3 + eps)
                    ) -
                    (
                        (dx1**2 - r4 * arg16) /
                        (r4 * arg16**2 + dx1**2 * r4 + eps)
                    ) +
                    (
                        (dx2**2 - r5 * arg21) /
                        (r5 * arg21**2 + dx2**2 * r5 + eps)
                    ) -
                    (
                        (dx1**2 - r6 * arg26) /
                        (r6 * arg26**2 + dx1**2 * r6 + eps)
                    ) +
                    (
                        (dx1**2 - r7 * arg31) /
                        (r7 * arg31**2 + dx1**2 * r7 + eps)
                    ) -
                    (
                        (dx2**2 - r8 * arg36) /
                        (r8 * arg36**2 + dx2**2 * r8 + eps)
                    )
                )
            )

            dbx_dx[0, nC:2*nC] = (
                dx2 / (r5 * arg25 + eps) - dx2 / (r2 * arg10 + eps) +
                dx2 / (r3 * arg15 + eps) - dx2 / (r8 * arg40 + eps) +
                dx1 / (r1 * arg5 + eps) - dx1 / (r6 * arg30 + eps) +
                dx1 / (r7 * arg35 + eps) - dx1 / (r4 * arg20 + eps)
            )

            dbx_dx[0, 2*nC:] = (
                dx1 / (r1 * arg4 + eps) - dx2 / (r2 * arg9 + eps) +
                dx2 / (r3 * arg14 + eps) - dx1 / (r4 * arg19 + eps) +
                dx2 / (r5 * arg24 + eps) - dx1 / (r6 * arg29 + eps) +
                dx1 / (r7 * arg34 + eps) - dx2 / (r8 * arg39 + eps)
            )

            dbx_dx /= (4 * np.pi)

        if (comp == "dby_dy") or ("dbz_dz" in components):
            # dby_dy
            dby_dy = np.zeros((1, 3 * nC))

            dby_dy[0, 0:nC] = (dy2 / (r3 * arg15 + eps) - dy2 / (r2 * arg10 + eps) +
                        dy1 / (r5 * arg25 + eps) - dy1 / (r8 * arg40 + eps) +
                        dy2 / (r1 * arg5 + eps) - dy2 / (r4 * arg20 + eps) +
                        dy1 / (r7 * arg35 + eps) - dy1 / (r6 * arg30 + eps))
            dby_dy[0, nC:2*nC] = (2 * (((dy2**2 - r1 * arg2) / (r1 * arg2**2 + dy2**2 * r1 + eps)) -
                       ((dy2**2 - r2 * arg7) / (r2 * arg7**2 + dy2**2 * r2 + eps)) +
                       ((dy2**2 - r3 * arg12) / (r3 * arg12**2 + dy2**2 * r3 + eps)) -
                       ((dy2**2 - r4 * arg17) / (r4 * arg17**2 + dy2**2 * r4 + eps)) +
                       ((dy1**2 - r5 * arg22) / (r5 * arg22**2 + dy1**2 * r5 + eps)) -
                       ((dy1**2 - r6 * arg27) / (r6 * arg27**2 + dy1**2 * r6 + eps)) +
                       ((dy1**2 - r7 * arg32) / (r7 * arg32**2 + dy1**2 * r7 + eps)) -
                       ((dy1**2 - r8 * arg37) / (r8 * arg37**2 + dy1**2 * r8 + eps))))
            dby_dy[0, 2*nC:] = (dy2 / (r1 * arg3 + eps) - dy2 / (r2 * arg8 + eps) +
                         dy2 / (r3 * arg13 + eps) - dy2 / (r4 * arg18 + eps) +
                         dy1 / (r5 * arg23 + eps) - dy1 / (r6 * arg28 + eps) +
                         dy1 / (r7 * arg33 + eps) - dy1 / (r8 * arg38 + eps))

            dby_dy /= (4 * np.pi)

        if comp == "dby_dy":

            rows += [dby_dy]

        if comp == "dbx_dx":

            rows += [dbx_dx]

        if comp == "dbz_dz":

            dbz_dz = -dbx_dx - dby_dy
            rows += [dbz_dz]

        if comp == "dbx_dy":
            dbx_dy = np.zeros((1, 3 * nC))

            dbx_dy[0, 0:nC] = (2 * (((dx1 * arg4) / (r1 * arg1**2 + (dx1**2) * r1 + eps)) -
                        ((dx2 * arg9) / (r2 * arg6**2 + (dx2**2) * r2 + eps)) +
                        ((dx2 * arg14) / (r3 * arg11**2 + (dx2**2) * r3 + eps)) -
                        ((dx1 * arg19) / (r4 * arg16**2 + (dx1**2) * r4 + eps)) +
                        ((dx2 * arg24) / (r5 * arg21**2 + (dx2**2) * r5 + eps)) -
                        ((dx1 * arg29) / (r6 * arg26**2 + (dx1**2) * r6 + eps)) +
                        ((dx1 * arg34) / (r7 * arg31**2 + (dx1**2) * r7 + eps)) -
                        ((dx2 * arg39) / (r8 * arg36**2 + (dx2**2) * r8 + eps))))
            dbx_dy[0, nC:2*nC] = (dy2 / (r1 * arg5 + eps) - dy2 / (r2 * arg10 + eps) +
                           dy2 / (r3 * arg15 + eps) - dy2 / (r4 * arg20 + eps) +
                           dy1 / (r5 * arg25 + eps) - dy1 / (r6 * arg30 + eps) +
                           dy1 / (r7 * arg35 + eps) - dy1 / (r8 * arg40 + eps))
            dbx_dy[0, 2*nC:] = (1 / r1 - 1 / r2 +
                         1 / r3 - 1 / r4 +
                         1 / r5 - 1 / r6 +
                         1 / r7 - 1 / r8)

            dbx_dy /= (4 * np.pi)

            rows += [dbx_dy]

        if comp == "dbx_dz":
            dbx_dz = np.zeros((1, 3 * nC))

            dbx_dz[0, 0:nC] =(2 * (((dx1 * arg5) / (r1 * (arg1**2) + (dx1**2) * r1 + eps)) -
                        ((dx2 * arg10) / (r2 * (arg6**2) + (dx2**2) * r2 + eps)) +
                        ((dx2 * arg15) / (r3 * (arg11**2) + (dx2**2) * r3 + eps)) -
                        ((dx1 * arg20) / (r4 * (arg16**2) + (dx1**2) * r4 + eps)) +
                        ((dx2 * arg25) / (r5 * (arg21**2) + (dx2**2) * r5 + eps)) -
                        ((dx1 * arg30) / (r6 * (arg26**2) + (dx1**2) * r6 + eps)) +
                        ((dx1 * arg35) / (r7 * (arg31**2) + (dx1**2) * r7 + eps)) -
                        ((dx2 * arg40) / (r8 * (arg36**2) + (dx2**2) * r8 + eps))))
            dbx_dz[0, nC:2*nC] = (1 / r1 - 1 / r2 +
                           1 / r3 - 1 / r4 +
                           1 / r5 - 1 / r6 +
                           1 / r7 - 1 / r8)
            dbx_dz[0, 2*nC:] = (dz2 / (r1 * arg4 + eps) - dz2 / (r2 * arg9 + eps) +
                         dz1 / (r3 * arg14 + eps) - dz1 / (r4 * arg19 + eps) +
                         dz2 / (r5 * arg24 + eps) - dz2 / (r6 * arg29 + eps) +
                         dz1 / (r7 * arg34 + eps) - dz1 / (r8 * arg39 + eps))

            dbx_dz /= (4 * np.pi)

            rows += [dbx_dz]

        if comp == "dby_dz":
            dby_dz = np.zeros((1, 3 * nC))

            dby_dz[0, 0:nC] = (1 / r3 - 1 / r2 +
                        1 / r5 - 1 / r8 +
                        1 / r1 - 1 / r4 +
                        1 / r7 - 1 / r6)
            dby_dz[0, nC:2*nC] = (2 * ((((dy2 * arg5) / (r1 * (arg2**2) + (dy2**2) * r1 + eps))) -
                    (((dy2 * arg10) / (r2 * (arg7**2) + (dy2**2) * r2 + eps))) +
                    (((dy2 * arg15) / (r3 * (arg12**2) + (dy2**2) * r3 + eps))) -
                    (((dy2 * arg20) / (r4 * (arg17**2) + (dy2**2) * r4 + eps))) +
                    (((dy1 * arg25) / (r5 * (arg22**2) + (dy1**2) * r5 + eps))) -
                    (((dy1 * arg30) / (r6 * (arg27**2) + (dy1**2) * r6 + eps))) +
                    (((dy1 * arg35) / (r7 * (arg32**2) + (dy1**2) * r7 + eps))) -
                    (((dy1 * arg40) / (r8 * (arg37**2) + (dy1**2) * r8 + eps)))))
            dby_dz[0, 2*nC:] = (dz2 / (r1 * arg3  + eps) - dz2 / (r2 * arg8 + eps) +
                     dz1 / (r3 * arg13 + eps) - dz1 / (r4 * arg18 + eps) +
                     dz2 / (r5 * arg23 + eps) - dz2 / (r6 * arg28 + eps) +
                     dz1 / (r7 * arg33 + eps) - dz1 / (r8 * arg38 + eps))

            dby_dz /= (4 * np.pi)

            rows += [dby_dz]

        if (comp == "bx") or ("tmi" in components):
            bx = np.zeros((1, 3 * nC))

            bx[0, 0:nC] = ((-2 * np.arctan2(dx1, arg1 + eps)) - (-2 * np.arctan2(dx2, arg6 + eps)) +
                       (-2 * np.arctan2(dx2, arg11 + eps)) - (-2 * np.arctan2(dx1, arg16 + eps)) +
                       (-2 * np.arctan2(dx2, arg21 + eps)) - (-2 * np.arctan2(dx1, arg26 + eps)) +
                       (-2 * np.arctan2(dx1, arg31 + eps)) - (-2 * np.arctan2(dx2, arg36 + eps)))
            bx[0, nC:2*nC] = (np.log(arg5) - np.log(arg10) +
                          np.log(arg15) - np.log(arg20) +
                          np.log(arg25) - np.log(arg30) +
                          np.log(arg35) - np.log(arg40))
            bx[0, 2*nC:] = ((np.log(arg4) - np.log(arg9)) +
                        (np.log(arg14) - np.log(arg19)) +
                        (np.log(arg24) - np.log(arg29)) +
                        (np.log(arg34) - np.log(arg39)))
            bx /= (4 * np.pi)

            # rows += [bx]

        if (comp == "by") or ("tmi" in components):
            by = np.zeros((1, 3 * nC))

            by[0, 0:nC] = (np.log(arg5) - np.log(arg10) +
                       np.log(arg15) - np.log(arg20) +
                       np.log(arg25) - np.log(arg30) +
                       np.log(arg35) - np.log(arg40))
            by[0, nC:2*nC] = ((-2 * np.arctan2(dy2, arg2 + eps)) - (-2 * np.arctan2(dy2, arg7 + eps)) +
                              (-2 * np.arctan2(dy2, arg12 + eps)) - (-2 * np.arctan2(dy2, arg17 + eps)) +
                              (-2 * np.arctan2(dy1, arg22 + eps)) - (-2 * np.arctan2(dy1, arg27 + eps)) +
                              (-2 * np.arctan2(dy1, arg32 + eps)) - (-2 * np.arctan2(dy1, arg37 + eps)))
            by[0, 2*nC:] = ((np.log(arg3) - np.log(arg8)) +
                            (np.log(arg13) - np.log(arg18)) +
                            (np.log(arg23) - np.log(arg28)) +
                            (np.log(arg33) - np.log(arg38)))

            by /= (-4 * np.pi)

            # rows += [by]

        if (comp == "bz") or ("tmi" in components):
            bz = np.zeros((1, 3 * nC))

            bz[0, 0:nC] = (np.log(arg4) - np.log(arg9) +
                       np.log(arg14) - np.log(arg19) +
                       np.log(arg24) - np.log(arg29) +
                       np.log(arg34) - np.log(arg39))
            bz[0, nC:2*nC] = ((np.log(arg3) - np.log(arg8)) +
                              (np.log(arg13) - np.log(arg18)) +
                              (np.log(arg23) - np.log(arg28)) +
                              (np.log(arg33) - np.log(arg38)))
            bz[0, 2*nC:] = ((-2 * np.arctan2(dz2, arg1_ + eps)) - (-2 * np.arctan2(dz2, arg6_ + eps)) +
                            (-2 * np.arctan2(dz1, arg11_ + eps)) - (-2 * np.arctan2(dz1, arg16_ + eps)) +
                            (-2 * np.arctan2(dz2, arg21_ + eps)) - (-2 * np.arctan2(dz2, arg26_ + eps)) +
                            (-2 * np.arctan2(dz1, arg31_ + eps)) - (-2 * np.arctan2(dz1, arg36_ + eps)))
            bz /= (-4 * np.pi)

        if comp == "bx":

            rows += [bx]

        if comp == "by":

            rows += [by]

        if comp == "bz":

            rows += [bz]

        if comp == "tmi":

            rows += [np.dot(P, np.r_[bx, by, bz])]

    return np.vstack(rows)
