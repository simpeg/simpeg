import numpy as np
import scipy.sparse as sp
from scipy.constants import mu_0
import properties

from ... import props
from ...data import Data
from ...utils import mkvc
from ..base import BasePFSimulation
from .survey import MagneticSurvey
from ...simulation import LinearSimulation

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
    parallelized = False

    def __init__(self, mesh, **kwargs):

        assert mesh.dim == 3, 'Integral formulation only available for 3D mesh'
        super(LinearSimulation, self).__init__(mesh=mesh, **kwargs)

    def fields(self, m):

        if self.coordinate_system == 'cartesian':
            m = self.chiMap*(m)
        else:
            m = self.chiMap*(matutils.spherical2cartesian(m.reshape((int(len(m)/3), 3), order='F')))

        if self.forwardOnly:
            # Compute the linear operation without forming the full dense F
            fields = self.Intrgl_Fwr_Op(m=m)

        else:

            if getattr(self, '_Mxyz', None) is not None:

                fields = np.dot(self.G, (self.Mxyz*m).astype(np.float32))

            else:
                fields = np.dot(self.G, m.astype(np.float32))

            if self.modelType == 'amplitude':

                fields = self.calcAmpData(fields.astype(np.float64))

        return fields.astype(np.float64)

    def calcAmpData(self, Bxyz):
        """
            Compute amplitude of the field
        """

        amplitude = np.sum(
            Bxyz.reshape((3, self.nD), order='F')**2., axis=0
        )**0.5

        return amplitude

    @property
    def G(self):
        if not self.ispaired:
            raise Exception('Need to pair!')

        if getattr(self, '_G', None) is None:

            if self.modelType == 'vector':
                self.magType = 'full'

            self._G = self.Intrgl_Fwr_Op(magType=self.magType,
                                         rx_type=self.rx_type)

        return self._G

    @property
    def nD(self):
        """
            Number of data
        """
        self._nD = self.survey.srcField.rxList[0].locs.shape[0]

        return self._nD

    @property
    def ProjTMI(self):
        if not self.ispaired:
            raise Exception('Need to pair!')

        if getattr(self, '_ProjTMI', None) is None:

            # Convert Bdecination from north to cartesian
            self._ProjTMI = Utils.matutils.dip_azimuth2cartesian(
                self.survey.srcField.param[1],
                self.survey.srcField.param[2]
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

            self.gtgdiag = np.zeros(dmudm.shape[1])

            for ii in range(self.G.shape[0]):

                self.gtgdiag += (w[ii]*self.G[ii, :]*dmudm)**2.

        if self.coordinate_system == 'cartesian':
            if self.modelType == 'amplitude':
                return np.sum((W * self.dfdm * self.G * dmudm)**2., axis=0)
            else:
                return self.gtgdiag

        else:  # spherical
            if self.modelType == 'amplitude':
                return np.sum(((W * self.dfdm) * self.G * (self.dSdm * dmudm))**2., axis=0)
            else:
                Japprox = sdiag(mkvc(self.gtgdiag)**0.5*dmudm.T) * (self.dSdm * dmudm)
                return mkvc(np.sum(Japprox.power(2), axis=0))

    def getJ(self, m, f=None):
        """
            Sensitivity matrix
        """
        if self.coordinate_system == 'cartesian':
            dmudm = self.chiMap.deriv(m)
        else:  # spherical
            dmudm = self.dSdm * self.chiMap.deriv(m)

        if self.modelType == 'amplitude':
            return self.dfdm * (self.G * dmudm)
        else:
            return self.G * dmudm

    def Jvec(self, m, v, f=None):

        if self.coordinate_system == 'cartesian':
            dmudm = self.chiMap.deriv(m)
        else:
            dmudm = self.dSdm * self.chiMap.deriv(m)

        if getattr(self, '_Mxyz', None) is not None:

            vec = np.dot(self.G, (self.Mxyz*(dmudm*v)).astype(np.float32))

        else:
            vec = np.dot(self.G, (dmudm*v).astype(np.float32))

        if self.modelType == 'amplitude':
            return self.dfdm*vec.astype(np.float64)
        else:
            return vec.astype(np.float64)

    def Jtvec(self, m, v, f=None):

        if self.coordinate_system == 'spherical':
            dmudm = self.dSdm * self.chiMap.deriv(m)
        else:
            dmudm = self.chiMap.deriv(m)

        if self.modelType == 'amplitude':
            if getattr(self, '_Mxyz', None) is not None:

                vec = self.Mxyz.T*np.dot(self.G.T, (self.dfdm.T*v).astype(np.float32)).astype(np.float64)

            else:
                vec = np.dot(self.G.T, (self.dfdm.T*v).astype(np.float32))

        else:

            vec = np.dot(self.G.T, v.astype(np.float32))

        return dmudm.T * vec.astype(np.float64)

    @property
    def dSdm(self):

        if getattr(self, '_dSdm', None) is None:

            if self.model is None:
                raise Exception('Requires a chi')

            nC = int(len(self.model)/3)

            m_xyz = self.chiMap * matutils.spherical2cartesian(self.model.reshape((nC, 3), order='F'))

            nC = int(m_xyz.shape[0]/3.)
            m_atp = matutils.cartesian2spherical(m_xyz.reshape((nC, 3), order='F'))

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
                            sp.csr_matrix((nC, nC))])

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
            self._dfdm = sp.csr_matrix(( mkvc(Bxyz), (ii,jj)), shape=(self.survey.nD, 3*self.survey.nD))

        return self._dfdm

    def Bxyz_a(self, m):
        """
            Return the normalized B fields
        """

        # Get field data
        if self.coordinate_system == 'spherical':
            m = matutils.atp2xyz(m)

        if getattr(self, '_Mxyz', None) is not None:
            Bxyz = np.dot(self.G, (self.Mxyz*m).astype(np.float32))
        else:
            Bxyz = np.dot(self.G, m.astype(np.float32))

        amp = self.calcAmpData(Bxyz.astype(np.float64))
        Bamp = sp.spdiags(1./amp, 0, self.nD, self.nD)

        return (Bxyz.reshape((3, self.nD), order='F')*Bamp)

    def Intrgl_Fwr_Op(self, m=None, magType='H0', rx_type='tmi'):
        """

        Magnetic forward operator in integral form

        magType  = 'H0' | 'x' | 'y' | 'z'
        rx_type  = 'tmi' | 'x' | 'y' | 'z'

        Return
        _G = Linear forward operator | (forwardOnly)=data

         """
        if m is not None:
            self.model = self.chiMap*m

        # Find non-zero cells
        if getattr(self, 'actInd', None) is not None:
            if self.actInd.dtype == 'bool':
                inds = np.where(self.actInd)[0]
            else:
                inds = self.actInd

        else:

            inds = np.asarray(range(self.mesh.nC))

        nC = len(inds)

        # Create active cell projector
        P = sp.csr_matrix((np.ones(nC), (inds, range(nC))),
                          shape=(self.mesh.nC, nC))

        # Create vectors of nodal location
        # (lower and upper coners for each cell)
        if isinstance(self.mesh, Mesh.TreeMesh):
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

        # If equivalent source, use semi-infite prism
        if self.equiSourceLayer:
            zn1 -= 1000.

        self.Yn = P.T*np.c_[mkvc(yn1), mkvc(yn2)]
        self.Xn = P.T*np.c_[mkvc(xn1), mkvc(xn2)]
        self.Zn = P.T*np.c_[mkvc(zn1), mkvc(zn2)]

        # survey = self.survey
        self.rxLoc = self.survey.srcField.rxList[0].locs

        if magType == 'H0':
            if getattr(self, 'M', None) is None:
                self.M = matutils.dip_azimuth2cartesian(np.ones(nC) * self.survey.srcField.param[1],
                                      np.ones(nC) * self.survey.srcField.param[2])

            Mx = sdiag(self.M[:, 0] * self.survey.srcField.param[0])
            My = sdiag(self.M[:, 1] * self.survey.srcField.param[0])
            Mz = sdiag(self.M[:, 2] * self.survey.srcField.param[0])

            self.Mxyz = sp.vstack((Mx, My, Mz))

        elif magType == 'full':

            self.Mxyz = sp.identity(3*nC) * self.survey.srcField.param[0]

        else:
            raise Exception('magType must be: "H0" or "full"')

                # Loop through all observations and create forward operator (nD-by-nC)
        print("Begin forward: M=" + magType + ", Rx type= " + self.rx_type)

        # Switch to determine if the process has to be run in parallel
        job = Forward(
                rxLoc=self.rxLoc, Xn=self.Xn, Yn=self.Yn, Zn=self.Zn,
                n_cpu=self.n_cpu, forwardOnly=self.forwardOnly,
                model=self.model, rx_type=self.rx_type, Mxyz=self.Mxyz,
                P=self.ProjTMI, parallelized=self.parallelized
                )

        G = job.calculate()

        return G


class Forward(object):

    progressIndex = -1
    parallelized = False
    rxLoc = None
    Xn, Yn, Zn = None, None, None
    n_cpu = None
    forwardOnly = False
    model = None
    rx_type = 'z'
    Mxyz = None
    P = None

    def __init__(self, **kwargs):
        super(Forward, self).__init__()
        Utils.setKwargs(self, **kwargs)

    def calculate(self):
        self.nD = self.rxLoc.shape[0]

        if self.parallelized:
            if self.n_cpu is None:

                # By default take half the cores, turns out be faster
                # than running full threads
                self.n_cpu = int(multiprocessing.cpu_count()/2)

            pool = multiprocessing.Pool(self.n_cpu)

            result = pool.map(self.calcTrow, [self.rxLoc[ii, :] for ii in range(self.nD)])
            pool.close()
            pool.join()

        else:

            result = []
            for ii in range(self.nD):
                result += [self.calcTrow(self.rxLoc[ii, :])]
                self.progress(ii, self.nD)

        if self.forwardOnly:
            return mkvc(np.vstack(result))

        else:
            return np.vstack(result)

    def calcTrow(self, xyzLoc):
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
        tx, ty, tz = calcRow(self.Xn, self.Yn, self.Zn, xyzLoc)

        if self.rx_type == 'tmi':
            row = self.P.dot(np.vstack((tx, ty, tz)))*self.Mxyz

        elif self.rx_type == 'x':
            row = tx*self.Mxyz

        elif self.rx_type == 'y':
            row = ty*self.Mxyz

        elif self.rx_type == 'z':
            row = tz*self.Mxyz

        elif self.rx_type == 'xyz':
            row = tx*self.Mxyz
            row = np.r_[row, ty*self.Mxyz]
            row = np.r_[row, tz*self.Mxyz]
        else:
            raise Exception('rx_type must be: "tmi", "x", "y" or "z"')

        if self.forwardOnly:

            return np.dot(row, self.model)
        else:
            return np.float32(row)

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
