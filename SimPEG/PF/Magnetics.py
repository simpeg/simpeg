from __future__ import print_function

import numpy as np
import scipy.sparse as sp
from scipy.constants import mu_0

from SimPEG import Utils
from SimPEG import Problem
from SimPEG import Solver
from SimPEG import Props
from SimPEG import mkvc
from SimPEG.Utils.matutils import atp2xyz, dipazm_2_xyz, xyz2atp
import matplotlib.pyplot as plt
import gc
from SimPEG import Mesh

from . import BaseMag as MAG
from .MagAnalytics import spheremodel, CongruousMagBC
import properties
from scipy.interpolate import griddata
from SimPEG import Solver as SimpegSolver


class MagneticIntegral(Problem.LinearProblem):

    chi, chiMap, chiDeriv = Props.Invertible(
        "Magnetic Susceptibility (SI)",
        default=1.
    )

    forwardOnly = False  # If false, matrix is store to memory (watch your RAM)
    actInd = None  #: Active cell indices provided
    M = None  #: Magnetization matrix provided, otherwise all induced
    recType = 'tmi'  #: Receiver type either "tmi" | "xyz"
    magType = 'H0'
    equiSourceLayer = False
    silent = False  # Don't display progress on screen
    W = None
    memory_saving_mode = False

    def __init__(self, mesh, **kwargs):

        assert mesh.dim == 3, 'Integral formulation only available for 3D mesh'
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

    def fwr_ind(self, m):

        if self.forwardOnly:

            # Compute the linear operation without forming the full dense F
            fwr_d = self.Intrgl_Fwr_Op(m=m)

            return fwr_d

        else:

            vec = np.dot(self.F, m.astype(np.float32))
            return vec.astype(np.float64)

    def fields(self, chi, **kwargs):

        m = self.chiMap*(chi)
        u = self.fwr_ind(m=m)

        return u

    @property
    def F(self):
        if not self.ispaired:
            raise Exception('Need to pair!')

        if getattr(self, '_F', None) is None:
            self._F = self.Intrgl_Fwr_Op(magType=self.magType,
                                         recType=self.recType)

        return self._F

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
            D = (450.-float(self.survey.srcField.param[2])) % 360.
            I = self.survey.srcField.param[1]
            # Projection matrix
            self._ProjTMI = Utils.mkvc(np.r_[np.cos(np.deg2rad(I))*np.cos(np.deg2rad(D)),
                              np.cos(np.deg2rad(I))*np.sin(np.deg2rad(D)),
                              np.sin(np.deg2rad(I))], 2).T

        return self._ProjTMI

    def getJtJdiag(self, m, W=None):
        """
            Return the diagonal of JtJ
        """

        if W is None:
            W = sp.speye(self.F.shape[0])

        dmudm = self.chiMap.deriv(m)

        if self.memory_saving_mode:
            wd = W.diagonal()
            JtJdiag = np.zeros_like(m)
            for ii in range(self.F.shape[0]):
                JtJdiag += ((wd[ii] * self.F[ii, :]) * dmudm)**2.

            return JtJdiag

        else:
            return np.sum((W * self.F * dmudm)**2., axis=0)

    def getJ(self, m, f):
        """
            Sensitivity matrix
        """

        dmudm = self.chiMap.deriv(m)
        return self.F * dmudm

    def Jvec(self, m, v, f=None):
        dmudm = self.chiMap.deriv(m)
        return self.F.dot(dmudm*v)

    def Jtvec(self, m, v, f=None):
        dmudm = self.chiMap.deriv(m)
        return dmudm.T * (self.F.T.dot(v))

    def Intrgl_Fwr_Op(self, m=None, magType='H0', recType='tmi'):
        """

        Magnetic forward operator in integral form

        magType  = 'H0' | 'x' | 'y' | 'z'
        recType  = 'tmi' | 'x' | 'y' | 'z'

        Return
        _F = Linear forward operator | (forwardOnly)=data

         """

        # Find non-zero cells
        if getattr(self, 'actInd', None) is not None:
            if self.actInd.dtype == 'bool':
                inds = np.asarray([inds for inds,
                                  elem in enumerate(self.actInd, 1) if elem],
                                  dtype=int) - 1
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
            bsw = (self.mesh.gridCC -
                   np.kron(self.mesh.vol.T**(1/3)/2,
                           np.ones(3)).reshape((self.mesh.nC, 3)))
            tne = (self.mesh.gridCC +
                   np.kron(self.mesh.vol.T**(1/3)/2,
                           np.ones(3)).reshape((self.mesh.nC, 3)))

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

        Yn = P.T*np.c_[Utils.mkvc(yn1), Utils.mkvc(yn2)]
        Xn = P.T*np.c_[Utils.mkvc(xn1), Utils.mkvc(xn2)]
        Zn = P.T*np.c_[Utils.mkvc(zn1), Utils.mkvc(zn2)]

        survey = self.survey
        rxLoc = survey.srcField.rxList[0].locs

        # Pre-allocate space and create Magnetization matrix if required
        # If assumes uniform Magnetization direction
        if magType == 'H0':
            if getattr(self, 'M', None) is None:
                self.M = dipazm_2_xyz(np.ones(nC) * survey.srcField.param[1],
                                      np.ones(nC) * survey.srcField.param[2])

            Mx = Utils.sdiag(self.M[:, 0]*survey.srcField.param[0])
            My = Utils.sdiag(self.M[:, 1]*survey.srcField.param[0])
            Mz = Utils.sdiag(self.M[:, 2]*survey.srcField.param[0])

            Mxyz = sp.vstack((Mx, My, Mz))

        elif magType == 'x':

            Mxyz = sp.vstack((sp.identity(nC)*survey.srcField.param[0],
                              sp.csr_matrix((nC, nC)),
                              sp.csr_matrix((nC, nC))))

        elif magType == 'y':

            Mxyz = sp.vstack((sp.csr_matrix((nC, nC)),
                              sp.identity(nC)*survey.srcField.param[0],
                              sp.csr_matrix((nC, nC))))

        elif magType == 'z':

            Mxyz = sp.vstack((sp.csr_matrix((nC, nC)),
                              sp.csr_matrix((nC, nC)),
                              sp.identity(nC)*survey.srcField.param[0]))

        elif magType == 'full':

            Mxyz = sp.identity(3*nC)*survey.srcField.param[0]

        else:
            raise Exception('magType must be: "H0", "x", "y", "z" or "full"')

        # Check if we need to store the forward operator and pre-allocate memory
        gc.collect()
        if self.forwardOnly:

            F = np.empty(self.survey.nRx, dtype='float64')

        else:

            if recType != 'xyz':
                F = np.empty((self.nD, Mxyz.shape[1]), dtype=np.float32)
            else:
                F = np.empty((3*self.nD, Mxyz.shape[1]), dtype=np.float32)

        # Loop through all observations and create forward operator (nD-by-nC)
        print("Begin forward: M=" + magType + ", Rx type= " + recType)

        # Add counter to dsiplay progress. Good for large problems
        count = -1
        for ii in range(self.nD):

            tx, ty, tz = get_T_mat(Xn, Yn, Zn, rxLoc[ii, :])

            if self.forwardOnly:

                if recType == 'tmi':
                    F[ii] = (self.ProjTMI.dot(np.vstack((tx, ty, tz)))*Mxyz).dot(m)

                elif recType == 'x':
                    F[ii] = (tx*Mxyz).dot(m)
                elif recType == 'y':
                    F[ii] = (ty*Mxyz).dot(m)
                elif recType == 'z':
                    F[ii] = (tz*Mxyz).dot(m)
                else:
                    raise Exception('recType must be: "tmi", "x", "y" or "z"')

            else:

                if recType == 'tmi':
                    F[ii, :] = self.ProjTMI.dot(np.vstack((tx, ty, tz)))*Mxyz

                elif recType == 'x':
                    F[ii, :] = tx*Mxyz

                elif recType == 'y':
                    F[ii, :] = ty*Mxyz

                elif recType == 'z':
                    F[ii, :] = tz*Mxyz

                elif recType == 'xyz':
                    F[ii, :] = tx*Mxyz
                    F[ii+self.nD, :] = ty*Mxyz
                    F[ii+2*self.nD, :] = tz*Mxyz
                else:
                    raise Exception('recType must be: "tmi", "x", "y" or "z"')

            if not self.silent:
                # Display progress
                count = progress(ii, count, self.nD)

        print("Done 100% ...forward operator completed!!\n")

        return F

    @property
    def modelMap(self):
        """
            Call for general mapping of the problem
        """
        return self.chiMap


class MagneticVector(MagneticIntegral):

    forwardOnly = False  # If false, matric is store to memory (watch your RAM)
    actInd = None  #: Active cell indices provided
    M = None  #: magType matrix provided, otherwise all induced
    # coordinate_system = 'cartesian'  # Formulation either "cartesian" | "spherical"
    magType = 'full'  # magType component
    silent = False  # Don't display progress on screen
    scale = 1.
    W = None
    threshold = None
    memory_saving_mode = False
    coordinate_system = properties.StringChoice(
    "Type of coordinate system we are regularizing in",
    choices=['cartesian', 'spherical'],
    default='cartesian' )

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

    def fwr_ind(self, m):

        if self.forwardOnly:

            # Compute the linear operation without forming the full dense G
            fwr_d = Intrgl_Fwr_Op(m=m, magType=self.magType)

            return fwr_d

        else:

            vec = np.dot(self.F, m.astype(np.float32))
            return vec.astype(np.float64)

    @property
    def F(self):
        if not self.ispaired:
            raise Exception('Need to pair!')

        if getattr(self, '_F', None) is None:

            self._F = self.Intrgl_Fwr_Op(magType=self.magType)

        return self._F

    def fields(self, chi, **kwargs):

        if self.coordinate_system == 'cartesian':
            m = self.chiMap*(chi)
        else:
            m = self.chiMap*(atp2xyz(chi.reshape((int(len(chi)/3), 3), order='F')))

        u = self.fwr_ind(m=m)

        return u

    def getJtJdiag(self, m, W=None):
        """
            Return the diagonal of JtJ
        """

        if W is None:
            W = Utils.speye(self.F.shape[0])

        dmudm = self.chiMap.deriv(m)

        if self.coordinate_system == 'cartesian':
            return np.sum((W * self.F * dmudm)**2., axis=0)

        else:

            gtgdiag = np.sum((W * self.F)**2., axis=0)
            Japprox = Utils.sdiag(mkvc(gtgdiag)**0.5) * (self.S * dmudm)

            return (Japprox.T*Japprox).diagonal()


    def getJ(self, chi, f=None):

        if self.coordinate_system == 'cartesian':

            return self.F*self.chiMap.deriv(chi)

        else:
            dmudm = self.S*self.chiMap.deriv(chi)

            return self.F * dmudm

    def Jvec(self, chi, v, f=None):

        if self.coordinate_system == 'cartesian':

            vec = np.dot(self.F, (self.chiMap.deriv(chi)*v).astype(np.float32))
            return vec.astype(np.float64)

        else:
            dmudm = self.S*self.chiMap.deriv(chi)
            vec = np.dot(self.F, (dmudm.dot(v)).astype(np.float32))
            return vec.astype(np.float64)

    def Jtvec(self, chi, v, f=None):

        vec = np.dot(self.F.T, v.astype(np.float32))

        vec = vec.astype(np.float64)
        if self.coordinate_system == 'cartesian':

            return self.chiMap.deriv(chi).T*(vec)

        else:

            dmudm = self.chiMap.deriv(chi).T * self.S.T

            return (dmudm).dot(vec)

    @property
    def S(self):

        if getattr(self, '_S', None) is None:

            if self.model is None:
                raise Exception('Requires a chi')

            # nC = int(self.mapPair().shape[0]/3)

            # TEST - CONVERT TO CARTESIAN FOR TILE INTERPOLATION
            nC = int(len(self.model)/3)
            m_xyz = self.chiMap * atp2xyz(self.model.reshape((nC, 3), order='F'))
            m_atp = xyz2atp(m_xyz.reshape((nC, 3), order='F'))

            nC = int(m_atp.shape[0]/3.)
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

            self._S = sp.vstack([Sx, Sy, Sz])

        return self._S


class MagneticAmplitude(MagneticIntegral):

    forwardOnly = False  # If false, matric is store to memory (watch your RAM)
    actInd = None  #: Active cell indices provided
    M = None  #: magType matrix provided, otherwise all induced
    magType = 'H0'  #: Option "H0", "x", "y", "z", "full" (for Joint)
    recType = 'xyz'
    silent = False  # Don't display progress on screen
    scale = 1.
    W = None
    coordinate_system = 'suscEffective'
    threshold = None

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

    def fwr_ind(self, chi):

        # Switch to avoid forming the dense matrix
        if self.forwardOnly:

            if self.coordinate_system == 'spherical':
                self.model = atp2xyz(chi)
            else:
                self.model = chi

            # Compute the linear operation without forming the full dense G
            m = self.chiMap * self.model

            if self.coordinate_system != 'suscEffective':
                self.magType = 'full'

            Bxyz = []
            for rtype in ['x', 'y', 'z']:
                Bxyz += [Intrgl_Fwr_Op(m=m, magType=self.magType,
                                       recType=rtype)]

            return self.calcAmpData(np.r_[Bxyz])

        else:
            if chi is None:

                if self.model is None:
                    raise Exception('Problem needs a chi chi')

                else:
                    m = self.chiMap * self.model

            else:

                self.model = chi
                m = self.chiMap * self.model

            if self.coordinate_system == 'spherical':
                m = atp2xyz(m)
            else:
                m = m


            if getattr(self, '_Mxyz', None) is not None:

                Bxyz = np.dot(self.F, (self.Mxyz*m).astype(np.float32))

            else:
                Bxyz = np.dot(self.F, m.astype(np.float32))

            return self.calcAmpData(Bxyz.astype(np.float64))

    def calcAmpData(self, Bxyz):
        """
            Compute amplitude of the field
        """

        Bamp = np.sum(Bxyz.reshape((self.nD, 3), order='F')**2., axis=1)**0.5

        return Bamp

    def fields(self, chi, **kwargs):

        ampB = self.fwr_ind(chi)

        return ampB

    def getJtJdiag(self, m, W=None):
        """
            Return the diagonal of JtJ
        """

        if W is None:
            W = 1.

        dmudm = self.chiMap.deriv(m)
        if self.coordinate_system == 'cartesian':
            return np.sum((W * self.dfdm * self.F * dmudm)**2., axis=0)

        else:
            return np.sum(((W * self.dfdm) * self.F * (self.S * dmudm))**2., axis=0)

    def getJ(self, chi, f=None):

        if self.coordinate_system == 'spherical':
            dmudm = self.S * self.chiMap.deriv(chi)
        else:
            dmudm = self.chiMap.deriv(chi)

        return self.dfdm * (self.F * dmudm)

    def Jvec(self, chi, v, f=None):

        if self.coordinate_system == 'spherical':
            dmudm = self.S * self.chiMap.deriv(chi)
        else:
            dmudm = self.chiMap.deriv(chi)

        # vec = np.empty(self.F.shape[0])
        # for ii in range(self.F.shape[0]):
        #     vec[ii] = self.F[ii, :].dot(dmudm*v)

        if getattr(self, '_Mxyz', None) is not None:

            vec = np.dot(self.F, (self.Mxyz*(dmudm*v)).astype(np.float32))

        else:
            vec = np.dot(self.F, (dmudm*v).astype(np.float32))

        return self.dfdm*vec.astype(np.float64)

    def Jtvec(self, chi, v, f=None):
        if self.coordinate_system == 'spherical':
            dmudm = self.S * self.chiMap.deriv(chi)
        else:
            dmudm = self.chiMap.deriv(chi)

        # vec = np.empty(self.F.shape[1])
        # for ii in range(self.F.shape[1]):
        #     vec[ii] = self.F[:, ii].dot(self.dfdm.T*v)
        if getattr(self, '_Mxyz', None) is not None:

            vec = self.Mxyz.T*np.dot(self.F.T, (self.dfdm.T*v).astype(np.float32)).astype(np.float64)

        else:
            vec = np.dot(self.F.T, (self.dfdm.T*v).astype(np.float32))

        return dmudm.T * vec.astype(np.float64)

    @property
    def F(self):
        if not self.ispaired:
            raise Exception('Need to pair!')

        if getattr(self, '_F', None) is None:
            if self.coordinate_system != 'suscEffective':
                self.magType = 'full'

            # self._F = []
            # for rtype in ['x', 'y', 'z']:
            #     self._F.append(self.Intrgl_Fwr_Op(magType=self.magType, recType=rtype))

            # self._F = np.vstack(self._F)
            self._F = self.Intrgl_Fwr_Op(magType=self.magType, recType=self.recType)
        return self._F

    @property
    def dfdm(self):

        if self.model is None:
            raise Exception('Problem needs a chi chi')

        if getattr(self, '_dfdm', None) is None:

            Bxyz = self.Bxyz_a(self.chiMap * self.model)

            Bx = sp.spdiags(Bxyz[:, 0], 0, self.nD, self.nD)
            By = sp.spdiags(Bxyz[:, 1], 0, self.nD, self.nD)
            Bz = sp.spdiags(Bxyz[:, 2], 0, self.nD, self.nD)

            self._dfdm = sp.hstack((Bx, By, Bz))

        return self._dfdm

    def Bxyz_a(self, m):
        """
            Return the normalized B fields
        """

        # Get field data
        if self.coordinate_system == 'spherical':
            m = atp2xyz(m)

        if getattr(self, '_Mxyz', None) is not None:
            Bxyz = np.dot(self.F, (self.Mxyz*m).astype(np.float32))
        else:
            Bxyz = np.dot(self.F, m.astype(np.float32))

        amp = self.calcAmpData(Bxyz.astype(np.float64))
        Bamp = sp.spdiags(1./amp, 0, self.nD, self.nD)

        return Bamp*Bxyz.reshape((self.nD, 3), order='F')

    @property
    def Mxyz(self):

        if getattr(self, '_Mxyz', None) is None:

            Mx = Utils.sdiag(self.M[:, 0])
            My = Utils.sdiag(self.M[:, 1])
            Mz = Utils.sdiag(self.M[:, 2])

            self._Mxyz = sp.vstack((Mx, My, Mz))

        return self._Mxyz

    @property
    def S(self):

        if getattr(self, '_S', None) is None:
            print('Updated S')
            if self.model is None:
                raise Exception('Requires a chi')

            nC = int(len(self.model)/3)

            a = self.model[:nC]
            t = self.model[nC:2*nC]
            p = self.model[2*nC:]

            Sx = sp.hstack([sp.diags(np.cos(t)*np.cos(p), 0),
                            sp.diags(-a*np.sin(t)*np.cos(p), 0),
                            sp.diags(-a*np.cos(t)*np.sin(p), 0)])

            Sy = sp.hstack([sp.diags(np.cos(t)*np.sin(p), 0),
                            sp.diags(-a*np.sin(t)*np.sin(p), 0),
                            sp.diags(a*np.cos(t)*np.cos(p), 0)])

            Sz = sp.hstack([sp.diags(np.sin(t), 0),
                            sp.diags(a*np.cos(t), 0),
                            sp.csr_matrix((nC, nC))])

            self._S = sp.vstack([Sx, Sy, Sz])

        return self._S


class Problem3D_DiffSecondary(Problem.BaseProblem):
    """
        Secondary field approach using differential equations!
    """

    surveyPair = MAG.BaseMagSurvey
    modelPair = MAG.BaseMagMap

    mu, muMap, muDeriv = Props.Invertible(
        "Magnetic Permeability (H/m)",
        default=mu_0
    )

    mui, muiMap, muiDeriv = Props.Invertible(
        "Inverse Magnetic Permeability (m/H)"
    )

    Props.Reciprocal(mu, mui)

    Solver = SimpegSolver  #: Type of solver to pair with
    solverOpts = {}  #: Solver options

    Ainv = None

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

        Pbc, Pin, self._Pout = \
            self.mesh.getBCProjWF('neumann', discretization='CC')

        Dface = self.mesh.faceDiv
        Mc = Utils.sdiag(self.mesh.vol)
        self._Div = Mc*Dface*Pin.T*Pin

    @property
    def MfMuI(self): return self._MfMuI

    @property
    def MfMui(self): return self._MfMui

    @property
    def MfMu0(self): return self._MfMu0

    def makeMassMatrices(self, m):
        mu = self.muMap*m
        self._MfMui = self.mesh.getFaceInnerProduct(1./mu)/self.mesh.dim
        # self._MfMui = self.mesh.getFaceInnerProduct(1./mu)
        # TODO: this will break if tensor mu
        self._MfMuI = Utils.sdiag(1./self._MfMui.diagonal())
        self._MfMu0 = self.mesh.getFaceInnerProduct(1./mu_0)/self.mesh.dim
        # self._MfMu0 = self.mesh.getFaceInnerProduct(1/mu_0)

    @Utils.requires('survey')
    def getB0(self):
        b0 = self.survey.B0
        B0 = np.r_[b0[0]*np.ones(self.mesh.nFx),
                   b0[1]*np.ones(self.mesh.nFy),
                   b0[2]*np.ones(self.mesh.nFz)]
        return B0

    def getRHS(self, m):
        """

        .. math ::

            \mathbf{rhs} = \Div(\MfMui)^{-1}\mathbf{M}^f_{\mu_0^{-1}}\mathbf{B}_0 - \Div\mathbf{B}_0+\diag(v)\mathbf{D} \mathbf{P}_{out}^T \mathbf{B}_{sBC}

        """
        B0 = self.getB0()
        Dface = self.mesh.faceDiv
        Mc = Utils.sdiag(self.mesh.vol)

        mu = self.muMap*m
        chi = mu/mu_0-1

        # Temporary fix
        Bbc, Bbc_const = CongruousMagBC(self.mesh, self.survey.B0, chi)
        self.Bbc = Bbc
        self.Bbc_const = Bbc_const
        # return self._Div*self.MfMuI*self.MfMu0*B0 - self._Div*B0 + Mc*Dface*self._Pout.T*Bbc
        return self._Div*self.MfMuI*self.MfMu0*B0 - self._Div*B0

    def getA(self, m):
        """
        GetA creates and returns the A matrix for the Magnetics problem

        The A matrix has the form:

        .. math ::

            \mathbf{A} =  \Div(\MfMui)^{-1}\Div^{T}

        """
        return self._Div*self.MfMuI*self._Div.T

    def fields(self, m=None):
        """
            Return magnetic potential (u) and flux (B)
            u: defined on the cell center [nC x 1]
            B: defined on the cell center [nF x 1]

            After we compute u, then we update B.

            .. math ::

                \mathbf{B}_s = (\MfMui)^{-1}\mathbf{M}^f_{\mu_0^{-1}}\mathbf{B}_0-\mathbf{B}_0 -(\MfMui)^{-1}\Div^T \mathbf{u}

        """

        if m is not None:
            self.model = m

        if self.Ainv is not None:
            self.Ainv.clean()

        self.makeMassMatrices(m)
        A = self.getA(m)

        self.Ainv = self.Solver(A, **self.solverOpts)

        RHS = self.getRHS(m)

        # m1 = sp.linalg.interface.aslinearoperator(Utils.sdiag(1/A.diagonal()))
        u = self.Ainv * RHS
        B0 = self.getB0()
        B = self.MfMuI*self.MfMu0*B0-B0-self.MfMuI*self._Div.T*u

        return {'B': B, 'u': u}


    @Utils.timeIt
    def Jvec(self, m, v, f=None):
        """
            Computing Jacobian multiplied by vector

            By setting our problem as

            .. math ::

                \mathbf{C}(\mathbf{m}, \mathbf{u}) = \mathbf{A}\mathbf{u} - \mathbf{rhs} = 0

            And taking derivative w.r.t m

            .. math ::

                \\nabla \mathbf{C}(\mathbf{m}, \mathbf{u}) = \\nabla_m \mathbf{C}(\mathbf{m}) \delta \mathbf{m} +
                                                             \\nabla_u \mathbf{C}(\mathbf{u}) \delta \mathbf{u} = 0

                \\frac{\delta \mathbf{u}}{\delta \mathbf{m}} = - [\\nabla_u \mathbf{C}(\mathbf{u})]^{-1}\\nabla_m \mathbf{C}(\mathbf{m})

            With some linear algebra we can have

            .. math ::

                \\nabla_u \mathbf{C}(\mathbf{u}) = \mathbf{A}

                \\nabla_m \mathbf{C}(\mathbf{m}) =
                \\frac{\partial \mathbf{A}}{\partial \mathbf{m}}(\mathbf{m})\mathbf{u} - \\frac{\partial \mathbf{rhs}(\mathbf{m})}{\partial \mathbf{m}}

            .. math ::

                \\frac{\partial \mathbf{A}}{\partial \mathbf{m}}(\mathbf{m})\mathbf{u} =
                \\frac{\partial \mathbf{\mu}}{\partial \mathbf{m}} \left[\Div \diag (\Div^T \mathbf{u}) \dMfMuI \\right]

                \dMfMuI = \diag(\MfMui)^{-1}_{vec} \mathbf{Av}_{F2CC}^T\diag(\mathbf{v})\diag(\\frac{1}{\mu^2})

                \\frac{\partial \mathbf{rhs}(\mathbf{m})}{\partial \mathbf{m}} =  \\frac{\partial \mathbf{\mu}}{\partial \mathbf{m}} \left[
                \Div \diag(\M^f_{\mu_{0}^{-1}}\mathbf{B}_0) \dMfMuI \\right] - \diag(\mathbf{v})\mathbf{D} \mathbf{P}_{out}^T\\frac{\partial B_{sBC}}{\partial \mathbf{m}}

            In the end,

            .. math ::

                \\frac{\delta \mathbf{u}}{\delta \mathbf{m}} =
                - [ \mathbf{A} ]^{-1}\left[ \\frac{\partial \mathbf{A}}{\partial \mathbf{m}}(\mathbf{m})\mathbf{u}
                - \\frac{\partial \mathbf{rhs}(\mathbf{m})}{\partial \mathbf{m}} \\right]

            A little tricky point here is we are not interested in potential (u), but interested in magnetic flux (B).
            Thus, we need sensitivity for B. Now we take derivative of B w.r.t m and have

            .. math ::

                \\frac{\delta \mathbf{B}} {\delta \mathbf{m}} = \\frac{\partial \mathbf{\mu} } {\partial \mathbf{m} }
                \left[
                \diag(\M^f_{\mu_{0}^{-1} } \mathbf{B}_0) \dMfMuI  \\
                 -  \diag (\Div^T\mathbf{u})\dMfMuI
                \\right ]

                 -  (\MfMui)^{-1}\Div^T\\frac{\delta\mathbf{u}}{\delta \mathbf{m}}

            Finally we evaluate the above, but we should remember that

            .. note ::

                We only want to evalute

                .. math ::

                    \mathbf{J}\mathbf{v} = \\frac{\delta \mathbf{P}\mathbf{B}} {\delta \mathbf{m}}\mathbf{v}

                Since forming sensitivity matrix is very expensive in that this monster is "big" and "dense" matrix!!


        """
        if f is None:
            f = self.fields(m)

        B, u = f['B'], f['u']
        mu = self.muMap*(m)
        dmudm = self.muDeriv
        dchidmu = Utils.sdiag(1/mu_0*np.ones(self.mesh.nC))

        vol = self.mesh.vol
        Div = self._Div
        Dface = self.mesh.faceDiv
        P = self.survey.projectFieldsDeriv(B)  # Projection matrix
        B0 = self.getB0()

        MfMuIvec = 1/self.MfMui.diagonal()
        dMfMuI = (Utils.sdiag(MfMuIvec**2)*self.mesh.aveF2CC.T *
                  Utils.sdiag(vol*1./mu**2))

        dCdm_A = Div * (Utils.sdiag(Div.T * u) * dMfMuI * dmudm)
        dCdm_RHS1 = Div * (Utils.sdiag(self.MfMu0 * B0) * dMfMuI)
        temp1 = (Dface*(self._Pout.T*self.Bbc_const*self.Bbc))
        dCdm_RHS2v = (Utils.sdiag(vol)*temp1)*np.inner(vol, dchidmu*dmudm*v)

        # dCdm_RHSv =  dCdm_RHS1*(dmudm*v) +  dCdm_RHS2v
        dCdm_RHSv = dCdm_RHS1 * (dmudm * v)
        dCdm_v = dCdm_A * v - dCdm_RHSv

        sol = self.Ainv*dCdm_v

        dudm = -sol
        dBdmv = (Utils.sdiag(self.MfMu0*B0)*(dMfMuI * (dmudm*v)) -
                 Utils.sdiag(Div.T*u)*(dMfMuI * (dmudm*v)) -
                 self.MfMuI*(Div.T * (dudm)))

        return Utils.mkvc(P*dBdmv)

    @Utils.timeIt
    def Jtvec(self, m, v, f=None):
        """
            Computing Jacobian^T multiplied by vector.

        .. math ::

            (\\frac{\delta \mathbf{P}\mathbf{B}} {\delta \mathbf{m}})^{T} = \left[ \mathbf{P}_{deriv}\\frac{\partial \mathbf{\mu} } {\partial \mathbf{m} }
            \left[
            \diag(\M^f_{\mu_{0}^{-1} } \mathbf{B}_0) \dMfMuI  \\
             -  \diag (\Div^T\mathbf{u})\dMfMuI
            \\right ]\\right]^{T}

             -  \left[\mathbf{P}_{deriv}(\MfMui)^{-1}\Div^T\\frac{\delta\mathbf{u}}{\delta \mathbf{m}} \\right]^{T}

        where

        .. math ::

            \mathbf{P}_{derv} = \\frac{\partial \mathbf{P}}{\partial\mathbf{B}}

        .. note ::

            Here we only want to compute

            .. math ::

                \mathbf{J}^{T}\mathbf{v} = (\\frac{\delta \mathbf{P}\mathbf{B}} {\delta \mathbf{m}})^{T} \mathbf{v}

        """
        if f is None:
            f = self.fields(m)

        B, u = f['B'], f['u']
        mu = self.muMap*(m)
        dmudm = self.muMap.deriv(m)
        dchidmu = Utils.sdiag(1/mu_0*np.ones(self.mesh.nC))

        vol = self.mesh.vol
        Div = self._Div
        Dface = self.mesh.faceDiv
        P = self.survey.projectFieldsDeriv(B)                 # Projection matrix
        B0 = self.getB0()

        MfMuIvec = 1/self.MfMui.diagonal()
        dMfMuI = Utils.sdiag(MfMuIvec**2)*self.mesh.aveF2CC.T*Utils.sdiag(vol*1./mu**2)

        s = Div * (self.MfMuI.T * (P.T*v))
        sol = self.Ainv*s

        dCdm_Atsol = (dmudm.T * dMfMuI.T*(Utils.sdiag(Div.T * u) * Div.T)) * sol

        dCdm_RHS1tsol = (dmudm.T * dMfMuI.T*(Utils.sdiag(self.MfMu0*B0)) * Div.T) * sol

        temp1sol = (Dface.T*(Utils.sdiag(vol)*sol))
        temp2 = self.Bbc_const*(self._Pout.T*self.Bbc).T
        dCdm_RHS2tsol = (dmudm.T*dchidmu.T*vol)*np.inner(temp2, temp1sol)

        # temporary fix
        # dCdm_RHStsol = dCdm_RHS1tsol - dCdm_RHS2tsol
        dCdm_RHStsol = dCdm_RHS1tsol

        # dCdm_RHSv =  dCdm_RHS1*(dmudm*v) +  dCdm_RHS2v
        # dCdm_v = dCdm_A*v - dCdm_RHSv

        Ctv = dCdm_Atsol - dCdm_RHStsol

        # B = self.MfMuI*self.MfMu0*B0-B0-self.MfMuI*self._Div.T*u
        # dBdm = d\mudm*dBd\mu
        # dPBdm^T*v = Atemp^T*P^T*v - Btemp^T*P^T*v - Ctv

        Atemp = Utils.sdiag(self.MfMu0*B0)*(dMfMuI * (dmudm))
        Btemp = Utils.sdiag(Div.T*u)*(dMfMuI * (dmudm))
        Jtv = Atemp.T*(P.T*v) - Btemp.T*(P.T*v) - Ctv

        return Utils.mkvc(Jtv)


# def MagneticsDiffSecondaryInv(mesh, model, data, **kwargs):

#     """
#         Inversion module for MagneticsDiffSecondary

#     """
#     from SimPEG import Optimization, Regularization, Parameters, ObjFunction, Inversion
#     prob = MagneticsDiffSecondary(mesh, model)

#     miter = kwargs.get('maxIter', 10)

#     if prob.ispaired:
#         prob.unpair()
#     if data.ispaired:
#         data.unpair()
#     prob.pair(data)

#     # Create an optimization program
#     opt = Optimization.InexactGaussNewton(maxIter=miter)
#     opt.bfgsH0 = Solver(sp.identity(model.nP), flag='D')
#     # Create a regularization program
#     reg = Regularization.Tikhonov(model)
#     # Create an objective function
#     beta = Parameters.BetaSchedule(beta0=1e0)
#     obj = ObjFunction.BaseObjFunction(data, reg, beta=beta)
#     # Create an inversion object
#     inv = Inversion.BaseInversion(obj, opt)

#     return inv, reg


def get_T_mat(Xn, Yn, Zn, rxLoc):
    """
    Load in the active nodes of a tensor mesh and computes the magnetic tensor
    for a given observation location rxLoc[obsx, obsy, obsz]

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

    Created on Oct, 20th 2015

    @author: dominiquef

     """

    eps = 1e-8  # add a small value to the locations to avoid /0

    nC = Xn.shape[0]

    # Pre-allocate space for 1D array
    Tx = np.zeros((1, 3*nC))
    Ty = np.zeros((1, 3*nC))
    Tz = np.zeros((1, 3*nC))

    dz2 = rxLoc[2] - Zn[:, 0]
    dz1 = rxLoc[2] - Zn[:, 1]

    dy2 = Yn[:, 1] - rxLoc[1]
    dy1 = Yn[:, 0] - rxLoc[1]

    dx2 = Xn[:, 1] - rxLoc[0]
    dx1 = Xn[:, 0] - rxLoc[0]

    R1 = (dy2**2 + dx2**2)
    R2 = (dy2**2 + dx1**2)
    R3 = (dy1**2 + dx2**2)
    R4 = (dy1**2 + dx1**2)

    arg1 = np.sqrt(dz2**2 + R2)
    arg2 = np.sqrt(dz2**2 + R1)
    arg3 = np.sqrt(dz1**2 + R1)
    arg4 = np.sqrt(dz1**2 + R2)
    arg5 = np.sqrt(dz2**2 + R3)
    arg6 = np.sqrt(dz2**2 + R4)
    arg7 = np.sqrt(dz1**2 + R4)
    arg8 = np.sqrt(dz1**2 + R3)

    Tx[0, 0:nC] = (
        np.arctan2(dy1 * dz2, (dx2 * arg5 + eps)) -
        np.arctan2(dy2 * dz2, (dx2 * arg2 + eps)) +
        np.arctan2(dy2 * dz1, (dx2 * arg3 + eps)) -
        np.arctan2(dy1 * dz1, (dx2 * arg8 + eps)) +
        np.arctan2(dy2 * dz2, (dx1 * arg1 + eps)) -
        np.arctan2(dy1 * dz2, (dx1 * arg6 + eps)) +
        np.arctan2(dy1 * dz1, (dx1 * arg7 + eps)) -
        np.arctan2(dy2 * dz1, (dx1 * arg4 + eps))
    )

    Ty[0, 0:nC] = (
        np.log((dz2 + arg2) / (dz1 + arg3 + eps)) -
        np.log((dz2 + arg1) / (dz1 + arg4 + eps)) +
        np.log((dz2 + arg6) / (dz1 + arg7 + eps)) -
        np.log((dz2 + arg5) / (dz1 + arg8 + eps))
    )

    Ty[0, nC:2*nC] = (
        np.arctan2(dx1 * dz2, (dy2 * arg1 + eps)) -
        np.arctan2(dx2 * dz2, (dy2 * arg2 + eps)) +
        np.arctan2(dx2 * dz1, (dy2 * arg3 + eps)) -
        np.arctan2(dx1 * dz1, (dy2 * arg4 + eps)) +
        np.arctan2(dx2 * dz2, (dy1 * arg5 + eps)) -
        np.arctan2(dx1 * dz2, (dy1 * arg6 + eps)) +
        np.arctan2(dx1 * dz1, (dy1 * arg7 + eps)) -
        np.arctan2(dx2 * dz1, (dy1 * arg8 + eps))
    )

    R1 = (dy2**2 + dz1**2)
    R2 = (dy2**2 + dz2**2)
    R3 = (dy1**2 + dz1**2)
    R4 = (dy1**2 + dz2**2)

    Ty[0, 2*nC:] = (
        np.log((dx1 + np.sqrt(dx1**2 + R1) + eps) /
               (dx2 + np.sqrt(dx2**2 + R1) + eps)) -
        np.log((dx1 + np.sqrt(dx1**2 + R2) + eps) /
               (dx2 + np.sqrt(dx2**2 + R2) + eps)) +
        np.log((dx1 + np.sqrt(dx1**2 + R4) + eps) /
               (dx2 + np.sqrt(dx2**2 + R4) + eps)) -
        np.log((dx1 + np.sqrt(dx1**2 + R3) + eps) /
               (dx2 + np.sqrt(dx2**2 + R3) + eps))
    )

    R1 = (dx2**2 + dz1**2)
    R2 = (dx2**2 + dz2**2)
    R3 = (dx1**2 + dz1**2)
    R4 = (dx1**2 + dz2**2)

    Tx[0, 2*nC:] = (
        np.log((dy1 + np.sqrt(dy1**2 + R1) + eps) /
               (dy2 + np.sqrt(dy2**2 + R1) + eps)) -
        np.log((dy1 + np.sqrt(dy1**2 + R2) + eps) /
               (dy2 + np.sqrt(dy2**2 + R2) + eps)) +
        np.log((dy1 + np.sqrt(dy1**2 + R4) + eps) /
               (dy2 + np.sqrt(dy2**2 + R4) + eps)) -
        np.log((dy1 + np.sqrt(dy1**2 + R3) + eps) /
               (dy2 + np.sqrt(dy2**2 + R3) + eps))
    )

    Tz[0, 2*nC:] = -(Ty[0, nC:2*nC] + Tx[0, 0:nC])
    Tz[0, nC:2*nC] = Ty[0, 2*nC:]
    Tx[0, nC:2*nC] = Ty[0, 0:nC]
    Tz[0, 0:nC] = Tx[0, 2*nC:]

    Tx = Tx/(4*np.pi)
    Ty = Ty/(4*np.pi)
    Tz = Tz/(4*np.pi)

    return Tx, Ty, Tz


def progress(iter, prog, final):
    """
    progress(iter,prog,final)

    Function measuring the progress of a process and print to screen the %.
    Useful to estimate the remaining runtime of a large problem.

    Created on Dec, 20th 2015

    @author: dominiquef
    """
    arg = np.floor(float(iter)/float(final)*10.)

    if arg > prog:

        print("Done " + str(arg*10) + " %")
        prog = arg

    return prog


def get_dist_wgt(mesh, rxLoc, actv, R, R0):
    """
    get_dist_wgt(xn,yn,zn,rxLoc,R,R0)

    Function creating a distance weighting function required for the magnetic
    inverse problem.

    INPUT
    xn, yn, zn : Node location
    rxLoc       : Observation locations [obsx, obsy, obsz]
    actv        : Active cell vector [0:air , 1: ground]
    R           : Decay factor (mag=3, grav =2)
    R0          : Small factor added (default=dx/4)

    OUTPUT
    wr       : [nC] Vector of distance weighting

    Created on Dec, 20th 2015

    @author: dominiquef
    """

    # Find non-zero cells
    if actv.dtype == 'bool':
        inds = np.asarray([inds for inds,
                          elem in enumerate(actv, 1) if elem], dtype=int) - 1
    else:
        inds = actv

    nC = len(inds)

    # Create active cell projector
    P = sp.csr_matrix((np.ones(nC), (inds, range(nC))),
                      shape=(mesh.nC, nC))

    # Geometrical constant
    p = 1/np.sqrt(3)

    # Create cell center location
    Ym, Xm, Zm = np.meshgrid(mesh.vectorCCy, mesh.vectorCCx, mesh.vectorCCz)
    hY, hX, hZ = np.meshgrid(mesh.hy, mesh.hx, mesh.hz)

    # Remove air cells
    Xm = P.T*Utils.mkvc(Xm)
    Ym = P.T*Utils.mkvc(Ym)
    Zm = P.T*Utils.mkvc(Zm)

    hX = P.T*Utils.mkvc(hX)
    hY = P.T*Utils.mkvc(hY)
    hZ = P.T*Utils.mkvc(hZ)

    V = P.T * Utils.mkvc(mesh.vol)
    wr = np.zeros(nC)

    ndata = rxLoc.shape[0]
    count = -1
    print("Begin calculation of distance weighting for R= " + str(R))

    for dd in range(ndata):

        nx1 = (Xm - hX * p - rxLoc[dd, 0])**2
        nx2 = (Xm + hX * p - rxLoc[dd, 0])**2

        ny1 = (Ym - hY * p - rxLoc[dd, 1])**2
        ny2 = (Ym + hY * p - rxLoc[dd, 1])**2

        nz1 = (Zm - hZ * p - rxLoc[dd, 2])**2
        nz2 = (Zm + hZ * p - rxLoc[dd, 2])**2

        R1 = np.sqrt(nx1 + ny1 + nz1)
        R2 = np.sqrt(nx1 + ny1 + nz2)
        R3 = np.sqrt(nx2 + ny1 + nz1)
        R4 = np.sqrt(nx2 + ny1 + nz2)
        R5 = np.sqrt(nx1 + ny2 + nz1)
        R6 = np.sqrt(nx1 + ny2 + nz2)
        R7 = np.sqrt(nx2 + ny2 + nz1)
        R8 = np.sqrt(nx2 + ny2 + nz2)

        temp = (R1 + R0)**-R + (R2 + R0)**-R + (R3 + R0)**-R + \
            (R4 + R0)**-R + (R5 + R0)**-R + (R6 + R0)**-R + \
            (R7 + R0)**-R + (R8 + R0)**-R

        wr = wr + (V*temp/8.)**2.

        count = progress(dd, count, ndata)

    wr = np.sqrt(wr)/V
    wr = Utils.mkvc(wr)
    wr = np.sqrt(wr/(np.max(wr)))

    print("Done 100% ...distance weighting completed!!\n")

    return wr


def writeUBCobs(filename, survey, d=None):
    """
    writeUBCobs(filename,B,M,rxLoc,d,wd)

    Function writing an observation file in UBC-MAG3D format.

    INPUT
    filename    : Name of out file including directory
    survey
    flag          : dobs | dpred

    OUTPUT
    Obsfile

    Created on Dec, 27th 2015

    @author: dominiquef
    """

    B = survey.srcField.param

    rxLoc = survey.srcField.rxList[0].locs

    wd = survey.std

    if d is None:
        d = survey.dobs

    data = np.c_[rxLoc, d, wd]
    head = ('%6.2f %6.2f %6.2f\n' % (B[1], B[2], B[0]) +
            '%6.2f %6.2f %6.2f\n' % (B[1], B[2], 1) +
            '%i' % len(d))
    np.savetxt(filename, data, fmt='%e', delimiter=' ', newline='\n',
               header=head, comments='')

    #print("Observation file saved to: " + filename)


def plot_obs_2D(rxLoc, d=None, title=None,
                vmin=None, vmax=None, levels=None, fig=None, ax=None,
                colorbar=True, marker=True, cmap="plasma_r", npts=100):

    """ Function plot_obs(rxLoc,d)
    Generate a 2d interpolated plot from scatter points of data

    INPUT
    rxLoc       : Observation locations [x,y,z]
    d           : Data vector

    OUTPUT
    figure()

    Created on Dec, 27th 2015

    @author: dominiquef

    """

    from scipy.interpolate import griddata
    import pylab as plt

    # Plot result
    if fig is None:
        fig = plt.figure()

    if ax is None:
        ax = plt.subplot()

    plt.sca(ax)
    if marker:
        plt.scatter(rxLoc[:, 0], rxLoc[:, 1], c='k', s=10)

    if d is not None:

        if (vmin is None):
            vmin = d.min()

        if (vmax is None):
            vmax = d.max()


        # Create grid of points
        x = np.linspace(rxLoc[:, 0].min(), rxLoc[:, 0].max(), npts)
        y = np.linspace(rxLoc[:, 1].min(), rxLoc[:, 1].max(), npts)

        X, Y = np.meshgrid(x, y)

        # Interpolate
        d_grid = griddata(rxLoc[:, 0:2], d, (X, Y), method='linear')
        im = plt.imshow(d_grid, extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)

        if colorbar:
            plt.colorbar(fraction=0.02)

        if levels is None:

            if vmin != vmax:
                plt.contour(X, Y, d_grid, 10, vmin=vmin, vmax=vmax, cmap=cmap)
        else:
            plt.contour(X, Y, d_grid, levels=levels, colors='k',
                        vmin=vmin, vmax=vmax)

    if title is not None:
        plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')

    return fig, im


def plotModelSections(mesh, m, normal='x', ind=0, vmin=None, vmax=None,
                      subFact=2, scale=1., xlim=None, ylim=None, vec='k',
                      title=None, axs=None, ndv=-100, contours=None,
                      orientation='vertical', cmap='pink_r'):

    """
    Plot section through a 3D tensor model
    """
    # plot recovered model
    nC = mesh.nC

    if vmin is None:
        vmin = m.min()

    if vmax is None:
        vmax = m.max()

    if len(m) == 3*nC:
        m_lpx = m[0:nC]
        m_lpy = m[nC:2*nC]
        m_lpz = -m[2*nC:]

        m_lpx[m_lpx == ndv] = np.nan
        m_lpy[m_lpy == ndv] = np.nan
        m_lpz[m_lpz == ndv] = np.nan

        amp = np.sqrt(m_lpx**2. + m_lpy**2. + m_lpz**2.)

        m_lpx = (m_lpx).reshape(mesh.vnC, order='F')
        m_lpy = (m_lpy).reshape(mesh.vnC, order='F')
        m_lpz = (m_lpz).reshape(mesh.vnC, order='F')
        amp = amp.reshape(mesh.vnC, order='F')
    else:
        amp = m.reshape(mesh.vnC, order='F')

    xx = mesh.gridCC[:, 0].reshape(mesh.vnC, order="F")
    zz = mesh.gridCC[:, 2].reshape(mesh.vnC, order="F")
    yy = mesh.gridCC[:, 1].reshape(mesh.vnC, order="F")

    if axs is None:
        fig, axs = plt.figure(), plt.subplot()

    if normal == 'x':
        xx = yy[ind, :, :].T
        yy = zz[ind, :, :].T
        model = amp[ind, :, :].T

        if len(m) == 3*nC:
            mx = m_lpy[ind, ::subFact, ::subFact].T
            my = m_lpz[ind, ::subFact, ::subFact].T

    elif normal == 'y':
        xx = xx[:, ind, :].T
        yy = zz[:, ind, :].T
        model = amp[:, ind, :].T

        if len(m) == 3*nC:
            mx = m_lpx[::subFact, ind, ::subFact].T
            my = m_lpz[::subFact, ind, ::subFact].T

    elif normal == 'z':
        xx = xx[:, :, ind].T
        yy = yy[:, :, ind].T
        model = amp[:, :, ind].T

        if len(m) == 3*nC:
            mx = m_lpx[::subFact, ::subFact, ind].T
            my = m_lpy[::subFact, ::subFact, ind].T

    im2 = axs.contourf(xx, yy, model,
                       15, vmin=vmin, vmax=vmax, clim=[vmin, vmax],
                       cmap=cmap)

    if contours is not None:
        axs.contour(xx, yy, model, contours, colors='k')

    if len(m) == 3*nC:
        pos = mkvc(mx**2.+my**2.) > 0
        axs.quiver(mkvc(xx[::subFact, ::subFact])[pos],
                   mkvc(yy[::subFact, ::subFact])[pos],
                   mkvc(mx)[pos],
                   mkvc(my)[pos],
                   pivot='mid',
                   scale_units="inches", scale=scale, linewidths=(1,),
                   edgecolors=(vec),
                   headaxislength=0.1, headwidth=10, headlength=30)
    cbar = plt.colorbar(im2, orientation=orientation, ax=axs,
                 ticks=np.linspace(im2.vmin, im2.vmax, 4),
                 format="${%.3f}$", shrink=0.5)
    axs.set_aspect('equal')

    if xlim is not None:
        axs.set_xlim(xlim[0], xlim[1])

    if ylim is not None:
        axs.set_ylim(ylim[0], ylim[1])

    if title is not None:
        axs.set_title(title)

    return axs, im2, cbar


def plotProfile(xyzd, a, b, npts, data=None,
                fig=None, ax=None, plotStr=None,
                coordinate_system='local'):
    """
    Plot the data and line profile inside the spcified limits
    """
    def linefun(x1, x2, y1, y2, nx, tol=1e-3):
        dx = x2-x1
        dy = y2-y1

        if np.abs(dx) <= tol:
            y = np.linspace(y1, y2, nx)
            x = np.ones_like(y)*x1
        elif np.abs(dy) <= tol:
            x = np.linspace(x1, x2, nx)
            y = np.ones_like(x)*y1
        else:
            x = np.linspace(x1, x2, nx)
            slope = (y2-y1)/(x2-x1)
            y = slope*(x-x1)+y1
        return x, y

    if fig is None:
        fig = plt.figure(figsize=(6, 9))

        plt.rcParams.update({'font.size': 14})

    if ax is None:
        ax = plt.subplot()

    x, y = linefun(a[0], b[0], a[1], b[1], npts)
    distance = np.sqrt((x-a[0])**2.+(y-a[1])**2.)
    dline = griddata(xyzd[:, :2], xyzd[:, -1], (x, y), method='linear')

    if coordinate_system == 'xProfile':
        distance += a[0]
    elif coordinate_system == 'yProfile':
        distance += a[1]

    ax.plot(distance, dline)

    if data is not None:

        # if len(plotStr) == len(data):
        for ii, d in enumerate(data):

            dline = griddata(xyzd[:, :2], d, (x, y), method='linear')

            if plotStr[ii]:
                ax.plot(distance, dline, plotStr[ii])
            else:
                ax.plot(distance, dline)

    ax.set_xlim(distance.min(), distance.max())

    # ax.set_xlabel("Distance (m)")
    # ax.set_ylabel("Magnetic field (nT)")

    #ax.text(distance.min(), dline.max()*0.8, 'A', fontsize = 16)
    # ax.text(distance.max()*0.97, out_linei.max()*0.8, 'B', fontsize = 16)
    # ax.legend(("Observed", "Simulated"), bbox_to_anchor=(0.5, -0.3))
    # ax.grid(True)

    return ax

def readMagneticsObservations(obs_file):
        """
            Read and write UBC mag file format

            INPUT:
            :param fileName, path to the UBC obs mag file

            OUTPUT:
            :param survey
            :param M, magnetization orentiaton (MI, MD)
        """

        fid = open(obs_file, 'r')

        # First line has the inclination,declination and amplitude of B0
        line = fid.readline()
        B = np.array(line.split(), dtype=float)

        # Second line has the magnetization orientation and a flag
        line = fid.readline()
        M = np.array(line.split(), dtype=float)

        # Third line has the number of rows
        line = fid.readline()
        ndat = int(line.strip())

        # Pre-allocate space for obsx, obsy, obsz, data, uncert
        line = fid.readline()
        temp = np.array(line.split(), dtype=float)

        d = np.zeros(ndat, dtype=float)
        wd = np.zeros(ndat, dtype=float)
        locXYZ = np.zeros((ndat, 3), dtype=float)

        for ii in range(ndat):

            temp = np.array(line.split(), dtype=float)
            locXYZ[ii, :] = temp[:3]

            if len(temp) > 3:
                d[ii] = temp[3]

                if len(temp) == 5:
                    wd[ii] = temp[4]

            line = fid.readline()

        rxLoc = MAG.RxObs(locXYZ)
        srcField = MAG.SrcField([rxLoc], param=(B[2], B[0], B[1]))
        survey = MAG.LinearSurvey(srcField)
        survey.dobs = d
        survey.std = wd
        return survey

def readVectorModel(mesh, modelFile):
    """
    Read UBC vector model
    """

    with open(modelFile) as f:
                magmodel = f.read()

    magmodel = magmodel.splitlines()
    M = []

    for line in magmodel:
        M.append([float(x) for x in line.split()])

    # Convert list to 2d array
    M = np.vstack(M)

    # Cycle through three components and permute from UBC to SimPEG
    for ii in range(3):
        m = np.reshape(M[:, ii],
                       (mesh.nCz, mesh.nCx, mesh.nCy),
                       order='F')

        m = m[::-1, :, :]
        m = np.transpose(m, (1, 2, 0))
        M[:, ii] = Utils.mkvc(m)

    return M
