from __future__ import print_function

import numpy as np
import scipy.sparse as sp
from scipy.constants import mu_0

from SimPEG import Utils
from SimPEG import Problem
from SimPEG import Solver
from SimPEG import Props
from SimPEG import Mesh
import multiprocessing
import properties
from SimPEG.Utils import mkvc, matutils, sdiag
from . import BaseMag as MAG
from .MagAnalytics import spheremodel, CongruousMagBC


class MagneticIntegral(Problem.LinearProblem):

    chi, chiMap, chiDeriv = Props.Invertible(
        "Magnetic Susceptibility (SI)",
        default=1.
    )

    forwardOnly = False  # If false, matrix is store to memory (watch your RAM)
    actInd = None  #: Active cell indices provided
    M = None  #: Magnetization matrix provided, otherwise all induced
    rx_type = 'tmi'  #: Receiver type either "tmi" | "xyz"
    magType = 'H0'
    equiSourceLayer = False
    silent = False  # Don't display progress on screen
    W = None
    gtgdiag = None
    memory_saving_mode = False
    n_cpu = None
    parallelized = False
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

    def __init__(self, mesh, **kwargs):

        assert mesh.dim == 3, 'Integral formulation only available for 3D mesh'
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

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

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

        Pbc, Pin, self._Pout = \
            self.mesh.getBCProjWF('neumann', discretization='CC')

        Dface = self.mesh.faceDiv
        Mc = sdiag(self.mesh.vol)
        self._Div = Mc * Dface * Pin.T * Pin

    @property
    def MfMuI(self): return self._MfMuI

    @property
    def MfMui(self): return self._MfMui

    @property
    def MfMu0(self): return self._MfMu0

    def makeMassMatrices(self, m):
        mu = self.muMap * m
        self._MfMui = self.mesh.getFaceInnerProduct(1. / mu) / self.mesh.dim
        # self._MfMui = self.mesh.getFaceInnerProduct(1./mu)
        # TODO: this will break if tensor mu
        self._MfMuI = sdiag(1. / self._MfMui.diagonal())
        self._MfMu0 = self.mesh.getFaceInnerProduct(1. / mu_0) / self.mesh.dim
        # self._MfMu0 = self.mesh.getFaceInnerProduct(1/mu_0)

    @Utils.requires('survey')
    def getB0(self):
        b0 = self.survey.B0
        B0 = np.r_[
            b0[0] * np.ones(self.mesh.nFx),
            b0[1] * np.ones(self.mesh.nFy),
            b0[2] * np.ones(self.mesh.nFz)
        ]
        return B0

    def getRHS(self, m):
        """

        .. math ::

            \mathbf{rhs} = \Div(\MfMui)^{-1}\mathbf{M}^f_{\mu_0^{-1}}\mathbf{B}_0 - \Div\mathbf{B}_0+\diag(v)\mathbf{D} \mathbf{P}_{out}^T \mathbf{B}_{sBC}

        """
        B0 = self.getB0()
        Dface = self.mesh.faceDiv
        # Mc = sdiag(self.mesh.vol)

        mu = self.muMap * m
        chi = mu / mu_0 - 1

        # Temporary fix
        Bbc, Bbc_const = CongruousMagBC(self.mesh, self.survey.B0, chi)
        self.Bbc = Bbc
        self.Bbc_const = Bbc_const
        # return self._Div*self.MfMuI*self.MfMu0*B0 - self._Div*B0 +
        # Mc*Dface*self._Pout.T*Bbc
        return self._Div * self.MfMuI * self.MfMu0 * B0 - self._Div * B0

    def getA(self, m):
        """
        GetA creates and returns the A matrix for the Magnetics problem

        The A matrix has the form:

        .. math ::

            \mathbf{A} =  \Div(\MfMui)^{-1}\Div^{T}

        """
        return self._Div * self.MfMuI * self._Div.T

    def fields(self, m):
        """
            Return magnetic potential (u) and flux (B)
            u: defined on the cell center [nC x 1]
            B: defined on the cell center [nG x 1]

            After we compute u, then we update B.

            .. math ::

                \mathbf{B}_s = (\MfMui)^{-1}\mathbf{M}^f_{\mu_0^{-1}}\mathbf{B}_0-\mathbf{B}_0 -(\MfMui)^{-1}\Div^T \mathbf{u}

        """
        self.makeMassMatrices(m)
        A = self.getA(m)
        rhs = self.getRHS(m)
        m1 = sp.linalg.interface.aslinearoperator(
            sdiag(1 / A.diagonal())
        )
        u, info = sp.linalg.bicgstab(A, rhs, tol=1e-6, maxiter=1000, M=m1)
        B0 = self.getB0()
        B = self.MfMuI * self.MfMu0 * B0 - B0 - self.MfMuI * self._Div.T * u

        return {'B': B, 'u': u}

    @Utils.timeIt
    def Jvec(self, m, v, u=None):
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
        if u is None:
            u = self.fields(m)

        B, u = u['B'], u['u']
        mu = self.muMap * (m)
        dmudm = self.muDeriv
        # dchidmu = sdiag(1 / mu_0 * np.ones(self.mesh.nC))

        vol = self.mesh.vol
        Div = self._Div
        Dface = self.mesh.faceDiv
        P = self.survey.projectFieldsDeriv(B)  # Projection matrix
        B0 = self.getB0()

        MfMuIvec = 1 / self.MfMui.diagonal()
        dMfMuI = sdiag(MfMuIvec**2) * \
            self.mesh.aveF2CC.T * sdiag(vol * 1. / mu**2)

        # A = self._Div*self.MfMuI*self._Div.T
        # RHS = Div*MfMuI*MfMu0*B0 - Div*B0 + Mc*Dface*Pout.T*Bbc
        # C(m,u) = A*m-rhs
        # dudm = -(dCdu)^(-1)dCdm

        dCdu = self.getA(m)
        dCdm_A = Div * (sdiag(Div.T * u) * dMfMuI * dmudm)
        dCdm_RHS1 = Div * (sdiag(self.MfMu0 * B0) * dMfMuI)
        # temp1 = (Dface * (self._Pout.T * self.Bbc_const * self.Bbc))
        # dCdm_RHS2v = (sdiag(vol) * temp1) * \
        #    np.inner(vol, dchidmu * dmudm * v)

        # dCdm_RHSv =  dCdm_RHS1*(dmudm*v) +  dCdm_RHS2v
        dCdm_RHSv = dCdm_RHS1 * (dmudm * v)
        dCdm_v = dCdm_A * v - dCdm_RHSv

        m1 = sp.linalg.interface.aslinearoperator(
            sdiag(1 / dCdu.diagonal())
        )
        sol, info = sp.linalg.bicgstab(dCdu, dCdm_v,
                                       tol=1e-6, maxiter=1000, M=m1)

        if info > 0:
            print("Iterative solver did not work well (Jvec)")
            # raise Exception ("Iterative solver did not work well")

        # B = self.MfMuI*self.MfMu0*B0-B0-self.MfMuI*self._Div.T*u
        # dBdm = d\mudm*dBd\mu

        dudm = -sol
        dBdmv = (
            sdiag(self.MfMu0 * B0) * (dMfMuI * (dmudm * v))
            - sdiag(Div.T * u) * (dMfMuI * (dmudm * v))
            - self.MfMuI * (Div.T * (dudm))
        )

        return mkvc(P * dBdmv)

    @Utils.timeIt
    def Jtvec(self, m, v, u=None):
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
        if u is None:
            u = self.fields(m)

        B, u = u['B'], u['u']
        mu = self.mapping * (m)
        dmudm = self.mapping.deriv(m)
        # dchidmu = sdiag(1 / mu_0 * np.ones(self.mesh.nC))

        vol = self.mesh.vol
        Div = self._Div
        Dface = self.mesh.faceDiv
        P = self.survey.projectFieldsDeriv(
            B)                 # Projection matrix
        B0 = self.getB0()

        MfMuIvec = 1 / self.MfMui.diagonal()
        dMfMuI = sdiag(MfMuIvec**2) * \
            self.mesh.aveF2CC.T * sdiag(vol * 1. / mu**2)

        # A = self._Div*self.MfMuI*self._Div.T
        # RHS = Div*MfMuI*MfMu0*B0 - Div*B0 + Mc*Dface*Pout.T*Bbc
        # C(m,u) = A*m-rhs
        # dudm = -(dCdu)^(-1)dCdm

        dCdu = self.getA(m)
        s = Div * (self.MfMuI.T * (P.T * v))

        m1 = sp.linalg.interface.aslinearoperator(
            sdiag(1 / (dCdu.T).diagonal())
        )
        sol, info = sp.linalg.bicgstab(dCdu.T, s, tol=1e-6, maxiter=1000, M=m1)

        if info > 0:
            print("Iterative solver did not work well (Jtvec)")
            # raise Exception ("Iterative solver did not work well")

        # dCdm_A = Div * ( sdiag( Div.T * u )* dMfMuI *dmudm  )
        # dCdm_Atsol = ( dMfMuI.T*( sdiag( Div.T * u ) * (Div.T * dmudm)) ) * sol
        dCdm_Atsol = (dmudm.T * dMfMuI.T *
                      (sdiag(Div.T * u) * Div.T)) * sol

        # dCdm_RHS1 = Div * (sdiag( self.MfMu0*B0  ) * dMfMuI)
        # dCdm_RHS1tsol = (dMfMuI.T*( sdiag( self.MfMu0*B0  ) ) * Div.T * dmudm) * sol
        dCdm_RHS1tsol = (
            dmudm.T * dMfMuI.T *
            (sdiag(self.MfMu0 * B0)) * Div.T
        ) * sol

        # temp1 = (Dface*(self._Pout.T*self.Bbc_const*self.Bbc))
        # temp1sol = (Dface.T * (sdiag(vol) * sol))
        # temp2 = self.Bbc_const * (self._Pout.T * self.Bbc).T
        # dCdm_RHS2v  = (sdiag(vol)*temp1)*np.inner(vol, dchidmu*dmudm*v)
        # dCdm_RHS2tsol = (dmudm.T * dchidmu.T * vol) * np.inner(temp2, temp1sol)

        # dCdm_RHSv =  dCdm_RHS1*(dmudm*v) +  dCdm_RHS2v

        # temporary fix
        # dCdm_RHStsol = dCdm_RHS1tsol - dCdm_RHS2tsol
        dCdm_RHStsol = dCdm_RHS1tsol

        # dCdm_RHSv =  dCdm_RHS1*(dmudm*v) +  dCdm_RHS2v
        # dCdm_v = dCdm_A*v - dCdm_RHSv

        Ctv = dCdm_Atsol - dCdm_RHStsol

        # B = self.MfMuI*self.MfMu0*B0-B0-self.MfMuI*self._Div.T*u
        # dBdm = d\mudm*dBd\mu
        # dPBdm^T*v = Atemp^T*P^T*v - Btemp^T*P^T*v - Ctv

        Atemp = sdiag(self.MfMu0 * B0) * (dMfMuI * (dmudm))
        Btemp = sdiag(Div.T * u) * (dMfMuI * (dmudm))
        Jtv = Atemp.T * (P.T * v) - Btemp.T * (P.T * v) - Ctv

        return mkvc(Jtv)


def MagneticsDiffSecondaryInv(mesh, model, data, **kwargs):
    """
        Inversion module for MagneticsDiffSecondary

    """
    from SimPEG import (
        Optimization, Regularization,
        Parameters, ObjFunction, Inversion
    )
    prob = MagneticsDiffSecondary(mesh, model)

    miter = kwargs.get('maxIter', 10)

    if prob.ispaired:
        prob.unpair()
    if data.ispaired:
        data.unpair()
    prob.pair(data)

    # Create an optimization program
    opt = Optimization.InexactGaussNewton(maxIter=miter)
    opt.bfgsH0 = Solver(sp.identity(model.nP), flag='D')
    # Create a regularization program
    reg = Regularization.Tikhonov(model)
    # Create an objective function
    beta = Parameters.BetaSchedule(beta0=1e0)
    obj = ObjFunction.BaseObjFunction(data, reg, beta=beta)
    # Create an inversion object
    inv = Inversion.BaseInversion(obj, opt)

    return inv, reg


def calcRow(Xn, Yn, Zn, rxLoc):
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

    dz2 = Zn[:, 1] - rxLoc[2] + eps
    dz1 = Zn[:, 0] - rxLoc[2] + eps

    dy2 = Yn[:, 1] - rxLoc[1] + eps
    dy1 = Yn[:, 0] - rxLoc[1] + eps

    dx2 = Xn[:, 1] - rxLoc[0] + eps
    dx1 = Xn[:, 0] - rxLoc[0] + eps

    dx2dx2 = dx2**2.
    dx1dx1 = dx1**2.

    dy2dy2 = dy2**2.
    dy1dy1 = dy1**2.

    dz2dz2 = dz2**2.
    dz1dz1 = dz1**2.

    R1 = (dy2dy2 + dx2dx2)
    R2 = (dy2dy2 + dx1dx1)
    R3 = (dy1dy1 + dx2dx2)
    R4 = (dy1dy1 + dx1dx1)

    arg1 = np.sqrt(dz2dz2 + R2)
    arg2 = np.sqrt(dz2dz2 + R1)
    arg3 = np.sqrt(dz1dz1 + R1)
    arg4 = np.sqrt(dz1dz1 + R2)
    arg5 = np.sqrt(dz2dz2 + R3)
    arg6 = np.sqrt(dz2dz2 + R4)
    arg7 = np.sqrt(dz1dz1 + R4)
    arg8 = np.sqrt(dz1dz1 + R3)

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
        np.log((dz2 + arg2 + eps) / (dz1 + arg3 + eps)) -
        np.log((dz2 + arg1 + eps) / (dz1 + arg4 + eps)) +
        np.log((dz2 + arg6 + eps) / (dz1 + arg7 + eps)) -
        np.log((dz2 + arg5 + eps) / (dz1 + arg8 + eps))
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

    R1 = (dy2dy2 + dz1dz1)
    R2 = (dy2dy2 + dz2dz2)
    R3 = (dy1dy1 + dz1dz1)
    R4 = (dy1dy1 + dz2dz2)

    Ty[0, 2*nC:] = (
        np.log((dx1 + np.sqrt(dx1dx1 + R1) + eps) /
               (dx2 + np.sqrt(dx2dx2 + R1) + eps)) -
        np.log((dx1 + np.sqrt(dx1dx1 + R2) + eps) /
               (dx2 + np.sqrt(dx2dx2 + R2) + eps)) +
        np.log((dx1 + np.sqrt(dx1dx1 + R4) + eps) /
               (dx2 + np.sqrt(dx2dx2 + R4) + eps)) -
        np.log((dx1 + np.sqrt(dx1dx1 + R3) + eps) /
               (dx2 + np.sqrt(dx2dx2 + R3) + eps))
    )

    R1 = (dx2dx2 + dz1dz1)
    R2 = (dx2dx2 + dz2dz2)
    R3 = (dx1dx1 + dz1dz1)
    R4 = (dx1dx1 + dz2dz2)

    Tx[0, 2*nC:] = (
        np.log((dy1 + np.sqrt(dy1dy1 + R1) + eps) /
               (dy2 + np.sqrt(dy2dy2 + R1) + eps)) -
        np.log((dy1 + np.sqrt(dy1dy1 + R2) + eps) /
               (dy2 + np.sqrt(dy2dy2 + R2) + eps)) +
        np.log((dy1 + np.sqrt(dy1dy1 + R4) + eps) /
               (dy2 + np.sqrt(dy2dy2 + R4) + eps)) -
        np.log((dy1 + np.sqrt(dy1dy1 + R3) + eps) /
               (dy2 + np.sqrt(dy2dy2 + R3) + eps))
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
        inds = np.where(actv)[0]
    else:
        inds = actv

    nC = len(inds)

    # Create active cell projector
    P = sp.csr_matrix((np.ones(nC), (inds, range(nC))),
                      shape=(mesh.nC, nC))

    # Geometrical constant
    p = 1 / np.sqrt(3)

    # Create cell center location
    Ym, Xm, Zm = np.meshgrid(mesh.vectorCCy, mesh.vectorCCx, mesh.vectorCCz)
    hY, hX, hZ = np.meshgrid(mesh.hy, mesh.hx, mesh.hz)

    # Remove air cells
    Xm = P.T * mkvc(Xm)
    Ym = P.T * mkvc(Ym)
    Zm = P.T * mkvc(Zm)

    hX = P.T * mkvc(hX)
    hY = P.T * mkvc(hY)
    hZ = P.T * mkvc(hZ)

    V = P.T * mkvc(mesh.vol)
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

        wr = wr + (V * temp / 8.)**2.

        count = progress(dd, count, ndata)

    wr = np.sqrt(wr) / V
    wr = mkvc(wr)
    wr = np.sqrt(wr / (np.max(wr)))

    print("Done 100% ...distance weighting completed!!\n")

    return wr
