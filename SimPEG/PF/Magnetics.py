from __future__ import print_function

import numpy as np
import scipy.sparse as sp
from scipy.constants import mu_0

from SimPEG import Utils
from SimPEG import Problem
from SimPEG import Solver
from SimPEG import Props

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
    rtype = 'tmi'  #: Receiver type either "tmi" | "xyz"

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

    def fwr_ind(self, m):

        if self.forwardOnly:

            # Compute the linear operation without forming the full dense G
            fwr_d = self.Intrgl_Fwr_Op(m=m)

            return fwr_d

        else:

            return self.G.dot(m)

    def fwr_rem(self):
        # TODO check if we are inverting for M
        return self.G.dot(self.chiMap(m))

    def fields(self, m, **kwargs):
        self.model = m

        if self.rtype == 'tmi':
            u = np.zeros(self.survey.nRx)
        else:
            u = np.zeros(3*self.survey.nRx)

        u = self.fwr_ind(m=m)
        # rem = self.rem

        # if induced is not None:
        #     total += induced

        return u

    # def Jvec(self, m, v, f=None):
    #     dmudm = self.chiMap.deriv(m)
    #     return self.G.dot(dmudm*v)

    # def Jtvec(self, m, v, f=None):
    #     dmudm = self.chiMap.deriv(m)
    #     return dmudm.T * (self.G.T.dot(v))

    @property
    def G(self):
        if not self.ispaired:
            raise Exception('Need to pair!')

        if getattr(self, '_G', None) is None:
            self._G = self.Intrgl_Fwr_Op()

        return self._G

    def Intrgl_Fwr_Op(self, m=None, Magnetization="ind"):

        """

        Magnetic forward operator in integral form

        flag        = 'ind' | 'xyz'

          1- ind : Magnetization fixed by user

          3- xyz: xyz tensor matrix stored with shape([3*ndata, 3*nc])

        Return
        _G = Linear forward modeling operation

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
        xn = self.mesh.vectorNx
        yn = self.mesh.vectorNy
        zn = self.mesh.vectorNz

        yn2, xn2, zn2 = np.meshgrid(yn[1:], xn[1:], zn[1:])
        yn1, xn1, zn1 = np.meshgrid(yn[0:-1], xn[0:-1], zn[0:-1])

        Yn = P.T*np.c_[Utils.mkvc(yn1), Utils.mkvc(yn2)]
        Xn = P.T*np.c_[Utils.mkvc(xn1), Utils.mkvc(xn2)]
        Zn = P.T*np.c_[Utils.mkvc(zn1), Utils.mkvc(zn2)]

        survey = self.survey
        rxLoc = survey.srcField.rxList[0].locs
        ndata = rxLoc.shape[0]

        # Pre-allocate space and create magnetization matrix if required


        # # If assumes uniform magnetization direction
        # if M.shape != (nC,3):

        #     print('Magnetization vector must be Nc x 3')
        #     return
        if getattr(self, 'M', None) is None:
            M = dipazm_2_xyz(np.ones(nC) * survey.srcField.param[1],
                             np.ones(nC) * survey.srcField.param[2])

        Mx = Utils.sdiag(M[:, 0]*survey.srcField.param[0])
        My = Utils.sdiag(M[:, 1]*survey.srcField.param[0])
        Mz = Utils.sdiag(M[:, 2]*survey.srcField.param[0])

        Mxyz = sp.vstack((Mx, My, Mz))

        if survey.srcField.rxList[0].rxType == 'tmi':


            # Convert Bdecination from north to cartesian
            D = (450.-float(survey.srcField.param[2])) % 360.
            I = survey.srcField.param[1]
            # Projection matrix
            Ptmi = Utils.mkvc(np.r_[np.cos(np.deg2rad(I))*np.cos(np.deg2rad(D)),
                              np.cos(np.deg2rad(I))*np.sin(np.deg2rad(D)),
                              np.sin(np.deg2rad(I))], 2).T

        if self.forwardOnly:

            if self.rtype == 'tmi':

                fwr_out = np.zeros(self.survey.nRx)

            else:

                fwr_out = np.zeros(3*self.survey.nRx)

        else:

            if (Magnetization == 'ind'):

                if survey.srcField.rxList[0].rxType == 'tmi':
                    fwr_out = np.zeros((ndata, nC))

                elif survey.srcField.rxList[0].rxType == 'xyz':

                    fwr_out = np.zeros((int(3*ndata), nC))

            elif Magnetization == 'xyz':
                if survey.srcField.rxList[0].rxType == 'tmi':
                    fwr_out = np.zeros((int(ndata), int(3*nC)))

                elif survey.srcField.rxList[0].rxType == 'xyz':
                    fwr_out = np.zeros((int(3*ndata), int(3*nC)))

            else:
                print("""Flag must be either 'ind' | 'xyz', please revised""")
                return

            # Loop through all observations and create forward operator (nD-by-nC)
            print("Begin calculation of forward operator: " + Magnetization)

        # Add counter to dsiplay progress. Good for large problems
        count = -1
        for ii in range(ndata):

            tx, ty, tz = get_T_mat(Xn, Yn, Zn, rxLoc[ii, :])

            if self.forwardOnly:

                if self.rtype == 'tmi':
                    fwr_out[ii] = (Ptmi.dot(np.vstack((tx, ty, tz)))*Mxyz).dot(m)

                elif self.rtype == 'xyz':
                    fwr_out[ii] = (tx*Mxyz).dot(m)
                    fwr_out[ii+ndata] = (ty*Mxyz).dot(m)
                    fwr_out[ii+2*ndata] = (tz*Mxyz).dot(m)

            else:

                if Magnetization == 'ind':

                    if survey.srcField.rxList[0].rxType == 'tmi':
                        fwr_out[ii, :] = Ptmi.dot(np.vstack((tx, ty, tz)))*Mxyz

                    elif survey.srcField.rxList[0].rxType == 'xyz':
                        fwr_out[ii, :] = tx*Mxyz
                        fwr_out[ii+ndata, :] = ty*Mxyz
                        fwr_out[ii+2*ndata, :] = tz*Mxyz

                elif Magnetization == 'xyz':

                    if survey.srcField.rxList[0].rxType == 'tmi':
                        fwr_out[ii, :] = Ptmi.dot(np.vstack((tx, ty, tz)) *
                                                  survey.srcField.param[0])

                    elif survey.srcField.rxList[0].rxType == 'xyz':
                        fwr_out[ii, :] = tx * survey.srcField.param[0]
                        fwr_out[ii+ndata, :] = ty * survey.srcField.param[0]
                        fwr_out[ii+2*ndata, :] = tz * survey.srcField.param[0]

            # Display progress
            count = progress(ii, count, ndata)

        print("Done 100% ...forward operator completed!!\n")

        return fwr_out


class MagneticVector(MagneticIntegral):

    forwardOnly = False  # If false, matric is store to memory (watch your RAM)
    actInd = None  #: Active cell indices provided
    M = None  #: Magnetization matrix provided, otherwise all induced
    rtype = 'tmi'  #: Receiver type either "tmi" | "xyz"

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

    def fwr_ind(self, m):

        if self.forwardOnly:

            # Compute the linear operation without forming the full dense G
            fwr_d = Intrgl_Fwr_Op(m=m, Magnetization='xyz')

            return fwr_d

        else:

            # m = np.hstack([m, mii])

            return self.G.dot(m)

    @property
    def G(self):
        if not self.ispaired:
            raise Exception('Need to pair!')

        if getattr(self, '_G', None) is None:
            self._G = self.Intrgl_Fwr_Op(Magnetization='xyz')

        return self._G


class MagneticAmplitude(MagneticIntegral):

    forwardOnly = False  # If false, matric is store to memory (watch your RAM)
    actInd = None  #: Active cell indices provided
    M = None  #: Magnetization matrix provided, otherwise all induced
    rtype = 'xyz'  #: Receivers must be "xyz"

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

    def fwr_ind(self, m):

        self.survey.srcField.rxList[0].rxType = 'xyz'

        if self.forwardOnly:

            # Compute the linear operation without forming the full dense G
            Bxyz = Intrgl_Fwr_Op(m=m)

            return self.calcAmpData(Bxyz)

        else:
            if m is None:
                m = self.chiMap*self.model

            Bxyz = self.G.dot(m)

            return self.calcAmpData(Bxyz)

    def calcAmpData(self, Bxyz):

        ndata = self.survey.srcField.rxList[0].locs.shape[0]

        Bamp = np.sqrt(Bxyz[:ndata]**2. +
                       Bxyz[ndata:2*ndata]**2. +
                       Bxyz[2*ndata:]**2.)

        return Bamp

    def fields(self, m, **kwargs):

        self.model = m

        ampB = self.fwr_ind(m)

        return ampB

    def Jvec(self, m, v, f=None):
        dmudm = self.chiMap.deriv(m)
        return self.dfdm*(self.G.dot(dmudm*v))

    def Jtvec(self, m, v, f=None):
        dmudm = self.chiMap.deriv(m)
        return dmudm.T * (self.G.T.dot(self.dfdm.T*v))

    @property
    def G(self):
        if not self.ispaired:
            raise Exception('Need to pair!')

        if getattr(self, '_G', None) is None:
            self._G = self.Intrgl_Fwr_Op()

        return self._G

    @property
    def dfdm(self):

        if getattr(self, '_dfdm', None) is None:

            ndata = self.survey.srcField.rxList[0].locs.shape[0]

            # Get field data
            m = self.chiMap*self.model

            Bxyz = self.G.dot(m)

            Bamp = self.calcAmpData(Bxyz)

            Bx = sp.spdiags(Bxyz[:ndata]/Bamp, 0, ndata, ndata)
            By = sp.spdiags(Bxyz[ndata:2*ndata]/Bamp, 0, ndata, ndata)
            Bz = sp.spdiags(Bxyz[2*ndata:]/Bamp, 0, ndata, ndata)
            self._dfdm = sp.hstack((Bx, By, Bz))

        return self._dfdm


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

    def fields(self, m):
        """
            Return magnetic potential (u) and flux (B)
            u: defined on the cell center [nC x 1]
            B: defined on the cell center [nF x 1]

            After we compute u, then we update B.

            .. math ::

                \mathbf{B}_s = (\MfMui)^{-1}\mathbf{M}^f_{\mu_0^{-1}}\mathbf{B}_0-\mathbf{B}_0 -(\MfMui)^{-1}\Div^T \mathbf{u}

        """
        self.makeMassMatrices(m)
        A = self.getA(m)
        rhs = self.getRHS(m)
        m1 = sp.linalg.interface.aslinearoperator(Utils.sdiag(1/A.diagonal()))
        u, info = sp.linalg.bicgstab(A, rhs, tol=1e-6, maxiter=1000, M=m1)
        B0 = self.getB0()
        B = self.MfMuI*self.MfMu0*B0-B0-self.MfMuI*self._Div.T*u

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
        mu = self.muMap*(m)
        dmudm = self.muDeriv
        dchidmu = Utils.sdiag(1/mu_0*np.ones(self.mesh.nC))

        vol = self.mesh.vol
        Div = self._Div
        Dface = self.mesh.faceDiv
        P = self.survey.projectFieldsDeriv(B)  # Projection matrix
        B0 = self.getB0()

        MfMuIvec = 1/self.MfMui.diagonal()
        dMfMuI = Utils.sdiag(MfMuIvec**2)*self.mesh.aveF2CC.T*Utils.sdiag(vol*1./mu**2)

        # A = self._Div*self.MfMuI*self._Div.T
        # RHS = Div*MfMuI*MfMu0*B0 - Div*B0 + Mc*Dface*Pout.T*Bbc
        # C(m,u) = A*m-rhs
        # dudm = -(dCdu)^(-1)dCdm

        dCdu = self.getA(m)
        dCdm_A = Div * (Utils.sdiag(Div.T * u) * dMfMuI * dmudm)
        dCdm_RHS1 = Div * (Utils.sdiag(self.MfMu0 * B0) * dMfMuI)
        temp1 = (Dface*(self._Pout.T*self.Bbc_const*self.Bbc))
        dCdm_RHS2v = (Utils.sdiag(vol)*temp1)*np.inner(vol, dchidmu*dmudm*v)

        # dCdm_RHSv =  dCdm_RHS1*(dmudm*v) +  dCdm_RHS2v
        dCdm_RHSv = dCdm_RHS1 * (dmudm * v)
        dCdm_v = dCdm_A * v - dCdm_RHSv

        m1 = sp.linalg.interface.aslinearoperator(Utils.sdiag(1/dCdu.diagonal()))
        sol, info = sp.linalg.bicgstab(dCdu, dCdm_v,
                                       tol=1e-6, maxiter=1000, M=m1)

        if info > 0:
            print("Iterative solver did not work well (Jvec)")
            # raise Exception ("Iterative solver did not work well")

        # B = self.MfMuI*self.MfMu0*B0-B0-self.MfMuI*self._Div.T*u
        # dBdm = d\mudm*dBd\mu

        dudm = -sol
        dBdmv =     (  Utils.sdiag(self.MfMu0*B0)*(dMfMuI * (dmudm*v)) \
                     - Utils.sdiag(Div.T*u)*(dMfMuI* (dmudm*v)) \
                     - self.MfMuI*(Div.T* (dudm)) )

        return Utils.mkvc(P*dBdmv)

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
        mu = self.mapping*(m)
        dmudm = self.mapping.deriv(m)
        dchidmu = Utils.sdiag(1/mu_0*np.ones(self.mesh.nC))

        vol = self.mesh.vol
        Div = self._Div
        Dface = self.mesh.faceDiv
        P = self.survey.projectFieldsDeriv(B)                 # Projection matrix
        B0 = self.getB0()

        MfMuIvec = 1/self.MfMui.diagonal()
        dMfMuI = Utils.sdiag(MfMuIvec**2)*self.mesh.aveF2CC.T*Utils.sdiag(vol*1./mu**2)

        # A = self._Div*self.MfMuI*self._Div.T
        # RHS = Div*MfMuI*MfMu0*B0 - Div*B0 + Mc*Dface*Pout.T*Bbc
        # C(m,u) = A*m-rhs
        # dudm = -(dCdu)^(-1)dCdm

        dCdu = self.getA(m)
        s = Div * (self.MfMuI.T * (P.T*v))

        m1 = sp.linalg.interface.aslinearoperator(Utils.sdiag(1/(dCdu.T).diagonal()))
        sol, info = sp.linalg.bicgstab(dCdu.T, s, tol=1e-6, maxiter=1000, M=m1)

        if info > 0:
            print("Iterative solver did not work well (Jtvec)")
            # raise Exception ("Iterative solver did not work well")


        # dCdm_A = Div * ( Utils.sdiag( Div.T * u )* dMfMuI *dmudm  )
        # dCdm_Atsol = ( dMfMuI.T*( Utils.sdiag( Div.T * u ) * (Div.T * dmudm)) ) * sol
        dCdm_Atsol = (dmudm.T * dMfMuI.T*(Utils.sdiag(Div.T * u) * Div.T)) * sol

        # dCdm_RHS1 = Div * (Utils.sdiag( self.MfMu0*B0  ) * dMfMuI)
        # dCdm_RHS1tsol = (dMfMuI.T*( Utils.sdiag( self.MfMu0*B0  ) ) * Div.T * dmudm) * sol
        dCdm_RHS1tsol = (dmudm.T * dMfMuI.T*(Utils.sdiag( self.MfMu0*B0)) * Div.T ) * sol


        # temp1 = (Dface*(self._Pout.T*self.Bbc_const*self.Bbc))
        temp1sol = ( Dface.T*( Utils.sdiag(vol)*sol ) )
        temp2 = self.Bbc_const*(self._Pout.T*self.Bbc).T
        # dCdm_RHS2v  = (Utils.sdiag(vol)*temp1)*np.inner(vol, dchidmu*dmudm*v)
        dCdm_RHS2tsol  = (dmudm.T*dchidmu.T*vol)*np.inner(temp2, temp1sol)

        # dCdm_RHSv =  dCdm_RHS1*(dmudm*v) +  dCdm_RHS2v

        #temporary fix
        # dCdm_RHStsol = dCdm_RHS1tsol - dCdm_RHS2tsol
        dCdm_RHStsol = dCdm_RHS1tsol

        # dCdm_RHSv =  dCdm_RHS1*(dmudm*v) +  dCdm_RHS2v
        # dCdm_v = dCdm_A*v - dCdm_RHSv

        Ctv = dCdm_Atsol - dCdm_RHStsol

        # B = self.MfMuI*self.MfMu0*B0-B0-self.MfMuI*self._Div.T*u
        # dBdm = d\mudm*dBd\mu
        # dPBdm^T*v = Atemp^T*P^T*v - Btemp^T*P^T*v - Ctv

        Atemp = Utils.sdiag(self.MfMu0*B0)*(dMfMuI * (dmudm))
        Btemp = Utils.sdiag(Div.T*u)*(dMfMuI* (dmudm))
        Jtv = Atemp.T*(P.T*v) - Btemp.T*(P.T*v) - Ctv

        return Utils.mkvc(Jtv)


def MagneticsDiffSecondaryInv(mesh, model, data, **kwargs):

    """
        Inversion module for MagneticsDiffSecondary

    """
    from SimPEG import Optimization, Regularization, Parameters, ObjFunction, Inversion
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

    eps = 1e-10  # add a small value to the locations to avoid /0

    nC = Xn.shape[0]

    # Pre-allocate space for 1D array
    Tx = np.zeros((1, 3*nC))
    Ty = np.zeros((1, 3*nC))
    Tz = np.zeros((1, 3*nC))

    dz2 = rxLoc[2] - Zn[:, 0] + eps
    dz1 = rxLoc[2] - Zn[:, 1] + eps

    dy2 = Yn[:, 1] - rxLoc[1] + eps
    dy1 = Yn[:, 0] - rxLoc[1] + eps

    dx2 = Xn[:, 1] - rxLoc[0] + eps
    dx1 = Xn[:, 0] - rxLoc[0] + eps

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

    Tx[0, 0:nC] = np.arctan2(dy1 * dz2, (dx2 * arg5)) +\
        - np.arctan2(dy2 * dz2, (dx2 * arg2)) +\
        np.arctan2(dy2 * dz1, (dx2 * arg3)) +\
        - np.arctan2(dy1 * dz1, (dx2 * arg8)) +\
        np.arctan2(dy2 * dz2, (dx1 * arg1)) +\
        - np.arctan2(dy1 * dz2, (dx1 * arg6)) +\
        np.arctan2(dy1 * dz1, (dx1 * arg7)) +\
        - np.arctan2(dy2 * dz1, (dx1 * arg4))

    Ty[0, 0:nC] = np.log((dz2 + arg2) / (dz1 + arg3)) +\
        -np.log((dz2 + arg1) / (dz1 + arg4)) +\
        np.log((dz2 + arg6) / (dz1 + arg7)) +\
        -np.log((dz2 + arg5) / (dz1 + arg8))

    Ty[0, nC:2*nC] = np.arctan2(dx1 * dz2, (dy2 * arg1)) +\
        - np.arctan2(dx2 * dz2, (dy2 * arg2)) +\
        np.arctan2(dx2 * dz1, (dy2 * arg3)) +\
        - np.arctan2(dx1 * dz1, (dy2 * arg4)) +\
        np.arctan2(dx2 * dz2, (dy1 * arg5)) +\
        - np.arctan2(dx1 * dz2, (dy1 * arg6)) +\
        np.arctan2(dx1 * dz1, (dy1 * arg7)) +\
        - np.arctan2(dx2 * dz1, (dy1 * arg8))

    R1 = (dy2**2 + dz1**2)
    R2 = (dy2**2 + dz2**2)
    R3 = (dy1**2 + dz1**2)
    R4 = (dy1**2 + dz2**2)

    Ty[0, 2*nC:] = np.log((dx1 + np.sqrt(dx1**2 + R1)) /
                          (dx2 + np.sqrt(dx2**2 + R1))) +\
        -np.log((dx1 + np.sqrt(dx1**2 + R2)) / (dx2 + np.sqrt(dx2**2 + R2))) +\
        np.log((dx1 + np.sqrt(dx1**2 + R4)) / (dx2 + np.sqrt(dx2**2 + R4))) +\
        -np.log((dx1 + np.sqrt(dx1**2 + R3)) / (dx2 + np.sqrt(dx2**2 + R3)))

    R1 = (dx2**2 + dz1**2)
    R2 = (dx2**2 + dz2**2)
    R3 = (dx1**2 + dz1**2)
    R4 = (dx1**2 + dz2**2)

    Tx[0, 2*nC:] = np.log((dy1 + np.sqrt(dy1**2 + R1)) /
                          (dy2 + np.sqrt(dy2**2 + R1))) +\
        -np.log((dy1 + np.sqrt(dy1**2 + R2)) / (dy2 + np.sqrt(dy2**2 + R2))) +\
        np.log((dy1 + np.sqrt(dy1**2 + R4)) / (dy2 + np.sqrt(dy2**2 + R4))) +\
        -np.log((dy1 + np.sqrt(dy1**2 + R3)) / (dy2 + np.sqrt(dy2**2 + R3)))

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


def dipazm_2_xyz(dip, azm_N):
    """
    dipazm_2_xyz(dip,azm_N)

    Function converting degree angles for dip and azimuth from north to a
    3-components in cartesian coordinates.

    INPUT
    dip     : Value or vector of dip from horizontal in DEGREE
    azm_N   : Value or vector of azimuth from north in DEGREE

    OUTPUT
    M       : [n-by-3] Array of xyz components of a unit vector in cartesian

    Created on Dec, 20th 2015

    @author: dominiquef
    """
    nC = len(azm_N)

    M = np.zeros((nC, 3))

    # Modify azimuth from North to Cartesian-X
    azm_X = (450. - np.asarray(azm_N)) % 360.

    D = np.deg2rad(np.asarray(dip))
    I = np.deg2rad(azm_X)

    M[:, 0] = np.cos(D) * np.cos(I)
    M[:, 1] = np.cos(D) * np.sin(I)
    M[:, 2] = np.sin(D)

    return M


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


def writeUBCobs(filename, survey, d):
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

    data = np.c_[rxLoc, d, wd]
    head = ('%6.2f %6.2f %6.2f\n' % (B[1], B[2], B[0])+
              '%6.2f %6.2f %6.2f\n' % (B[1], B[2], 1)+
              '%i\n' % len(d))
    np.savetxt(filename, data, fmt='%e', delimiter=' ', newline='\n',header=head,comments='')

    print("Observation file saved to: " + filename)


def plot_obs_2D(rxLoc, d=None, title=None,
                vmin=None, vmax=None, levels=None, fig=None, ax=None,
                colorbar=True, marker=True, cmap="plasma_r"):
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
        x = np.linspace(rxLoc[:, 0].min(), rxLoc[:, 0].max(), 100)
        y = np.linspace(rxLoc[:, 1].min(), rxLoc[:, 1].max(), 100)

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
