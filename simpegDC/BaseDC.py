from SimPEG import *


class DipoleTx(Survey.BaseTx):
    """A dipole transmitter, locA and locB are moved to the closest cell-centers"""

    current = 1

    def __init__(self, locA, locB, rxList, **kwargs):
        super(DipoleTx, self).__init__((locA, locB), 'dipole', rxList, **kwargs)
        self._rhsDict = {}

    def getRhs(self, mesh):
        if mesh not in self._rhsDict:
            pts = [self.loc[0], self.loc[1]]
            inds = Utils.closestPoints(mesh, pts)
            q = np.zeros(mesh.nC)
            q[inds] = - self.current * ( np.r_[1., -1.] / mesh.vol[inds] )
            self._rhsDict[mesh] = q
        return self._rhsDict[mesh]


class DipoleRx(Survey.BaseRx):
    """A dipole transmitter, locA and locB are moved to the closest cell-centers"""
    def __init__(self, locsM, locsN, **kwargs):
        locs = (locsM, locsN)
        assert locsM.shape == locsN.shape, 'locs must be the same shape.'
        super(DipoleRx, self).__init__(locs, 'dipole', storeProjections=False, **kwargs)

    @property
    def nD(self):
        """Number of data in the receiver."""
        return self.locs[0].shape[0]

    def getP(self, mesh):
        P0 = mesh.getInterpolationMat(self.locs[0], self.projGLoc)
        P1 = mesh.getInterpolationMat(self.locs[1], self.projGLoc)
        return P0 - P1


class SurveyDC(Survey.BaseSurvey):
    """
        **SurveyDC**

        Geophysical DC resistivity data.

    """

    def __init__(self, txList, **kwargs):
        self.txList = txList
        Survey.BaseSurvey.__init__(self, **kwargs)
        self._rhsDict = {}
        self._Ps = {}

    def projectFields(self, u):
        """
            Predicted data.

            .. math::
                d_\\text{pred} = Pu(m)
        """
        P = self.getP(self.prob.mesh)
        return P*mkvc(u)

    def getRhs(self, mesh):
        if mesh not in self._rhsDict:
            RHS = np.array([tx.getRhs(mesh) for tx in self.txList]).T
            self._rhsDict[mesh] = RHS
        return self._rhsDict[mesh]

    def getP(self, mesh):
        if mesh in self._Ps:
            return self._Ps[mesh]

        P_tx = [sp.vstack([rx.getP(mesh) for rx in tx.rxList]) for tx in self.txList]

        self._Ps[mesh] = sp.block_diag(P_tx)
        return self._Ps[mesh]




class ProblemDC(Problem.BaseProblem):
    """
        **ProblemDC**

        Geophysical DC resistivity problem.

    """

    surveyPair = SurveyDC
    Solver = Solver

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh)
        self.mesh.setCellGradBC('neumann')
        Utils.setKwargs(self, **kwargs)


    deleteTheseOnModelUpdate = ['_A', '_Msig', '_dMdsig']

    @property
    def Msig(self):
        if getattr(self, '_Msig', None) is None:
            sigma = self.curModel.transform
            Av = self.mesh.aveF2CC
            self._Msig = Utils.sdInv(Utils.sdiag(self.mesh.dim * Av.T * (1/sigma)))
        return self._Msig

    @property
    def dMdsig(self):
        if getattr(self, '_dMdsig', None) is None:
            sigma = self.curModel.transform
            Av = self.mesh.aveF2CC
            dMdprop = self.mesh.dim * Utils.sdiag(self.Msig.diagonal()**2) * Av.T * Utils.sdiag(1./sigma**2)
            self._dMdsig = lambda Gu: Utils.sdiag(Gu) * dMdprop
        return self._dMdsig

    @property
    def A(self):
        """
            Makes the matrix A(m) for the DC resistivity problem.

            :param numpy.array m: model
            :rtype: scipy.csc_matrix
            :return: A(m)

            .. math::
                c(m,u) = A(m)u - q = G\\text{sdiag}(M(mT(m)))Du - q = 0

            Where M() is the mass matrix and mT is the model transform.
        """
        if getattr(self, '_A', None) is None:
            D = self.mesh.faceDiv
            G = self.mesh.cellGrad
            self._A = D*self.Msig*G
            # Remove the null space from the matrix.
            self._A[-1,-1] /= self.mesh.vol[-1]
            self._A = self._A.tocsc()
        return self._A

    def fields(self, m):
        self.curModel = m
        A    = self.A
        Ainv = self.Solver(A)
        Q    = self.survey.getRhs(self.mesh)
        Phi  = Ainv * Q
        for ii in range(Phi.shape[1]):
            # Remove the static shift for each phi column.
            Phi[:,ii] -= Phi[-1, ii]
        return Phi

    def Jvec(self, m, v, u=None):
        """
            :param numpy.array m: model
            :param numpy.array v: vector to multiply
            :param numpy.array u: fields
            :rtype: numpy.array
            :return: Jv

            .. math::
                c(m,u) = A(m)u - q = G\\text{sdiag}(M(mT(m)))Du - q = 0

                \\nabla_u (A(m)u - q) = A(m)

                \\nabla_m (A(m)u - q) = G\\text{sdiag}(Du)\\nabla_m(M(mT(m)))

            Where M() is the mass matrix and mT is the model transform.

            .. math::
                J = - P \left( \\nabla_u c(m, u) \\right)^{-1} \\nabla_m c(m, u)

                J(v) = - P ( A(m)^{-1} ( G\\text{sdiag}(Du)\\nabla_m(M(mT(m))) v ) )
        """
        # Set current model; clear dependent property $\mathbf{A(m)}$
        self.curModel = m
        sigma = self.curModel.transform # $\sigma = \mathcal{M}(\m)$
        if u is None:
            # Run forward simulation if $u$ not provided
            u = self.fields(self.curModel)
        else:
            shp = (self.mesh.nC, self.survey.nTx)
            u = u.reshape(shp, order='F')

        D = self.mesh.faceDiv
        G = self.mesh.cellGrad
        # Derivative of model transform, $\deriv{\sigma}{\m}$
        dsigdm_x_v = self.curModel.transformDeriv * v

        # Take derivative of $C(m,u)$ w.r.t. $m$
        dCdm_x_v = np.empty_like(u)
        # loop over fields for each transmitter
        for i in range(self.survey.nTx):
            # Derivative of inner product, $\left(\mathbf{M}_{1/\sigma}^f\right)^{-1}$
            dAdsig         = D * self.dMdsig( G * u[:,i] )
            dCdm_x_v[:, i] = dAdsig *  dsigdm_x_v

        # Take derivative of $C(m,u)$ w.r.t. $u$
        dCdu = self.A
        # Solve for $\deriv{u}{m}$
        dCdu_inv = self.Solver(dCdu, **self.solverOpts)
        P        = self.survey.getP(self.mesh)
        J_x_v    = - P * mkvc( dCdu_inv * dCdm_x_v )
        return J_x_v # Make $\mathbf{Jv}$ a vector.

    def Jtvec(self, m, v, u=None):

        self.curModel = m
        sigma = self.curModel.transform # $\sigma = \mathcal{M}(\m)$
        if u is None:
            u = self.fields(self.curModel)

        shp = (self.mesh.nC, self.survey.nTx)
        u = u.reshape(shp, order='F')
        P = self.survey.getP(self.mesh)
        PT_x_v = (P.T*v).reshape(shp, order='F')

        D = self.mesh.faceDiv
        G = self.mesh.cellGrad
        A = self.A
        mT_dm = self.mapping.deriv(m)

        dCdu = A.T
        Ainv = self.Solver(dCdu)
        w = Ainv * PT_x_v

        Jtv = 0
        for i, ui in enumerate(u.T):  # loop over each column
            Jtv += self.dMdsig( G * ui ).T * ( D.T * w[:,i] )

        Jtv = - mT_dm.T * ( Jtv )
        return Jtv
