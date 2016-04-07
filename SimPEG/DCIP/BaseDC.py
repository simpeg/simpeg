from SimPEG import *

class FieldsDC_CC(Problem.Fields):
    knownFields = {'phi_sol':'CC'}
    aliasFields = {
                    'phi' : ['phi_sol','CC','_phi'],
                    'e' : ['phi_sol','F','_e'],
                    'j' : ['phi_sol','F','_j']
                  }

    def __init__(self,mesh,survey,**kwargs):
        super(FieldsDC_CC, self).__init__(mesh, survey, **kwargs)

    def startup(self):
        self._cellGrad = self.survey.prob.mesh.cellGrad
        self._Mfinv = self.survey.prob.mesh.getFaceInnerProduct(invMat=True)

    def _phi(self, phi_sol, srcList):
        phi = phi_sol
        # for i, src in enumerate(srcList):
        #     phi_p = src.phi_p(self.survey.prob)
        #     if phi_p is not None:
        #         phi[:,i] += phi_p
        return phi

    def _e(self, phi_sol, srcList):
        e = -self._cellGrad*phi_sol
        # for i, src in enumerate(srcList):
        #     e_p = src.e_p(self.survey.prob)
        #     if e_p is not None:
        #         e[:,i] += e_p
        return e

    def _j(self, phi_sol, srcList):

        j = -self._Mfinv*self.survey.prob.Msig*self._cellGrad*phi_sol
        # for i, src in enumerate(srcList):
        #     j_p = src.j_p(self.survey.prob)
        #     if j_p is not None:
        #         j[:,i] += j_p
        return j



class SrcDipole(Survey.BaseSrc):
    """A dipole source, locA and locB are moved to the closest cell-centers"""

    current = 1
    loc  = None
    # _rhsDict = None

    def __init__(self, rxList, locA, locB, **kwargs):
        self.loc = (locA, locB)
        super(SrcDipole, self).__init__(rxList, **kwargs)

    def eval(self, prob):
        # Recompute rhs
        # if getattr(self, '_rhsDict', None) is None:
        #     self._rhsDict = {}
        # if mesh not in self._rhsDict:
        pts = [self.loc[0], self.loc[1]]
        inds = Utils.closestPoints(prob.mesh, pts)
        q = np.zeros(prob.mesh.nC)
        q[inds] = - self.current * ( np.r_[1., -1.] / prob.mesh.vol[inds] )
        # self._rhsDict[mesh] = q
        # return self._rhsDict[mesh]
        return q


class RxDipole(Survey.BaseRx):
    """A dipole source, locA and locB are moved to the closest cell-centers"""
    def __init__(self, locsM, locsN, **kwargs):
        locs = (locsM, locsN)
        assert locsM.shape == locsN.shape, 'locs must be the same shape.'
        super(RxDipole, self).__init__(locs, 'dipole', storeProjections=False, **kwargs)

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
    uncert = None
    def __init__(self, srcList, **kwargs):
        self.srcList = srcList
        Survey.BaseSurvey.__init__(self, **kwargs)
        # self._rhsDict = {}
        self._Ps = {}

    def eval(self, u):
        """
            Predicted data.

            .. math::
                d_\\text{pred} = Pu(m)
        """
        P = self.getP(self.prob.mesh)
        return P*mkvc(u[self.srcList, 'phi_sol'])

    def getP(self, mesh):
        if mesh in self._Ps:
            return self._Ps[mesh]

        P_src = [sp.vstack([rx.getP(mesh) for rx in src.rxList]) for src in self.srcList]

        self._Ps[mesh] = sp.block_diag(P_src)
        return self._Ps[mesh]


class ProblemDC_CC(Problem.BaseProblem):
    """
        **ProblemDC**

        Geophysical DC resistivity problem.

    """

    surveyPair = SurveyDC
    Solver     = Solver
    fieldsPair = FieldsDC_CC
    Ainv = None

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
            self._Msig = Utils.sdiag(1/(self.mesh.dim * Av.T * (1/sigma)))
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
            self._A[0,0] /= self.mesh.vol[0]
            self._A = self._A.tocsc()
        return self._A

    def getRHS(self):
        # if self.mesh not in self._rhsDict:
        RHS = np.array([src.eval(self) for src in self.survey.srcList]).T
        # self._rhsDict[mesh] = RHS
        # return self._rhsDict[mesh]
        return RHS

    def fields(self, m):

        F = self.fieldsPair(self.mesh, self.survey)
        self.curModel = m
        A    = self.A
        self.Ainv = self.Solver(A, **self.solverOpts)
        RHS    = self.getRHS()
        Phi  = self.Ainv * RHS
        Srcs = self.survey.srcList
        F[Srcs, 'phi_sol'] = Phi

        return F

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
            u = self.fields(self.curModel)[self.survey.srcList, 'phi_sol']
        else:
            u = u[self.survey.srcList, 'phi_sol']

        D = self.mesh.faceDiv
        G = self.mesh.cellGrad
        # Derivative of model transform, $\deriv{\sigma}{\m}$
        dsigdm_x_v = self.curModel.transformDeriv * v

        # Take derivative of $C(m,u)$ w.r.t. $m$
        dCdm_x_v = np.empty_like(u)
        # loop over fields for each source
        for i in range(self.survey.nSrc):
            # Derivative of inner product, $\left(\mathbf{M}_{1/\sigma}^f\right)^{-1}$
            dAdsig         = D * self.dMdsig( G * u[:,i] )
            dCdm_x_v[:, i] = dAdsig *  dsigdm_x_v

        # Take derivative of $C(m,u)$ w.r.t. $u$
        dA_du = self.A
        # Solve for $\deriv{u}{m}$
        # dCdu_inv = self.Solver(dCdu, **self.solverOpts)
        if self.Ainv is None:
            self.Ainv = self.Solver(dA_du, **self.solverOpts)

        P        = self.survey.getP(self.mesh)
        Jv    = - P * mkvc( self.Ainv * dCdm_x_v )
        return Jv

    def Jtvec(self, m, v, u=None):

        self.curModel = m
        sigma = self.curModel.transform # $\sigma = \mathcal{M}(\m)$
        if u is None:
            # Run forward simulation if $u$ not provided
            u = self.fields(self.curModel)[self.survey.srcList, 'phi_sol']
        else:
            u = u[self.survey.srcList, 'phi_sol']

        shp = u.shape
        P = self.survey.getP(self.mesh)
        PT_x_v = (P.T*v).reshape(shp, order='F')

        D = self.mesh.faceDiv
        G = self.mesh.cellGrad
        dA_du = self.A
        mT_dm = self.mapping.deriv(m)

        # We probably always need this due to the linesearch .. (?)
        self.Ainv = self.Solver(dA_du.T, **self.solverOpts)
        # if self.Ainv is None:
        #     self.Ainv = self.Solver(dCdu, **self.solverOpts)

        w = self.Ainv * PT_x_v

        Jtv = 0
        for i, ui in enumerate(u.T):  # loop over each column
            Jtv += self.dMdsig( G * ui ).T * ( D.T * w[:,i] )

        Jtv = - mT_dm.T * ( Jtv )
        return Jtv





