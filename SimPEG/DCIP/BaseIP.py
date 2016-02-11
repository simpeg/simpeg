from SimPEG import *
from BaseDC import SurveyDC, FieldsDC_CC

class SurveyIP(SurveyDC):
    """
        **SurveyDC**

        Geophysical DC resistivity data.

    """

    def __init__(self, srcList, **kwargs):
        self.srcList = srcList
        Survey.BaseSurvey.__init__(self, **kwargs)
        self._Ps = {}

    def dpred(self, m, u=None):
        """
            Predicted data.

            .. math::
                d_\\text{pred} = Pu(m)
        """

        return self.prob.forward(m)


class ProblemIP(Problem.BaseProblem):
    """
        **ProblemIP**

        Geophysical IP resistivity problem.

    """

    surveyPair = SurveyDC
    Solver     = Solver
    sigma = None
    Ainv = None
    u = None

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh)
        self.mesh.setCellGradBC('neumann')
        Utils.setKwargs(self, **kwargs)

    # deleteTheseOnModelUpdate = ['_A', '_Msig', '_dMdsig']

    @property
    def Msig(self):
        if getattr(self, '_Msig', None) is None:
            # sigma = self.curModel.transform
            sigma = self.sigma
            Av = self.mesh.aveF2CC
            self._Msig = Utils.sdiag(1/(self.mesh.dim * Av.T * (1/sigma)))
        return self._Msig

    @property
    def dMdsig(self):
        if getattr(self, '_dMdsig', None) is None:
            # sigma = self.curModel.transform
            sigma = self.sigma
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

    def getRHS(self):
        # if self.mesh not in self._rhsDict:
        RHS = np.array([src.eval(self) for src in self.survey.srcList]).T
        # self._rhsDict[mesh] = RHS
        # return self._rhsDict[mesh]
        return RHS

    def fields(self, m):
        if self.u is None:
	        A    = self.A
	        if self.Ainv == None:
	        	self.Ainv = self.Solver(A, **self.solverOpts)
	        Q    = self.getRHS()
	        self.u  = self.Ainv * Q
        return self.u

    def forward(self, m, u=None):
        # Set current model; clear dependent property $\mathbf{A(m)}$
        self.curModel = m
        # sigma = self.curModel.transform # $\sigma = \mathcal{M}(\m)$
        sigma = self.sigma
        if self.u is None:
            # Run forward simulation if $u$ not provided
            u = self.fields(sigma)

        shp = (self.mesh.nC, self.survey.nSrc)
        u = self.u.reshape(shp, order='F')

        D = self.mesh.faceDiv
        G = self.mesh.cellGrad
        # Derivative of model transform, $\deriv{\sigma}{\m}$
        # dsigdm_x_v = self.curModel.transformDeriv * v

        dsigdm_x_v = Utils.sdiag(sigma) * self.curModel.transformDeriv *  m

        # Take derivative of $C(m,u)$ w.r.t. $m$
        dCdm_x_v = np.empty_like(u)
        # loop over fields for each source
        for i in range(self.survey.nSrc):
            # Derivative of inner product, $\left(\mathbf{M}_{1/\sigma}^f\right)^{-1}$
            dAdsig         = D * self.dMdsig( G * u[:,i] )
            dCdm_x_v[:, i] = dAdsig *  dsigdm_x_v

        # Take derivative of $C(m,u)$ w.r.t. $u$

        if self.Ainv == None:
        	self.Ainv = self.Solver(A, **self.solverOpts)

        # dCdu = self.A
        # Solve for $\deriv{u}{m}$
        # dCdu_inv = self.Solver(dCdu, **self.solverOpts)
        P        = self.survey.getP(self.mesh)
        J_x_v    = - P * mkvc( self.Ainv * dCdm_x_v )
        return -J_x_v

    def Jvec(self, m, v, u=None):
        return self.forward(v)

    def Jtvec(self, m, v, u=None):

        self.curModel = m
        # sigma = self.curModel.transform # $\sigma = \mathcal{M}(\m)$
        sigma = self.sigma
        if self.u is None:
            u = self.fields(sigma)
        else:
        	u = self.u
        shp = (self.mesh.nC, self.survey.nSrc)
        u = u.reshape(shp, order='F')
        P = self.survey.getP(self.mesh)
        PT_x_v = (P.T*v).reshape(shp, order='F')

        D = self.mesh.faceDiv
        G = self.mesh.cellGrad
        A = self.A
        mT_dm = Utils.sdiag(sigma)*self.mapping.deriv(m)
        # mT_dm = self.mapping.deriv(m)

        # dCdu = A.T
        # Ainv = self.Solver(dCdu, **self.solverOpts)
        # if self.Ainv == None:
    	self.Ainv = self.Solver(A.T, **self.solverOpts)

        w = self.Ainv * PT_x_v

        Jtv = 0
        for i, ui in enumerate(u.T):  # loop over each column
            Jtv += self.dMdsig( G * ui ).T * ( D.T * w[:,i] )

        Jtv = - mT_dm.T * ( Jtv )
        return -Jtv

