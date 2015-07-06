from BaseTDEM import BaseTDEMProblem, FieldsTDEM
from SimPEG.Utils import mkvc, sdiag
import numpy as np
from SurveyTDEM import SurveyTDEM


class FieldsTDEM_e_from_b(FieldsTDEM):
    """Fancy Field Storage for a TDEM survey."""
    knownFields = {'b': 'F'}
    aliasFields = {'e': ['b','E','e_from_b']}

    def startup(self):
        self.MeSigmaI  = self.survey.prob.MeSigmaI
        self.edgeCurlT = self.survey.prob.mesh.edgeCurl.T
        self.MfMui     = self.survey.prob.MfMui

    def e_from_b(self, b, srcInd, timeInd):
        # TODO: implement non-zero js
        return self.MeSigmaI*(self.edgeCurlT*(self.MfMui*b))

class FieldsTDEM_e_from_b_Ah(FieldsTDEM):
    """Fancy Field Storage for a TDEM survey.

        This is used when solving Ahat and AhatT
    """
    knownFields = {'b': 'F'}
    aliasFields = {'e': ['b','E','e_from_b']}
    p = None

    def startup(self):
        self.MeSigmaI  = self.survey.prob.MeSigmaI
        self.edgeCurlT = self.survey.prob.mesh.edgeCurl.T
        self.MfMui     = self.survey.prob.MfMui

    def e_from_b(self, y_b, srcInd, tInd):
        y_e = self.MeSigmaI*(self.edgeCurlT*(self.MfMui*y_b))
        if 'e' in self.p:
            y_e = y_e - self.MeSigmaI*self.p[srcInd,'e',tInd]
        return y_e

class ProblemTDEM_b(BaseTDEMProblem):
    """
        Time-Domain EM problem - B-formulation

        TDEM_b treats the following discretization of Maxwell's equations

        .. math::
            \dcurl \e^{(t+1)} + \\frac{\\b^{(t+1)} - \\b^{(t)}}{\delta t} = 0 \\\\
            \dcurl^\\top \MfMui \\b^{(t+1)} - \MeSig \e^{(t+1)} = \Me \j_s^{(t+1)}

        with \\\(\\b\\\) defined on cell faces and \\\(\e\\\) defined on edges.
    """
    def __init__(self, mesh, mapping=None, **kwargs):
        BaseTDEMProblem.__init__(self, mesh, mapping=mapping, **kwargs)

    solType = 'b' #: Type of the solution, in this case the 'b' field

    surveyPair = SurveyTDEM
    _FieldsForward_pair = FieldsTDEM_e_from_b     #: used for the forward calculation only

    ####################################################
    # Internal Methods
    ####################################################

    def getA(self, tInd):
        """
            :param int tInd: Time index
            :rtype: scipy.sparse.csr_matrix
            :return: A
        """
        dt = self.timeSteps[tInd]
        return self.MfMui*self.mesh.edgeCurl*self.MeSigmaI*self.mesh.edgeCurl.T*self.MfMui + (1.0/dt)*self.MfMui

    def getRHS(self, tInd, F):
        dt = self.timeSteps[tInd]
        B_n = np.c_[[F[src,'b',tInd] for src in self.survey.srcList]].T
        if B_n.shape[0] is not 1:
            raise NotImplementedError('getRHS not implemented for this shape of B_n')
        RHS = (1.0/dt)*self.MfMui*B_n[0,:,:] #TODO: This is a hack
        return RHS

    ####################################################
    # Derivatives
    ####################################################

    def Gvec(self, m, vec, u=None):
        """
            :param numpy.array m: Conductivity model
            :param numpy.array vec: vector (like a model)
            :param simpegEM.TDEM.FieldsTDEM u: Fields resulting from m
            :rtype: simpegEM.TDEM.FieldsTDEM
            :return: f

            Multiply G by a vector
        """
        if u is None:
            u = self.fields(m)
        self.curModel = m

        # Note: Fields has shape (nF/E, nSrc, nT+1)
        #       However, p will only really fill (:,:,1:nT+1)
        #       meaning the 'initial fields' are zero (:,:,0)
        p = FieldsTDEM(self.mesh, self.survey)
        # 'b' at all times is zero.
        #       However, to save memory we will **not** do:
        #
        #               p[:, 'b', :] = 0.0

        # fake initial 'e' fields
        p[:, 'e', 0] = 0.0
        dMdsig = self.MeSigmaDeriv
        # self.mesh.getEdgeInnerProductDeriv(self.curModel.transform)
        # dsigdm_x_v = self.curModel.sigmaDeriv*vec
        # dsigdm_x_v = self.curModel.transformDeriv*vec
        for i in range(1,self.nT+1):
            # TODO: G[1] may be dependent on the model
            #       for a galvanic source (deriv of the dc problem)
            #
            # Do multiplication for all src in self.survey.srcList
            for src in self.survey.srcList:
                p[src, 'e', i] = - dMdsig(u[src,'e',i]) *  vec
        return p

    def Gtvec(self, m, vec, u=None):
        """
            :param numpy.array m: Conductivity model
            :param numpy.array vec: vector (like a fields)
            :param simpegEM.TDEM.FieldsTDEM u: Fields resulting from m
            :rtype: np.ndarray (like a model)
            :return: p

            Multiply G.T by a vector
        """
        if u is None:
            u = self.fields(m)
        self.curModel = m
        # dMdsig = self.mesh.getEdgeInnerProductDeriv(self.curModel.transform)
        # dsigdm = self.curModel.transformDeriv
        MeSigmaDeriv = self.MeSigmaDeriv

        nSrc = self.survey.nSrc
        VUs = None
        # Here we can do internal multiplications of Gt*v and then multiply by MsigDeriv.T in one go.
        for i in range(1,self.nT+1):
            vu = None
            for src in self.survey.srcList:
                vusrc = MeSigmaDeriv(u[src,'e',i]).T * vec[src,'e',i]
                vu = vusrc if vu is None else vu + vusrc
            VUs = vu if VUs is None else VUs + vu
        # p = -dsigdm.T*VUs
        return -VUs

    def solveAh(self, m, p):
        """
            :param numpy.array m: Conductivity model
            :param simpegEM.TDEM.FieldsTDEM p: Fields object
            :rtype: simpegEM.TDEM.FieldsTDEM
            :return: y

            Solve the block-matrix system \\\(\\\hat{A} \\\hat{y} = \\\hat{p}\\\):

            .. math::
                \mathbf{\hat{A}} = \left[
                    \\begin{array}{cccc}
                        A & 0 & & \\\\
                        B & A & & \\\\
                        & \ddots & \ddots & \\\\
                        & & B & A
                    \end{array}
                \\right] \\\\
                \mathbf{A} =
                \left[
                    \\begin{array}{cc}
                        \\frac{1}{\delta t} \MfMui & \MfMui\dcurl \\\\
                        \dcurl^\\top \MfMui & -\MeSig
                    \end{array}
                \\right] \\\\
                \mathbf{B} =
                \left[
                    \\begin{array}{cc}
                        -\\frac{1}{\delta t} \MfMui & 0 \\\\
                        0 & 0
                    \end{array}
                \\right] \\\\
        """

        def AhRHS(tInd, y):
            rhs = self.MfMui*(self.mesh.edgeCurl*(self.MeSigmaI*p[:,'e',tInd+1]))
            if 'b' in p:
                rhs = rhs + p[:,'b',tInd+1]
            if tInd == 0:
                return rhs
            dt = self.timeSteps[tInd]
            return rhs + 1.0/dt*self.MfMui*y[:,'b',tInd]

        F = FieldsTDEM_e_from_b_Ah(self.mesh, self.survey, p=p)

        return self.forward(m, AhRHS, F)

    def solveAht(self, m, p):
        """
            :param numpy.array m: Conductivity model
            :param simpegEM.TDEM.FieldsTDEM p: Fields object
            :rtype: simpegEM.TDEM.FieldsTDEM
            :return: y

            Solve the block-matrix system \\\(\\\hat{A}^\\\\top \\\hat{y} = \\\hat{p}\\\):

            .. math::
                \mathbf{\hat{A}}^\\top = \left[
                    \\begin{array}{cccc}
                        A & B & & \\\\
                          & \ddots & \ddots & \\\\
                          & & A & B \\\\
                          & & 0 & A
                    \end{array}
                \\right] \\\\
                \mathbf{A} =
                \left[
                    \\begin{array}{cc}
                        \\frac{1}{\delta t} \MfMui & \MfMui\dcurl \\\\
                        \dcurl^\\top \MfMui & -\MeSig
                    \end{array}
                \\right] \\\\
                \mathbf{B} =
                \left[
                    \\begin{array}{cc}
                        -\\frac{1}{\delta t} \MfMui & 0 \\\\
                        0 & 0
                    \end{array}
                \\right] \\\\
        """

        #  Mini Example:
        #
        #       nT = 3, len(times) == 4, fields stored in F[:,:,1:4]
        #
        #       0 is held for initial conditions (this shifts the storage by +1)
        #       ^
        #  fLoc 0     1     2     3
        #       |-----|-----|-----|
        #  tInd    0     1     2
        #                        / ___/
        #                      2          (tInd=2 uses fields 3 and would use 4 but it doesn't exist)
        #                 / ___/
        #                1                (tInd=1 uses fields 2 and 3)

        def AhtRHS(tInd, y):
            nSrc, nF = self.survey.nSrc, self.mesh.nF
            rhs = np.zeros((nF,1) if nSrc == 1 else (nF, nSrc))

            if 'e' in p:
                rhs += self.MfMui*(self.mesh.edgeCurl*(self.MeSigmaI*p[:,'e',tInd+1]))
            if 'b' in p:
                rhs += p[:,'b',tInd+1]

            if tInd == self.nT-1:
                return rhs
            dt = self.timeSteps[tInd+1]
            return rhs + 1.0/dt*self.MfMui*y[:,'b',tInd+2]

        F = FieldsTDEM_e_from_b_Ah(self.mesh, self.survey, p=p)

        return self.adjoint(m, AhtRHS, F)

    ####################################################
    # Functions for tests
    ####################################################

    def _AhVec(self, m, vec):
        """
            :param numpy.array m: Conductivity model
            :param simpegEM.TDEM.FieldsTDEM vec: Fields object
            :rtype: simpegEM.TDEM.FieldsTDEM
            :return: f

            Multiply the matrix \\\(\\\hat{A}\\\) by a fields vector where

            .. math::
                \mathbf{\hat{A}} = \left[
                    \\begin{array}{cccc}
                        A & 0 & & \\\\
                        B & A & & \\\\
                        & \ddots & \ddots & \\\\
                        & & B & A
                    \end{array}
                \\right] \\\\
                \mathbf{A} =
                \left[
                    \\begin{array}{cc}
                        \\frac{1}{\delta t} \MfMui & \MfMui\dcurl \\\\
                        \dcurl^\\top \MfMui & -\MeSig
                    \end{array}
                \\right] \\\\
                \mathbf{B} =
                \left[
                    \\begin{array}{cc}
                        -\\frac{1}{\delta t} \MfMui & 0 \\\\
                        0 & 0
                    \end{array}
                \\right] \\\\
        """

        self.curModel = m
        f = FieldsTDEM(self.mesh, self.survey)
        for i in range(1,self.nT+1):
            dt = self.timeSteps[i-1]
            b = 1.0/dt*self.MfMui*vec[:,'b',i] + self.MfMui*(self.mesh.edgeCurl*vec[:,'e',i])
            if i > 1:
                b = b - 1.0/dt*self.MfMui*vec[:,'b',i-1]
            f[:,'b',i] = b
            f[:,'e',i] = self.mesh.edgeCurl.T*(self.MfMui*vec[:,'b',i]) - self.MeSigma*vec[:,'e',i]
        return f

    def _AhtVec(self, m, vec):
        """
            :param numpy.array m: Conductivity model
            :param simpegEM.TDEM.FieldsTDEM vec: Fields object
            :rtype: simpegEM.TDEM.FieldsTDEM
            :return: f

            Multiply the matrix \\\(\\\hat{A}\\\) by a fields vector where

            .. math::
                \mathbf{\hat{A}}^\\top = \left[
                    \\begin{array}{cccc}
                        A & B & & \\\\
                          & \ddots & \ddots & \\\\
                          & & A & B \\\\
                          & & 0 & A
                    \end{array}
                \\right] \\\\
                \mathbf{A} =
                \left[
                    \\begin{array}{cc}
                        \\frac{1}{\delta t} \MfMui & \MfMui\dcurl \\\\
                        \dcurl^\\top \MfMui & -\MeSig
                    \end{array}
                \\right] \\\\
                \mathbf{B} =
                \left[
                    \\begin{array}{cc}
                        -\\frac{1}{\delta t} \MfMui & 0 \\\\
                        0 & 0
                    \end{array}
                \\right] \\\\
        """
        self.curModel = m
        f = FieldsTDEM(self.mesh, self.survey)
        for i in range(self.nT):
            b = 1.0/self.timeSteps[i]*self.MfMui*vec[:,'b',i+1] + self.MfMui*(self.mesh.edgeCurl*vec[:,'e',i+1])
            if i < self.nT-1:
                b = b - 1.0/self.timeSteps[i+1]*self.MfMui*vec[:,'b',i+2]
            f[:,'b', i+1] = b
            f[:,'e', i+1] = self.mesh.edgeCurl.T*(self.MfMui*vec[:,'b',i+1]) - self.MeSigma*vec[:,'e',i+1]
        return f
