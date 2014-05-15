from BaseTDEM import BaseTDEMProblem
from SimPEG.Utils import mkvc
import numpy as np
from SurveyTDEM import SurveyTDEM, FieldsTDEM

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
        B_n = np.c_[[F[tx,'b',tInd] for tx in self.survey.txList]].T
        RHS = (1.0/dt)*self.MfMui*B_n
        return RHS

    def calcFields(self, sol, tInd):

        if self.solType == 'b':
            b = sol
            e = self.MeSigmaI*(self.mesh.edgeCurl.T*(self.MfMui*b))
            # Todo: implement non-zero js
        else:
            raise NotImplementedError('solType "%s" is not implemented in CalcFields.' % self.solType)

        return {'b':b, 'e':e}


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

        # Note: Fields has shape (nF/E, nTx, nT+1)
        #       However, p will only really fill (:,:,1:nT+1)
        #       meaning the 'initial fields' are zero (:,:,0)
        p = FieldsTDEM(self.mesh, self.survey)
        # 'b' at all times is zero.
        #       However, to save memory we will **not** do:
        #
        #               p[:, 'b', :] = 0.0

        # fake initial 'e' fields
        p[:, 'e', 0] = 0.0
        c = self.mesh.getEdgeInnerProductDeriv(self.curTModel)*(self.curTModelDeriv*vec)
        for i in range(1,self.nT+1):
            # TODO: G[1] may be dependent on the model
            #       for a galvanic source (deriv of the dc problem)
            for tx in self.survey.txList:
                p[tx, 'e', i] = -u[tx,'e',i]*c # i.e.: - diag(e) * MsigDeriv * v
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

        nTx, nE = self.survey.nTx, self.mesh.nE
        tmp = np.zeros(nE)
        # Here we can do internal multiplications of Gt*v and then multiply by MsigDeriv.T in one go.
        for i in range(1,self.nT+1):
            vu = vec[:,'e',i]*u[:,'e',i]
            if nTx > 1:
                vu = vu.sum(axis=1)
            tmp += vu
        p = -mkvc(self.curTModelDeriv.T*(self.mesh.getEdgeInnerProductDeriv(self.curTModel).T*tmp))
        return p

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

        def AhCalcFields(sol, tInd):
            y_b = sol
            if self.survey.nTx == 1:
                y_b = mkvc(y_b)
            y_e = self.MeSigmaI*(self.mesh.edgeCurl.T*(self.MfMui*y_b)) - self.MeSigmaI*p[:,'e',tInd+1]
            return {'b':y_b, 'e':y_e}

        return self.forward(m, AhRHS, AhCalcFields)

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
            nTx, nF = self.survey.nTx, self.mesh.nF
            rhs = np.zeros(nF if nTx == 1 else (nF, nTx))

            if 'e' in p:
                rhs += self.MfMui*(self.mesh.edgeCurl*(self.MeSigmaI*p[:,'e',tInd+1]))
            if 'b' in p:
                rhs += p[:,'b',tInd+1]

            if tInd == self.nT-1:
                return rhs
            dt = self.timeSteps[tInd+1]
            return rhs + 1.0/dt*self.MfMui*y[:,'b',tInd+2]

        def AhtCalcFields(sol, tInd):
            y_b = sol
            if self.survey.nTx == 1:
                y_b = mkvc(y_b)
            y_e = self.MeSigmaI*(self.mesh.edgeCurl.T*(self.MfMui*y_b))
            if 'e' in p:
                y_e += - self.MeSigmaI*p[:,'e',tInd+1]
            return {'b':y_b, 'e':y_e}

        return self.adjoint(m, AhtRHS, AhtCalcFields)

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
