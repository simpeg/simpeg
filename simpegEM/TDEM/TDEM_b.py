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

    solType = 'b'

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


    ####################################################
    # Derivatives
    ####################################################

    def Jvec(self, m, v, u=None):
        if u is None:
            u = self.fields(m)
        p = self.Gvec(m, v, u)
        y = self.solveAh(m, p)
        Jv = self.survey.projectFieldsDeriv(u, v=y)
        return mkvc(Jv)

    def Jtvec(self, m, v, u=None):
        if u is None:
            u = self.fields(m)

        if not isinstance(v, self.dataPair):
            v = self.dataPair(self.survey, v)

        p = self.survey.projectFieldsDeriv(u, v=v, adjoint=True)
        y = self.solveAht(m, p)
        w = self.Gtvec(m, y, u)
        return w

    def Gvec(self, m, vec, u=None):
        """
            :param numpy.array m: Conductivity model
            :param numpy.array vec: vector (like a model)
            :param simpegEM.TDEM.FieldsTDEM u: Fields resulting from m
            :rtype: simpegEM.TDEM.FieldsTDEM
            :return: f

            Multiply G by a vector where
        """
        if u is None:
            u = self.fields(m)

        p = FieldsTDEM(self.mesh, self.survey)
        p[:, 'b', :] = 0.0 #np.zeros((self.mesh.nF, self.survey.nTx, self.prob.nT))
        p[:, 'e', 0] = 0.0 #np.zeros((self.mesh.nF, self.survey.nTx))
        # p = FieldsTDEM(self.mesh, 1, self.nT, 'b')
        curModel = self.mapping.transform(m)
        c = self.mesh.getEdgeInnerProductDeriv(curModel)*self.mapping.transformDeriv(m)*vec
        for i in range(self.nT):
            for tx in self.survey.txList:
                p[tx, 'e', i+1] = -u[tx,'e',i+1]*c
        return p

    def Gtvec(self, m, v, u=None):
        if u is None:
            u = self.fields(m)
        nTx, nE = self.survey.nTx, self.mesh.nE
        tmp = np.zeros(nE if nTx == 1 else (nE,nTx))
        for i in range(1,self.nT+1):
            tmp += v[:,'e',i]*u[:,'e',i]

        curModel = self.mapping.transform(m)
        p = -mkvc(self.mapping.transformDeriv(m).T*self.mesh.getEdgeInnerProductDeriv(curModel).T*tmp)
        return p

    def solveAh(self, m, p):
        def AhRHS(tInd, u):
            rhs = self.MfMui*self.mesh.edgeCurl*self.MeSigmaI*p[:,'e',tInd+1] + p[:,'b',tInd+1]
            if tInd == 0:
                return rhs
            dt = self.timeSteps[tInd]
            return rhs + 1.0/dt*self.MfMui*u[:,'b',tInd]

        def AhCalcFields(sol, solType, tInd):
            b = sol
            if self.survey.nTx == 1:
                b = mkvc(b)
            e = self.MeSigmaI*self.mesh.edgeCurl.T*self.MfMui*b - self.MeSigmaI*p[:,'e',tInd+1]
            return {'b':b, 'e':e}

        self.curModel = m
        return self.forward(m, AhRHS, AhCalcFields)

    def solveAht(self, m, p):

        def AhtRHS(tInd, u):
            rhs = self.MfMui*self.mesh.edgeCurl*self.MeSigmaI*p[:,'e',tInd] + p[:,'b',tInd]
            if tInd == self.nT-1:
                return rhs
            dt = self.timeSteps[tInd+1]
            return rhs + 1.0/dt*self.MfMui*u[:,'b',tInd+1]

        def AhtCalcFields(sol, solType, tInd):
            b = sol
            if self.survey.nTx == 1:
                b = mkvc(b)
            e = self.MeSigmaI*self.mesh.edgeCurl.T*self.MfMui*b - self.MeSigmaI*p[:,'e',tInd]
            return {'b':b, 'e':e}

        self.curModel = m
        return self.adjoint(m, AhtRHS, AhtCalcFields)

    ####################################################
    # Functions for tests
    ####################################################

    def AhVec(self, m, vec):
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
            b = 1.0/dt*self.MfMui*vec[:,'b',i] + self.MfMui*self.mesh.edgeCurl*vec[:,'e',i]
            if i > 1:
                b = b - 1.0/dt*self.MfMui*vec[:,'b',i-1]
            f[:,'b',i] = b
            f[:,'e',i] = self.mesh.edgeCurl.T*self.MfMui*vec[:,'b',i] - self.MeSigma*vec[:,'e',i]
        return f

    def AhtVec(self, m, vec):
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
        for i in range(1,self.nT+1):
            b = 1.0/self.timeSteps[i-1]*self.MfMui*vec[:,'b',i] + self.MfMui*self.mesh.edgeCurl*vec[:,'e',i]
            if i < self.nT:
                b = b - 1.0/self.timeSteps[i]*self.MfMui*vec[:,'b',i+1]
            f[:,'b', i] = b
            f[:,'e', i] = self.mesh.edgeCurl.T*self.MfMui*vec[:,'b',i] - self.MeSigma*vec[:,'e',i]
        return f



if __name__ == '__main__':
    from SimPEG import *
    import simpegEM as EM
    from simpegEM.Utils.Ana import hzAnalyticDipoleT
    from scipy.constants import mu_0
    import matplotlib.pyplot as plt

    cs, ncx, ncz, npad = 5., 20, 6, 20
    hx = [(cs, ncx), (cs, npad, 1.3)]
    hz = [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
    mesh = Mesh.CylMesh([hx,1,hz], '00C')
    mapping = Maps.Vertical1DMap(mesh)

    opts = {'txLoc':0.,
            'txType':'VMD_MVP',
            'rxLoc':np.r_[150., 0., 0.],
            'rxType':'bz',
            'timeCh':np.logspace(-4,-2,20),
            }
    survey = EM.TDEM.SurveyTDEM1D(**opts)

    prb = EM.TDEM.ProblemTDEM_b(mesh, mapping=mapping)
    # prb.setTimes([1e-5, 5e-5, 2.5e-4], [150, 150, 150])
    # prb.setTimes([1e-5, 5e-5, 2.5e-4], [10, 10, 10])
    prb.timeSteps = [(1e-5, 10)]
    prb.pair(survey)
    m = np.random.rand(mesh.nCz)

    print survey.dpred(m)




