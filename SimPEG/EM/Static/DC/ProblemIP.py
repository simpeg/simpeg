from SimPEG import Problem, Utils
from SimPEG.EM.Base import BaseEMProblem
from SurveyDC import Survey
from FieldsDC import Fields, Fields_CC, Fields_N
from SimPEG.Utils import sdiag
import numpy as np
from SimPEG.Utils import Zero
from BoundaryUtils import getxBCyBC_CC

class IPPropMap(Maps.PropMap):
    """
        Property Map for IP Problems. The electrical chargeability,
        (\\(\\eta\\)) is the default inversion property
    """

    eta = Maps.Property("Electrical Chargeability", defaultInvProp = True)
    sigma = Maps.Property("Electrical Conductivity", defaultInvProp = False, propertyLink=('rho',Maps.ReciprocalMap))
    rho = Maps.Property("Electrical Resistivity", propertyLink=('sigma', Maps.ReciprocalMap))


class BaseIPProblem(BaseEMProblem):

    surveyPair = Survey
    fieldsPair = Fields
    PropMap = IPPropMap
    Ainv = None
    f = None

    def fields(self, m):
        self.curModel = m

        if self.f is None:
            f = self.fieldsPair(self.mesh, self.survey)
            if self.Ainv == None:
                A = self.getA()
                self.Ainv = self.Solver(A, **self.solverOpts)
            RHS = self.getRHS()
            u = self.Ainv * RHS
            Srcs = self.survey.srcList
            f[Srcs, self._solutionType] = u
        return f

    def Jvec(self, m, v, f=None):

        if f is None:
            f = self.fields(m)

        self.curModel = m

        Jv = self.dataPair(self.survey) #same size as the data

        A = self.getA()

        for src in self.survey.srcList:
            u_src = f[src, self._solutionType] # solution vector
            dA_dm_v = self.getADeriv(u_src, v)
            dRHS_dm_v = self.getRHSDeriv(src, v)
            du_dm_v = self.Ainv * ( - dA_dm_v + dRHS_dm_v )

            for rx in src.rxList:
                df_dmFun = getattr(f, '_%sDeriv'%rx.projField, None)
                df_dm_v = df_dmFun(src, du_dm_v, v, adjoint=False)
                Jv[src, rx] = rx.evalDeriv(src, self.mesh, f, df_dm_v)
        return Utils.mkvc(Jv)

    def Jtvec(self, m, v, f=None):
        if f is None:
            f = self.fields(m)

        self.curModel = m

        # Ensure v is a data object.
        if not isinstance(v, self.dataPair):
            v = self.dataPair(self.survey, v)

        Jtv = np.zeros(m.size)
        AT = self.getA()


        for src in self.survey.srcList:
            u_src = f[src, self._solutionType]
            for rx in src.rxList:
                PTv = rx.evalDeriv(src, self.mesh, f, v[src, rx], adjoint=True) # wrt f, need possibility wrt m
                df_duTFun = getattr(f, '_%sDeriv'%rx.projField, None)
                df_duT, df_dmT = df_duTFun(src, None, PTv, adjoint=True)
                ATinvdf_duT = self.Ainv * df_duT
                dA_dmT = self.getADeriv(u_src, ATinvdf_duT, adjoint=True)
                dRHS_dmT = self.getRHSDeriv(src, ATinvdf_duT, adjoint=True)
                du_dmT = -dA_dmT + dRHS_dmT
                Jtv += df_dmT + du_dmT

        return Utils.mkvc(Jtv)

    def getSourceTerm(self):
        """
        takes concept of source and turns it into a matrix
        """
        """
        Evaluates the sources, and puts them in matrix form

        :rtype: (numpy.ndarray, numpy.ndarray)
        :return: q (nC or nN, nSrc)
        """

        Srcs = self.survey.srcList

        if self._formulation is 'EB':
            n = self.mesh.nN
            # return NotImplementedError

        elif self._formulation is 'HJ':
            n = self.mesh.nC

        q = np.zeros((n, len(Srcs)))

        for i, src in enumerate(Srcs):
            q[:,i] = src.eval(self)
        return q

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        return toDelete

    # assume log rho or log cond

    def MfRhoIDeriv(self,u):
        """
            Derivative of :code:`MfRhoI` with respect to the model.
        """

        dMfRhoI_dI = -self.MfRhoI**2
        dMf_drho = self.mesh.getFaceInnerProductDeriv(self.curModel.rho)(u)
        drho_dlogrho = Utils.sdiag(self.curModel.rho)
        return dMfRhoI_dI * ( dMf_drho * ( drho_dlogrho))

    # TODO: This should take a vector
    def MeSigmaDeriv(self, u):
        """
            Derivative of MeSigma with respect to the model
        """
        dsigma_dlogsigma = Utils.sdiag(self.curModel.sigma)
        return self.mesh.getEdgeInnerProductDeriv(self.curModel.sigma)(u) * dsigma_dlogsigma

