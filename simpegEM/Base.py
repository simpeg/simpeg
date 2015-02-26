from SimPEG import Survey, Problem, Utils, Models, np, sp, Solver as SimpegSolver
from scipy.constants import mu_0

class BaseEMProblem(Problem.BaseProblem):

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

    solType = None
    storeTheseFields = ['e', 'b']

    surveyPair = Survey.BaseSurvey
    dataPair = Survey.Data

    Solver = SimpegSolver
    solverOpts = {}

    verbose = False


    ####################################################
    # Mu Model
    ####################################################
    @property
    def mu(self):
        if getattr(self, '_mu', None) is None:
            self._mu = mu_0
        return self._mu
    @mu.setter
    def mu(self, value):
        if getattr(self, '_MfMui', None) is not None:
            del self._MfMui
        if getattr(self, '_MeMu', None) is not None:
            del delf._MeMu 
        if getattr(self, '_MeMuI', None) is not None:
            del self._MeMuI
        self._mu = value
        


    ####################################################
    # Mass Matrices
    ####################################################

    @property
    def MfMui(self):
        if getattr(self, '_MfMui', None) is None:
            self._MfMui = self.mesh.getFaceInnerProduct(1/self.mu)
        return self._MfMui

    @property
    def MeMuI(self):
        # Assuming isotropic mu! 
        if getattr(self, '_MeMuI', None) is None:
            self._MeMui = self.mesh.getEdgeInnerProduct(1/self.mu)
        return self._MeMuI

    @property
    def MeMu(self):
        #TODO: assuming constant mu
        if getattr(self, '_MeMu', None) is None:
            self._MeMu = self.mesh.getEdgeInnerProduct(self.mu)
        return self._MeMu

    @property
    def Me(self):
        if getattr(self, '_Me', None) is None:
            self._Me = self.mesh.getEdgeInnerProduct()
        return self._Me

    @property
    def MeSigma(self):
        #TODO: hardcoded to sigma as the model
        if getattr(self, '_MeSigma', None) is None:
            sigma = self.curModel.transform
            self._MeSigma = self.mesh.getEdgeInnerProduct(sigma)
        return self._MeSigma

    @property
    def MeSigmaI(self):
        #TODO: hardcoded to sigma as the model
        if getattr(self, '_MeSigmaI', None) is None:
            sigma = self.curModel.transform
            self._MeSigmaI = self.mesh.getEdgeInnerProduct(sigma, invMat=True)
        return self._MeSigmaI

    @property
    def MfSigmai(self):
        #TODO: hardcoded to sigma as the model
        #TODO: hardcoded to sigma diagonal 
        if getattr(self, '_MfSigmai', None) is None:
            sigma = self.curModel.transform
            self._MfSigmai = self.mesh.getFaceInnerProduct(1/sigma)
        return self._MfSigmai

    deleteTheseOnModelUpdate = ['_MeSigma', '_MeSigmaI','_MfSigmai']

    def fields(self, m):
        self.curModel = m
        F = self.forward(m, self.getRHS, self.calcFields)
        return F
