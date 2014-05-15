from SimPEG import Survey, Problem, Utils, np, sp, Solver as SimpegSolver
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
    # Mass Matrices
    ####################################################

    @property
    def MfMui(self):
        #TODO: assuming constant mu
        if getattr(self, '_MfMui', None) is None:
            self._MfMui = self.mesh.getFaceInnerProduct(1/mu_0)
        return self._MfMui

    @property
    def Me(self):
        if getattr(self, '_Me', None) is None:
            self._Me = self.mesh.getEdgeInnerProduct()
        return self._Me

    @property
    def MeSigma(self):
        #TODO: hardcoded to sigma as the model
        if getattr(self, '_MeSigma', None) is None:
            sigma = self.curTModel
            self._MeSigma = self.mesh.getEdgeInnerProduct(sigma)
        return self._MeSigma

    @property
    def MeSigmaI(self):
        #TODO: hardcoded to sigma as the model
        if getattr(self, '_MeSigmaI', None) is None:
            sigma = self.curTModel
            self._MeSigmaI = self.mesh.getEdgeInnerProduct(sigma, invMat=True)
        return self._MeSigmaI

    curModel = Utils.dependentProperty('_curModel', None, ['_MeSigma', '_MeSigmaI', '_curTModel', '_curTModelDeriv'], 'Sets the current model, and removes dependent mass matrices.')

    @property
    def curTModel(self):
        if getattr(self, '_curTModel', None) is None:
            self._curTModel = self.mapping.transform(self.curModel)
        return self._curTModel

    @property
    def curTModelDeriv(self):
        if getattr(self, '_curTModelDeriv', None) is None:
            self._curTModelDeriv = self.mapping.transformDeriv(self.curModel)
        return self._curTModelDeriv

    def fields(self, m):
        self.curModel = m
        F = self.forward(m, self.getRHS, self.calcFields)
        return F
