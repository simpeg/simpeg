from SimPEG import Survey, Problem, Utils, Models, np, sp, Solver as SimpegSolver
from scipy.constants import mu_0

class BaseEMProblem(Problem.BaseProblem):

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)


    surveyPair = Survey.BaseSurvey
    dataPair = Survey.Data

    Solver = SimpegSolver
    solverOpts = {}

    verbose = False

    ####################################################
    # Make A Symmetric
    ####################################################
    @property
    def _makeASymmetric(self):
        if getattr(self, '__makeASymmetric', None) is None:
            self.__makeASymmetric = True
        return self.__makeASymmetric

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
    def Me(self):
        if getattr(self, '_Me', None) is None:
            self._Me = self.mesh.getEdgeInnerProduct()
        return self._Me

    @property
    def Mf(self):
        if getattr(self, '_Mf', None) is None:
            self._Mf = self.mesh.getFaceInnerProduct()
        return self._Mf


    # ----- Magnetic Permeability ----- # 
    @property
    def MfMui(self):
        # TODO: hardcoded to assume diagonal mu
        if getattr(self, '_MfMui', None) is None:
            self._MfMui = self.mesh.getFaceInnerProduct(1/self.mu)
        return self._MfMui

    @property
    def MeMuI(self):
        if getattr(self, '_MeMuI', None) is None:
            self._MeMuI = self.mesh.getEdgeInnerProduct(self.mu, invMat=True)
        return self._MeMuI

    @property
    def MeMu(self):
        if getattr(self, '_MeMu', None) is None:
            self._MeMu = self.mesh.getEdgeInnerProduct(self.mu)
        return self._MeMu


    # ----- Electrical Conductivity ----- # 
    #TODO: hardcoded to sigma as the model
    @property
    def MeSigma(self):
        if getattr(self, '_MeSigma', None) is None:
            sigma = self.curModel.transform
            self._MeSigma = self.mesh.getEdgeInnerProduct(sigma)
        return self._MeSigma

    
    @property
    def MeSigmaI(self):
        if getattr(self, '_MeSigmaI', None) is None:
            sigma = self.curModel.transform
            self._MeSigmaI = self.mesh.getEdgeInnerProduct(sigma, invMat=True)
        return self._MeSigmaI

    @property
    def dMeSigmaI_dI(self):
        # TODO: hardcoded that sigma is diagonal
        if getattr(self, '_dMeSigmaI_dI', None) is None:
            self._dMeSigmaI_dI = - self.MeSigmaI**2
        return self._dMeSigmaI_dI

    @property
    def MfSigmai(self):
        #TODO: hardcoded to sigma diagonal 
        if getattr(self, '_MfSigmai', None) is None:
            sigma = self.curModel.transform
            self._MfSigmai = self.mesh.getFaceInnerProduct(1/sigma)
        return self._MfSigmai
    

    deleteTheseOnModelUpdate = ['_MeSigma', '_MeSigmaI','_MfSigmai']


    ####################################################
    # Fields
    ####################################################

    def fields(self, m):
        self.curModel = m
        F = self.forward(m)
        return F
