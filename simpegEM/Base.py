from SimPEG import Survey, Problem, Utils, Models, Maps, PropMaps, np, sp, Solver as SimpegSolver
from scipy.constants import mu_0

class EMPropMap(Maps.PropMap):
    sigma = Maps.Property("Electrical Conductivity", defaultInvProp = True, propertyLink=('rho',Maps.ReciprocalMap))
    mu = Maps.Property("Inverse Magnetic Permeability", defaultVal = mu_0, propertyLink=('mui',Maps.ReciprocalMap))

    rho = Maps.Property("Electrical Resistivity", propertyLink=('sigma', Maps.ReciprocalMap)) 
    mui = Maps.Property("Inverse Magnetic Permeability", defaultVal = 1./mu_0, propertyLink=('mu', Maps.ReciprocalMap))


class BaseEMProblem(Problem.BaseProblem):

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)


    surveyPair = Survey.BaseSurvey
    dataPair = Survey.Data
    
    PropMap = EMPropMap

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
    # Mass Matrices
    ####################################################

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        if self.mapping.sigmaMap is not None or self.mapping.rhoMap is not None:
            toDelete += ['_MeSigma', '_MeSigmaI','_MfRho','_MfRhoI']
        if self.mapping.muMap is not None or self.mapping.muiMap is not None:
            toDelete += ['_MeMu', '_MeMuI','_MfMui','_MfMuiI']
        return toDelete
    
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
        if getattr(self, '_MfMui', None) is None:
            self._MfMui = self.mesh.getFaceInnerProduct(self.curModel.mui)
        return self._MfMui

    @property
    def MfMuiI(self):
        if getattr(self, '_MfMuiI', None) is None:
            self._MfMuiI = self.mesh.getFaceInnerProduct(self.curModel.mui, invMat=True)
        return self._MfMuiI

    @property
    def MeMu(self):
        if getattr(self, '_MeMu', None) is None:
            self._MeMu = self.mesh.getEdgeInnerProduct(self.curModel.mu)
        return self._MeMu

    @property
    def MeMuI(self):
        if getattr(self, '_MeMuI', None) is None:
            self._MeMuI = self.mesh.getEdgeInnerProduct(self.curModel.mu, invMat=True)
        return self._MeMuI

    # ----- Electrical Conductivity ----- # 
    #TODO: hardcoded to sigma as the model
    @property
    def MeSigma(self):
        if getattr(self, '_MeSigma', None) is None:
            self._MeSigma = self.mesh.getEdgeInnerProduct(self.curModel.sigma)
        return self._MeSigma

    def MeSigmaDeriv(self, u):
        """
        Deriv of MeSigma wrt sigma
        """ 
        return self.mesh.getEdgeInnerProductDeriv(self.curModel.sigma)(u)
    

    @property
    def MeSigmaI(self):
        if getattr(self, '_MeSigmaI', None) is None:
            self._MeSigmaI = self.mesh.getEdgeInnerProduct(self.curModel.sigma, invMat=True)
        return self._MeSigmaI

    def MeSigmaIDeriv(self, u):
        """
        Deriv of MeSigma wrt sigma
        """ 
        return self.mesh.getEdgeInnerProductDeriv(self.curModel.sigma, invMat=True)(u)


    @property
    def MfRho(self):
        if getattr(self, '_MfRho', None) is None:
            self._MfRho = self.mesh.getFaceInnerProduct(self.curModel.rho)
        return self._MfRho

    def MfRhoDeriv(self,u):
        return self.mesh.getFaceInnerProductDeriv(self.curModel.rho)(u)

    @property
    def MfRhoI(self):
        if getattr(self, '_MfRhoI', None) is None:
            self._MfRhoI = self.mesh.getFaceInnerProduct(self.curModel.rho, invMat=True)
        return self._MfRhoI

    def dMfRhoIDeriv(self,u):
        return self.mesh.getFaceInnerProductDeriv(self.curModel.rho, invMat=True)
