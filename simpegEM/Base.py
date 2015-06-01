from SimPEG import Survey, Problem, Utils, Models, Maps, PropMaps, np, sp, Solver as SimpegSolver
from scipy.constants import mu_0

class EMPropMap(Maps.PropMap):
    sigma = Maps.Property("Electrical Conductivity", defaultInvProp = True)
    mui = Maps.Property("Inverse Magnetic Permeability", defaultVal = 1./mu_0)

    # rho = Maps.Property("Electrical Resistivity") 
    # mu = Maps.Property("Inverse Magnetic Permeability", defaultVal = 1./mu_0)

    # Do some error checking: only 1 of sigma, rho can be InvProp similar story with mu and mui 
    # Also ensure that sigma and rho are reciprocals of one another "" 

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
    # Phys Props
    ####################################################

    # Mu
    # @property
    # def mu(self):
    #     if getattr(self, '_mu', None) is None:
    #         # if getattr(self, '_mui', None) is not None:
    #         #     self._mu = sel
    #         self._mu = mu_0
    #     return self._mu
    # @mu.setter
    # def mu(self, value):
    #     if getattr(self, '_MfMui', None) is not None:
    #         del self._MfMui
    #     if getattr(self, '_MfMuiI', None) is not None:
    #         del self._MfMuiI
    #     if getattr(self, '_MeMu', None) is not None:
    #         del delf._MeMu 
    #     if getattr(self, '_MeMuI', None) is not None:
    #         del self._MeMuI
    #     self._mu = value
    
    # TODO: hardcoded to assume diagonal mu
    # @property
    # def mui(self):
    #     if getattr(self, '_mui', None) is None:
    #         self._mui = 1./mu_0
    #     return self._mui
    # @mui.setter
    # def mui(self, value):
    #     if getattr(self, '_MfMui', None) is not None:
    #         del self._MfMui
    #     if getattr(self, '_MfMuiI', None) is not None:
    #         del self._MfMuiI
    #     if getattr(self, '_MeMu', None) is not None:
    #         del delf._MeMu 
    #     if getattr(self, '_MeMuI', None) is not None:
    #         del self._MeMuI
    #     self._mui = value

    # Sigma
    # @property
    # def sigma(self):
    #     if getattr(self, '_sigma', None) is None:
    #         self._sigma = self.curModel.transform
    #     return self._sigma
    # @sigma.setter
    # def sigma(self, value):
    #     if getattr(self, '_MeSigma', None) is not None:
    #         del self._MeSigma
    #     if getattr(self, '_MeSigmaI', None) is not None:
    #         del self._MeSigmaI
    #     if getattr(self, '_MfSigmai', None) is not None:
    #         del delf._MfSigmai 
    #     if getattr(self, '_MfSigmaiI', None) is not None:
    #         del self._MfSigmaiI 
    #     self._sigma = value

    # def dsigma_dm(self):
    #     return self.curModel.transformDeriv


    # TODO: hardcoded to assume diagonal sigma
    # @property
    # def sigmai(self):
    #     if getattr(self, '_sigmai', None) is None:
    #         self._sigmai = 1./self.curModel.transform
    #     return self._sigmai
    # @sigmai.setter
    # def sigmai(self, value):
    #     if getattr(self, '_MeSigma', None) is not None:
    #         del self._MeSigma
    #     if getattr(self, '_MeSigmaI', None) is not None:
    #         del self._MeSigmaI
    #     if getattr(self, '_MfSigmai', None) is not None:
    #         del delf._MfSigmai 
    #     if getattr(self, '_MfSigmaiI', None) is not None:
    #         del self._MfSigmaiI 
    #     self._sigma = value


    ####################################################
    # Mass Matrices
    ####################################################

    # TODO: Link to EMPropMap 
    # if Prop
    # deleteTheseOnModelUpdate = ['_MeSigma', '_MeSigmaI','_MfSigmai','_MfSigmaiI']
    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        if self.mapping.sigmaMap is not None:
            toDelete += ['_MeSigma', '_MeSigmaI','_MfSigmai','_MfSigmaiI']
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
            self._MeMu = self.mesh.getEdgeInnerProduct(self.mu)
        return self._MeMu

    @property
    def MeMuI(self):
        if getattr(self, '_MeMuI', None) is None:
            self._MeMuI = self.mesh.getEdgeInnerProduct(self.mu, invMat=True)
        return self._MeMuI

    # ----- Electrical Conductivity ----- # 
    #TODO: hardcoded to sigma as the model
    @property
    def MeSigma(self):
        if getattr(self, '_MeSigma', None) is None:
            self._MeSigma = self.mesh.getEdgeInnerProduct(self.curModel.sigma)
        return self._MeSigma

    # def dMeSigma_dsigma(self, u):
    #     return self.mesh.getEdgeInnerProductDeriv(self.sigma)(u)
    
    @property
    def MeSigmaI(self):
        if getattr(self, '_MeSigmaI', None) is None:
            self._MeSigmaI = self.mesh.getEdgeInnerProduct(self.curModel.sigma, invMat=True)
        return self._MeSigmaI

    # def dMeSigmaI_dsigma(self,u)

    @property
    def MfSigmai(self):
        if getattr(self, '_MfSigmai', None) is None:
            self._MfSigmai = self.mesh.getFaceInnerProduct(self.sigmai)
        return self._MfSigmai

    # def dMfSigmai_dsigmai(self,u)

    @property
    def MfSigmaiI(self):
        if getattr(self, '_MfSigmaiI', None) is None:
            self._MfSigmaiI = self.mesh.getFaceInnerProduct(self.sigmai, invMat=True)
        return self._MfSigmaiI

    # def dMfSigmaiI(self,u)
    

    ####################################################
    # Fields
    ####################################################

    def fields(self, m):
        self.curModel = m
        F = self.forward(m)
        return F
