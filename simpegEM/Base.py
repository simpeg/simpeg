from SimPEG import Survey, Problem, Utils, Models, Maps, PropMaps, np, sp, Solver as SimpegSolver
from scipy.constants import mu_0

class EMPropMap(Maps.PropMap):
    """ 
        Property Map for EM Problems. The electrical conductivity (\\(\\sigma\\)) is the default inversion property, and the default value of the magnetic permeability is that of free space (\\(\\mu = 4\\pi\\times 10^{-7} \\) H/m)
    """

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
        """
            Edge inner product matrix
        """
        if getattr(self, '_Me', None) is None:
            self._Me = self.mesh.getEdgeInnerProduct()
        return self._Me

    @property
    def Mf(self):
        """
            Face inner product matrix
        """
        if getattr(self, '_Mf', None) is None:
            self._Mf = self.mesh.getFaceInnerProduct()
        return self._Mf


    # ----- Magnetic Permeability ----- # 
    @property
    def MfMui(self):
        """
            Face inner product matrix for \\(\\mu^{-1}\\). Used in the E-B formulation
        """
        if getattr(self, '_MfMui', None) is None:
            self._MfMui = self.mesh.getFaceInnerProduct(self.curModel.mui)
        return self._MfMui

    @property
    def MfMuiI(self):
        """
            Inverse of :code:`MfMui`.
        """
        if getattr(self, '_MfMuiI', None) is None:
            self._MfMuiI = self.mesh.getFaceInnerProduct(self.curModel.mui, invMat=True)
        return self._MfMuiI

    @property
    def MeMu(self):
        """
            Edge inner product matrix for \\(\\mu\\). Used in the H-J formulation
        """
        if getattr(self, '_MeMu', None) is None:
            self._MeMu = self.mesh.getEdgeInnerProduct(self.curModel.mu)
        return self._MeMu

    @property
    def MeMuI(self):
        """
            Inverse of :code:`MeMu`
        """
        if getattr(self, '_MeMuI', None) is None:
            self._MeMuI = self.mesh.getEdgeInnerProduct(self.curModel.mu, invMat=True)
        return self._MeMuI


    # ----- Electrical Conductivity ----- # 
    #TODO: hardcoded to sigma as the model
    @property
    def MeSigma(self):
        """
            Edge inner product matrix for \\(\\sigma\\). Used in the E-B formulation
        """
        if getattr(self, '_MeSigma', None) is None:
            self._MeSigma = self.mesh.getEdgeInnerProduct(self.curModel.sigma)
        return self._MeSigma

    # TODO: This should take a vector 
    def MeSigmaDeriv(self, u):
        """
            Derivative of MeSigma with respect to the model
        """ 
        return self.mesh.getEdgeInnerProductDeriv(self.curModel.sigma)(u) * self.curModel.sigmaDeriv
    

    @property
    def MeSigmaI(self):
        """
            Inverse of the edge inner product matrix for \\(\\sigma\\). 
        """
        if getattr(self, '_MeSigmaI', None) is None:
            self._MeSigmaI = self.mesh.getEdgeInnerProduct(self.curModel.sigma, invMat=True)
        return self._MeSigmaI

    # TODO: This should take a vector
    def MeSigmaIDeriv(self, u):
        """
            Derivative of :code:`MeSigma` with respect to the model 
        """ 
        # TODO: only works for diagonal tensors. getEdgeInnerProductDeriv, invMat=True should be implemented in SimPEG

        dMeSigmaI_dI = -self.MeSigmaI**2
        dMe_dsig = self.mesh.getEdgeInnerProductDeriv(self.curModel.sigma)(u)
        dsig_dm = self.curModel.sigmaDeriv
        return dMeSigmaI_dI * ( dMe_dsig * ( dsig_dm))
        # return self.mesh.getEdgeInnerProductDeriv(self.curModel.sigma, invMat=True)(u)


    @property
    def MfRho(self):
        """
            Face inner product matrix for \\(\\rho\\). Used in the H-J formulation
        """
        if getattr(self, '_MfRho', None) is None:
            self._MfRho = self.mesh.getFaceInnerProduct(self.curModel.rho)
        return self._MfRho

    # TODO: This should take a vector
    def MfRhoDeriv(self,u):
        """
            Derivative of :code:`MfRho` with respect to the model. 
        """
        return self.mesh.getFaceInnerProductDeriv(self.curModel.rho)(u) * (-Utils.sdiag(self.curModel.rho**2) * self.curModel.sigmaDeriv)
        # self.curModel.rhoDeriv

    @property
    def MfRhoI(self):
        """
            Inverse of :code:`MfRho`
        """
        if getattr(self, '_MfRhoI', None) is None:
            self._MfRhoI = self.mesh.getFaceInnerProduct(self.curModel.rho, invMat=True)
        return self._MfRhoI

    # TODO: This isn't going to work yet
    # TODO: This should take a vector
    def MfRhoIDeriv(self,u):
        """
            Derivative of :code:`MfRhoI` with respect to the model. 
        """
        return self.mesh.getFaceInnerProductDeriv(self.curModel.rho, invMat=True)(u) * self.curModel.rhoDeriv
