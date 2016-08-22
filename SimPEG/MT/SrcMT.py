from SimPEG import Utils, Problem, Maps, np, sp, mkvc
from SimPEG.EM.FDEM.SrcFDEM import BaseSrc as FDEMBaseSrc
from SimPEG.EM.Utils import omega
from scipy.constants import mu_0
from numpy.lib import recfunctions as recFunc
from .Utils.sourceUtils import homo1DModelSource
from .Utils import rec2ndarr
import sys

#################
###   Sources ###
#################

class BaseMTSrc(FDEMBaseSrc):
    '''
    Sources for the MT problem.
    Use the SimPEG BaseSrc, since the source fields share properties with the transmitters.

    :param float freq: The frequency of the source
    :param list rxList: A list of receivers associated with the source
    '''

    freq = None #: Frequency (float)


    def __init__(self, rxList, freq):

        self.freq = float(freq)
        FDEMBaseSrc.__init__(self, rxList)

# 1D sources
class polxy_1DhomotD(BaseMTSrc):
    """
    MT source for both polarizations (x and y) for the total Domain.

    It calculates fields calculated based on conditions on the boundary of the domain.
    """
    def __init__(self, rxList, freq):
        BaseMTSrc.__init__(self, rxList, freq)


    # TODO: need to add the  primary fields calc and source terms into the problem.

# Need to implement such that it works for all dims.
class polxy_1Dprimary(BaseMTSrc):
    """
    MT source for both polarizations (x and y) given a 1D primary models.
    It assigns fields calculated from the 1D model as fields in the full space of the problem.
    """
    def __init__(self, rxList, freq):
        # assert mkvc(self.mesh.hz.shape,1) == mkvc(sigma1d.shape,1),'The number of values in the 1D background model does not match the number of vertical cells (hz).'
        self.sigma1d = None
        BaseMTSrc.__init__(self, rxList, freq)
        # Hidden property of the ePrimary
        self._ePrimary = None

    def ePrimary(self,problem):
        # Get primary fields for both polarizations
        if self.sigma1d is None:
            # Set the sigma1d as the 1st column in the background model
            if len(problem._sigmaPrimary) == problem.mesh.nC:
                if problem.mesh.dim == 1:
                    self.sigma1d = problem.mesh.r(problem._sigmaPrimary,'CC','CC','M')[:]
                elif problem.mesh.dim == 3:
                    self.sigma1d = problem.mesh.r(problem._sigmaPrimary,'CC','CC','M')[0,0,:]
            # Or as the 1D model that matches the vertical cell number
            elif len(problem._sigmaPrimary) == problem.mesh.nCz:
                self.sigma1d = problem._sigmaPrimary

        if self._ePrimary is None:
            self._ePrimary = homo1DModelSource(problem.mesh,self.freq,self.sigma1d)
        return self._ePrimary

    def bPrimary(self,problem):
        # Project ePrimary to bPrimary
        # Satisfies the primary(background) field conditions
        if problem.mesh.dim == 1:
            C = problem.mesh.nodalGrad
        elif problem.mesh.dim == 3:
            C = problem.mesh.edgeCurl
        bBG_bp = (- C * self.ePrimary(problem) )*(1/( 1j*omega(self.freq) ))
        return bBG_bp

    def S_e(self,problem):
        """
        Get the electrical field source
        """
        e_p = self.ePrimary(problem)
        Map_sigma_p = Maps.SurjectVertical1D(problem.mesh)
        sigma_p = Map_sigma_p._transform(self.sigma1d)
        # Make mass matrix
        # Note: M(sig) - M(sig_p) = M(sig - sig_p)
        # Need to deal with the edge/face discrepencies between 1d/2d/3d
        if problem.mesh.dim == 1:
            Mesigma = problem.mesh.getFaceInnerProduct(problem.curModel.sigma)
            Mesigma_p = problem.mesh.getFaceInnerProduct(sigma_p)
        if problem.mesh.dim == 2:
            pass
        if problem.mesh.dim == 3:
            Mesigma = problem.MeSigma
            Mesigma_p = problem.mesh.getEdgeInnerProduct(sigma_p)
        return (Mesigma - Mesigma_p) * e_p

    def S_eDeriv_m(self, problem, v, adjoint = False):
        '''
        Get the derivative of S_e wrt to sigma (m)
        '''
        # Need to deal with
        if problem.mesh.dim == 1:
            # Need to use the faceInnerProduct
            MsigmaDeriv = problem.mesh.getFaceInnerProductDeriv(problem.curModel.sigma)(self.ePrimary(problem)[:,1]) * problem.curModel.sigmaDeriv
            # MsigmaDeriv = ( MsigmaDeriv * MsigmaDeriv.T)**2
        if problem.mesh.dim == 2:
            pass
        if problem.mesh.dim == 3:
            # Need to take the derivative of both u_px and u_py
            ePri = self.ePrimary(problem)
            # MsigmaDeriv = problem.MeSigmaDeriv(ePri[:,0]) + problem.MeSigmaDeriv(ePri[:,1])
            # MsigmaDeriv = problem.MeSigmaDeriv(np.sum(ePri,axis=1))
            if adjoint:
                return sp.hstack(( problem.MeSigmaDeriv(ePri[:,0]).T, problem.MeSigmaDeriv(ePri[:,1]).T ))*v
            else:
                return np.hstack(( mkvc(problem.MeSigmaDeriv(ePri[:,0]) * v,2), mkvc(problem.MeSigmaDeriv(ePri[:,1])*v,2) ))
        if adjoint:
            #
            return MsigmaDeriv.T * v
        else:
            # v should be nC size
            return MsigmaDeriv * v

class polxy_3Dprimary(BaseMTSrc):
    """
    MT source for both polarizations (x and y) given a 3D primary model. It assigns fields calculated from the 1D model
    as fields in the full space of the problem.
    """
    def __init__(self, rxList, freq):
        # assert mkvc(self.mesh.hz.shape,1) == mkvc(sigma1d.shape,1),'The number of values in the 1D background model does not match the number of vertical cells (hz).'
        self.sigmaPrimary = None
        BaseMTSrc.__init__(self, rxList, freq)
        # Hidden property of the ePrimary
        self._ePrimary = None

    def ePrimary(self,problem):
        # Get primary fields for both polarizations
        self.sigmaPrimary = problem._sigmaPrimary

        if self._ePrimary is None:
            self._ePrimary = homo3DModelSource(problem.mesh,self.sigmaPrimary,self.freq)
        return self._ePrimary

    def bPrimary(self,problem):
        # Project ePrimary to bPrimary
        # Satisfies the primary(background) field conditions
        if problem.mesh.dim == 1:
            C = problem.mesh.nodalGrad
        elif problem.mesh.dim == 3:
            C = problem.mesh.edgeCurl
        bBG_bp = (- C * self.ePrimary(problem) )*(1/( 1j*omega(self.freq) ))
        return bBG_bp

    def S_e(self,problem):
        """
        Get the electrical field source
        """
        e_p = self.ePrimary(problem)
        Map_sigma_p = Maps.SurjectVertical1D(problem.mesh)
        sigma_p = Map_sigma_p._transform(self.sigma1d)
        # Make mass matrix
        # Note: M(sig) - M(sig_p) = M(sig - sig_p)
        # Need to deal with the edge/face discrepencies between 1d/2d/3d
        if problem.mesh.dim == 1:
            Mesigma = problem.mesh.getFaceInnerProduct(problem.curModel.sigma)
            Mesigma_p = problem.mesh.getFaceInnerProduct(sigma_p)
        if problem.mesh.dim == 2:
            pass
        if problem.mesh.dim == 3:
            Mesigma = problem.MeSigma
            Mesigma_p = problem.mesh.getEdgeInnerProduct(sigma_p)
        return (Mesigma - Mesigma_p) * e_p

    def S_eDeriv_m(self, problem, v, adjoint = False):
        '''
        Get the derivative of S_e wrt to sigma (m)
        '''
        # Need to deal with
        if problem.mesh.dim == 1:
            # Need to use the faceInnerProduct
            MsigmaDeriv = problem.mesh.getFaceInnerProductDeriv(problem.curModel.sigma)(self.ePrimary(problem)[:,1]) * problem.curModel.sigmaDeriv
            # MsigmaDeriv = ( MsigmaDeriv * MsigmaDeriv.T)**2
        if problem.mesh.dim == 2:
            pass
        if problem.mesh.dim == 3:
            # Need to take the derivative of both u_px and u_py
            ePri = self.ePrimary(problem)
            # MsigmaDeriv = problem.MeSigmaDeriv(ePri[:,0]) + problem.MeSigmaDeriv(ePri[:,1])
            # MsigmaDeriv = problem.MeSigmaDeriv(np.sum(ePri,axis=1))
            if adjoint:
                return sp.hstack(( problem.MeSigmaDeriv(ePri[:,0]).T, problem.MeSigmaDeriv(ePri[:,1]).T ))*v
            else:
                return np.hstack(( mkvc(problem.MeSigmaDeriv(ePri[:,0]) * v,2), mkvc(problem.MeSigmaDeriv(ePri[:,1])*v,2) ))
        if adjoint:
            #
            return MsigmaDeriv.T * v
        else:
            # v should be nC size
            return MsigmaDeriv * v
