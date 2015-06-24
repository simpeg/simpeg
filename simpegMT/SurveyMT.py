from SimPEG import Survey, Utils, Problem, Maps, np, sp, mkvc
from simpegEM.FDEM.SurveyFDEM import SrcFDEM
from simpegEM.Utils.EMUtils import omega
from scipy.constants import mu_0
import sys
from numpy.lib import recfunctions as recFunc
from DataMT import DataMT
from simpegMT.Sources import homo1DModelSource
#################
### Receivers ###
#################

class RxMT(Survey.BaseRx):

    knownRxTypes = {
                    # 3D impedance
                    'zxxr':['Z3D', 'real'],
                    'zxyr':['Z3D', 'real'],
                    'zyxr':['Z3D', 'real'],
                    'zyyr':['Z3D', 'real'],
                    'zxxi':['Z3D', 'imag'],
                    'zxyi':['Z3D', 'imag'],
                    'zyxi':['Z3D', 'imag'],
                    'zyyi':['Z3D', 'imag'],
                    # 2D impedance
                    # TODO:
                    # 1D impedance
                    'z1dr':['Z1D', 'real'],
                    'z1di':['Z1D', 'imag']
                    #TODO: Add tipper fractions as well. Bz/B(x|y)
                    # 'exi':['e', 'Ex', 'imag'],
                    # 'eyi':['e', 'Ey', 'imag'],
                    # 'ezi':['e', 'Ez', 'imag'],

                    # 'bxr':['b', 'Fx', 'real'],
                    # 'byr':['b', 'Fy', 'real'],
                    # 'bzr':['b', 'Fz', 'real'],
                    # 'bxi':['b', 'Fx', 'imag'],
                    # 'byi':['b', 'Fy', 'imag'],
                    # 'bzi':['b', 'Fz', 'imag'],
                   }
    # TODO: Have locs as single or double coordinates for both or numerator and denominator separately, respectively.
    def __init__(self, locs, rxType):
        Survey.BaseRx.__init__(self, locs, rxType)

    @property
    def projField(self):
        """
        Field Type projection (e.g. e b ...)
        :param str fracPos: Position of the field in the data ratio

        """
        if 'numerator' in fracPos:
            return self.knownRxTypes[self.rxType][0][0]
        elif 'denominator' in fracPos:
            return self.knownRxTypes[self.rxType][1][0]
        else:
            raise Exception('{s} is an unknown option. Use numerator or denominator.')

    @property
    def projGLoc(self):
        """
        Grid Location projection (e.g. Ex Fy ...)
        :param str fracPos: Position of the field in the data ratio

        """
        if 'numerator' in fracPos:
            return self.knownRxTypes[self.rxType][0][1]
        elif 'denominator' in fracPos:
            return self.knownRxTypes[self.rxType][0][1]
        else:
            raise Exception('{s} is an unknown option. Use numerator or denominator.')
    @property
    def projType(self):
        """
        Receiver type for projection.

        """
        return self.knownRxTypes[self.rxType][0]

    @property
    def projComp(self):
        """Component projection (real/imag)"""
        return self.knownRxTypes[self.rxType][1]

    def projectFields(self, src, mesh, f):
        '''
        Project the fields and return the correct data.
        '''

        if self.projType is 'Z1D':
            Pex = mesh.getInterpolationMat(self.locs,'Fx')
            Pbx = mesh.getInterpolationMat(self.locs,'Ex')
            ex = Pex*mkvc(f[src,'e_1d'],2)
            bx = Pbx*mkvc(f[src,'b_1d'],2)/mu_0
            f_part_complex = ex/bx
        # elif self.projType is 'Z2D':
        elif self.projType is 'Z3D':
            # Get the projection
            Pex = mesh.getInterpolationMat(self.locs,'Ex')
            Pey = mesh.getInterpolationMat(self.locs,'Ey')
            Pbx = mesh.getInterpolationMat(self.locs,'Fx')
            Pby = mesh.getInterpolationMat(self.locs,'Fy')
            # Get the fields at location
            # px: x-polaration and py: y-polaration.
            ex_px = Pex*f[src,'e_px']
            ey_px = Pey*f[src,'e_px']
            ex_py = Pex*f[src,'e_py']
            ey_py = Pey*f[src,'e_py']
            hx_px = Pbx*f[src,'b_px']/mu_0
            hy_px = Pby*f[src,'b_px']/mu_0
            hx_py = Pbx*f[src,'b_py']/mu_0
            hy_py = Pby*f[src,'b_py']/mu_0
            # Make the complex data
            if 'zxx' in self.rxType:
                f_part_complex = (ex_px*hy_py - ex_py*hy_px)/(hx_px*hy_py - hx_py*hy_px)
            elif 'zxy' in self.rxType:
                f_part_complex  = (-ex_px*hx_py + ex_py*hx_px)/(hx_px*hy_py - hx_py*hy_px)
            elif 'zyx' in self.rxType:
                f_part_complex  = (ey_px*hy_py - ey_py*hy_px)/(hx_px*hy_py - hx_py*hy_px)
            elif 'zyy' in self.rxType:
                f_part_complex  = (-ey_px*hx_py + ey_py*hx_px)/(hx_px*hy_py - hx_py*hy_px)
        else:
            NotImplementedError('Projection of {:s} receiver type is not implemented.'.format(self.rxType))
        # Get the real or imag component
        real_or_imag = self.projComp
        f_part = getattr(f_part_complex, real_or_imag)
        # print f_part
        return f_part

    def projectFieldsDeriv(self, src, mesh, f, v, adjoint=False):
        """
        The derivative of the projection wrt u

        :param MTsrc src: MT source
        :param TensorMesh mesh: Mesh defining the topology of the problem
        :param MTfields f: MT fields object of the source
        :param numpy.ndarray v: Random vector of size
        """

        real_or_imag = self.projComp

        if not adjoint:
            if self.projType is 'Z1D':
                Pex = mesh.getInterpolationMat(self.locs,'Fx')
                Pbx = mesh.getInterpolationMat(self.locs,'Ex')
                # ex = Pex*mkvc(f[src,'e_1d'],2)
                # bx = Pbx*mkvc(f[src,'b_1d'],2)/mu_0
                dP_de = Utils.sdiag(1./(Pbx*mkvc(f[src,'b_1d'],2)/mu_0))*(Pex*v)
                dP_db = - Utils.sdiag(Pex*mkvc(f[src,'e_1d'],2))*(Utils.sdiag(1./(Pbx*mkvc(f[src,'b_1d'],2)/mu_0)).T*Utils.sdiag(1./(Pbx*mkvc(f[src,'b_1d'],2)/mu_0)))*(Pbx*f._bDeriv_u(src,v)/mu_0)
                PDeriv_complex = np.sum((dP_de,dP_db))
            # elif self.projType is 'Z2D
            elif self.projType is 'Z3D':
                raise NotImplementedError('Has not be implement for full impedance tensor')
            Pv = np.array(getattr(PDeriv_complex, real_or_imag))
        elif adjoint:
            if self.projType is 'Z1D':
                Pex = mesh.getInterpolationMat(self.locs,'Fx')
                Pbx = mesh.getInterpolationMat(self.locs,'Ex')
                # ex = Pex*mkvc(f[src,'e_1d'],2)
                # bx = Pbx*mkvc(f[src,'b_1d'],2)/mu_0
                dP_deTv = mkvc(Pex.T*Utils.sdiag(1./(Pbx*mkvc(f[src,'b_1d'],2)/mu_0)).T*v,2)
                db_duv = Pbx.T/mu_0*Utils.sdiag(1./(Pbx*mkvc(f[src,'b_1d'],2)/mu_0))*(Utils.sdiag(1./(Pbx*mkvc(f[src,'b_1d'],2)/mu_0))).T*Utils.sdiag(Pex*mkvc(f[src,'e_1d'],2)).T
                dP_dbTv = -mkvc(f._bDeriv_u(src,db_duv,adjoint=True)*v,2)
                PDeriv_complex = np.sum((dP_deTv,dP_dbTv))
            elif self.projType is 'Z3D':
                raise NotImplementedError('must be real or imag')
            Pv = np.array(getattr(PDeriv_complex, real_or_imag))
        return Pv


###############
### Sources ###
###############

class srcMT(SrcFDEM): # Survey.BaseSrc):
    '''
    Sources for the MT problem.
    Use the SimPEG BaseSrc, since the source fields share properties with the transmitters.

    :param float freq: The frequency of the source
    :param list rxList: A list of receivers associated with the source
    '''

    freq = None #: Frequency (float)
    rxPair = RxMT

    def __init__(self, rxList, freq):
        self.freq = float(freq)
        Survey.BaseSrc.__init__(self, rxList)

# 1D sources
class srcMT_polxy_1DhomotD(srcMT):
    """
    MT source for both polarizations (x and y) for the total Domain. It calculates fields calculated based on conditions on the boundary of the domain.
    """
    def __init__(self, rxList, freq):
        srcMT.__init__(self, rxList, freq)


    # TODO: need to add the  primary fields calc and source terms into the problem.


# Need to implement such that it works for all dims.
class srcMT_polxy_1Dprimary(srcMT):
    """
    MT source for both polarizations (x and y) given a 1D primary models. It assigns fields calculated from the 1D model
    as fields in the full space of the problem.
    """
    def __init__(self, rxList, freq, sigma1d):
        # assert mkvc(self.mesh.hz.shape,1) == mkvc(sigma1d.shape,1),'The number of values in the 1D background model does not match the number of vertical cells (hz).'
        self.sigma1d = sigma1d
        srcMT.__init__(self, rxList, freq)



    def ePrimary(self,problem):
        # Get primary fields for both polarizations
        eBG_bp = homo1DModelSource(problem.mesh,self.freq,self.sigma1d)
        return eBG_bp

    def bPrimary(self,problem):
        # Project ePrimary to bPrimary
        # Satisfies the primary(background) field conditions
        if problem.mesh.dim == 1:
            C = problem.mesh.nodalGrad
        elif problem.mesh.dim == 3:
            C = problem.mesh.edgeCurl
        bBG_bp = (- C * self.ePrimary(problem) )/( 1j*omega(self.freq) )
        return bBG_bp

    def S_e(self,problem):
        """
        Get the electrical field source
        """
        e_p = self.ePrimary(problem)
        Map_sigma_p = Maps.Vertical1DMap(problem.mesh)
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

    def S_eDeriv(self, problem, v, adjoint = False):
        # Need to deal with
        if problem.mesh.dim == 1:
            # Need to use the faceInnerProduct
            MsigmaDeriv = problem.mesh.getFaceInnerProductDeriv(problem.curModel.sigma)(self.ePrimary(problem)[:,-1]) * problem.curModel.sigmaDeriv
            # MsigmaDeriv = ( MsigmaDeriv * MsigmaDeriv.T)**2
        if problem.mesh.dim == 2:
            pass
        if problem.mesh.dim == 3:
            MsigmaDeriv = problem.MeSigmaDeriv(self.ePrimary(problem))
        if adjoint:
            #
            return MsigmaDeriv.T * v
        else:
            # v should be nC size
            return MsigmaDeriv * v


##############
### Survey ###
##############
class SurveyMT(Survey.BaseSurvey):
    """
        Survey class for MT. Contains all the sources associated with the survey.

        :param list srcList: List of sources associated with the survey

    """

    srcPair = srcMT

    def __init__(self, srcList, **kwargs):
        # Sort these by frequency
        self.srcList = srcList
        Survey.BaseSurvey.__init__(self, **kwargs)

        _freqDict = {}
        for src in srcList:
            if src.freq not in _freqDict:
                _freqDict[src.freq] = []
            _freqDict[src.freq] += [src]

        self._freqDict = _freqDict
        self._freqs = sorted([f for f in self._freqDict])

    @property
    def freqs(self):
        """Frequencies"""
        return self._freqs

    @property
    def nFreq(self):
        """Number of frequencies"""
        return len(self._freqDict)

    # TODO: Rename to getSources
    def getSrcByFreq(self, freq):
        """Returns the sources associated with a specific frequency."""
        assert freq in self._freqDict, "The requested frequency is not in this survey."
        return self._freqDict[freq]

    def projectFields(self, u):
        data = DataMT(self)
        for src in self.srcList:
            print 'Project at freq: {:.3e}'.format(src.freq)
            sys.stdout.flush()
            for rx in src.rxList:
                data[src, rx] = rx.projectFields(src, self.mesh, u)
        return data

    def projectFieldsDeriv(self, u):
        raise Exception('Use Transmitters to project fields deriv.')

