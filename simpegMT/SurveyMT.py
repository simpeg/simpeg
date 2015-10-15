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
                    'z1di':['Z1D', 'imag'],
                    # Tipper
                    'tzxr':['T3D','real'],
                    'tzxi':['T3D','imag'],
                    'tzyr':['T3D','real'],
                    'tzyi':['T3D','imag']
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
            Pex = mesh.getInterpolationMat(self.locs[:,-1],'Fx')
            Pbx = mesh.getInterpolationMat(self.locs[:,-1],'Ex')
            ex = Pex*mkvc(f[src,'e_1d'],2)
            bx = Pbx*mkvc(f[src,'b_1d'],2)/mu_0
            # Note: Has a minus sign in front, to comply with quadrant calculations.
            # Can be derived from zyx case for the 3D case.
            f_part_complex = -ex/bx
        # elif self.projType is 'Z2D':
        elif self.projType is 'Z3D':
            if self.locs.ndim == 3:
                eFLocs = self.locs[:,:,0]
                bFLocs = self.locs[:,:,1]
            else:
                eFLocs = self.locs
                bFLocs = self.locs
            # Get the projection
            Pex = mesh.getInterpolationMat(eFLocs,'Ex')
            Pey = mesh.getInterpolationMat(eFLocs,'Ey')
            Pbx = mesh.getInterpolationMat(bFLocs,'Fx')
            Pby = mesh.getInterpolationMat(bFLocs,'Fy')
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
                f_part_complex = ( ex_px*hy_py - ex_py*hy_px)/(hx_px*hy_py - hx_py*hy_px)
            elif 'zxy' in self.rxType:
                f_part_complex = (-ex_px*hx_py + ex_py*hx_px)/(hx_px*hy_py - hx_py*hy_px)
            elif 'zyx' in self.rxType:
                f_part_complex = ( ey_px*hy_py - ey_py*hy_px)/(hx_px*hy_py - hx_py*hy_px)
            elif 'zyy' in self.rxType:
                f_part_complex = (-ey_px*hx_py + ey_py*hx_px)/(hx_px*hy_py - hx_py*hy_px)
        elif self.projType is 'T3D':
            if self.locs.ndim == 3:
                horLoc = self.locs[:,:,0]
                vertLoc = self.locs[:,:,1]
            else:
                horLoc = self.locs
                vertLoc = self.locs
            Pbx = mesh.getInterpolationMat(horLoc,'Fx')
            Pby = mesh.getInterpolationMat(horLoc,'Fy')
            Pbz = mesh.getInterpolationMat(vertLoc,'Fz')
            bx_px = Pbx*f[src,'b_px']
            by_px = Pby*f[src,'b_px']
            bz_px = Pbz*f[src,'b_px']
            bx_py = Pbx*f[src,'b_py']
            by_py = Pby*f[src,'b_py']
            bz_py = Pbz*f[src,'b_py']
            if 'tzx' in self.rxType:
                f_part_complex = (- by_px*bz_py + by_py*bz_px)/(bx_px*by_py - bx_py*by_px)
            if 'tzy' in self.rxType:
                f_part_complex = (  bx_px*bz_py - bx_py*bz_px)/(bx_px*by_py - bx_py*by_px)

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
                Pex = mesh.getInterpolationMat(self.locs[:,-1],'Fx')
                Pbx = mesh.getInterpolationMat(self.locs[:,-1],'Ex')
                # ex = Pex*mkvc(f[src,'e_1d'],2)
                # bx = Pbx*mkvc(f[src,'b_1d'],2)/mu_0
                dP_de = -mkvc(Utils.sdiag(1./(Pbx*mkvc(f[src,'b_1d'],2)/mu_0))*(Pex*v),2)
                dP_db = mkvc( Utils.sdiag(Pex*mkvc(f[src,'e_1d'],2))*(Utils.sdiag(1./(Pbx*mkvc(f[src,'b_1d'],2)/mu_0)).T*Utils.sdiag(1./(Pbx*mkvc(f[src,'b_1d'],2)/mu_0)))*(Pbx*f._bDeriv_u(src,v)/mu_0),2)
                PDeriv_complex = np.sum(np.hstack((dP_de,dP_db)),1)
            elif self.projType is 'Z2D':
                raise NotImplementedError('Has not be implement for 2D impedance tensor')
            elif self.projType is 'Z3D':
                if self.locs.ndim == 3:
                    eFLocs = self.locs[:,:,0]
                    bFLocs = self.locs[:,:,1]
                else:
                    eFLocs = self.locs
                    bFLocs = self.locs
                # Get the projection
                Pex = mesh.getInterpolationMat(eFLocs,'Ex')
                Pey = mesh.getInterpolationMat(eFLocs,'Ey')
                Pbx = mesh.getInterpolationMat(bFLocs,'Fx')
                Pby = mesh.getInterpolationMat(bFLocs,'Fy')
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
                # Derivatives as lambda functions
                ex_px_u = lambda vec: Pex*f._e_pxDeriv_u(src,vec)
                ey_px_u = lambda vec: Pey*f._e_pxDeriv_u(src,vec)
                ex_py_u = lambda vec: Pex*f._e_pyDeriv_u(src,vec)
                ey_py_u = lambda vec: Pey*f._e_pyDeriv_u(src,vec)
                hx_px_u = lambda vec: Pbx*f._b_pxDeriv_u(src,vec)/mu_0
                hy_px_u = lambda vec: Pby*f._b_pxDeriv_u(src,vec)/mu_0
                hx_py_u = lambda vec: Pbx*f._b_pyDeriv_u(src,vec)/mu_0
                hy_py_u = lambda vec: Pby*f._b_pyDeriv_u(src,vec)/mu_0

                # Define the components of the derivative
                Hd = (hx_px*hy_py - hx_py*hy_px)
                Hd_uV = hx_px_u(hy_py*v) - hx_py*hy_px_u(v) + hx_px*hy_py_u(v) - hx_py_u(hy_px*v)

                if 'zxx' in self.rxType:
                    Zij = ( ex_px*hy_py - ex_py*hy_px)/Hd
                    ZijN_uV = ex_px_u(hy_py*v) - ex_py*hy_px_u(v) + ex_px*hy_py_u(v) - ex_py_u(hy_px*v)
                elif 'zxy' in self.rxType:
                    Zij = (-ex_px*hx_py + ex_py*hx_px)/Hd
                    ZijN_uV = -ex_px_u(hx_py*v) + ex_py*hx_px_u(v) - ex_px*hx_py_u(v) + ex_py_u(hx_px*v)
                elif 'zyx' in self.rxType:
                    Zij = ( ey_px*hy_py - ey_py*hy_px)/Hd
                    ZijN_uV = ey_px_u(hy_py*v) - ey_py*hy_px_u(v) + ey_px*hy_py_u(v) - ey_py_u(hy_px*v)
                elif 'zyy' in self.rxType:
                    Zij = (-ey_px*hx_py + ey_py*hx_px)/Hd
                    ZijN_uV = -ey_px_u(hx_py*v) + ey_py*hx_px_u(v) - ey_px*hx_py_u(v) +ey_py_u(hx_px*v)

                # Calculate the complex derivative
                PDeriv_complex = ZijN_uV/Hd + Zij * (Hd_uV/Hd)

            # Extract the real number for the real/imag components.
            Pv = np.array(getattr(PDeriv_complex, real_or_imag))
        elif adjoint:
            # Note: The v vector is real and the return should be complex
            if self.projType is 'Z1D':
                Pex = mesh.getInterpolationMat(self.locs[:,-1],'Fx')
                Pbx = mesh.getInterpolationMat(self.locs[:,-1],'Ex')
                # ex = Pex*mkvc(f[src,'e_1d'],2)
                # bx = Pbx*mkvc(f[src,'b_1d'],2)/mu_0
                dP_deTv = -mkvc(Pex.T*Utils.sdiag(1./(Pbx*mkvc(f[src,'b_1d'],2)/mu_0)).T*v,2)
                db_duv = Pbx.T/mu_0*Utils.sdiag(1./(Pbx*mkvc(f[src,'b_1d'],2)/mu_0))*(Utils.sdiag(1./(Pbx*mkvc(f[src,'b_1d'],2)/mu_0))).T*Utils.sdiag(Pex*mkvc(f[src,'e_1d'],2)).T*v
                dP_dbTv = mkvc(f._bDeriv_u(src,db_duv,adjoint=True),2)
                PDeriv_real = np.sum(np.hstack((dP_deTv,dP_dbTv)),1)
            elif self.projType is 'Z2D':
                raise NotImplementedError('Has not be implement for 2D impedance tensor')
            elif self.projType is 'Z3D':
                if self.locs.ndim == 3:
                    eFLocs = self.locs[:,:,0]
                    bFLocs = self.locs[:,:,1]
                else:
                    eFLocs = self.locs
                    bFLocs = self.locs
                # Get the projection
                Pex = mesh.getInterpolationMat(eFLocs,'Ex')
                Pey = mesh.getInterpolationMat(eFLocs,'Ey')
                Pbx = mesh.getInterpolationMat(bFLocs,'Fx')
                Pby = mesh.getInterpolationMat(bFLocs,'Fy')
                # Get the fields at location
                # px: x-polaration and py: y-polaration.
                aex_px = f[src,'e_px'].T*Pex.T
                aey_px = f[src,'e_px'].T*Pey.T
                aex_py = f[src,'e_py'].T*Pex.T
                aey_py = f[src,'e_py'].T*Pey.T
                ahx_px = f[src,'b_px'].T/mu_0*Pbx.T
                ahy_px = f[src,'b_px'].T/mu_0*Pby.T
                ahx_py = f[src,'b_py'].T/mu_0*Pbx.T
                ahy_py = f[src,'b_py'].T/mu_0*Pby.T
                # Derivatives as lambda functions
                aex_px_u = lambda vec: f._e_pxDeriv_u(src,Pex*vec,adjoint=True)
                aey_px_u = lambda vec: f._e_pxDeriv_u(src,Pey*vec,adjoint=True)
                aex_py_u = lambda vec: f._e_pyDeriv_u(src,Pex*vec,adjoint=True)
                aey_py_u = lambda vec: f._e_pyDeriv_u(src,Pey*vec,adjoint=True)
                ahx_px_u = lambda vec: f._b_pxDeriv_u(src,Pbx*vec,adjoint=True)/mu_0
                ahy_px_u = lambda vec: f._b_pxDeriv_u(src,Pby*vec,adjoint=True)/mu_0
                ahx_py_u = lambda vec: f._b_pyDeriv_u(src,Pbx*vec,adjoint=True)/mu_0
                ahy_py_u = lambda vec: f._b_pyDeriv_u(src,Pby*vec,adjoint=True)/mu_0

                # Define the components of the derivative
                aHd = (ahy_py*ahx_px - ahy_px*ahx_py)
                aHd_uV = ahx_px_u(ahy_py*v) - ahx_py*ahy_px_u(v) + ahx_py_u(ahy_py*v) - ahx_py_u(ahy_px*v)
                # Need to fix this to reflect the adjoint
                if 'zxx' in self.rxType:
                    Zij = ( ahy_py*aex_px - ahy_px*aex_py)/aHd
                    ZijN_uV =  ahy_py*aex_px_u(v) - ahy_px_u(aex_py*v) + ahy_py_u(aex_px*v) - ahy_px*aex_py_u(v)
                elif 'zxy' in self.rxType:
                    Zij = (-ahx_py*aex_px + ahx_px*aex_py)/aHd
                    ZijN_uV = -ahx_py*aex_px_u(v) + ahx_px_u(aex_py*v) - ahx_py_u(aex_px*v) + ahx_px*aex_py_u(v)
                elif 'zyx' in self.rxType:
                    Zij = ( ahy_py*aey_px - ahy_px*aey_py)/aHd
                    ZijN_uV =  ahy_py*aey_px_u(v) - ahy_px_u(aey_py*v) + ahy_py_u(aey_px*v) - ahy_px*aey_py_u(v)
                elif 'zyy' in self.rxType:
                    Zij = (-ahx_py*aey_px + ahx_px*aey_py)/aHd
                    ZijN_uV = -ahx_py*aey_px_u(v) + ahx_px_u(aey_py*v) - ahx_py_u(aey_px*v) + ahx_px*aey_py_u(v)

                # Calculate the complex derivative
                PDeriv_real = ZijN_uV/aHd + (aHd_uV/aHd)*Zij^T
            # Extract the data
            if real_or_imag == 'imag':
                Pv = 1j*PDeriv_real
            elif real_or_imag == 'real':
                Pv = PDeriv_real.astype(complex)

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
    def __init__(self, rxList, freq):
        # assert mkvc(self.mesh.hz.shape,1) == mkvc(sigma1d.shape,1),'The number of values in the 1D background model does not match the number of vertical cells (hz).'
        self.sigma1d = None
        srcMT.__init__(self, rxList, freq)
        # Hidden property of the ePrimary
        self._ePrimary = None

    def ePrimary(self,problem):
        # Get primary fields for both polarizations
        if self.sigma1d is None:
            # Set the sigma1d as the 1st column in the background model
            if len(problem._sigmaPrimary) == problem.mesh.nC:
                self.sigma1d = problem.mesh.r(problem._sigmaPrimary,'CC','CC','M')[0,0,:]
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
            sys.stdout.flush()
            for rx in src.rxList:
                data[src, rx] = rx.projectFields(src, self.mesh, u)
        return data

    def projectFieldsDeriv(self, u):
        raise Exception('Use Transmitters to project fields deriv.')

