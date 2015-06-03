from SimPEG import Survey, Utils, Problem, np, sp, mkvc
from scipy.constants import mu_0
import sys
from numpy.lib import recfunctions as recFunc
from DataMT import DataMT
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

    def projectFields(self, src, mesh, u):
        '''
        Project the fields and return the 
        '''

        if self.projType is 'Z1D':
            Pex = mesh.getInterpolationMat(self.locs,'Fx')
            Pbx = mesh.getInterpolationMat(self.locs,'Ex')   
            ex = Pex*mkvc(u[src,'e_1d'],2)
            bx = Pbx*mkvc(u[src,'b_1d'],2)/mu_0
            f_part_complex = ex/bx
        elif self.projType is 'Z3D':
            # Get the projection
            Pex = mesh.getInterpolationMat(self.locs,'Ex')
            Pey = mesh.getInterpolationMat(self.locs,'Ey')
            Pbx = mesh.getInterpolationMat(self.locs,'Fx')
            Pby = mesh.getInterpolationMat(self.locs,'Fy')
            # Get the fields at location
            # px: x-polaration and py: y-polaration.
            ex_px = Pex*u[src,'e_px']
            ey_px = Pey*u[src,'e_px']
            ex_py = Pex*u[src,'e_py']
            ey_py = Pey*u[src,'e_py']
            hx_px = Pbx*u[src,'b_px']/mu_0
            hy_px = Pby*u[src,'b_px']/mu_0
            hx_py = Pbx*u[src,'b_py']/mu_0
            hy_py = Pby*u[src,'b_py']/mu_0
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
        return f_part

    def projectFieldsDeriv(self, src, mesh, u, v, adjoint=False):
        P = self.getP(mesh)

        if not adjoint:
            Pv_complex = P * v
            real_or_imag = self.projComp
            Pv = getattr(Pv_complex, real_or_imag)
        elif adjoint:
            Pv_real = P.T * v

            real_or_imag = self.projComp
            if real_or_imag == 'imag':
                Pv = 1j*Pv_real
            elif real_or_imag == 'real':
                Pv = Pv_real.astype(complex)
            else:
                raise NotImplementedError('must be real or imag')

        return Pv


# Note: Might need to add tests to make sure that both polarization have the same rxList. 

###############
### Sources ###
###############
class srcMT(Survey.BaseSrc):
    '''
    Sources for the MT problem. 
    Use the SimPEG BaseSrc, since the source fields share properties with the transmitters.

    :param float freq: The frequency of the source
    :param list rxList: A list of receivers associated with the source
    :param str srcPol: The polarization of the source
    '''

    freq = None #: Frequency (float)

    rxPair = RxMT

    knownSrcTypes = ['pol_xy','pol_x','pol_y'] # ORThogonal POLarization

    def __init__(self, freq, rxList, srcPol = 'pol_xy'): # remove rxType? hardcode to one thing. always polarizations
        self.freq = float(freq)
        Survey.BaseSrc.__init__(self, None, srcPol, rxList)


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
    def getSources(self, freq):
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

