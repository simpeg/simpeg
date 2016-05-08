import SimPEG
from SimPEG.EM.Utils import *
from SimPEG.EM.Base import BaseEMSurvey
from scipy.constants import mu_0
from SimPEG.Utils import Zero, Identity
import SrcFDEM as Src
import RxFDEM as Rx
from SimPEG import sp

class BaseRx(SimPEG.Survey.BaseRx):
    """
    Frequency domain receivers

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    """

    def __init__(self, locs, orientation=None, real_or_imag=None):
        assert(orientation in ['x','y','z']), "Orientation %s not known. Orientation must be in 'x', 'y', 'z'. Arbitrary orientations have not yet been implemented."%orientation
        assert(real_or_imag in ['real', 'imag']), "'real_or_imag' must be 'real' or 'imag', not %s"%real_or_imag

        self.projComp = orientation
        self.real_or_imag = real_or_imag

        SimPEG.Survey.BaseRx.__init__(self, locs, rxType=None) #TODO: remove rxType from baseRx

    def projGLoc(self, u):
        """Grid Location projection (e.g. Ex Fy ...)"""
        return u._GLoc(self.projField) + self.projComp

    def eval(self, src, mesh, f):
        """
        Project fields to recievers to get data.

        :param Source src: FDEM source
        :param Mesh mesh: mesh used
        :param Fields f: fields object
        :rtype: numpy.ndarray
        :return: fields projected to recievers
        """

        P = self.getP(mesh, self.projGLoc(f))
        f_part_complex = f[src, self.projField]
        # get the real or imag component
        f_part = getattr(f_part_complex, self.real_or_imag)

        return P*f_part

    def evalDeriv(self, src, mesh, f, v, adjoint=False):
        """
        Derivative of projected fields with respect to the inversion model times a vector.

        :param Source src: FDEM source
        :param Mesh mesh: mesh used
        :param Fields f: fields object
        :param numpy.ndarray v: vector to multiply
        :rtype: numpy.ndarray
        :return: fields projected to recievers
        """

        P = self.getP(mesh, self.projGLoc(f))

        if not adjoint:
            Pv_complex = P * v
            # real_or_imag = self.projComp
            Pv = getattr(Pv_complex, self.real_or_imag)
        elif adjoint:
            Pv_real = P.T * v

            # real_or_imag = self.projComp
            if self.real_or_imag == 'imag':
                Pv = 1j*Pv_real
            elif self.real_or_imag == 'real':
                Pv = Pv_real.astype(complex)
            else:
                raise NotImplementedError('must be real or imag')

        return Pv


class eField(BaseRx):

    def __init__(self, locs, orientation=None, real_or_imag=None):
        self.projField = 'e'
        BaseRx.__init__(self, locs, orientation, real_or_imag)


class bField(BaseRx):

    def __init__(self, locs, orientation=None, real_or_imag=None):
        self.projField = 'b'
        BaseRx.__init__(self, locs, orientation, real_or_imag)


class hField(BaseRx):

    def __init__(self, locs, orientation=None, real_or_imag=None):
        self.projField = 'h'
        BaseRx.__init__(self, locs, orientation, real_or_imag)


class jField(BaseRx):

    def __init__(self, locs, orientation=None, real_or_imag=None):
        self.projField = 'j'
        BaseRx.__init__(self, locs, orientation, real_or_imag)
