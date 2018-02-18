import properties

from ...NewSurvey import BaseRx

__all__ = [
    'Point_e', 'Point_b', 'Point_h', 'Point_j', 'Point_bSecondary'
]


class BaseFDEMRx(BaseRx):
    """
    Base frequency domain electromagnetic receiver. Inherit this to build a
    frequency domain electromagnetic receiver.
    """

    # TODO: eventually, this should be a Vector3 and we can allow arbitraty
    # orientations
    orientation = properties.StringChoice(
        "orientation of the receiver 'x', 'y', or 'z'",
        choices=['x', 'y', 'z'],
        required=True
    )

    component = properties.StringChoice(
        "'real' or 'imag' component of the field to be measured",
        choices={
            'real': ['re', 'in-phase', 'inphase'],
            'imag': ['im', 'quadrature', 'quad', 'out-of-phase']
        },
        required=True
    )

    def __init__(self, **kwargs):
        super(BaseFDEMRx, self).__init__(**kwargs)

    @property
    def projComp(self):
        # TODO generalize for arbitrary orientations
        if getattr(self, '_projComp', None) is None:
            if self.orientation == "x":
                projComp = "x"
            elif self.orientation == "y":
                projComp = "y"
            elif self.orientation == "z":
                projComp = "z"
            else:
                raise NotImplementedError(
                    "Arbitrary receiver orientations have not yet been implemented"
                )
            self._projComp = projComp
        return self._projComp

    def projGLoc(self, f):
        """Grid Location projection (e.g. Ex Fy ...)"""

        return f._GLoc(self.projField) + self.projComp

    def eval(self, src, mesh, f):
        """
        Project fields to receivers to get data.

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: FDEM source
        :param BaseMesh mesh: mesh used
        :param Fields f: fields object
        :rtype: numpy.ndarray
        :return: fields projected to recievers
        """

        P = self.getP(mesh, self.projGLoc(f))
        f_part_complex = f[src, self.projField]
        f_part = getattr(f_part_complex, self.component) # real or imag component

        return P*f_part

    def evalDeriv(self, src, mesh, f, du_dm_v=None, v=None, adjoint=False):
        """
        Derivative of projected fields with respect to the inversion model times a vector.

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: FDEM source
        :param BaseMesh mesh: mesh used
        :param Fields f: fields object
        :param numpy.ndarray v: vector to multiply
        :rtype: numpy.ndarray
        :return: fields projected to recievers
        """

        df_dmFun = getattr(f, '_{0}Deriv'.format(self.projField), None)

        assert v is not None, (
            'v must be provided to compute the deriv or adjoint'
        )

        P = self.getP(mesh, self.projGLoc(f))

        if not adjoint:
            assert du_dm_v is not None, (
                'du_dm_v must be provided to evaluate the receiver deriv')
            df_dm_v = df_dmFun(src, du_dm_v, v, adjoint=False)
            Pv_complex = P * df_dm_v
            Pv = getattr(Pv_complex, self.component)

            return Pv

        elif adjoint:
            PTv_real = P.T * v

            if self.component == 'imag':
                PTv = 1j*PTv_real
            elif self.component == 'real':
                PTv = PTv_real.astype(complex)
            else:
                raise NotImplementedError('must be real or imag')

            df_duT, df_dmT = df_dmFun(src, None, PTv, adjoint=True)

            return df_duT, df_dmT


class Point_e(BaseFDEMRx):
    """
    Electric field FDEM receiver

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    :param string component: real or imaginary component 'real' or 'imag'
    """

    def __init__(self, **kwargs):
        super(Point_e, self).__init__(**kwargs)
        self.projField = 'e'


class Point_b(BaseFDEMRx):
    """
    Magnetic flux FDEM receiver

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    :param string component: real or imaginary component 'real' or 'imag'
    """

    def __init__(self, **kwargs):
        super(Point_b, self).__init__(**kwargs)
        self.projField = 'b'


class Point_bSecondary(BaseFDEMRx):
    """
    Magnetic flux FDEM receiver

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    :param string component: real or imaginary component 'real' or 'imag'
    """

    def __init__(self, **kwargs):
        super(Point_bSecondary, self).__init__(**kwargs)
        self.projField = 'bSecondary'


class Point_h(BaseFDEMRx):
    """
    Magnetic field FDEM receiver

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    :param string component: real or imaginary component 'real' or 'imag'
    """

    def __init__(self, **kwargs):
        super(Point_h, self).__init__(**kwargs)
        self.projField = 'h'


class Point_j(BaseFDEMRx):
    """
    Current density FDEM receiver

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    :param string component: real or imaginary component 'real' or 'imag'
    """

    def __init__(self, **kwargs):
        super(Point_j, self).__init__(**kwargs)
        self.projField = 'j'

