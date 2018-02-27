import SimPEG



class BaseRx(SimPEG.Survey.BaseRx):
    """
    Frequency domain receiver base class

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    :param string component: real or imaginary component 'real' or 'imag'
    """

    def __init__(self, locs, orientation=None, component=None):
        assert(
            orientation in ['x', 'y', 'z']
        ), "Orientation {0!s} not known. Orientation must be in 'x', 'y', 'z'."
        " Arbitrary orientations have not yet been implemented.".format(
            orientation
        )
        assert(
            component in ['real', 'imag']
            ), "'component' must be 'real' or 'imag', not {0!s}".format(
                component
            )

        self.projComp = orientation
        self.component = component

        # TODO: remove rxType from baseRx
        SimPEG.Survey.BaseRx.__init__(self, locs, rxType=None)

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


class Point_e(BaseRx):
    """
    Electric field FDEM receiver

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    :param string component: real or imaginary component 'real' or 'imag'
    """

    def __init__(self, locs, orientation=None, component=None):
        self.projField = 'e'
        super(Point_e, self).__init__(locs, orientation, component)


class Point_b(BaseRx):
    """
    Magnetic flux FDEM receiver

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    :param string component: real or imaginary component 'real' or 'imag'
    """

    def __init__(self, locs, orientation=None, component=None):
        self.projField = 'b'
        super(Point_b, self).__init__(locs, orientation, component)


class Point_bSecondary(BaseRx):
    """
    Magnetic flux FDEM receiver

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    :param string component: real or imaginary component 'real' or 'imag'
    """

    def __init__(self, locs, orientation=None, component=None):
        self.projField = 'bSecondary'
        super(Point_bSecondary, self).__init__(locs, orientation, component)


class Point_h(BaseRx):
    """
    Magnetic field FDEM receiver

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    :param string component: real or imaginary component 'real' or 'imag'
    """

    def __init__(self, locs, orientation=None, component=None):
        self.projField = 'h'
        super(Point_h, self).__init__(locs, orientation, component)


class Point_j(BaseRx):
    """
    Current density FDEM receiver

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    :param string component: real or imaginary component 'real' or 'imag'
    """

    def __init__(self, locs, orientation=None, component=None):
        self.projField = 'j'
        super(Point_j, self).__init__(locs, orientation, component)
