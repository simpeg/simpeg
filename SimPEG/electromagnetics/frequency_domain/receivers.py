import properties
import numpy as np
from ...utils.code_utils import deprecate_class, deprecate_property

from ... import survey


class BaseRx(survey.BaseRx):
    """
    Frequency domain receiver base class

    :param numpy.ndarray locations: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    :param string component: real or imaginary component 'real' or 'imag'
    """

    orientation = properties.StringChoice(
        "orientation of the receiver (x, y, z, rotated)",
        {
            "x": [],
            "y": [],
            "z": [],
            "rotated": ["rot", "arbitrary"],
        },
    )

    component = properties.StringChoice(
        "component of the field (real, imag, complex, amplitude, phase)",
        {
            "real": ["re", "in-phase", "in phase"],
            "imag": ["imaginary", "im", "out-of-phase", "out of phase"],
            "complex": ["comp", "both"],
            "amplitude": ["amp"],
            "phase": ["pha"],
        },
    )

    projComp = deprecate_property(
        orientation, "projComp", new_name="orientation", removal_version="0.15.0"
    )

    def __init__(self, locations, orientation=None, component=None, **kwargs):
        proj = kwargs.pop("projComp", None)
        if proj is not None:
            self.projComp = proj
        else:
            self.orientation = orientation

        self.component = component

        super(BaseRx, self).__init__(locations, **kwargs)

    def projGLoc(self, f):
        """Grid Location projection (e.g. Ex Fy ...)"""
        return f._GLoc(self.projField) + self.orientation
    
    def evalDataComplex(self, data_complex):
        if self.component == 'amplitude':
            return abs(data_complex)
        elif self.component == 'phase':
            return np.angle(data_complex)
        else:
            return data_complex

    def eval(self, src, mesh, f):
        """
        Project fields to receivers to get data.

        :param SimPEG.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: FDEM source
        :param discretize.base.BaseMesh mesh: mesh used
        :param Fields f: fields object
        :rtype: numpy.ndarray
        :return: fields projected to recievers
        """

        P = self.getP(mesh, self.projGLoc(f))
        f_part_complex = f[src, self.projField]
        f_part = getattr(f_part_complex, self.component)  # real or imag component

        return P * f_part

    def evalDeriv(self, src, mesh, f, du_dm_v=None, v=None, adjoint=False):
        """
        Derivative of projected fields with respect to the inversion model times a vector.

        :param SimPEG.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: FDEM source
        :param discretize.base.BaseMesh mesh: mesh used
        :param Fields f: fields object
        :param numpy.ndarray v: vector to multiply
        :rtype: numpy.ndarray
        :return: fields projected to recievers
        """

        df_dmFun = getattr(f, "_{0}Deriv".format(self.projField), None)

        assert v is not None, "v must be provided to compute the deriv or adjoint"

        P = self.getP(mesh, self.projGLoc(f))

        if not adjoint:
            assert (
                du_dm_v is not None
            ), "du_dm_v must be provided to evaluate the receiver deriv"
            df_dm_v = df_dmFun(src, du_dm_v, v, adjoint=False)
            Pv_complex = P * df_dm_v
            Pv = getattr(Pv_complex, self.component)

            return Pv

        elif adjoint:
            PTv_real = P.T * v

            if self.component == "imag":
                PTv = 1j * PTv_real
            elif self.component == "real":
                PTv = PTv_real.astype(complex)
            else:
                raise NotImplementedError("must be real or imag")

            df_duT, df_dmT = df_dmFun(src, None, PTv, adjoint=True)

            return df_duT, df_dmT


class PointElectricField(BaseRx):
    """
    Electric field FDEM receiver

    :param numpy.ndarray locations: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string orientation: receiver orientation 'x', 'y', 'z', or 'rotated'
    :param string component: 'real', 'imag', 'complex', 'amplitude', or 'phase'
    :param float azimuth: azimuth, only used if `orientation='rotated'`
    :param float elevation: elevation, only used if `orientation='rotated'`
    """

    # TODO : the current implementation of azimuth/elevation is not good. It
    #        only allows for one azimuth/elevation for all locations. Ideally
    #        the angles should have the same size as locations (but 1D).

    azimuth = properties.Float("azimuth (anticlockwise from Easting)", default=0, min=-360.0, max=360)

    elevation = properties.Float("elevation (positive up)", default=0, min=-180.0, max=180)

    def __init__(self, locations, orientation="x", component="real", **kwargs):
        angles = kwargs.get("azimuth", None) or kwargs.get("elevation", None)
        if orientation in ["x", "y", "z"] and angles:
            raise ValueError(
                "orientation must be 'rotated' if angles are provided."
            )
        self.projField = "e"
        super(PointElectricField, self).__init__(locations, orientation, component, **kwargs)


class PointMagneticFluxDensity(BaseRx):
    """
    Magnetic flux FDEM receiver

    :param numpy.ndarray locations: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    :param string component: real or imaginary component 'real' or 'imag'
    """

    def __init__(self, locations, orientation="x", component="real"):
        self.projField = "b"
        super(PointMagneticFluxDensity, self).__init__(
            locations, orientation, component
        )


class PointMagneticFluxDensitySecondary(BaseRx):
    """
    Magnetic flux FDEM receiver

    :param numpy.ndarray locations: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    :param string component: real or imaginary component 'real' or 'imag'
    """

    def __init__(self, locations, orientation="x", component="real"):
        self.projField = "bSecondary"
        super(PointMagneticFluxDensitySecondary, self).__init__(
            locations, orientation, component
        )


class PointMagneticField(BaseRx):
    """
    Magnetic field FDEM receiver

    :param numpy.ndarray locations: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string orientation: receiver orientation 'x', 'y', 'z', or 'rotated'
    :param string component: 'real', 'imag', 'complex', 'amplitude', or 'phase'
    :param float azimuth: azimuth, only used if `orientation='rotated'`
    :param float elevation: elevation, only used if `orientation='rotated'`
    """

    # TODO : the current implementation of azimuth/elevation is not good. It
    #        only allows for one azimuth/elevation for all locations. Ideally
    #        the angles should have the same size as locations (but 1D).

    azimuth = properties.Float("azimuth (anticlockwise from Easting)", default=0, min=-360.0, max=360)

    elevation = properties.Float("elevation (positive up)", default=0, min=-180.0, max=180)

    def __init__(self, locations, orientation="x", component="real", **kwargs):
        angles = kwargs.get("azimuth", None) or kwargs.get("elevation", None)
        if orientation in ["x", "y", "z"] and angles:
            raise ValueError(
                "orientation must be 'rotated' if angles are provided."
            )
        self.projField = "h"
        super(PointMagneticField, self).__init__(locations, orientation, component, **kwargs)


class PointCurrentDensity(BaseRx):
    """
    Current density FDEM receiver

    :param numpy.ndarray locations: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    :param string component: real or imaginary component 'real' or 'imag'
    """

    def __init__(self, locations, orientation="x", component="real"):
        self.projField = "j"
        super(PointCurrentDensity, self).__init__(locations, orientation, component)


############
# Deprecated
############
@deprecate_class(removal_version="0.15.0")
class Point_e(PointElectricField):
    pass


@deprecate_class(removal_version="0.15.0")
class Point_b(PointMagneticFluxDensity):
    pass


@deprecate_class(removal_version="0.15.0")
class Point_bSecondary(PointMagneticFluxDensitySecondary):
    pass


@deprecate_class(removal_version="0.15.0")
class Point_h(PointMagneticField):
    pass


@deprecate_class(removal_version="0.15.0")
class Point_j(PointCurrentDensity):
    pass
