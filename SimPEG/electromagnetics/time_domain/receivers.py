import scipy.sparse as sp
from ...utils.code_utils import deprecate_class, deprecate_property
import properties

from ...utils import mkvc
from ...survey import BaseTimeRx


class BaseRx(BaseTimeRx):
    """Base TDEM receiver class

    Parameters
    ----------
    locations : (n_loc, n_dim) np.ndarray
        Receiver locations. 
    orientation : str, default = 'z'
        Receiver orientation. Must be one of: 'x', 'y' or 'z'
    times : (n_times) np.ndarray
        Time channels
    """
    def __init__(self, locations, times, orientation='z', **kwargs):
        proj = kwargs.pop("projComp", None)
        if proj is not None:
            warnings.warn(
                "'projComp' overrides the 'orientation' property which automatically"
                " handles the projection from the mesh the receivers!!! "
                "'projComp' is deprecated and will be removed in SimPEG 0.16.0."
            )
            self.projComp = proj

        self.orientation = orientation
        super().__init__(locations=locations, times=times, **kwargs)

    # orientation = properties.StringChoice(
    #     "orientation of the receiver. Must currently be 'x', 'y', 'z'", ["x", "y", "z"]
    # )

    @property
    def orientation(self):
        """Orientation of the receiver.

        Returns
        -------
        str
            Orientation of the receiver. One of {'x', 'y', 'z'}
        """
        return self._orientation

    @orientation.setter
    def orientation(self, var):

        if isinstance(var, str):
            var = var.lower()
            if var not in ('x', 'y', 'z'):
                raise ValueError(f"orientation must be either 'x', 'y' or 'z'. Got {var}")
        else:
            raise TypeError(f"orientation must be a str. Got {type(var)}")

        self._orientation = var

    projComp = deprecate_property(
        orientation,
        "projComp",
        new_name="orientation",
        removal_version="0.16.0",
        error=True,
    )

    # def projGLoc(self, f):
    #     """Grid Location projection (e.g. Ex Fy ...)"""
    #     return f._GLoc(self.projField) + self.orientation

    # def projTLoc(self, f):
    #     """Time Location projection (e.g. CC N)"""
    #     return f._TLoc(self.projField)

    def getSpatialP(self, mesh, f):
        """
        Returns the spatial projection matrix.

        .. note::

            This is not stored in memory, but is created on demand.
        """
        if getattr(self, 'projGLoc', None) is None:
            self.projGLoc = f._GLoc(self.projField) + self.orientation
        return mesh.getInterpolationMat(self.locations, self.projGLoc)

    def getTimeP(self, time_mesh, f):
        """
        Returns the time projection matrix.

        .. note::

            This is not stored in memory, but is created on demand.
        """
        if getattr(self, 'projTLoc', None) is None:
            self.projTLoc = f._TLoc(self.projField)
        return time_mesh.getInterpolationMat(self.times, self.projTLoc)

    def getP(self, mesh, time_mesh, f):
        """Returns the projection matrices as a
        list for all components collected by
        the receivers.

        .. note::
            Projection matrices are stored as a dictionary (mesh, time_mesh) if storeProjections is True
        """
        if (mesh, time_mesh) in self._Ps:
            return self._Ps[(mesh, time_mesh)]

        Ps = self.getSpatialP(mesh, f)
        Pt = self.getTimeP(time_mesh, f)
        P = sp.kron(Pt, Ps)

        if self.storeProjections:
            self._Ps[(mesh, time_mesh)] = P

        return P

    def eval(self, src, mesh, time_mesh, f):
        """
        Project fields to receivers to get data.

        :param SimPEG.electromagnetics.time_domain.sources.BaseWaveform src: TDEM source
        :param discretize.base.BaseMesh mesh: mesh used
        :param Fields f: fields object
        :rtype: numpy.ndarray
        :return: fields projected to recievers
        """

        P = self.getP(mesh, time_mesh, f)
        f_part = mkvc(f[src, self.projField, :])
        return P * f_part

    def evalDeriv(self, src, mesh, time_mesh, f, v, adjoint=False):
        """
        Derivative of projected fields with respect to the inversion model times a vector.

        :param SimPEG.electromagnetics.time_domain.sources.BaseWaveform src: TDEM source
        :param discretize.base.BaseMesh mesh: mesh used
        :param discretize.base.BaseMesh time_mesh: time mesh
        :param Fields f: fields object
        :param numpy.ndarray v: vector to multiply
        :rtype: numpy.ndarray
        :return: fields projected to recievers
        """

        P = self.getP(mesh, time_mesh, f)
        if not adjoint:
            return P * v
        elif adjoint:
            # dP_dF_T = P.T * v #[src, self]
            # newshape = (len(dP_dF_T)/time_mesh.nN, time_mesh.nN )
            return P.T * v  # np.reshape(dP_dF_T, newshape, order='F')


class PointElectricField(BaseRx):
    """
    Electric field TDEM receiver

    :param numpy.ndarray locations: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param numpy.ndarray times: times
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    """

    def __init__(self, locations=None, times=None, orientation="x", **kwargs):
        self.projField = "e"
        super(PointElectricField, self).__init__(
            locations, times, orientation, **kwargs
        )


class PointMagneticFluxDensity(BaseRx):
    """
    Magnetic flux TDEM receiver

    :param numpy.ndarray locations: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param numpy.ndarray times: times
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    """

    def __init__(self, locations=None, times=None, orientation="x", **kwargs):
        self.projField = "b"
        super(PointMagneticFluxDensity, self).__init__(
            locations, times, orientation, **kwargs
        )


class PointMagneticFluxTimeDerivative(BaseRx):
    """
    dbdt TDEM receiver

    :param numpy.ndarray locations: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param numpy.ndarray times: times
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    """

    def __init__(self, locations=None, times=None, orientation="x", **kwargs):
        self.projField = "dbdt"
        super(PointMagneticFluxTimeDerivative, self).__init__(
            locations, times, orientation, **kwargs
        )

    def eval(self, src, mesh, time_mesh, f):

        if self.projField in f.aliasFields:
            return super(PointMagneticFluxTimeDerivative, self).eval(
                src, mesh, time_mesh, f
            )

        P = self.getP(mesh, time_mesh, f)
        f_part = mkvc(f[src, "b", :])
        return P * f_part

    def projGLoc(self, f):
        """Grid Location projection (e.g. Ex Fy ...)"""
        if self.projField in f.aliasFields:
            return super(PointMagneticFluxTimeDerivative, self).projGLoc(f)
        return f._GLoc(self.projField) + self.orientation

    def getTimeP(self, time_mesh, f):
        """
        Returns the time projection matrix.

        .. note::

            This is not stored in memory, but is created on demand.
        """
        if self.projField in f.aliasFields:
            return super(PointMagneticFluxTimeDerivative, self).getTimeP(time_mesh, f)

        return time_mesh.getInterpolationMat(self.times, "CC") * time_mesh.faceDiv


class PointMagneticField(BaseRx):
    """
    Magnetic field TDEM receiver

    :param numpy.ndarray locations: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param numpy.ndarray times: times
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    """

    def __init__(self, locations=None, times=None, orientation="x", **kwargs):
        self.projField = "h"
        super(PointMagneticField, self).__init__(
            locations, times, orientation, **kwargs
        )


class PointCurrentDensity(BaseRx):
    """
    Current density TDEM receiver

    :param numpy.ndarray locations: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param numpy.ndarray times: times
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    """

    def __init__(self, locations=None, times=None, orientation="x", **kwargs):
        self.projField = "j"
        super(PointCurrentDensity, self).__init__(
            locations, times, orientation, **kwargs
        )


class PointMagneticFieldTimeDerivative(BaseRx):
    def __init__(self, locations=None, times=None, orientation="x", **kwargs):
        self.projField = "dhdt"
        super(PointMagneticFieldTimeDerivative, self).__init__(
            locations, times, orientation, **kwargs
        )


############
# Deprecated
############


@deprecate_class(removal_version="0.16.0", error=True)
class Point_e(PointElectricField):
    pass


@deprecate_class(removal_version="0.16.0", error=True)
class Point_b(PointMagneticFluxDensity):
    pass


@deprecate_class(removal_version="0.16.0", error=True)
class Point_h(PointMagneticField):
    pass


@deprecate_class(removal_version="0.16.0", error=True)
class Point_j(PointCurrentDensity):
    pass


@deprecate_class(removal_version="0.16.0", error=True)
class Point_dbdt(PointMagneticFluxTimeDerivative):
    pass


@deprecate_class(removal_version="0.16.0", error=True)
class Point_dhdt(PointMagneticFieldTimeDerivative):
    pass
