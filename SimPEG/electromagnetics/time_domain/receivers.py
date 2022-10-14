import scipy.sparse as sp

from ...utils import mkvc, validate_string, validate_type, validate_direction
from discretize.utils import Zero
from ...survey import BaseTimeRx
import warnings


class BaseRx(BaseTimeRx):
    """Base TDEM receiver class

    Parameters
    ----------
    locations : (n_loc, n_dim) numpy.ndarray
        Receiver locations.
    orientation : {'z', 'x', 'y'} or numpy.ndarray
        Receiver orientation.
    times : (n_times) numpy.ndarray
        Time channels
    """

    def __init__(
        self,
        locations,
        times,
        orientation="z",
        use_source_receiver_offset=False,
        **kwargs
    ):
        proj = kwargs.pop("projComp", None)
        if proj is not None:
            warnings.warn(
                "'projComp' overrides the 'orientation' property which automatically"
                " handles the projection from the mesh the receivers!!! "
                "'projComp' is deprecated and will be removed in SimPEG 0.16.0."
            )
            self.projComp = proj

        if locations is None:
            raise AttributeError("'locations' are required. Cannot be 'None'")

        if times is None:
            raise AttributeError("'times' are required. Cannot be 'None'")

        self.orientation = orientation
        self.use_source_receiver_offset = use_source_receiver_offset
        super().__init__(locations=locations, times=times, **kwargs)

    # orientation = properties.StringChoice(
    #     "orientation of the receiver. Must currently be 'x', 'y', 'z'", ["x", "y", "z"]
    # )

    @property
    def orientation(self):
        """Orientation of the receiver.

        Returns
        -------
        numpy.ndarray
            Orientation of the receiver.
        """
        return self._orientation

    @orientation.setter
    def orientation(self, var):
        self._orientation = validate_direction("orientation", var, dim=3)

    # def projected_grid(self, f):
    #     """Grid Location projection (e.g. Ex Fy ...)"""
    #     return f._GLoc(self.projField) + self.orientation

    @property
    def use_source_receiver_offset(self):
        """Use source-receiver offset.

        Whether to interpret the location as a source-receiver offset.

        Returns
        -------
        bool

        Notes
        -----
        This is currently only implemented for the 1D layered code.
        """
        return self._use_source_receiver_offset

    @use_source_receiver_offset.setter
    def use_source_receiver_offset(self, val):
        self._use_source_receiver_offset = validate_type(
            "use_source_receiver_offset", val, bool
        )

    # def projected_time_grid(self, f):
    #     """Time Location projection (e.g. CC N)"""
    #     return f._TLoc(self.projField)

    def getSpatialP(self, mesh, f):
        """Get spatial projection matrix from mesh to receivers.

        Only constructed when called.

        Parameters
        ----------
        mesh : discretize.BaseMesh
            A discretize mesh
        f : SimPEG.electromagnetics.time_domain.fields.FieldsTDEM

        Returns
        -------
        scipy.sparse.csr_matrix
            P, the interpolation matrix
        """
        P = Zero()
        field = f._GLoc(self.projField)
        for strength, comp in zip(self.orientation, ["x", "y", "z"]):
            if strength != 0.0:
                P = P + strength * mesh.get_interpolation_matrix(self.locations, field + comp)
        return P

    def getTimeP(self, time_mesh, f):
        """Get time projection matrix from mesh to receivers.

        Only constructed when called.

        Parameters
        ----------
        time_mesh : discretize.TensorMesh
            A 1D ``TensorMesh`` defining the time discretization
        f : SimPEG.electromagnetics.time_domain.fields.FieldsTDEM

        Returns
        -------
        scipy.sparse.csr_matrix
            P, the interpolation matrix
        """
        projected_time_grid = f._TLoc(self.projField)
        return time_mesh.getInterpolationMat(self.times, projected_time_grid)

    def getP(self, mesh, time_mesh, f):
        """Returns projection matrices as a list for all components collected by the receivers.

        Parameters
        ----------
        mesh : discretize.BaseMesh
            A discretize mesh defining spatial discretization
        time_mesh : discretize.TensorMesh
            A 1D ``TensorMesh`` defining the time discretization
        f : SimPEG.electromagnetics.time_domain.fields.FieldsTDEM

        Returns
        -------
        scipy.sparse.csr_matrix
            Returns full projection matrix from fields to receivers.

        Notes
        -----
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
        """Project fields to receivers to get data.

        Parameters
        ----------
        src : SimPEG.electromagnetics.frequency_domain.sources.BaseTDEMSrc
            A time-domain EM source
        mesh : discretize.base.BaseMesh
            The mesh on which the discrete set of equations is solved
        time_mesh : discretize.TensorMesh
            A 1D ``TensorMesh`` defining the time discretization
        f : SimPEG.electromagnetic.time_domain.fields.FieldsTDEM
            The solution for the fields defined on the mesh

        Returns
        -------
        numpy.ndarray
            Fields projected to the receiver(s)
        """
        P = self.getP(mesh, time_mesh, f)
        f_part = mkvc(f[src, self.projField, :])
        return P * f_part

    def evalDeriv(self, src, mesh, time_mesh, f, v, adjoint=False):
        """Derivative of projected fields with respect to the inversion model times a vector.

        Parameters
        ----------
        src : SimPEG.electromagnetics.frequency_domain.sources.BaseTDEMSrc
            A time-domain EM source
        mesh : discretize.base.BaseMesh
            The mesh on which the discrete set of equations is solved
        time_mesh : discretize.TensorMesh
            A 1D ``TensorMesh`` defining the time discretization
        f : SimPEG.electromagnetic.time_domain.fields.FieldsTDEM
            The solution for the fields defined on the mesh
        v : numpy.ndarray
            A vector
        adjoint : bool, default = ``False``
            If ``True``, return the adjoint

        Returns
        -------
        numpy.ndarray
            derivative of fields times a vector projected to the receiver(s)
        """
        P = self.getP(mesh, time_mesh, f)

        if not adjoint:
            return P * v
        elif adjoint:
            # dP_dF_T = P.T * v #[src, self]
            # newshape = (len(dP_dF_T)/time_mesh.nN, time_mesh.nN )
            return P.T * v  # np.reshape(dP_dF_T, newshape, order='F')


class PointElectricField(BaseRx):
    """Measure TDEM electric field at a point.

    Parameters
    ----------
    locations : (n_loc, n_dim) numpy.ndarray
        Receiver locations.
    times : (n_times) numpy.ndarray
        Time channels
    orientation : {'z', 'x', 'y'}
        Receiver orientation.
    """

    def __init__(self, locations=None, times=None, orientation="z", **kwargs):
        self.projField = "e"
        super(PointElectricField, self).__init__(
            locations, times, orientation, **kwargs
        )


class PointMagneticFluxDensity(BaseRx):
    """Measure TDEM magnetic flux density at a point.

    Parameters
    ----------
    locations : (n_loc, n_dim) numpy.ndarray
        Receiver locations.
    times : (n_times) numpy.ndarray
        Time channels
    orientation : {'z', 'x', 'y'}
        Receiver orientation.
    """

    def __init__(self, locations=None, times=None, orientation="z", **kwargs):
        self.projField = "b"
        super(PointMagneticFluxDensity, self).__init__(
            locations, times, orientation, **kwargs
        )


class PointMagneticFluxTimeDerivative(BaseRx):
    """Measure time-derivative of magnetic flux density at a point.

    Parameters
    ----------
    locations : (n_loc, n_dim) numpy.ndarray
        Receiver locations.
    times : (n_times) numpy.ndarray
        Time channels
    orientation : {'z', 'x', 'y'}
        Receiver orientation.
    """

    def __init__(self, locations=None, times=None, orientation="z", **kwargs):
        self.projField = "dbdt"
        super(PointMagneticFluxTimeDerivative, self).__init__(
            locations, times, orientation, **kwargs
        )

    def eval(self, src, mesh, time_mesh, f):
        """Project solution of fields to receivers to get data.

        Parameters
        ----------
        src : SimPEG.electromagnetics.frequency_domain.sources.BaseTDEMSrc
            A time-domain EM source
        mesh : discretize.base.BaseMesh
            The mesh on which the discrete set of equations is solved
        time_mesh : discretize.TensorMesh
            A 1D ``TensorMesh`` defining the time discretization
        f : SimPEG.electromagnetic.time_domain.fields.FieldsTDEM
            The solution for the fields defined on the mesh

        Returns
        -------
        numpy.ndarray
            Fields projected to the receiver(s)
        """

        if self.projField in f.aliasFields:
            return super(PointMagneticFluxTimeDerivative, self).eval(
                src, mesh, time_mesh, f
            )

        P = self.getP(mesh, time_mesh, f)
        f_part = mkvc(f[src, "b", :])
        return P * f_part

    # def projected_grid(self, f):
    #     """Grid Location projection (e.g. Ex Fy ...)"""
    #     if self.projField in f.aliasFields:
    #         return super(PointMagneticFluxTimeDerivative, self).projected_grid(f)
    #     return f._GLoc(self.projField) + self.orientation

    def getTimeP(self, time_mesh, f):
        """Get time projection matrix from mesh to receivers.

        Only constructed when called.

        Parameters
        ----------
        time_mesh : discretize.TensorMesh
            A 1D ``TensorMesh`` defining the time discretization
        f : SimPEG.electromagnetics.time_domain.fields.FieldsTDEM

        Returns
        -------
        scipy.sparse.csr_matrix
            P, the interpolation matrix
        """
        if self.projField in f.aliasFields:
            return super(PointMagneticFluxTimeDerivative, self).getTimeP(time_mesh, f)

        return time_mesh.getInterpolationMat(self.times, "CC") * time_mesh.faceDiv


class PointMagneticField(BaseRx):
    """Measure TDEM magnetic field at a point.

    Parameters
    ----------
    locations : (n_loc, n_dim) numpy.ndarray
        Receiver locations.
    times : (n_times) numpy.ndarray
        Time channels
    orientation : {'z', 'x', 'y'}
        Receiver orientation.
    """

    def __init__(self, locations=None, times=None, orientation="x", **kwargs):
        self.projField = "h"
        super(PointMagneticField, self).__init__(
            locations, times, orientation, **kwargs
        )


class PointCurrentDensity(BaseRx):
    """Measure TDEM current density at a point.

    Parameters
    ----------
    locations : (n_loc, n_dim) numpy.ndarray
        Receiver locations.
    times : (n_times) numpy.ndarray
        Time channels
    orientation : {'z', 'x', 'y'}
        Receiver orientation.
    """

    def __init__(self, locations=None, times=None, orientation="x", **kwargs):
        self.projField = "j"
        super(PointCurrentDensity, self).__init__(
            locations, times, orientation, **kwargs
        )


class PointMagneticFieldTimeDerivative(BaseRx):
    """Measure time-derivative of magnet field at a point.

    Parameters
    ----------
    locations : (n_loc, n_dim) numpy.ndarray
        Receiver locations.
    times : (n_times) numpy.ndarray
        Time channels
    orientation : {'z', 'x', 'y'}
        Receiver orientation.
    """

    def __init__(self, locations=None, times=None, orientation="x", **kwargs):
        self.projField = "dhdt"
        super(PointMagneticFieldTimeDerivative, self).__init__(
            locations, times, orientation, **kwargs
        )
