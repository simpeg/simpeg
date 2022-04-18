import scipy.sparse as sp
# import properties
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

        if locations is None:
            raise AttributeError("'locations' are required. Cannot be 'None'")

        if times is None:
            raise AttributeError("'times' are required. Cannot be 'None'")

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

    # def projGLoc(self, f):
    #     """Grid Location projection (e.g. Ex Fy ...)"""
    #     return f._GLoc(self.projField) + self.orientation

    # def projTLoc(self, f):
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
        if getattr(self, 'projGLoc', None) is None:
            self.projGLoc = f._GLoc(self.projField) + self.orientation
        return mesh.getInterpolationMat(self.locations, self.projGLoc)

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
        if getattr(self, 'projTLoc', None) is None:
            self.projTLoc = f._TLoc(self.projField)
        return time_mesh.getInterpolationMat(self.times, self.projTLoc)

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
        np.ndarray
            Fields projected to the receiver(s)
        """

        if getattr(self, 'projGLoc', None) is None:
            self.projGLoc = f._GLoc(self.projField) + self.orientation
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
        v : np.ndarray
            A vector
        adjoint : bool, default = ``False``
            If ``True``, return the adjoint
        
        Returns
        -------
        np.ndarray
            derivative of fields times a vector projected to the receiver(s)
        """

        if getattr(self, 'projGLoc', None) is None:
            self.projGLoc = f._GLoc(self.projField) + self.orientation
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
    locations : (n_loc, n_dim) np.ndarray
        Receiver locations.
    times : (n_times) np.ndarray
        Time channels
    orientation : str, default = 'z'
        Receiver orientation. Must be one of: 'x', 'y' or 'z'
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
    locations : (n_loc, n_dim) np.ndarray
        Receiver locations.
    times : (n_times) np.ndarray
        Time channels
    orientation : str, default = 'z'
        Receiver orientation. Must be one of: 'x', 'y' or 'z'
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
    locations : (n_loc, n_dim) np.ndarray
        Receiver locations.
    times : (n_times) np.ndarray
        Time channels
    orientation : str, default = 'z'
        Receiver orientation. Must be one of: 'x', 'y' or 'z'
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
        np.ndarray
            Fields projected to the receiver(s)
        """

        if self.projField in f.aliasFields:
            return super(PointMagneticFluxTimeDerivative, self).eval(
                src, mesh, time_mesh, f
            )

        P = self.getP(mesh, time_mesh, f)
        f_part = mkvc(f[src, "b", :])
        return P * f_part

    # def projGLoc(self, f):
    #     """Grid Location projection (e.g. Ex Fy ...)"""
    #     if self.projField in f.aliasFields:
    #         return super(PointMagneticFluxTimeDerivative, self).projGLoc(f)
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
    locations : (n_loc, n_dim) np.ndarray
        Receiver locations.
    times : (n_times) np.ndarray
        Time channels
    orientation : str, default = 'z'
        Receiver orientation. Must be one of: 'x', 'y' or 'z'
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
    locations : (n_loc, n_dim) np.ndarray
        Receiver locations.
    times : (n_times) np.ndarray
        Time channels
    orientation : str, default = 'z'
        Receiver orientation. Must be one of: 'x', 'y' or 'z'
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
    locations : (n_loc, n_dim) np.ndarray
        Receiver locations.
    times : (n_times) np.ndarray
        Time channels
    orientation : str, default = 'z'
        Receiver orientation. Must be one of: 'x', 'y' or 'z'
    """
    def __init__(self, locations=None, times=None, orientation="x", **kwargs):
        self.projField = "dhdt"
        super(PointMagneticFieldTimeDerivative, self).__init__(
            locations, times, orientation, **kwargs
        )
