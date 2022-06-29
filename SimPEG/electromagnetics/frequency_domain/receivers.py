# import properties
from ... import survey
from ...utils import validate_string_property
import warnings


class BaseRx(survey.BaseRx):
    """Base FDEM receivers class.

    Parameters
    ----------
    locations : (n_loc, n_dim) np.ndarray
        Receiver locations. 
    orientation : str, default = 'z'
        Receiver orientation. Must be one of: 'x', 'y' or 'z'
    component : str, default = 'real'
        Real or imaginary component. Choose one of: 'real' or 'imag'
    """

    def __init__(self, locations, orientation='z', component='real', **kwargs):
        proj = kwargs.pop("projComp", None)
        if proj is not None:
            warnings.warn(
                "'projComp' overrides the 'orientation' property which automatically"
                " handles the projection from the mesh the receivers!!! "
                "'projComp' is deprecated and will be removed in SimPEG 0.16.0."
            )
            self.projComp = proj

        self.orientation = orientation
        self.component = component

        super(BaseRx, self).__init__(locations, **kwargs)

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
        var = validate_string_property('orientation', var, string_list=('x', 'y', 'z'))
        self._orientation = var.lower()

    # component = properties.StringChoice(
    #     "component of the field (real or imag)",
    #     {
    #         "real": ["re", "in-phase", "in phase"],
    #         "imag": ["imaginary", "im", "out-of-phase", "out of phase"],
    #     },
    # )

    @property
    def component(self):
        """Data component; i.e. real or imaginary.

        Returns
        -------
        str
            Orientation of the receiver; i.e. 'real' or 'imag'
        """
        return self._component

    @component.setter
    def component(self, var):

        if isinstance(var, str):
            if var.lower() in ('real', 're', 'in-phase', 'in phase'):
                self._component = 'real'
            elif var.lower() in ('imag', 'imaginary', 'im', 'out-of-phase', 'out of phase', 'quadrature'):
                self._component = 'imag'
            else:
                raise ValueError(f"orientation must be either 'real' or 'imag'. Got {var}")
        else:
            raise TypeError(f"orientation must be a str. Got {type(var)}")

    # def projected_grid(self, f):
    #     """Grid Location projection (e.g. Ex Fy ...)"""
    #     return f._GLoc(self.projField) + self.orientation


    def eval(self, src, mesh, f):
        """Project fields from the mesh to the receiver(s).

        Parameters
        ----------
        src : SimPEG.electromagnetics.frequency_domain.sources.BaseFDEMSrc
            A frequency-domain EM source
        mesh : discretize.base.BaseMesh
            The mesh on which the discrete set of equations is solved
        f : SimPEG.electromagnetic.frequency_domain.fields.FieldsFDEM
            The solution for the fields defined on the mesh
        
        Returns
        -------
        np.ndarray
            Fields projected to the receiver(s)
        """
        projected_grid = f._GLoc(self.projField) + self.orientation
        P = self.getP(mesh, projected_grid)
        f_part_complex = f[src, self.projField]
        f_part = getattr(f_part_complex, self.component)  # real or imag component

        return P * f_part

    def evalDeriv(self, src, mesh, f, du_dm_v=None, v=None, adjoint=False):
        """Derivative of the projected fields with respect to the model, times a vector.

        Parameters
        ----------
        src : SimPEG.electromagnetics.frequency_domain.sources.BaseFDEMSrc
            A frequency-domain EM source
        mesh : discretize.base.BaseMesh
            The mesh on which the discrete set of equations is solved
        f : SimPEG.electromagnetic.frequency_domain.fields.FieldsFDEM
            The solution for the fields defined on the mesh
        du_dm_v : np.ndarray, default = ``None``
            The derivative of the fields on the mesh with respect to the model,
            times a vector.
        v : np.ndarray
            The vector which being multiplied
        adjoint : bool, default = ``False``
            If ``True``, return the ajoint
        
        Returns
        -------
        np.ndarray
            The derivative times a vector at the receiver(s)
        """

        df_dmFun = getattr(f, "_{0}Deriv".format(self.projField), None)

        assert v is not None, "v must be provided to compute the deriv or adjoint"
        
        projected_grid = f._GLoc(self.projField) + self.orientation
        P = self.getP(mesh, projected_grid)

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
                PTv = -1j * PTv_real
            elif self.component == "real":
                PTv = PTv_real.astype(complex)
            else:
                raise NotImplementedError("must be real or imag")

            df_duT, df_dmT = df_dmFun(src, None, PTv, adjoint=True)
            # if self.component == "imag":  # conjugate
            #     df_duT *= -1
            #     df_dmT *= -1

            return df_duT, df_dmT


class PointElectricField(BaseRx):
    """Measure FDEM electric field at a point.

    Parameters
    ----------
    locations : (n_loc, n_dim) np.ndarray
        Receiver locations. 
    orientation : str, default = 'z'
        Receiver orientation. Must be one of: 'x', 'y' or 'z'
    component : str, default = 'real'
        Real or imaginary component. Choose one of: 'real' or 'imag'
    """

    def __init__(self, locations, orientation="x", component="real"):
        self.projField = "e"
        super(PointElectricField, self).__init__(locations, orientation, component)


class PointMagneticFluxDensity(BaseRx):
    """Measure FDEM total field magnetic flux density at a point.

    Parameters
    ----------
    locations : (n_loc, n_dim) np.ndarray
        Receiver locations. 
    orientation : str, default = 'z'
        Receiver orientation. Must be one of: 'x', 'y' or 'z'
    component : str, default = 'real'
        Real or imaginary component. Choose one of: 'real' or 'imag'
    """

    def __init__(self, locations, orientation="x", component="real"):
        self.projField = "b"
        super(PointMagneticFluxDensity, self).__init__(
            locations, orientation, component
        )


class PointMagneticFluxDensitySecondary(BaseRx):
    """Measure FDEM secondary magnetic flux density at a point.

    Parameters
    ----------
    locations : (n_loc, n_dim) np.ndarray
        Receiver locations. 
    orientation : str, default = 'z'
        Receiver orientation. Must be one of: 'x', 'y' or 'z'
    component : str, default = 'real'
        Real or imaginary component. Choose one of: 'real' or 'imag'
    """

    def __init__(self, locations, orientation="x", component="real"):
        self.projField = "bSecondary"
        super(PointMagneticFluxDensitySecondary, self).__init__(
            locations, orientation, component
        )


class PointMagneticField(BaseRx):
    """Measure FDEM total magnetic field at a point.

    Parameters
    ----------
    locations : (n_loc, n_dim) np.ndarray
        Receiver locations. 
    orientation : str, default = 'z'
        Receiver orientation. Must be one of: 'x', 'y' or 'z'
    component : str, default = 'real'
        Real or imaginary component. Choose one of: 'real' or 'imag'
    """

    def __init__(self, locations, orientation="x", component="real"):
        self.projField = "h"
        super(PointMagneticField, self).__init__(locations, orientation, component)


class PointCurrentDensity(BaseRx):
    """Measure FDEM current density at a point.

    Parameters
    ----------
    locations : (n_loc, n_dim) np.ndarray
        Receiver locations. 
    orientation : str, default = 'z'
        Receiver orientation. Must be one of: 'x', 'y' or 'z'
    component : str, default = 'real'
        Real or imaginary component. Choose one of: 'real' or 'imag'
    """

    def __init__(self, locations, orientation="x", component="real"):
        self.projField = "j"
        super(PointCurrentDensity, self).__init__(locations, orientation, component)
