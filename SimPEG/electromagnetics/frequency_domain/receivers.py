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

    def __init__(
        self,
        locations,
        orientation="z",
        component="real",
        data_type="field",
        use_source_receiver_offset=False,
        **kwargs,
    ):

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
        self.data_type = data_type
        self.use_source_receiver_offset = use_source_receiver_offset

        super().__init__(locations, **kwargs)

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
        var = validate_string_property("orientation", var, string_list=("x", "y", "z"))
        self._orientation = var.lower()

    @property
    def component(self):
        """Data component; i.e. real or imaginary.

        Returns
        -------
        str : ['real', 'imag', 'both', 'complex']
            Orientation of the receiver; i.e. 'real' or 'imag'. The options 'both' and
            'complex' are only available for the 1D layered simulations.
        """
        return self._component

    @component.setter
    def component(self, val):

        if isinstance(val, str):
            val = val.lower()
            if val in ("real", "re", "in-phase", "in phase"):
                val = "real"
            elif val in (
                "imag",
                "imaginary",
                "im",
                "out-of-phase",
                "out of phase",
                "quadrature",
            ):
                val = "imag"

            if val.lower() in ["real", "imag", "both", "complex"]:
                self._component = val
            else:
                raise ValueError(
                    f"orientation must be either 'real', 'imag', 'both', or 'complex'. Got {val}"
                )
        else:
            raise TypeError(f"orientation must be a str. Got {type(val)}")

    @property
    def data_type(self):
        """The type of data for this receiver.

        The data type is either a field measurement or a part per million (ppm) measurement
        of the primary field.

        Returns
        -------
        str : ['field', 'ppm']

        Notes
        -----
        This is currently only implemented for the 1D layered simulations.
        """
        return self._data_type

    @data_type.setter
    def data_type(self, val):
        if isinstance(val, str):
            val = val.lower()
            if val in ["field", "ppm"]:
                self._data_type = val
            else:
                raise ValueError(
                    f"data_type must be either 'field' or 'ppm'. Got {val}"
                )
        else:
            raise TypeError(f"data_type must be a str. Got {type(val)}")

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
        if not isinstance(val, bool):
            raise TypeError("use_source_receiver_offset must be a bool")
        self._use_source_receiver_offset = val

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

    @property
    def nD(self):
        if self.component == "both":
            return int(self.locations.shape[0] * 2)
        else:
            return self.locations.shape[0]


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

    def __init__(self, locations, orientation="x", component="real", **kwargs):
        self.projField = "e"
        super().__init__(locations, orientation, component, **kwargs)


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

    def __init__(self, locations, orientation="x", component="real", **kwargs):
        self.projField = "b"
        super().__init__(locations, orientation, component, **kwargs)


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

    def __init__(self, locations, orientation="x", component="real", **kwargs):
        self.projField = "bSecondary"
        super().__init__(locations, orientation, component, **kwargs)


class PointMagneticField(BaseRx):
    """Measure FDEM total magnetic field at a point.

    Parameters
    ----------
    locations : (n_loc, n_dim) np.ndarray
        Receiver locations.
    orientation : {'x', 'y', 'z'}
        Receiver orientation.
    component : {'real', 'imag'}
        Real or imaginary component.
    """

    def __init__(self, locations, orientation="x", component="real", **kwargs):
        self.projField = "h"
        super().__init__(locations, orientation, component, **kwargs)


class PointMagneticFieldSecondary(BaseRx):
    """
    Magnetic flux FDEM receiver


    locations : (n_loc, n_dim) np.ndarray
        Receiver locations.
    orientation : {'x', 'y', 'z'}
        Receiver orientation
    component : {'real', 'imag', 'complex', 'both'}
        Real or imaginary component.

    """

    def __init__(self, locations, orientation="x", component="real", **kwargs):
        self.projField = "hSecondary"
        super().__init__(
            locations, orientation=orientation, component=component, **kwargs
        )


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

    def __init__(self, locations, orientation="x", component="real", **kwargs):
        self.projField = "j"
        super().__init__(locations, orientation, component, **kwargs)
