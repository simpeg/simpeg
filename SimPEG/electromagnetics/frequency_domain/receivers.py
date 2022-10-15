# import properties
from ... import survey
from ...utils import validate_string, validate_type, validate_direction
import warnings
from discretize.utils import Zero


class BaseRx(survey.BaseRx):
    """Base FDEM receivers class.

    Parameters
    ----------
    locations : (n_loc, n_dim) numpy.ndarray
        Receiver locations.
    orientation : {'z', 'x', 'y'} or numpy.ndarray
        Receiver orientation.
    component : {'real', 'imag', 'both', 'complex'}
        Component of the receiver; i.e. 'real' or 'imag'. The options 'both' and
        'complex' are only available for the 1D layered simulations.
    data_type : {'field', 'ppm'}
        Data type observed by the receiver, either field, or ppm secondary
        of the total field.
    use_source_receiver_offset : bool, optional
        Whether to interpret the receiver locations as defining the source and receiver
        offset.
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
        numpy.ndarray
        """
        return self._orientation

    @orientation.setter
    def orientation(self, var):
        self._orientation = validate_direction("orientation", var, dim=3)

    @property
    def component(self):
        """Data component; i.e. real or imaginary.

        Returns
        -------
        str : {'real', 'imag', 'both', 'complex'}
            Component of the receiver; i.e. 'real' or 'imag'. The options 'both' and
            'complex' are only available for the 1D layered simulations.
        """
        return self._component

    @component.setter
    def component(self, val):
        self._component = validate_string(
            "component",
            val,
            (
                ("real", "re", "in-phase", "in phase"),
                (
                    "imag",
                    "imaginary",
                    "im",
                    "out-of-phase",
                    "out of phase",
                    "quadrature",
                ),
                "both",
                "complex",
            ),
        )

    @property
    def data_type(self):
        """The type of data for this receiver.

        The data type is either a field measurement or a part per million (ppm) measurement
        of the primary field.

        Returns
        -------
        str : {'field', 'ppm'}

        Notes
        -----
        This is currently only implemented for the 1D layered simulations.
        """
        return self._data_type

    @data_type.setter
    def data_type(self, val):
        self._data_type = validate_string(
            "data_type", val, string_list=("field", "ppm")
        )

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

    def getP(self, mesh, projected_grid):
        """Get projection matrix from mesh to receivers

        Parameters
        ----------
        mesh : discretize.BaseMesh
            A discretize mesh
        projected_grid : str
            Define what part of the mesh (i.e. edges, faces, centers, nodes) to
            project from. Must be one of::

                'E', 'edges_'           -> field defined on edges
                'F', 'faces_'           -> field defined on faces
                'CCV', 'cell_centers_'  -> vector field defined on cell centers

        Returns
        -------
        scipy.sparse.csr_matrix
            P, the interpolation matrix
        """
        if (mesh, projected_grid) in self._Ps:
            return self._Ps[(mesh, projected_grid)]

        P = Zero()
        for strength, comp in zip(self.orientation, ["x", "y", "z"]):
            if strength != 0.0:
                P = P + strength * mesh.get_interpolation_matrix(
                    self.locations, projected_grid + comp
                )

        if self.storeProjections:
            self._Ps[(mesh, projected_grid)] = P
        return P

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
        numpy.ndarray
            Fields projected to the receiver(s)
        """
        projected_grid = f._GLoc(self.projField)
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
        du_dm_v : numpy.ndarray
            The derivative of the fields on the mesh with respect to the model,
            times a vector.
        v : numpy.ndarray, optional
            The vector which being multiplied
        adjoint : bool
            If ``True``, return the ajoint

        Returns
        -------
        numpy.ndarray
            The derivative times a vector at the receiver(s)
        """

        df_dmFun = getattr(f, "_{0}Deriv".format(self.projField), None)

        assert v is not None, "v must be provided to compute the deriv or adjoint"

        projected_grid = f._GLoc(self.projField)
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
    locations : (n_loc, n_dim) numpy.ndarray
        Receiver locations.
    orientation : {'x', 'y', 'z'}
        Receiver orientation.
    component : {'real', 'imag'}
        Real or imaginary component.
    """

    def __init__(self, locations, orientation="x", component="real", **kwargs):
        self.projField = "e"
        super().__init__(locations, orientation, component, **kwargs)


class PointMagneticFluxDensity(BaseRx):
    """Measure FDEM total field magnetic flux density at a point.

    Parameters
    ----------
    locations : (n_loc, n_dim) numpy.ndarray
        Receiver locations.
    orientation : {'x', 'y', 'z'}
        Receiver orientation.
    component : {'real', 'imag'}
        Real or imaginary component.
    """

    def __init__(self, locations, orientation="x", component="real", **kwargs):
        self.projField = "b"
        super().__init__(locations, orientation, component, **kwargs)


class PointMagneticFluxDensitySecondary(BaseRx):
    """Measure FDEM secondary magnetic flux density at a point.

    Parameters
    ----------
    locations : (n_loc, n_dim) numpy.ndarray
        Receiver locations.
    orientation : {'x', 'y', 'z'}
        Receiver orientation.
    component : {'real', 'imag'}
        Real or imaginary component.
    """

    def __init__(self, locations, orientation="x", component="real", **kwargs):
        self.projField = "bSecondary"
        super().__init__(locations, orientation, component, **kwargs)


class PointMagneticField(BaseRx):
    """Measure FDEM total magnetic field at a point.

    Parameters
    ----------
    locations : (n_loc, n_dim) numpy.ndarray
        Receiver locations.
    orientation : {'x', 'y', 'z'}
        Receiver orientation.
    component : {'real', 'imag', 'both', 'complex'}
        Component of the receiver; i.e. 'real' or 'imag'. The options 'both' and
        'complex' are only available for the 1D layered simulations.
    data_type : {'field', 'ppm'}
        Data type observed by the receiver, either field, or ppm secondary
        of the total field.
    use_source_receiver_offset : bool, optional
        Whether to interpret the receiver locations as defining the source and receiver
        offset.

    Notes
    -----
    `data_type`, `use_source_receiver_offset`, and the options of `'both'` and
    `'complex'` for component are only implemented for the `Simulation1DLayered`.
    """

    def __init__(self, locations, orientation="x", component="real", **kwargs):
        self.projField = "h"
        super().__init__(locations, orientation, component, **kwargs)


class PointMagneticFieldSecondary(BaseRx):
    """
    Magnetic flux FDEM receiver


    locations : (n_loc, n_dim) numpy.ndarray
        Receiver locations.
    orientation : {'x', 'y', 'z'}
        Receiver orientation
    component : {'real', 'imag', 'both', 'complex'}
        Component of the receiver; i.e. 'real' or 'imag'. The options 'both' and
        'complex' are only available for the 1D layered simulations.
    data_type : {'field', 'ppm'}
        Data type observed by the receiver, either field, or ppm secondary
        of the total field.
    use_source_receiver_offset : bool, optional
        Whether to interpret the receiver locations as defining the source and receiver
        offset.

    Notes
    -----
    `data_type`, `use_source_receiver_offset`, and the options of `'both'` and
    `'complex'` for component are only implemented for the `Simulation1DLayered`.
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
    locations : (n_loc, n_dim) numpy.ndarray
        Receiver locations.
    orientation : {'x', 'y', 'z'}
        Receiver orientation.
    component : {'real', 'imag'}
        Real or imaginary component.
    """

    def __init__(self, locations, orientation="x", component="real", **kwargs):
        self.projField = "j"
        super().__init__(locations, orientation, component, **kwargs)
