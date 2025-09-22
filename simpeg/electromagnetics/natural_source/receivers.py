from ...utils.code_utils import (
    validate_string,
    validate_type,
    validate_ndarray_with_shape,
    deprecate_class,
)
import numpy as np
from scipy.constants import mu_0
from ...survey import BaseRx


def _alpha(src):
    return 1 / (2 * np.pi * mu_0 * src.frequency)


class BaseNaturalSourceRx(BaseRx):
    """
    Base class for natural source electromagnetic receivers.

    Parameters
    ----------
    locations1, locations2 : (n_loc, n_dim) array_like
        Locations where the two fields are measured.
    **kwargs
        Additional keyword arguments passed to `simpeg.BaseRx`.
    """

    _loc_names = ("First", "Second")

    def __init__(self, locations1, locations2, **kwargs):
        super().__init__(locations=(locations1, locations2), **kwargs)

    @property
    def locations(self):
        """Locations of the two field measurements.

        Locations where the two fields are measured for the receiver.
        The name of the field is dependant upon the MT receiver, but
        for common MT receivers, these would be the electric field
        and magnetic field measurement locations.

        Returns
        -------
        locations1, locations2 : (n_loc, n_dim) numpy.ndarray
        """
        return self._locations

    @locations.setter
    def locations(self, locs):
        locs = validate_type("locations", locs, tuple)
        try:
            loc0, loc1 = locs
        except ValueError:
            raise ValueError(
                f"locations must have two values to unpack, got {len(locs)}"
            )
        # check that they are both numpy arrays and have the same shape.
        loc0 = validate_ndarray_with_shape(
            f"{self._loc_names[0]} locations", loc0, shape=("*", "*")
        )
        loc1 = validate_ndarray_with_shape(
            f"{self._loc_names[1]} locations", loc1, shape=loc0.shape
        )
        self._locations = (loc0, loc1)
        # make sure projection matrices are cleared
        self._Ps = {}

    @property
    def nD(self):
        """Number of data associated with the receiver object.

        Returns
        -------
        int
            Number of data associated with the receiver object.
        """

        return self._locations[0].shape[0]

    def getP(self, mesh, projected_grid, location_id=0):
        """Get projection matrix from mesh to specified receiver locations.

        Natural source electromagnetic data may be computed from field measurements
        at one or two locations. The `getP` method returns the projection matrix from
        the mesh to the appropriate receiver locations. `location_id=0` is used to
        project from the mesh to the set of roving receiver locations. `location_id=1`
        is used when horizontal fields used to compute NSEM data are measured at a
        base station.

        Parameters
        ----------
        mesh : discretize.BaseMesh
            A discretize mesh.
        projected_grid : str
            Define what part of the mesh (i.e. edges, faces, centers, nodes) to
            project from. Must be one of::

                'Ex', 'edges_x'           -> x-component of field defined on x edges
                'Ey', 'edges_y'           -> y-component of field defined on y edges
                'Ez', 'edges_z'           -> z-component of field defined on z edges
                'Fx', 'faces_x'           -> x-component of field defined on x faces
                'Fy', 'faces_y'           -> y-component of field defined on y faces
                'Fz', 'faces_z'           -> z-component of field defined on z faces
                'N', 'nodes'              -> scalar field defined on nodes
                'CC', 'cell_centers'      -> scalar field defined on cell centers
                'CCVx', 'cell_centers_x'  -> x-component of vector field defined on cell centers
                'CCVy', 'cell_centers_y'  -> y-component of vector field defined on cell centers
                'CCVz', 'cell_centers_z'  -> z-component of vector field defined on cell centers

        locations_id : int
            Receiver locations ID. 0 used for roving locations. 1 used for base station locations.

        Returns
        -------
        scipy.sparse.csr_matrix
            P, the interpolation matrix.
        """
        key = (mesh, projected_grid, location_id)
        if key in self._Ps:
            return self._Ps[key]
        locs = self._locations[location_id]
        P = mesh.get_interpolation_matrix(locs, projected_grid)
        if self.storeProjections:
            self._Ps[key] = P
        return P


class _ElectricAndMagneticReceiver(BaseNaturalSourceRx):
    """
    Intermediate class for MT receivers that measure an electric and magnetic field
    """

    _loc_names = ("Electric field", "Magnetic field")

    @property
    def locations_e(self):
        """Electric field measurement locations

        Returns
        -------
        numpy.ndarray
            Location where the electric field is measured for all receiver data
        """
        return self._locations[0]

    @property
    def locations_h(self):
        """Magnetic field measurement locations

        Returns
        -------
        numpy.ndarray
            Location where the magnetic field is measured for all receiver data
        """
        return self._locations[1]


class Impedance(_ElectricAndMagneticReceiver):
    r"""Receiver class for 1D, 2D and 3D impedance data.

    This class is used to simulate data types that can be derived from the impedance tensor:

    .. math::
        \begin{bmatrix} Z_{xx} & Z_{xy} \\ Z_{yx} & Z_{yy} \end{bmatrix} =
        \begin{bmatrix} E_x^{(x)} & E_x^{(y)} \\ E_y^{(x)} & E_y^{(y)} \end{bmatrix} \,
        \begin{bmatrix} H_x^{(x)} & H_x^{(y)} \\ H_y^{(x)} & H_y^{(y)} \end{bmatrix}^{-1}

    where superscripts :math:`(x)` and :math:`(y)` denote signals corresponding to
    incident planewaves whose electric fields are polarized along the x and y-directions
    respectively. Electric and magnetic fields do not need to be simulated at the same
    location, so this class can be used to simulate quasi-impedance data; i.e. where
    the electric fields are measured at a base station.

    Note that in ``simpeg``, natural source EM data are defined according to
    standard xyz coordinates; i.e. (x,y,z) is (Easting, Northing, Z +ve up).

    In addition to measuring the real or imaginary component of an impedance tensor
    element :math:`Z_{ij}`, the receiver object can be set to measure the
    the apparent resistivity:

    .. math::
        \rho_{ij} = \dfrac{| Z_{ij} \, |^2}{\mu_0 \omega}

    or the phase angle:

    .. math::
        \phi_{ij} = \frac{180}{\pi} \,
        \tan^{-1} \Bigg ( \dfrac{Im[Z_{ij}]}{Re[Z_{ij}]} \Bigg )

    where :math:`\mu_0` is the permeability of free-space and :math:`\omega` is the
    angular frequency in rad/s. The phase angle is represented in degrees and
    is computed by:

    Parameters
    ----------
    locations_e : (n_loc, n_dim) array_like
        Locations where the electric fields are measured.
    locations_h : (n_loc, n_dim) array_like, optional
        Locations where the magnetic fields are measured. Defaults to the same
        locations as electric field measurements, `locations_e`.
    orientation : {'xx', 'xy', 'yx', 'yy'}
        Receiver orientation. Specifies whether the receiver's data correspond to
        the :math:`Z_{xx}`, :math:`Z_{xy}`, :math:`Z_{yx}` or :math:`Z_{yy}` impedance.
        The data type is specified by the `component` input argument.
    component : {'real', 'imag', 'apparent_resistivity', 'phase', 'complex'}
        Data type. For the impedance element :math:`Z_{ij}` specified by the `orientation`
        input argument, the receiver can be set to compute the following:
        - 'real': Real component of the impedance (V/A)
        - 'imag': Imaginary component of the impedance (V/A)
        - 'rho': Apparent resistivity (:math:`\Omega m`)
        - 'phase': Phase angle (degrees)
        - 'complex': The complex impedance is returned. Do not use for inversion!
    storeProjections : bool
        Whether to cache to internal projection matrices.
    """

    def __init__(
        self,
        locations_e,
        locations_h=None,
        orientation="xx",
        component="real",
        storeProjections=False,
    ):
        if locations_h is None:
            locations_h = locations_e
        super().__init__(
            locations1=locations_e,
            locations2=locations_h,
            storeProjections=storeProjections,
        )
        self.orientation = orientation
        self.component = component

    @property
    def component(self):
        r"""Data type; i.e. "real", "imag", "apparent_resistivity", "phase"

        For the impedance element :math:`Z_{ij}`, the `component` property specifies
        whether the data are:
        - 'real': Real component of the impedance (V/A)
        - 'imag': Imaginary component of the impedance (V/A)
        - 'rho': Apparent resistivity (:math:`\Omega m`)
        - 'phase': Phase angle (degrees)
        - 'complex': Complex impedance (V/A)

        Returns
        -------
        str
            Data type; i.e. "real", "imag", "apparent_resistivity", "phase"
        """
        return self._component

    @component.setter
    def component(self, var):
        self._component = validate_string(
            "component",
            var,
            [
                ("real", "re", "in-phase", "in phase"),
                ("imag", "imaginary", "im", "out-of-phase", "out of phase"),
                (
                    "apparent_resistivity",
                    "apparent resistivity",
                    "appresistivity",
                    "apparentresistivity",
                    "apparent-resistivity",
                    "apparent_resistivity",
                    "appres",
                    "app_res",
                    "rho",
                    "rhoa",
                ),
                ("phase", "phi"),
                "complex",
            ],
        )

    @property
    def orientation(self):
        """Receiver orientation.

        Specifies whether the receiver's data correspond to
        the :math:`Z_{xx}`, :math:`Z_{xy}`, :math:`Z_{yx}` or :math:`Z_{yy}` impedance.
        The data type is specified by the `component` input argument.

        Returns
        -------
        str
            Receiver orientation. One of {'xx', 'xy', 'yx', 'yy'}
        """
        return self._orientation

    @orientation.setter
    def orientation(self, var):
        self._orientation = validate_string(
            "orientation", var, string_list=("xx", "xy", "yx", "yy")
        )

    def _eval_impedance(self, src, mesh, f):
        if mesh.dim < 3 and self.orientation in ["xx", "yy"]:
            return np.zeros((self.nD, 1), dtype=complex)
        e = f[src, "e"]
        h = f[src, "h"]
        if mesh.dim == 3:
            if self.orientation[0] == "x":
                e = self.getP(mesh, "Ex", 0) @ e
            else:
                e = self.getP(mesh, "Ey", 0) @ e

            hx = self.getP(mesh, "Fx", 1) @ h
            hy = self.getP(mesh, "Fy", 1) @ h
            if self.orientation[1] == "x":
                h = hy
            else:
                h = -hx

            top = e[:, 0] * h[:, 1] - e[:, 1] * h[:, 0]
            bot = hx[:, 0] * hy[:, 1] - hx[:, 1] * hy[:, 0]
        else:
            if mesh.dim == 1:
                e_loc = f.aliasFields["e"][1]
                h_loc = f.aliasFields["h"][1]
                PE = self.getP(mesh, e_loc)
                PH = self.getP(mesh, h_loc)
            elif mesh.dim == 2:
                if self.orientation == "xy":
                    PE = self.getP(mesh, "Ex")
                    PH = self.getP(mesh, "CC")
                elif self.orientation == "yx":
                    PE = self.getP(mesh, "CC")
                    PH = self.getP(mesh, "Ex")
            top = PE @ e[:, 0]
            bot = PH @ h[:, 0]

            # need to negate if 'yx' and fields are xy
            # and as well if 'xy' and fields are 'yx'
            if mesh.dim == 1 and self.orientation != f.field_directions:
                bot *= -1
        return top / bot

    def _eval_impedance_deriv(self, src, mesh, f, du_dm_v=None, v=None, adjoint=False):
        if mesh.dim < 3 and self.orientation in ["xx", "yy"]:
            if adjoint:
                return 0 * v
            else:
                return 0 * du_dm_v
        e = f[src, "e"]
        h = f[src, "h"]
        if mesh.dim == 3:
            if self.orientation[0] == "x":
                Pe = self.getP(mesh, "Ex", 0)
                e = Pe @ e
            else:
                Pe = self.getP(mesh, "Ey", 0)
                e = Pe @ e

            Phx = self.getP(mesh, "Fx", 1)
            Phy = self.getP(mesh, "Fy", 1)
            hx = Phx @ h
            hy = Phy @ h
            if self.orientation[1] == "x":
                h = hy
            else:
                h = -hx

            top = e[:, 0] * h[:, 1] - e[:, 1] * h[:, 0]
            bot = hx[:, 0] * hy[:, 1] - hx[:, 1] * hy[:, 0]
            imp = top / bot
        else:
            if mesh.dim == 1:
                e_loc = f.aliasFields["e"][1]
                h_loc = f.aliasFields["h"][1]
                PE = self.getP(mesh, e_loc)
                PH = self.getP(mesh, h_loc)
            elif mesh.dim == 2:
                if self.orientation == "xy":
                    PE = self.getP(mesh, "Ex")
                    PH = self.getP(mesh, "CC")
                elif self.orientation == "yx":
                    PE = self.getP(mesh, "CC")
                    PH = self.getP(mesh, "Ex")

            top = PE @ e[:, 0]
            bot = PH @ h[:, 0]

            if mesh.dim == 1 and self.orientation != f.field_directions:
                bot *= -1

            imp = top / bot

        if adjoint:
            if self.component == "phase":
                # gradient of arctan2(y, x) is (-y/(x**2 + y**2), x/(x**2 + y**2))
                v = 180 / np.pi * imp / (imp.real**2 + imp.imag**2) * v
                # switch real and imaginary, and negate real part of output
                v = -v.imag - 1j * v.real
                # imaginary part gets extra (-) due to conjugate transpose
            elif self.component == "apparent_resistivity":
                v = 2 * _alpha(src) * imp * v
                v = v.real - 1j * v.imag
            elif self.component == "imag":
                v = -1j * v

            # Work backwards!
            gtop_v = v / bot
            gbot_v = -imp * v / bot
            n_d = self.nD

            if mesh.dim == 3:
                ghx_v = np.c_[hy[:, 1], -hy[:, 0]] * gbot_v[..., None]
                ghy_v = np.c_[-hx[:, 1], hx[:, 0]] * gbot_v[..., None]
                ge_v = np.c_[h[:, 1], -h[:, 0]] * gtop_v[..., None]
                gh_v = np.c_[-e[:, 1], e[:, 0]] * gtop_v[..., None]

                if self.orientation[1] == "x":
                    ghy_v += gh_v
                else:
                    ghx_v -= gh_v

                if v.ndim == 2:
                    # collapse into a long list of n_d vectors
                    ghx_v = ghx_v.reshape((n_d, -1))
                    ghy_v = ghy_v.reshape((n_d, -1))
                    ge_v = ge_v.reshape((n_d, -1))

                gh_v = Phx.T @ ghx_v + Phy.T @ ghy_v
                ge_v = Pe.T @ ge_v
            else:
                if mesh.dim == 1 and self.orientation != f.field_directions:
                    gbot_v = -gbot_v

                gh_v = PH.T @ gbot_v
                ge_v = PE.T @ gtop_v

            gfu_h_v, gfm_h_v = f._hDeriv(src, None, gh_v, adjoint=True)
            gfu_e_v, gfm_e_v = f._eDeriv(src, None, ge_v, adjoint=True)

            return gfu_h_v + gfu_e_v, gfm_h_v + gfm_e_v

        if mesh.dim == 3:
            de_v = Pe @ f._eDeriv(src, du_dm_v, v, adjoint=False)
            dh_v = f._hDeriv(src, du_dm_v, v, adjoint=False)
            dhx_v = Phx @ dh_v
            dhy_v = Phy @ dh_v
            if self.orientation[1] == "x":
                dh_dm_v = dhy_v
            else:
                dh_dm_v = -dhx_v

            dtop_v = (
                e[:, 0] * dh_dm_v[:, 1]
                + de_v[:, 0] * h[:, 1]
                - e[:, 1] * dh_dm_v[:, 0]
                - de_v[:, 1] * h[:, 0]
            )
            dbot_v = (
                hx[:, 0] * dhy_v[:, 1]
                + dhx_v[:, 0] * hy[:, 1]
                - hx[:, 1] * dhy_v[:, 0]
                - dhx_v[:, 1] * hy[:, 0]
            )
            imp_deriv = (bot * dtop_v - top * dbot_v) / (bot * bot)
        else:
            de_v = PE @ f._eDeriv(src, du_dm_v, v, adjoint=False)
            dh_v = PH @ f._hDeriv(src, du_dm_v, v, adjoint=False)

            if mesh.dim == 1 and self.orientation != f.field_directions:
                dh_v *= -1

            imp_deriv = (de_v - imp * dh_v) / bot

        if self.component == "apparent_resistivity":
            rx_deriv = (
                2
                * _alpha(src)
                * (imp.real * imp_deriv.real + imp.imag * imp_deriv.imag)
            )
        elif self.component == "phase":
            amp2 = imp.imag**2 + imp.real**2
            deriv_re = -imp.imag / amp2 * imp_deriv.real
            deriv_im = imp.real / amp2 * imp_deriv.imag

            rx_deriv = (180 / np.pi) * (deriv_re + deriv_im)
        else:
            rx_deriv = getattr(imp_deriv, self.component)
        return rx_deriv

    def eval(self, src, mesh, f):  # noqa: A003
        """Compute receiver data from the discrete field solution.

        Parameters
        ----------
        src : .frequency_domain.sources.BaseFDEMSrc
            NSEM source.
        mesh : discretize.TensorMesh
            Mesh on which the discretize solution is obtained.
        f : simpeg.electromagnetics.frequency_domain.fields.FieldsFDEM
            NSEM fields object of the source.

        Returns
        -------
        numpy.ndarray
            Evaluated data for the receiver.
        """

        imp = self._eval_impedance(src, mesh, f)
        if self.component == "complex":
            return imp
        elif self.component == "apparent_resistivity":
            return _alpha(src) * (imp.real**2 + imp.imag**2)
        elif self.component == "phase":
            return 180 / np.pi * (np.arctan2(imp.imag, imp.real))
        else:
            return getattr(imp, self.component)

    def evalDeriv(self, src, mesh, f, du_dm_v=None, v=None, adjoint=False):
        r"""Derivative of data with respect to the fields.

        Let :math:`\mathbf{d}` represent the data corresponding the receiver object.
        And let :math:`\mathbf{u}` represent the discrete numerical solution of the
        fields on the mesh. Where :math:`\mathbf{P}` is a projection function that
        maps from the fields to the data, i.e.:

        .. math::
            \mathbf{d} = \mathbf{P}(\mathbf{u})

        this method computes and returns the derivative:

        .. math::
            \dfrac{\partial \mathbf{d}}{\partial \mathbf{u}} =
            \dfrac{\partial [ \mathbf{P} (\mathbf{u}) ]}{\partial \mathbf{u}}

        Parameters
        ----------
        str : .frequency_domain.sources.BaseFDEMSrc
            The NSEM source.
        mesh : discretize.TensorMesh
            Mesh on which the discretize solution is obtained.
        f : simpeg.electromagnetics.frequency_domain.fields.FieldsFDEM
            NSEM fields object for the source.
        du_dm_v : None, optional
            Supply pre-computed derivative?
        v : numpy.ndarray, optional
            Vector of size
        adjoint : bool, optional
            Whether to compute the ajoint operation.

        Returns
        -------
        numpy.ndarray
            Calculated derivative (n_data,) if `adjoint` is ``False``, and (n_param, 2) if `adjoint`
            is ``True``, for both polarizations.
        """
        if self.component == "complex":
            raise NotImplementedError(
                "complex valued data derivative is not implemented."
            )
        return self._eval_impedance_deriv(
            src, mesh, f, du_dm_v=du_dm_v, v=v, adjoint=adjoint
        )


class Tipper(BaseNaturalSourceRx):
    r"""Receiver class for tipper data (3D problems only).

    This class can be used to simulate AFMag tipper data, defined according to:

    .. math::
        \begin{bmatrix} T_{zx} & T_{zy} \end{bmatrix} =
        \begin{bmatrix} H_x^{(x)} & H_y^{(x)} \\ H_x^{(y)} & H_y^{(y)} \end{bmatrix}^{-1} \,
        \begin{bmatrix} H_z^{(x)} \\ H_z^{(y)} \end{bmatrix}

    where superscripts :math:`(x)` and :math:`(y)` denote signals corresponding to
    incident planewaves whose electric fields are polarized along the x and y-directions
    respectively. Note that in ``simpeg``, natural source EM data are defined according to
    standard xyz coordinates; i.e. (x,y,z) is (Easting, Northing, Z +ve up).

    The receiver class can also be used to simulate a diverse set of Tipper-like data types
    when horizontal magnetic fields are measured at a remote base station. These are defined
    according to:

    .. math::
        \begin{bmatrix} T_{xx} & T_{yx} & T_{zx} \\ T_{xy} & T_{yy} & T_{zy} \end{bmatrix} =
        \begin{bmatrix} H_x^{(x)} & H_y^{(x)} \\ H_x^{(y)} & H_y^{(y)} \end{bmatrix}_b^{-1} \,
        \begin{bmatrix} H_x^{(x)} & H_y^{(x)} & H_z^{(x)} \\ H_x^{(y)} & H_y^{(y)} & H_z^{(y)} \end{bmatrix}_r

    where subscript :math:`b` denotes the base station location and subscript
    :math:`r` denotes the mobile receiver location.

    Parameters
    ----------
    locations_h : (n_loc, n_dim) array_like
        Locations where the roving magnetic fields are measured.
    locations_base : (n_loc, n_dim) array_like, optional
        Locations where the base station magnetic fields are measured. Defaults to
        the same locations as the roving magnetic fields measurements,
        `locations_r`.
    orientation : {'xx', 'yx', 'zx', 'zy', 'yy', 'zy'}
        Specifies the tipper element :math:`T_{ij}` corresponding to the data.
    component : {'real', 'imag', 'complex'}
        Tipper data type. For the tipper element :math:`T_{ij}` specified by the `orientation`
        input argument, the receiver can be set to compute the following:
        - 'real': Real component of the tipper (unitless)
        - 'imag': Imaginary component of the tipper (unitless)
        - 'complex': The complex tipper is returned. Do not use for inversion!
    storeProjections : bool
        Whether to cache to internal projection matrices.
    """

    _loc_names = ("Roving magnetic field", "Base station magnetic field")

    def __init__(
        self,
        locations_h,
        locations_base=None,
        orientation="xx",
        component="real",
        storeProjections=False,
    ):
        if locations_base is None:
            locations_base = locations_h
        super().__init__(
            locations1=locations_h,
            locations2=locations_base,
            storeProjections=storeProjections,
        )
        self.orientation = orientation
        self.component = component

    @property
    def locations_h(self):
        """Roving magnetic field measurement locations.

        Returns
        -------
        numpy.ndarray
            Roving locations where the magnetic field is measured for all receiver data.
        """
        return self._locations[0]

    @property
    def locations_base(self):
        """Base station magnetic field measurement locations.

        Returns
        -------
        numpy.ndarray
            Base station locations where the horizontal magnetic fields are measured.
        """
        return self._locations[1]

    @property
    def component(self):
        r"""Tipper data type; i.e. "real", "imag"

        For the tipper element :math:`T_{ij}`, the `component` property specifies
        whether the data are:
        - 'real': Real component of the tipper (unitless)
        - 'imag': Imaginary component of the tipper (unitless)
        - 'complex': Complex tipper (unitless)

        Returns
        -------
        str
            Tipper data type; i.e. "real", "imag", "complex"
        """
        return self._component

    @component.setter
    def component(self, var):
        self._component = validate_string(
            "component",
            var,
            [
                ("real", "re", "in-phase", "in phase"),
                ("imag", "imaginary", "im", "out-of-phase", "out of phase"),
                "complex",
            ],
        )

    @property
    def orientation(self):
        """Specifies the tipper element :math:`T_{ij}` corresponding to the data.

        Returns
        -------
        str
            Specifies the tipper element :math:`T_{ij}` corresponding to the data.
            One of {'xx', 'yx', 'zx', 'zy', 'yy', 'zy'}.
        """
        return self._orientation

    @orientation.setter
    def orientation(self, var):
        self._orientation = validate_string(
            "orientation", var, string_list=("zx", "zy", "xx", "xy", "yx", "yy")
        )

    def _eval_tipper(self, src, mesh, f):
        # will grab both primary and secondary and sum them!
        h = f[src, "h"]

        Phx = self.getP(mesh, "Fx", 1)
        Phy = self.getP(mesh, "Fy", 1)
        Pho = self.getP(mesh, "F" + self.orientation[0], 0)

        hx = Phx @ h
        hy = Phy @ h
        ho = Pho @ h

        if self.orientation[1] == "x":
            h = -hy
        else:
            h = hx

        top = h[:, 0] * ho[:, 1] - h[:, 1] * ho[:, 0]
        bot = hx[:, 0] * hy[:, 1] - hx[:, 1] * hy[:, 0]
        return top / bot

    def _eval_tipper_deriv(self, src, mesh, f, du_dm_v=None, v=None, adjoint=False):
        # will grab both primary and secondary and sum them!
        h = f[src, "h"]

        Phx = self.getP(mesh, "Fx", 1)
        Phy = self.getP(mesh, "Fy", 1)
        Pho = self.getP(mesh, "F" + self.orientation[0], 0)

        hx = Phx @ h
        hy = Phy @ h
        ho = Pho @ h

        if self.orientation[1] == "x":
            h = -hy
        else:
            h = hx

        top = h[:, 0] * ho[:, 1] - h[:, 1] * ho[:, 0]
        bot = hx[:, 0] * hy[:, 1] - hx[:, 1] * hy[:, 0]
        tip = top / bot

        if adjoint:
            # Work backwards!
            gtop_v = (v / bot)[..., None]
            gbot_v = (-tip * v / bot)[..., None]
            n_d = self.nD

            ghx_v = np.c_[hy[:, 1], -hy[:, 0]] * gbot_v
            ghy_v = np.c_[-hx[:, 1], hx[:, 0]] * gbot_v
            gho_v = np.c_[-h[:, 1], h[:, 0]] * gtop_v
            gh_v = np.c_[ho[:, 1], -ho[:, 0]] * gtop_v

            if self.orientation[1] == "x":
                ghy_v -= gh_v
            else:
                ghx_v += gh_v

            if v.ndim == 2:
                # collapse into a long list of n_d vectors
                ghx_v = ghx_v.reshape((n_d, -1))
                ghy_v = ghy_v.reshape((n_d, -1))
                gho_v = gho_v.reshape((n_d, -1))

            gh_v = Phx.T @ ghx_v + Phy.T @ ghy_v + Pho.T @ gho_v
            return f._hDeriv(src, None, gh_v, adjoint=True)

        dh_v = f._hDeriv(src, du_dm_v, v, adjoint=False)
        dhx_v = Phx @ dh_v
        dhy_v = Phy @ dh_v
        dho_v = Pho @ dh_v
        if self.orientation[1] == "x":
            dh_v = -dhy_v
        else:
            dh_v = dhx_v

        dtop_v = (
            h[:, 0] * dho_v[:, 1]
            + dh_v[:, 0] * ho[:, 1]
            - h[:, 1] * dho_v[:, 0]
            - dh_v[:, 1] * ho[:, 0]
        )
        dbot_v = (
            hx[:, 0] * dhy_v[:, 1]
            + dhx_v[:, 0] * hy[:, 1]
            - hx[:, 1] * dhy_v[:, 0]
            - dhx_v[:, 1] * hy[:, 0]
        )

        return (bot * dtop_v - top * dbot_v) / (bot * bot)

    def eval(self, src, mesh, f):  # noqa: A003
        tip = self._eval_tipper(src, mesh, f)
        if self.component == "complex":
            return tip
        else:
            return getattr(tip, self.component)

    def evalDeriv(self, src, mesh, f, du_dm_v=None, v=None, adjoint=False):
        # Docstring inherited from parent class (Impedance).
        if self.component == "complex":
            raise NotImplementedError(
                "complex valued data derivative is not implemented."
            )
        if adjoint:
            if self.component == "imag":
                v = -1j * v
        imp_deriv = self._eval_tipper_deriv(
            src, mesh, f, du_dm_v=du_dm_v, v=v, adjoint=adjoint
        )
        if adjoint:
            return imp_deriv
        return getattr(imp_deriv, self.component)


class Admittance(_ElectricAndMagneticReceiver):
    r"""Receiver class for data types derived from the 3D admittance tensor.

    This class is used to simulate data types that can be derived from the admittance tensor:

    .. math::
        \begin{bmatrix} Y_{xx} & Y_{xy} \\ Y_{yx} & Y_{yy} \\ Y_{zx} & Y_{zy} \end{bmatrix} =
        \begin{bmatrix} H_x^{(x)} & H_x^{(y)} \\ H_y^{(x)} & H_y^{(y)} \\ H_z^{(x)} & H_z^{(y)} \end{bmatrix}_{\, r} \;
        \begin{bmatrix} E_x^{(x)} & E_x^{(y)} \\ E_y^{(x)} & E_y^{(y)} \end{bmatrix}_b^{-1}

    where superscripts :math:`(x)` and :math:`(y)` denote signals corresponding to
    incident planewaves whose electric fields are polarized along the x and y-directions
    respectively. Note that in simpeg, natural source EM data are defined according to
    standard xyz coordinates; i.e. (x,y,z) is (Easting, Northing, Z +ve up).

    Parameters
    ----------
    locations_e : (n_loc, n_dim) array_like
        Locations where the electric fields are measured.
    locations_h : (n_loc, n_dim) array_like, optional
        Locations where the magnetic fields are measured. Defaults to the same
        locations as electric field measurements, `locations_e`.
    orientation : {'xx', 'xy', 'yx', 'yy', 'zx', 'zy'}
        Admittance receiver orientation. Specifies the admittance tensor element
        :math:`Y_{ij}` corresponding to the data. The data type is specified by
        the `component` input argument.
    component : {'real', 'imag', 'complex'}
        Admittance data type. For the admittance element :math:`Y_{ij}` specified by the
        `orientation` input argument, the receiver can be set to compute the following:
        - 'real': Real component of the admittance (A/V)
        - 'imag': Imaginary component of the admittance (A/V)
        - 'complex': The complex admittance is returned. Do not use for inversion!
    storeProjections : bool
        Whether to cache to internal projection matrices.
    """

    def __init__(
        self,
        locations_e,
        locations_h=None,
        orientation="xx",
        component="real",
        storeProjections=False,
    ):
        if locations_h is None:
            locations_h = locations_e
        super().__init__(
            locations1=locations_e,
            locations2=locations_h,
            storeProjections=storeProjections,
        )
        self.orientation = orientation
        self.component = component

    @property
    def orientation(self):
        """Receiver orientation.

        Specifies whether the receiver's data correspond to
        the :math:`Y_{xx}`, :math:`Y_{xy}`, :math:`Y_{yx}`, :math:`Y_{yy}`,
        :math:`Y_{zx}`, or :math:`Y_{zy}` admittance.

        Returns
        -------
        str
            Receiver orientation. One of {'xx', 'xy', 'yx', 'yy', 'zx', 'zy'}
        """
        return self._orientation

    @orientation.setter
    def orientation(self, var):
        self._orientation = validate_string(
            "orientation", var, string_list=("xx", "xy", "yx", "yy", "zx", "zy")
        )

    @property
    def component(self):
        r"""Admittance data type.

        For the admittance element :math:`Y_{ij}`, the `component` property specifies
        whether the data are:
        - 'real': Real component of the admittance (A/V)
        - 'imag': Imaginary component of the admittance (A/V)
        - 'complex': Complex admittance (A/V)

        Returns
        -------
        str
            Data type; i.e. "real", "imag".
        """
        return self._component

    @component.setter
    def component(self, var):
        self._component = validate_string(
            "component",
            var,
            [
                ("real", "re", "in-phase", "in phase"),
                ("imag", "imaginary", "im", "out-of-phase", "out of phase"),
                "complex",
            ],
        )

    def _eval_admittance(self, src, mesh, f):
        if mesh.dim < 3:
            raise NotImplementedError(
                "Admittance receiver not implemented for dim < 3."
            )

        e = f[src, "e"]
        h = f[src, "h"]

        ex = self.getP(mesh, "Ex", 0) @ e
        ey = self.getP(mesh, "Ey", 0) @ e

        h = self.getP(mesh, "F" + self.orientation[0], 1) @ h

        if self.orientation[1] == "x":
            top = h[:, 0] * ey[:, 1] - h[:, 1] * ex[:, 1]
        else:
            top = -h[:, 0] * ey[:, 0] + h[:, 1] * ex[:, 0]

        bot = ex[:, 0] * ey[:, 1] - ex[:, 1] * ey[:, 0]

        return top / bot

    def _eval_admittance_deriv(self, src, mesh, f, du_dm_v=None, v=None, adjoint=False):
        if mesh.dim < 3:
            raise NotImplementedError(
                "Admittance receiver not implemented for dim < 3."
            )

        # Compute admittances
        e = f[src, "e"]
        h = f[src, "h"]

        Pex = self.getP(mesh, "Ex", 0)
        Pey = self.getP(mesh, "Ey", 0)
        Ph = self.getP(mesh, "F" + self.orientation[0], 1)

        ex = Pex @ e
        ey = Pey @ e
        h = Ph @ h

        if self.orientation[1] == "x":
            p_ind = 1
            fact = 1.0
        else:
            p_ind = 0
            fact = -1.0

        top = fact * (h[:, 0] * ey[:, p_ind] - h[:, 1] * ex[:, p_ind])
        bot = ex[:, 0] * ey[:, 1] - ex[:, 1] * ey[:, 0]
        adm = top / bot

        # ADJOINT
        if adjoint:
            if self.component == "imag":
                v = -1j * v

            # J_T * v = d_top_T * a_v + d_bot_T * b
            a_v = fact * v / bot  # term 1
            b_v = -adm * v / bot  # term 2

            ex_v = np.c_[ey[:, 1], -ey[:, 0]] * b_v[:, None]  # terms dex in bot
            ey_v = np.c_[-ex[:, 1], ex[:, 0]] * b_v[:, None]  # terms dey in bot
            ex_v[:, p_ind] -= h[:, 1] * a_v  # add terms dex in top
            ey_v[:, p_ind] += h[:, 0] * a_v  # add terms dey in top
            e_v = Pex.T @ ex_v + Pey.T @ ey_v

            h_v = np.c_[ey[:, p_ind], -ex[:, p_ind]] * a_v[:, None]  # h in top
            h_v = Ph.T @ h_v

            fu_e_v, fm_e_v = f._eDeriv(src, None, e_v, adjoint=True)
            fu_h_v, fm_h_v = f._hDeriv(src, None, h_v, adjoint=True)

            return fu_e_v + fu_h_v, fm_e_v + fm_h_v

        # JVEC
        de_v = f._eDeriv(src, du_dm_v, v, adjoint=False)
        dh_v = Ph @ f._hDeriv(src, du_dm_v, v, adjoint=False)

        dex_v = Pex @ de_v
        dey_v = Pey @ de_v

        dtop_v = fact * (
            h[:, 0] * dey_v[:, p_ind]
            + dh_v[:, 0] * ey[:, p_ind]
            - h[:, 1] * dex_v[:, p_ind]
            - dh_v[:, 1] * ex[:, p_ind]
        )
        dbot_v = (
            ex[:, 0] * dey_v[:, 1]
            + dex_v[:, 0] * ey[:, 1]
            - ex[:, 1] * dey_v[:, 0]
            - dex_v[:, 1] * ey[:, 0]
        )
        adm_deriv = (bot * dtop_v - top * dbot_v) / (bot * bot)

        return getattr(adm_deriv, self.component)

    def eval(self, src, mesh, f):  # noqa: A003
        # Docstring inherited from parent class (Impedance).
        adm = self._eval_admittance(src, mesh, f)
        if self.component == "complex":
            return adm
        return getattr(adm, self.component)

    def evalDeriv(self, src, mesh, f, du_dm_v=None, v=None, adjoint=False):
        # Docstring inherited from parent class (Impedance).
        if self.component == "complex":
            raise NotImplementedError(
                "complex valued data derivative is not implemented."
            )
        return self._eval_admittance_deriv(
            src, mesh, f, du_dm_v=du_dm_v, v=v, adjoint=adjoint
        )


class ApparentConductivity(_ElectricAndMagneticReceiver):
    r"""Receiver class for simulating apparent conductivity data (3D problems only).

    This class is used to simulate apparent conductivity data, in S/m, as defined by:

    .. math::
        \sigma_{app} = \mu_0 \omega \dfrac{\big | \vec{H} \big |^2}{\big | \vec{E} \big |^2}

    where :math:`\omega` is the angular frequency in rad/s,

    .. math::
        \big | \vec{H} \big | = \Big [ H_x^2 + H_y^2 + H_z^2 \Big ]^{1/2}

    and

    .. math::
        \big | \vec{E} \big | = \Big [ E_x^2 + E_y^2 \Big ]^{1/2}

    Parameters
    ----------
    locations_e : (n_loc, n_dim) array_like
        Locations where the electric fields are measured.
    locations_h : (n_loc, n_dim) array_like, optional
        Locations where the magnetic fields are measured. Defaults to the same
        locations as electric field measurements, `locations_e`.
    storeProjections : bool
        Whether to cache to internal projection matrices.
    """

    def __init__(self, locations_e, locations_h=None, storeProjections=False):
        if locations_h is None:
            locations_h = locations_e
        super().__init__(
            locations1=locations_e,
            locations2=locations_h,
            storeProjections=storeProjections,
        )

    def _eval_apparent_conductivity(self, src, mesh, f):
        if mesh.dim < 3:
            raise NotImplementedError(
                "ApparentConductivity receiver not implemented for dim < 3."
            )

        e = f[src, "e"]
        h = f[src, "h"]

        Pex = self.getP(mesh, "Ex", 0)
        Pey = self.getP(mesh, "Ey", 0)
        Phx = self.getP(mesh, "Fx", 1)
        Phy = self.getP(mesh, "Fy", 1)
        Phz = self.getP(mesh, "Fz", 1)

        ex = np.sum(Pex @ e, axis=-1)
        ey = np.sum(Pey @ e, axis=-1)
        hx = np.sum(Phx @ h, axis=-1)
        hy = np.sum(Phy @ h, axis=-1)
        hz = np.sum(Phz @ h, axis=-1)

        top = np.abs(hx) ** 2 + np.abs(hy) ** 2 + np.abs(hz) ** 2
        bot = np.abs(ex) ** 2 + np.abs(ey) ** 2

        return (2 * np.pi * src.frequency * mu_0) * top / bot

    def _eval_apparent_conductivity_deriv(
        self, src, mesh, f, du_dm_v=None, v=None, adjoint=False
    ):
        if mesh.dim < 3:
            raise NotImplementedError(
                "Admittance receiver not implemented for dim < 3."
            )

        # Compute admittances
        e = f[src, "e"]
        h = f[src, "h"]

        Pex = self.getP(mesh, "Ex", 0)
        Pey = self.getP(mesh, "Ey", 0)
        Phx = self.getP(mesh, "Fx", 1)
        Phy = self.getP(mesh, "Fy", 1)
        Phz = self.getP(mesh, "Fz", 1)

        ex = np.sum(Pex @ e, axis=-1)
        ey = np.sum(Pey @ e, axis=-1)
        hx = np.sum(Phx @ h, axis=-1)
        hy = np.sum(Phy @ h, axis=-1)
        hz = np.sum(Phz @ h, axis=-1)

        fact = 2 * np.pi * src.frequency * mu_0
        top = np.abs(hx) ** 2 + np.abs(hy) ** 2 + np.abs(hz) ** 2
        bot = np.abs(ex) ** 2 + np.abs(ey) ** 2

        # ADJOINT
        if adjoint:
            # Compute: J_T * v = d_top_T * a_v + d_bot_T * b
            a_v = fact * v / bot  # term 1
            b_v = -fact * top * v / bot**2  # term 2

            hx *= a_v
            hy *= a_v
            hz *= a_v
            ex *= b_v
            ey *= b_v

            e_v = 2 * (Pex.T @ ex + Pey.T @ ey).conjugate()
            h_v = 2 * (Phx.T @ hx + Phy.T @ hy + Phz.T @ hz).conjugate()

            fu_e_v, fm_e_v = f._eDeriv(src, None, e_v, adjoint=True)
            fu_h_v, fm_h_v = f._hDeriv(src, None, h_v, adjoint=True)

            return fu_e_v + fu_h_v, fm_e_v + fm_h_v

        # JVEC
        de_v = f._eDeriv(src, du_dm_v, v, adjoint=False)
        dh_v = f._hDeriv(src, du_dm_v, v, adjoint=False)

        dex_v = np.sum(Pex @ de_v, axis=-1)
        dey_v = np.sum(Pey @ de_v, axis=-1)
        dhx_v = np.sum(Phx @ dh_v, axis=-1)
        dhy_v = np.sum(Phy @ dh_v, axis=-1)
        dhz_v = np.sum(Phz @ dh_v, axis=-1)

        # Imaginary components cancel and its 2x the real
        dtop_v = (
            2
            * (
                hx * dhx_v.conjugate() + hy * dhy_v.conjugate() + hz * dhz_v.conjugate()
            ).real
        )

        dbot_v = 2 * (ex * dex_v.conjugate() + ey * dey_v.conjugate()).real

        return fact * (bot * dtop_v - top * dbot_v) / (bot * bot)

    def eval(self, src, mesh, f):  # noqa: A003
        """Compute receiver data from the discrete field solution.

        Parameters
        ----------
        src : .frequency_domain.sources.BaseFDEMSrc
            NSEM source.
        mesh : discretize.TensorMesh
            Mesh on which the discretize solution is obtained.
        f : simpeg.electromagnetics.frequency_domain.fields.FieldsFDEM
            NSEM fields object of the source.

        Returns
        -------
        numpy.ndarray
            Evaluated data for the receiver.
        """
        return self._eval_apparent_conductivity(src, mesh, f)

    def evalDeriv(self, src, mesh, f, du_dm_v=None, v=None, adjoint=False):
        r"""Derivative of data with respect to the fields.

        Let :math:`\mathbf{d}` represent the data corresponding the receiver object.
        And let :math:`\mathbf{u}` represent the discrete numerical solution of the
        fields on the mesh. Where :math:`\mathbf{P}` is a projection function that
        maps from the fields to the data, i.e.:

        .. math::
            \mathbf{d} = \mathbf{P}(\mathbf{u})

        this method computes and returns the derivative:

        .. math::
            \dfrac{\partial \mathbf{d}}{\partial \mathbf{u}} =
            \dfrac{\partial [ \mathbf{P} (\mathbf{u}) ]}{\partial \mathbf{u}}

        Parameters
        ----------
        src : .frequency_domain.sources.BaseFDEMSrc
            The NSEM source.
        mesh : discretize.TensorMesh
            Mesh on which the discretize solution is obtained.
        f : simpeg.electromagnetics.frequency_domain.fields.FieldsFDEM
            NSEM fields object for the source.
        du_dm_v : None, optional
            Supply pre-computed derivative?
        v : numpy.ndarray, optional
            Vector of size
        adjoint : bool, optional
            Whether to compute the ajoint operation.

        Returns
        -------
        numpy.ndarray
            Calculated derivative (n_data,) if `adjoint` is ``False``, and (n_param, 2) if `adjoint`
            is ``True``, for both polarizations.
        """
        return self._eval_apparent_conductivity_deriv(
            src, mesh, f, du_dm_v=du_dm_v, v=v, adjoint=adjoint
        )


@deprecate_class(removal_version="0.24.0", error=True, replace_docstring=False)
class PointNaturalSource(Impedance):
    """
    .. warning::
        This class was removed in SimPEG v0.24.0.
        Please use :class:`.natural_source.receivers.Impedance`.
    """


@deprecate_class(removal_version="0.24.0", error=True, replace_docstring=False)
class Point3DTipper(Tipper):
    """
    .. warning::
        This class was removed in SimPEG v0.24.0.
        Please use :class:`.natural_source.receivers.Tipper`.
    """
