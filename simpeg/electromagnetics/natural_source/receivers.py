"""Natural source EM receivers."""

from ...utils.code_utils import (
    validate_string,
    validate_type,
    validate_ndarray_with_shape,
    deprecate_class,
)
import numpy as np
from scipy.constants import mu_0
from ...survey import BaseRx
from simpeg.utils import mkvc


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
    """Intermediate class for MT receivers that measure an electric and magnetic field."""

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
        r"""Data type; i.e. "real", "imag", "apparent_resistivity", "phase".

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
                    PE = self.getP(mesh, "Ex", 0)
                    PH = self.getP(mesh, "CC", 1)
                elif self.orientation == "yx":
                    PE = self.getP(mesh, "CC", 0)
                    PH = self.getP(mesh, "Ex", 1)
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
                    PE = self.getP(mesh, "Ex", 0)
                    PH = self.getP(mesh, "CC", 1)
                elif self.orientation == "yx":
                    PE = self.getP(mesh, "CC", 0)
                    PH = self.getP(mesh, "Ex", 1)

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
            Calculated derivative (n_data,) if `adjoint` is ``False``,
            and (n_param, 2) if `adjoint` is ``True``, for both polarizations.
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
        \begin{bmatrix}
        T_{xx} & T_{yx} & T_{zx} \\ T_{xy} & T_{yy} & T_{zy}
        \end{bmatrix} = \begin{bmatrix}
        H_x^{(x)} & H_y^{(x)} \\ H_x^{(y)} & H_y^{(y)}
        \end{bmatrix}_b^{-1} \, \begin{bmatrix}
        H_x^{(x)} & H_y^{(x)} & H_z^{(x)} \\ H_x^{(y)} & H_y^{(y)} & H_z^{(y)}
        \end{bmatrix}_r

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
        r"""Tipper data type; i.e. "real", "imag".

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

        # Only Tzx
        if mesh.dim == 2:

            Phx = self.getP(mesh, "Ex", 1)
            Phz = self.getP(mesh, "Ey", 0)

            hz = Phz @ h[:, 0]
            hx = Phx @ h[:, 0]

            return hz / hx

        else:

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

        if mesh.dim == 2:

            Phx = self.getP(mesh, "Ex", 1)
            Phz = self.getP(mesh, "Ey", 0)

            hz = Phz @ h[:, 0]
            hx = Phx @ h[:, 0]

            tip = hz / hx

        else:

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

        # ADJOINT
        if adjoint:

            n_d = self.nD

            if mesh.dim == 2:

                ghz_v = v / hx
                ghx_v = -tip * v / hx
                gh_v = Phx.T @ ghx_v + Phz.T @ ghz_v

            else:
                # Work backwards!
                gtop_v = (v / bot)[..., None]
                gbot_v = (-tip * v / bot)[..., None]

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

        # JVEC
        dh_v = f._hDeriv(src, du_dm_v, v, adjoint=False)

        if mesh.dim == 2:

            dhx_v = Phx @ dh_v
            dhz_v = Phz @ dh_v

            return (dhz_v - tip * dhx_v) / hx

        else:

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
        \begin{bmatrix}
        Y_{xx} & Y_{xy} \\ Y_{yx} & Y_{yy} \\ Y_{zx} & Y_{zy}
        \end{bmatrix} = \begin{bmatrix}
        H_x^{(x)} & H_x^{(y)} \\ H_y^{(x)} & H_y^{(y)} \\ H_z^{(x)} & H_z^{(y)}
        \end{bmatrix}_{\, r} \; \begin{bmatrix}
        E_x^{(x)} & E_x^{(y)} \\ E_y^{(x)} & E_y^{(y)}
        \end{bmatrix}_b^{-1}

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
        if mesh.dim == 1:
            raise NotImplementedError(
                "Admittance receiver not implemented for dim == 1."
            )

        e = f[src, "e"]
        h = f[src, "h"]

        if mesh.dim == 2:
            if self.orientation == "yx":
                PE = self.getP(mesh, "Ex", 0)
                PH = self.getP(mesh, "CC", 1)
            elif self.orientation == "xy":
                PE = self.getP(mesh, "CC", 0)
                PH = self.getP(mesh, "Ex", 1)

            top = PH @ h[:, 0]
            bot = PE @ e[:, 0]

        else:

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
        if mesh.dim == 1:
            raise NotImplementedError(
                "Admittance receiver not implemented for dim == 1."
            )

        # COMPUTE ADMITTANCES
        e = f[src, "e"]
        h = f[src, "h"]

        if mesh.dim == 2:
            if self.orientation == "yx":
                Pe = self.getP(mesh, "Ex", 0)
                Ph = self.getP(mesh, "CC", 1)
            elif self.orientation == "xy":
                Pe = self.getP(mesh, "CC", 0)
                Ph = self.getP(mesh, "Ex", 1)

            top = Ph @ h[:, 0]
            bot = Pe @ e[:, 0]
            adm = top / bot

            fact = 1.0

        else:

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

            if mesh.dim == 2:

                h_v = Ph.T @ a_v
                e_v = Pe.T @ b_v

            else:

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
        if mesh.dim == 2:

            de_v = Pe @ f._eDeriv(src, du_dm_v, v, adjoint=False)
            dh_v = Ph @ f._hDeriv(src, du_dm_v, v, adjoint=False)

            adm_deriv = (dh_v - adm * de_v) / bot

        else:

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


class _BaseOrientationInvariant(BaseNaturalSourceRx):

    _loc_names = ("First", "Second")

    def __init__(  # noqa: D107
        self,
        locations1,
        locations2=None,
        base_type="magnetic",
        storeProjections=False,
    ):
        super().__init__(
            locations1=locations1,
            locations2=locations2,
            storeProjections=storeProjections,
        )
        self.base_type = base_type

    @property
    def base_type(self):
        r"""Whether a 'magnetic' or 'electric' base station is used.

        Returns
        -------
        str
            Base station type; i.e. "magnetic" or "electric"
        """
        return self._base_type

    @base_type.setter
    def base_type(self, var):
        self._base_type = validate_string("base_type", var, ["magnetic", "electric"])

    def _eval_root_gram_determinant(self, src, mesh, f):

        if mesh.dim < 3:
            raise NotImplementedError(
                "'RootGramDeterminant' transfer function only for 3D simulation."
            )

        h = f[src, "h"]
        hx = self.getP(mesh, "Fx", 0) @ h
        hy = self.getP(mesh, "Fy", 0) @ h
        hz = self.getP(mesh, "Fz", 0) @ h

        if self.base_type == "magnetic":
            bx = self.getP(mesh, "Fx", 1) @ h
            by = self.getP(mesh, "Fy", 1) @ h

        else:
            e = f[src, "e"]
            bx = self.getP(mesh, "Ex", 1) @ e
            by = self.getP(mesh, "Ey", 1) @ e

        # abs(det(H H*))
        top = (
            (np.abs(hx[:, 0] ** 2) + np.abs(hy[:, 0] ** 2) + np.abs(hz[:, 0] ** 2))
            * (np.abs(hx[:, 1] ** 2) + np.abs(hy[:, 1] ** 2) + np.abs(hz[:, 1] ** 2))
        ) - np.abs(
            hx[:, 0] * hx[:, 1].conjugate()
            + hy[:, 0] * hy[:, 1].conjugate()
            + hz[:, 0] * hz[:, 1].conjugate()
        ) ** 2

        # abs(det(B B*)) = abs(det(B))**2
        bot = np.abs(bx[:, 0] * by[:, 1] - bx[:, 1] * by[:, 0]) ** 2

        return np.sqrt(top / bot)

    def _eval_root_gram_determinant_deriv(
        self, src, mesh, f, du_dm_v=None, v=None, adjoint=False
    ):

        if mesh.dim < 3:
            raise NotImplementedError(
                "'AmplitudeSquared' transfer function only for 3D simulation."
            )

        h = f[src, "h"]
        Phx = self.getP(mesh, "Fx", 0)
        Phy = self.getP(mesh, "Fy", 0)
        Phz = self.getP(mesh, "Fz", 0)

        hx = Phx @ h
        hy = Phy @ h
        hz = Phz @ h

        if self.base_type == "magnetic":

            Pbx = self.getP(mesh, "Fx", 1)
            Pby = self.getP(mesh, "Fy", 1)

            bx = Pbx @ h
            by = Pby @ h

        else:

            Pbx = self.getP(mesh, "Ex", 1)
            Pby = self.getP(mesh, "Ey", 1)

            e = f[src, "e"]
            bx = Pbx @ e
            by = Pby @ e

        # Entries of HH*. Note that vec_h21 = conj(vec_h12)
        vec_h11 = np.abs(hx[:, 0]) ** 2 + np.abs(hy[:, 0]) ** 2 + np.abs(hz[:, 0]) ** 2
        vec_h22 = np.abs(hx[:, 1]) ** 2 + np.abs(hy[:, 1]) ** 2 + np.abs(hz[:, 1]) ** 2
        vec_h12 = (
            hx[:, 0] * hx[:, 1].conjugate()
            + hy[:, 0] * hy[:, 1].conjugate()
            + hz[:, 0] * hz[:, 1].conjugate()
        )
        top = vec_h11 * vec_h22 - np.abs(vec_h12) ** 2  # abs(det(H H*))

        vec_b11 = np.abs(bx[:, 0]) ** 2 + np.abs(by[:, 0]) ** 2
        vec_b22 = np.abs(bx[:, 1]) ** 2 + np.abs(by[:, 1]) ** 2
        vec_b12 = bx[:, 0] * bx[:, 1].conjugate() + by[:, 0] * by[:, 1].conjugate()
        bot = vec_b11 * vec_b22 - np.abs(vec_b12) ** 2  # abs(det(H H*))

        scale = 0.5 / np.sqrt(top / bot)
        # Scale by w*mu_0
        if isinstance(self, ApparentConductivity):
            scale /= _alpha(src)

        # ADJOINT
        if adjoint:

            # J_T * v = d_top_T * a_v + d_bot_T * b
            a_v = scale * v / bot  # term 1
            b_v = -scale * top * v / bot**2  # term 2

            a_v = np.repeat(mkvc(a_v, n_dims=2), 2, axis=-1)
            b_v = np.repeat(mkvc(b_v, n_dims=2), 2, axis=-1)

            # derivatives for det(HH*)
            px = (
                np.c_[
                    vec_h22 * hx[:, 0].conjugate() - (vec_h12 * hx[:, 1]).conjugate(),
                    vec_h11 * hx[:, 1].conjugate() - vec_h12 * hx[:, 0].conjugate(),
                ]
                * a_v
            )
            py = (
                np.c_[
                    vec_h22 * hy[:, 0].conjugate() - (vec_h12 * hy[:, 1]).conjugate(),
                    vec_h11 * hy[:, 1].conjugate() - vec_h12 * hy[:, 0].conjugate(),
                ]
                * a_v
            )
            pz = (
                np.c_[
                    vec_h22 * hz[:, 0].conjugate() - (vec_h12 * hz[:, 1]).conjugate(),
                    vec_h11 * hz[:, 1].conjugate() - vec_h12 * hz[:, 0].conjugate(),
                ]
                * a_v
            )

            # derivatives for det(BB*)
            qx = (
                np.c_[
                    vec_b22 * bx[:, 0].conjugate() - (vec_b12 * bx[:, 1]).conjugate(),
                    vec_b11 * bx[:, 1].conjugate() - vec_b12 * bx[:, 0].conjugate(),
                ]
                * b_v
            )
            qy = (
                np.c_[
                    vec_b22 * by[:, 0].conjugate() - (vec_b12 * by[:, 1]).conjugate(),
                    vec_b11 * by[:, 1].conjugate() - vec_b12 * by[:, 0].conjugate(),
                ]
                * b_v
            )

            h_v = 2 * (Phx.T @ px + Phy.T @ py + Phz.T @ pz)
            b_v = 2 * (Pbx.T @ qx + Pby.T @ qy)

            if self.base_type == "magnetic":

                return f._hDeriv(src, None, h_v + b_v, adjoint=True)

            else:

                fu_b_v, fm_b_v = f._eDeriv(src, None, b_v, adjoint=True)
                fu_h_v, fm_h_v = f._hDeriv(src, None, h_v, adjoint=True)
                return fu_b_v + fu_h_v, fm_b_v + fm_h_v

        # JVEC
        dh_v = f._hDeriv(src, du_dm_v, v, adjoint=False)
        dhx_v = Phx @ dh_v
        dhy_v = Phy @ dh_v
        dhz_v = Phz @ dh_v

        if self.base_type == "magnetic":
            db_v = dh_v
        else:
            db_v = f._eDeriv(src, du_dm_v, v, adjoint=False)

        dbx_v = Pbx @ db_v
        dby_v = Pby @ db_v

        # When taking derivative of hh* wrt the model, imaginary components
        # cancel and its 2x the real of the conjugate x the deriv
        dtop_v = (
            (
                2
                * vec_h11
                * (
                    hx[:, 1].conjugate() * dhx_v[:, 1]
                    + hy[:, 1].conjugate() * dhy_v[:, 1]
                    + hz[:, 1].conjugate() * dhz_v[:, 1]
                )
            ).real
            + (
                2
                * vec_h22
                * (
                    hx[:, 0].conjugate() * dhx_v[:, 0]
                    + hy[:, 0].conjugate() * dhy_v[:, 0]
                    + hz[:, 0].conjugate() * dhz_v[:, 0]
                )
            ).real
            - (
                2
                * vec_h12.conjugate()
                * (
                    hx[:, 1].conjugate() * dhx_v[:, 0]
                    + hy[:, 1].conjugate() * dhy_v[:, 0]
                    + hz[:, 1].conjugate() * dhz_v[:, 0]
                    + hx[:, 0] * dhx_v[:, 1].conjugate()
                    + hy[:, 0] * dhy_v[:, 1].conjugate()
                    + hz[:, 0] * dhz_v[:, 1].conjugate()
                )
            ).real
        )

        dbot_v = (
            (
                2
                * vec_b11
                * (
                    bx[:, 1].conjugate() * dbx_v[:, 1]
                    + by[:, 1].conjugate() * dby_v[:, 1]
                )
            ).real
            + (
                2
                * vec_b22
                * (
                    bx[:, 0].conjugate() * dbx_v[:, 0]
                    + by[:, 0].conjugate() * dby_v[:, 0]
                )
            ).real
            - (
                2
                * vec_b12.conjugate()
                * (
                    bx[:, 1].conjugate() * dbx_v[:, 0]
                    + by[:, 1].conjugate() * dby_v[:, 0]
                    + bx[:, 0] * dbx_v[:, 1].conjugate()
                    + by[:, 0] * dby_v[:, 1].conjugate()
                )
            ).real
        )

        return scale * (bot * dtop_v - top * dbot_v) / (bot * bot)

    def _eval_cross_product_amplitude(self, src, mesh, f):

        if mesh.dim < 3:
            raise NotImplementedError(
                "'CrossProductAmplitude' transfer function only for 3D simulation."
            )

        h = f[src, "h"]
        hx = self.getP(mesh, "Fx", 0) @ h
        hy = self.getP(mesh, "Fy", 0) @ h
        hz = self.getP(mesh, "Fz", 0) @ h

        if self.base_type == "magnetic":
            bx = self.getP(mesh, "Fx", 1) @ h
            by = self.getP(mesh, "Fy", 1) @ h

        else:
            e = f[src, "e"]
            bx = self.getP(mesh, "Ex", 1) @ e
            by = self.getP(mesh, "Ey", 1) @ e

        top_12 = (
            np.abs(hx[:, 0] * hy[:, 1] - hy[:, 0] * hx[:, 1]) ** 2
        )  # abs(det(H12))**2
        top_13 = (
            np.abs(hx[:, 0] * hz[:, 1] - hz[:, 0] * hx[:, 1]) ** 2
        )  # abs(det(H13))**2
        top_23 = (
            np.abs(hy[:, 0] * hz[:, 1] - hz[:, 0] * hy[:, 1]) ** 2
        )  # abs(det(H23))**2

        # abs(det(B B*)) = abs(det(B))**2
        bot = np.abs(bx[:, 0] * by[:, 1] - bx[:, 1] * by[:, 0]) ** 2

        return np.sqrt((top_12 + top_13 + top_23) / bot)

    def _eval_cross_product_amplitude_deriv(
        self, src, mesh, f, du_dm_v=None, v=None, adjoint=False
    ):

        if mesh.dim < 3:
            raise NotImplementedError(
                "'AmplitudeSquared' transfer function only for 3D simulation."
            )

        h = f[src, "h"]
        Phx = self.getP(mesh, "Fx", 0)
        Phy = self.getP(mesh, "Fy", 0)
        Phz = self.getP(mesh, "Fz", 0)

        hx = Phx @ h
        hy = Phy @ h
        hz = Phz @ h

        if self.base_type == "magnetic":

            Pbx = self.getP(mesh, "Fx", 1)
            Pby = self.getP(mesh, "Fy", 1)

            bx = Pbx @ h
            by = Pby @ h

        else:

            Pbx = self.getP(mesh, "Ex", 1)
            Pby = self.getP(mesh, "Ey", 1)

            e = f[src, "e"]
            bx = Pbx @ e
            by = Pby @ e

        # Entries of HH*. Note that vec_h21 = conj(vec_h12)
        det_h12 = hx[:, 0] * hy[:, 1] - hy[:, 0] * hx[:, 1]  # det(H12)
        det_h13 = hx[:, 0] * hz[:, 1] - hz[:, 0] * hx[:, 1]  # det(H13)
        det_h23 = hy[:, 0] * hz[:, 1] - hz[:, 0] * hy[:, 1]  # det(H23)
        top = np.abs(det_h12) ** 2 + np.abs(det_h13) ** 2 + np.abs(det_h23) ** 2

        # abs(det(B B*)) = abs(det(B))**2
        det_b = bx[:, 0] * by[:, 1] - bx[:, 1] * by[:, 0]
        bot = np.abs(det_b) ** 2

        scale = 0.5 / np.sqrt(top / bot)
        # Scale by w*mu_0
        if isinstance(self, ApparentConductivity):
            scale /= _alpha(src)

        # ADJOINT
        if adjoint:

            # J_T * v = d_top_T * a_v + d_bot_T * b
            a_v = scale * v / bot  # term 1
            b_v = -scale * top * v / bot**2  # term 2

            a_v = np.repeat(mkvc(a_v, n_dims=2), 2, axis=-1)
            b_v = np.repeat(mkvc(b_v, n_dims=2), 2, axis=-1)

            # derivatives for det(HH*)
            px = (
                np.c_[
                    det_h12.conjugate() * hy[:, 1] + det_h13.conjugate() * hz[:, 1],
                    -det_h12.conjugate() * hy[:, 0] - det_h13.conjugate() * hz[:, 0],
                ]
                * a_v
            )
            py = (
                np.c_[
                    -det_h12.conjugate() * hx[:, 1] + det_h23.conjugate() * hz[:, 1],
                    det_h12.conjugate() * hx[:, 0] - det_h23.conjugate() * hz[:, 0],
                ]
                * a_v
            )
            pz = (
                np.c_[
                    -det_h13.conjugate() * hx[:, 1] - det_h23.conjugate() * hy[:, 1],
                    det_h13.conjugate() * hx[:, 0] + det_h23.conjugate() * hy[:, 0],
                ]
                * a_v
            )

            # derivatives for det(BB*)
            qx = (
                np.c_[det_b.conjugate() * by[:, 1], -det_b.conjugate() * by[:, 0]] * b_v
            )
            qy = (
                np.c_[-det_b.conjugate() * bx[:, 1], det_b.conjugate() * bx[:, 0]] * b_v
            )

            h_v = 2 * (Phx.T @ px + Phy.T @ py + Phz.T @ pz)
            b_v = 2 * (Pbx.T @ qx + Pby.T @ qy)

            if self.base_type == "magnetic":

                return f._hDeriv(src, None, h_v + b_v, adjoint=True)

            else:

                fu_b_v, fm_b_v = f._eDeriv(src, None, b_v, adjoint=True)
                fu_h_v, fm_h_v = f._hDeriv(src, None, h_v, adjoint=True)
                return fu_b_v + fu_h_v, fm_b_v + fm_h_v

        # JVEC
        dh_v = f._hDeriv(src, du_dm_v, v, adjoint=False)
        dhx_v = Phx @ dh_v
        dhy_v = Phy @ dh_v
        dhz_v = Phz @ dh_v

        if self.base_type == "magnetic":
            db_v = dh_v
        else:
            db_v = f._eDeriv(src, du_dm_v, v, adjoint=False)

        dbx_v = Pbx @ db_v
        dby_v = Pby @ db_v

        # cancel and its 2x the real of the conjugate x the deriv
        dtop_v = (
            2
            * (
                det_h12.conjugate()
                * (
                    dhx_v[:, 0] * hy[:, 1]
                    + hx[:, 0] * dhy_v[:, 1]
                    - dhy_v[:, 0] * hx[:, 1]
                    - hy[:, 0] * dhx_v[:, 1]
                )
                + det_h13.conjugate()
                * (
                    dhx_v[:, 0] * hz[:, 1]
                    + hx[:, 0] * dhz_v[:, 1]
                    - dhz_v[:, 0] * hx[:, 1]
                    - hz[:, 0] * dhx_v[:, 1]
                )
                + det_h23.conjugate()
                * (
                    dhy_v[:, 0] * hz[:, 1]
                    + hy[:, 0] * dhz_v[:, 1]
                    - dhz_v[:, 0] * hy[:, 1]
                    - hz[:, 0] * dhy_v[:, 1]
                )
            ).real
        )

        dbot_v = (
            2
            * (
                det_b.conjugate()
                * (
                    dbx_v[:, 0] * by[:, 1]
                    + bx[:, 0] * dby_v[:, 1]
                    - dby_v[:, 0] * bx[:, 1]
                    - by[:, 0] * dbx_v[:, 1]
                )
            ).real
        )

        return scale * (bot * dtop_v - top * dbot_v) / (bot * bot)

    def _eval_horizontal_determinant(self, src, mesh, f):

        if mesh.dim < 3:
            raise NotImplementedError(
                "'AmplitudeRatio' transfer function only for 3D simulation."
            )

        h = f[src, "h"]
        hx = self.getP(mesh, "Fx", 0) @ h
        hy = self.getP(mesh, "Fy", 0) @ h

        if self.base_type == "magnetic":
            bx = self.getP(mesh, "Fx", 1) @ h
            by = self.getP(mesh, "Fy", 1) @ h

        else:
            e = f[src, "e"]
            bx = self.getP(mesh, "Ex", 1) @ e
            by = self.getP(mesh, "Ey", 1) @ e

        top = hx[:, 0] * hy[:, 1] - hx[:, 1] * hy[:, 0]
        bot = bx[:, 0] * by[:, 1] - bx[:, 1] * by[:, 0]

        return top / bot

    def _eval_horizontal_determinant_deriv(
        self, src, mesh, f, du_dm_v=None, v=None, adjoint=False
    ):

        if mesh.dim < 3:
            raise NotImplementedError(
                "'HorizontalDeterminant' transfer function only for 3D simulation."
            )

        h = f[src, "h"]
        Phx = self.getP(mesh, "Fx", 0)
        Phy = self.getP(mesh, "Fy", 0)

        hx = Phx @ h
        hy = Phy @ h

        if self.base_type == "magnetic":

            Pbx = self.getP(mesh, "Fx", 1)
            Pby = self.getP(mesh, "Fy", 1)

            bx = Pbx @ h
            by = Pby @ h

        else:

            Pbx = self.getP(mesh, "Ex", 1)
            Pby = self.getP(mesh, "Ey", 1)

            e = f[src, "e"]
            bx = Pbx @ e
            by = Pby @ e

        top = hx[:, 0] * hy[:, 1] - hx[:, 1] * hy[:, 0]
        bot = bx[:, 0] * by[:, 1] - bx[:, 1] * by[:, 0]

        # ADJOINT
        if adjoint:

            if isinstance(self, ApparentConductivity):
                scale = _alpha(src) ** -1 * top / bot
                v = (scale.real - 1j * scale.imag) * v / np.abs(top / bot)
            elif self.component == "amp":
                scale = _alpha(src) ** -1 * top / bot
                v = (scale.real - 1j * scale.imag) * v / np.abs(scale)
            elif self.component == "imag":
                v = -1j * v

            # J_T * v = d_top_T * a_v + d_bot_T * b
            a_v = v / bot  # term 1
            b_v = -top * v / bot**2  # term 2

            hx_v = np.c_[hy[:, 1], -hy[:, 0]] * a_v[:, None]  # terms dex in bot
            hy_v = np.c_[-hx[:, 1], hx[:, 0]] * a_v[:, None]  # terms dey in bot
            h_v = Phx.T @ hx_v + Phy.T @ hy_v

            bx_v = np.c_[by[:, 1], -by[:, 0]] * b_v[:, None]  # terms dex in bot
            by_v = np.c_[-bx[:, 1], bx[:, 0]] * b_v[:, None]  # terms dey in bot
            b_v = Pbx.T @ bx_v + Pby.T @ by_v

            if self.base_type == "magnetic":

                return f._hDeriv(src, None, h_v + b_v, adjoint=True)

            else:

                fu_b_v, fm_b_v = f._eDeriv(src, None, b_v, adjoint=True)
                fu_h_v, fm_h_v = f._hDeriv(src, None, h_v, adjoint=True)
                return fu_b_v + fu_h_v, fm_b_v + fm_h_v

        # JVEC
        dh_v = f._hDeriv(src, du_dm_v, v, adjoint=False)
        dhx_v = Phx @ dh_v
        dhy_v = Phy @ dh_v

        if self.base_type == "magnetic":
            db_v = dh_v
        else:
            db_v = f._eDeriv(src, du_dm_v, v, adjoint=False)

        dbx_v = Pbx @ db_v
        dby_v = Pby @ db_v

        dtop_v = (
            hx[:, 0] * dhy_v[:, 1]
            + dhx_v[:, 0] * hy[:, 1]
            - hy[:, 0] * dhx_v[:, 1]
            - dhy_v[:, 0] * hx[:, 1]
        )
        dbot_v = (
            bx[:, 0] * dby_v[:, 1]
            + dbx_v[:, 0] * by[:, 1]
            - by[:, 0] * dbx_v[:, 1]
            - dby_v[:, 0] * bx[:, 1]
        )

        deriv = (bot * dtop_v - top * dbot_v) / (bot * bot)

        if isinstance(self, ApparentConductivity):
            scale = _alpha(src) ** -1 * top / bot
            return (scale.real * deriv.real + scale.imag * deriv.imag) / np.abs(
                top / bot
            )
        elif self.component == "amp":
            scale = top / bot
            return (scale.real * deriv.real + scale.imag * deriv.imag) / np.abs(scale)
        else:
            return getattr(deriv, self.component)


# class RootGramDeterminant(BaseNaturalSourceRx):
class RootGramDeterminant(_BaseOrientationInvariant):
    r"""Orientation invariant transfer function using the root Gram matrix determinant.

    Receiver class for simulating a data that are invariant to sensor orientation
    for either an electric or magnetic base station. The datum is based on taking
    the amplitude of the determinant of Gram matrix for transfer functions
    derived from three-component airborne magnetic fields. For a magnetic base
    station the quantity is unitless. For an electric base station, the units
    are A$^2$/V$^2$. See the *Notes* section for a formal definition of the datum.

    Notes
    -----
    Consider an acquisition system that measures 3-component magnetic fields
    in the air and magnetic fields at a base station. The fundamental set of
    transfer functions (i.e. tippers) that can be generated from these
    measurements is given by:

    .. math::
        \begin{bmatrix}
        T_{xx} & T_{yx} & T_{zx} \\ T_{xy} & T_{yy} & T_{zy}
        \end{bmatrix} = \begin{bmatrix}
        H_x^{(x)} & H_y^{(x)} \\ H_x^{(y)} & H_y^{(y)}
        \end{bmatrix}_b^{-1} \, \begin{bmatrix}
        H_x^{(x)} & H_y^{(x)} & H_z^{(x)} \\ H_x^{(y)} & H_y^{(y)} & H_z^{(y)}
        \end{bmatrix}_r

    where subscript :math:`b` denotes the base station location and subscript
    :math:`r` denotes the mobile receiver location.

    For this system, the orientation invariant transfer function is defined as:

    .. math::
        \widehat{\mathbf{T}} = \bigg (
        \dfrac{det (\mathbf{H_r H_r^\dagger}) }{det (\mathbf{H_b H_b^\dagger})}
        \bigg )^{1/2}

    where $\dagger$ denotes the Hermitian.

    Now consider an acquisition system that measures 3-component magnetic fields
    in the air and electric fields at a base station. The fundamental set of
    transfer functions (i.e. admittances) that can be generated from these
    measurements is given by:

    .. math::
        \begin{bmatrix}
        Y_{xx} & Y_{yx} & Y_{zx} \\ Y_{xy} & Y_{yy} & Y_{zy}
        \end{bmatrix} = \begin{bmatrix}
        E_x^{(x)} & E_y^{(x)} \\ E_x^{(y)} & E_y^{(y)}
        \end{bmatrix}_b^{-1} \, \begin{bmatrix}
        H_x^{(x)} & H_y^{(x)} & H_z^{(x)} \\ H_x^{(y)} & H_y^{(y)} & H_z^{(y)}
        \end{bmatrix}_r

    For this system, the orientation invariant transfer function is defined as:

    .. math::
        \widehat{\mathbf{Y}} = \bigg (
        \dfrac{det (\mathbf{H_r H_r^\dagger}) }{det (\mathbf{E_b E_b^\dagger})}
        \bigg )^{1/2}

    Parameters
    ----------
    locations_h : (n_loc, n_dim) array_like
        Locations where the roving magnetic fields are measured.
    locations_base : (n_loc, n_dim) array_like, optional
        Locations where the base station magnetic fields are measured. Defaults to
        the same locations as the roving magnetic fields measurements,
        `locations_r`.
    base_type : {'magnetic', 'electric'}
        Whether magnetic or electric fields are measured at the base station.
        For magnetic fields, the quantity is unitless. For electric fields,
        the quantity has units A/V.
    storeProjections : bool
        Whether to cache to internal projection matrices.
    """

    _loc_names = ("Roving magnetic field", "Base station field")

    def __init__(  # noqa: D107
        self,
        locations_h,
        locations_base=None,
        base_type="magnetic",
        storeProjections=False,
    ):
        if locations_base is None:
            locations_base = locations_h
        super().__init__(
            locations1=locations_h,
            locations2=locations_base,
            storeProjections=storeProjections,
        )
        self.base_type = base_type

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

    def eval(self, src, mesh, f):  # noqa: D102 A003
        # Docstring inherited from parent class (BaseNaturalSourceRX).
        # return self._eval_transfer_function(src, mesh, f)
        return self._eval_root_gram_determinant(src, mesh, f)

    def evalDeriv(  # noqa: D102
        self, src, mesh, f, du_dm_v=None, v=None, adjoint=False
    ):
        # Docstring inherited from parent class (BaseNaturalSourceRX).
        # return self._eval_transfer_function_deriv(
        #     src, mesh, f, du_dm_v=du_dm_v, v=v, adjoint=adjoint
        # )
        return self._eval_root_gram_determinant_deriv(
            src, mesh, f, du_dm_v=du_dm_v, v=v, adjoint=adjoint
        )


class CrossProductAmplitude(RootGramDeterminant):
    r"""Orientation invariant transfer function from the cross-product amplitude.

    Receiver class for simulating a data that are invariant to sensor orientation
    for either an electric or magnetic base station. The datum is based on taking
    the amplitude of the determinant of the cross-product of transfer functions
    derived from three-component airborne magnetic fields. For a magnetic base
    station the quantity is unitless. For an electric base station, the units
    are A$^2$/V$^2$. See the *Notes* section for a formal definition of the datum.

    Notes
    -----
    Consider an acquisition system that measures 3-component magnetic fields
    in the air and magnetic fields at a base station. The fundamental set of
    transfer functions (i.e. tippers) that can be generated from these
    measurements is given by:

    .. math::
        \begin{bmatrix}
        T_{xx} & T_{yx} & T_{zx} \\ T_{xy} & T_{yy} & T_{zy}
        \end{bmatrix} = \begin{bmatrix}
        H_x^{(x)} & H_y^{(x)} \\ H_x^{(y)} & H_y^{(y)}
        \end{bmatrix}_b^{-1} \, \begin{bmatrix}
        H_x^{(x)} & H_y^{(x)} & H_z^{(x)} \\ H_x^{(y)} & H_y^{(y)} & H_z^{(y)}
        \end{bmatrix}_r

    where subscript :math:`b` denotes the base station location and subscript
    :math:`r` denotes the mobile receiver location.

    For this system, the orientation invariant transfer function is defined as:

    .. math::
        | \mathbf{T} | = \bigg (
        p_x p_x^\ast + p_y p_y^\ast + p_z p_z^\ast
        \bigg )^{1/2}

    where $\ast$ denotes the complex conjugate and

    .. math::
        \begin{split}
        p_x &= T_{yx}T_{zy} - T_{zx}T_{yy}\\
        p_y &= T_{zx}T_{xy} - T_{xx}T_{zy}\\
        p_z &= T_{xx}T_{yy} - T_{yx}T_{xy}
        \end{split}

    Now consider an acquisition system that measures 3-component magnetic fields
    in the air and electric fields at a base station. The fundamental set of
    transfer functions (i.e. admittances) that can be generated from these
    measurements is given by:

    .. math::
        \begin{bmatrix}
        Y_{xx} & Y_{yx} & Y_{zx} \\ Y_{xy} & Y_{yy} & Y_{zy}
        \end{bmatrix} = \begin{bmatrix}
        E_x^{(x)} & E_y^{(x)} \\ E_x^{(y)} & E_y^{(y)}
        \end{bmatrix}_b^{-1} \, \begin{bmatrix}
        H_x^{(x)} & H_y^{(x)} & H_z^{(x)} \\ H_x^{(y)} & H_y^{(y)} & H_z^{(y)}
        \end{bmatrix}_r

    For this system, the orientation invariant transfer function is defined as:

    .. math::
        | \mathbf{Y} | = \bigg (
        q_x q_x^\ast + q_y q_y^\ast + q_z q_z^\ast
        \bigg )^{1/2}

    where $\ast$ denotes the complex conjugate and

    .. math::
        \begin{split}
        p_x &= Y_{yx}Y_{zy} - Y_{zx}Y_{yy}\\
        p_y &= Y_{zx}Y_{xy} - Y_{xx}Y_{zy}\\
        p_z &= Y_{xx}Y_{yy} - Y_{yx}Y_{xy}
        \end{split}

    Parameters
    ----------
    locations_h : (n_loc, n_dim) array_like
        Locations where the roving magnetic fields are measured.
    locations_base : (n_loc, n_dim) array_like, optional
        Locations where the base station magnetic fields are measured. Defaults to
        the same locations as the roving magnetic fields measurements,
        `locations_r`.
    base_type : {'magnetic', 'electric'}
        Whether magnetic or electric fields are measured at the base station.
        For magnetic fields, the quantity is unitless. For electric fields,
        the quantity has units A/V.
    storeProjections : bool
        Whether to cache to internal projection matrices.
    """

    _loc_names = ("Roving magnetic field", "Base station field")

    def __init__(  # noqa: D107
        self,
        locations_h,
        locations_base=None,
        base_type="magnetic",
        storeProjections=False,
    ):
        super().__init__(
            locations_h=locations_h,
            locations_base=locations_base,
            base_type=base_type,
            storeProjections=storeProjections,
        )

    def eval(self, src, mesh, f):  # noqa: D102 A003
        # Docstring inherited from parent class (BaseNaturalSourceRX).
        # return self._eval_transfer_function(src, mesh, f)
        return self._eval_cross_product_amplitude(src, mesh, f)

    def evalDeriv(  # noqa: D102
        self, src, mesh, f, du_dm_v=None, v=None, adjoint=False
    ):
        # Docstring inherited from parent class (BaseNaturalSourceRX).
        # return self._eval_transfer_function_deriv(
        #     src, mesh, f, du_dm_v=du_dm_v, v=v, adjoint=adjoint
        # )
        return self._eval_cross_product_amplitude_deriv(
            src, mesh, f, du_dm_v=du_dm_v, v=v, adjoint=adjoint
        )


class HorizontalDeterminant(RootGramDeterminant):
    r"""Determinant of the horizontal transfer functions.

    Receiver class for simulating a data that are invariant to airborne sensor
    orientation about the tow cable for either an electric or magnetic base station.
    The datum is derived by taking the determinant of the horizontal transfer
    functions. For a magnetic base station the quantity is unitless. For an electric
    base station, the units are A$^2$/V$^2$. See the *Notes* section for a formal
    definition of the datum.

    Notes
    -----

    Consider an acquisition system that measures 3-component magnetic fields
    in the air and magnetic fields at a base station. The fundamental set of
    transfer functions (i.e. tippers) that can be generated from these
    measurements is given by:

    .. math::
        \begin{bmatrix}
        T_{xx} & T_{yx} & T_{zx} \\ T_{xy} & T_{yy} & T_{zy}
        \end{bmatrix} = \begin{bmatrix}
        H_x^{(x)} & H_y^{(x)} \\ H_x^{(y)} & H_y^{(y)}
        \end{bmatrix}_b^{-1} \, \begin{bmatrix}
        H_x^{(x)} & H_y^{(x)} & H_z^{(x)} \\ H_x^{(y)} & H_y^{(y)} & H_z^{(y)}
        \end{bmatrix}_r

    where subscript :math:`b` denotes the base station location and subscript
    :math:`r` denotes the mobile receiver location.

    For this system, the horizontal determinant transfer function is defined as:

    .. math::
        det(T_H) = T_{xx}T_{yy} - T_{yx}T_{xy}

    Now consider an acquisition system that measures 3-component magnetic fields
    in the air and electric fields at a base station. The fundamental set of
    transfer functions (i.e. admittances) that can be generated from these
    measurements is given by:

    .. math::
        \begin{bmatrix}
        Y_{xx} & Y_{yx} & Y_{zx} \\ Y_{xy} & Y_{yy} & Y_{zy}
        \end{bmatrix} = \begin{bmatrix}
        E_x^{(x)} & E_y^{(x)} \\ E_x^{(y)} & E_y^{(y)}
        \end{bmatrix}_b^{-1} \, \begin{bmatrix}
        H_x^{(x)} & H_y^{(x)} & H_z^{(x)} \\ H_x^{(y)} & H_y^{(y)} & H_z^{(y)}
        \end{bmatrix}_r

    For this system, the horizontal determinant transfer function is defined as:

    .. math::
        det(Y_H) = Y_{xx}Y_{yy} - Y_{yx}Y_{xy}

    Parameters
    ----------
    locations_h : (n_loc, n_dim) array_like
        Locations where the roving magnetic fields are measured.
    locations_base : (n_loc, n_dim) array_like, optional
        Locations where the base station magnetic fields are measured. Defaults to
        the same locations as the roving magnetic fields measurements,
        `locations_r`.
    base_type : {'magnetic', 'electric'}
        Whether magnetic or electric fields are measured at the base station.
        For magnetic fields, the quantity is unitless. For electric fields,
        the quantity has units $A^2/V^2$.
    component : {'real', 'imag'}
        Define the receiver to measure the real or imaginary component:
        - 'real': Real component
        - 'imag': Imaginary component
    storeProjections : bool
        Whether to cache to internal projection matrices.
    """
    _loc_names = ("Roving magnetic field", "Base station field")

    def __init__(  # noqa: D107
        self,
        locations_h,
        locations_base=None,
        base_type="magnetic",
        component="real",
        storeProjections=False,
    ):
        super().__init__(
            locations_h=locations_h,
            locations_base=locations_base,
            base_type=base_type,
            storeProjections=storeProjections,
        )
        self.component = component

    @property
    def component(self):
        r"""Data type; i.e. "real", "imag", "amp".

        The `component` property specifies
        whether the data are:
        - 'real': Real component
        - 'imag': Imaginary component
        - 'amp': Amplitude

        Returns
        -------
        str
            Data type; i.e. "real", "imag", "amp"
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
                ("amp", "ampl", "amplitude"),
            ],
        )

    def eval(self, src, mesh, f):  # noqa: A003 D102
        # Doctring inherited from parent class (BaseNaturalSourceRx
        # vals = self._eval_transfer_function(src, mesh, f)
        vals = self._eval_horizontal_determinant(src, mesh, f)
        if self.component == "complex":
            return vals
        elif self.component == "amp":
            return np.abs(vals)
        else:
            return getattr(vals, self.component)

    def evalDeriv(  # noqa: D102
        self, src, mesh, f, du_dm_v=None, v=None, adjoint=False
    ):
        # Doctring inherited from parent class (BaseNaturalSourceRx)
        if self.component == "complex":
            raise NotImplementedError(
                "complex valued data derivative is not implemented."
            )

        return self._eval_horizontal_determinant_deriv(
            src, mesh, f, du_dm_v=du_dm_v, v=v, adjoint=adjoint
        )


class ApparentConductivity(_BaseOrientationInvariant):
    r"""Receiver class for simulating apparent conductivity data (3D problems only).

    This class is used to simulate an apparent conductivity datum, in S/m.

    Parameters
    ----------
    locations_e : (n_loc, n_dim) array_like
        Locations where the electric fields are measured.
    locations_h : (n_loc, n_dim) array_like, optional
        Locations where the magnetic fields are measured. Defaults to the same
        locations as electric field measurements, `locations_e`.
    component : {"root_gram_determinant", "cross_product_amplitude", "horizontal_determinant"}
        The method used to generate the appparent conductivity datum.
    storeProjections : bool
        Whether to cache to internal projection matrices.
    """

    def __init__(
        self,
        locations_e,
        locations_h=None,
        component="cross_product_amplitude",
        storeProjections=False,
    ):  # noqa: D102
        if locations_h is None:
            locations_h = locations_e
        super().__init__(
            locations1=locations_h,
            locations2=locations_e,
            base_type="electric",
            storeProjections=storeProjections,
        )
        self.component = component

    @property
    def locations_h(self):
        """Roving magnetic field measurement locations.

        Returns
        -------
        numpy.ndarray
            Locations where the magnetic fields are measured.
        """
        return self._locations[0]

    @property
    def locations_e(self):
        """Electric field measurement locations.

        Returns
        -------
        numpy.ndarray
            Locations where the horizontal electric fields are measured.
        """
        return self._locations[1]

    @property
    def component(self):
        r"""Equation used to generate the apparent conductivity datum.

        The `component` property specifies
        whether the data are derived using:
        - 'root_gram_determinant': Root Gram determinant of the admittance matrix
        - 'cross_product_amplitude': Cross product amplitude of the admittance matrix
        - 'horizontal_determinant': Determinant of the horizontal admittances

        Returns
        -------
        str
            Data type; i.e. "root_gram_determinant", "cross_product_amplitude", or
            "horizontal_determinant".
        """
        return self._component

    @component.setter
    def component(self, var):
        self._component = validate_string(
            "component",
            var,
            [
                "root_gram_determinant",
                "cross_product_amplitude",
                "horizontal_determinant",
            ],
        )

    def eval(self, src, mesh, f):  # noqa: A003 D102
        # Docstring inherited from parent class
        if self._component == "root_gram_determinant":
            return _alpha(src) ** -1 * self._eval_root_gram_determinant(src, mesh, f)
        elif self._component == "cross_product_amplitude":
            return _alpha(src) ** -1 * self._eval_cross_product_amplitude(src, mesh, f)
        elif self._component == "horizontal_determinant":
            return _alpha(src) ** -1 * np.abs(
                self._eval_horizontal_determinant(src, mesh, f)
            )

    def evalDeriv(  # noqa: A003 D102
        self, src, mesh, f, du_dm_v=None, v=None, adjoint=False
    ):
        # Docstring inherited from parent class
        # scaling by w*mu_0 happens inside function
        if self._component == "root_gram_determinant":
            return self._eval_root_gram_determinant_deriv(
                src, mesh, f, du_dm_v=du_dm_v, v=v, adjoint=adjoint
            )
        elif self._component == "cross_product_amplitude":
            return self._eval_cross_product_amplitude_deriv(
                src, mesh, f, du_dm_v=du_dm_v, v=v, adjoint=adjoint
            )
        elif self._component == "horizontal_determinant":
            return self._eval_horizontal_determinant_deriv(
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
