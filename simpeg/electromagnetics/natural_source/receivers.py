from ...utils.code_utils import (
    deprecate_class,
    validate_string,
    validate_ndarray_with_shape,
)
import warnings
import numpy as np
from scipy.constants import mu_0
from ...survey import BaseRx


def _alpha(src):
    return 1 / (2 * np.pi * mu_0 * src.frequency)


# THIS SHOULD BE HIDDEN METHOD
def _getP(rx, mesh, projected_grid, field="e", is_tipper_bs=False):
    """Projection matrix from discrete field solution to measurement locations.

    Note projection matrices are stored as a dictionary listed by meshes.

    Parameters
    ----------
    rx : .natural_source.receivers.Impedance
        a NSEM receiver
    mesh : discretize.base.BaseMesh
        The mesh on which the discrete set of equations is solved.
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

    field : {"e", "h"}
        Whether to project electric or magnetic fields from the mesh.
    is_tipper_bs : bool
        Whether the projection is for a remote base station location where horizontal
        magnetic fields are measured. Ensures stash names for projection matrices
        aren't reused.
    """
    # Inherited method from BaseRx projecting to self.locations
    if mesh.dim < 3:
        return rx.getP(mesh, projected_grid)

    if is_tipper_bs:
        stash_name = projected_grid + "_bs"
    else:
        stash_name = projected_grid

    if (mesh, stash_name) in rx._Ps:
        return rx._Ps[(mesh, stash_name, field)]

    if field == "e":
        locs = rx.locations_e
    elif field == "h":
        if isinstance(rx, MobileMT):
            locs = rx.locations_h
        elif is_tipper_bs:
            locs = rx.locations_bs
        else:
            locs = rx.locations
    else:
        raise ValueError("Field type {} unrecognized. Use 'e' or 'h'".format(field))

    P = mesh.get_interpolation_matrix(locs, projected_grid)
    if rx.storeProjections:
        rx._Ps[(mesh, stash_name, field)] = P
    return P


class MobileMT(BaseRx):
    r"""Receiver class for MobileMT data (3D problems only).

    This class is used to simulate the apparent conductivity data, in S/m, collected
    by Expert Geophysics MobileMT systems:

    .. math::
        \sigma_{app} = \mu_0 \omega \dfrac{\big | \vec{H} \big |^2}{\big | \vec{E} \big |^2}

    where :math:`\omega` is the angular frequency in rad/s,

    .. math::
        \big | \vec{H} \big | = \Big [ H_x^2 + H_y^2 + H_z^2 \Big ]^{1/2}

    and

    .. math::
        \big | \vec{H} \big | = \Big [ H_x^2 + H_y^2 + H_z^2 \Big ]^{1/2}

    Parameters
    ----------
    locations : (n_loc, n_dim) numpy.ndarray
        Locations where the electric and magnetic fields are measured.
        If electric and magnetic fields are measured in different locations, please
        use `locations_e` and `locations_h` when instantiating.
    locations_e : (n_loc, n_dim) numpy.ndarray
        Locations where the horizontal electric fields are measured.
        Must be same size as `locations_h`.
    locations_h : (n_loc, n_dim) numpy.ndarray
        Locations where the magnetic fields are measured.
        Must be same size as `locations_e`.
    """

    def __init__(
        self,
        locations=None,
        locations_e=None,
        locations_h=None,
    ):
        # check if locations_e or h have been provided
        if (locations_e is not None) and (locations_h is not None):
            # check that locations are same size
            if locations_e.size == locations_h.size:
                self._locations_e = locations_e
                self._locations_h = locations_h
            else:
                raise Exception("location h needs to be same size as location e")

            locations = np.hstack([locations_e, locations_h])
        elif locations is not None:
            # check shape of locations
            if isinstance(locations, list):
                if len(locations) == 2:
                    self._locations_e = locations[0]
                    self._locations_h = locations[1]
                elif len(locations) == 1:
                    self._locations_e = locations[0]
                    self._locations_h = locations[0]
                else:
                    raise Exception("incorrect size of list, must be length of 1 or 2")
                locations = locations[0]
            elif isinstance(locations, np.ndarray):
                self._locations_e = locations
                self._locations_h = locations
            else:
                raise Exception("locations need to be either a list or numpy array")
        else:
            locations = np.array([[0.0]])

        super().__init__(locations)

    @property
    def locations_e(self):
        """Locations for the electric field measurements for each datum.

        Returns
        -------
        (n_data, dim) numpy.ndarray
            Location for the electric field measurements for each datum.
        """
        return self._locations_e

    @property
    def locations_h(self):
        """Locations for the magnetic field measurements for each datum.

        Returns
        -------
        (n_data, dim) numpy.ndarray
            Location for the magnetic field measurements for each datum.
        """
        return self._locations_h

    def _eval_mobilemt(self, src, mesh, f):
        if mesh.dim < 3:
            raise NotImplementedError("MobileMT receiver not implemented for dim < 3.")

        e = f[src, "e"]
        h = f[src, "h"]

        ex = np.sum(_getP(self, mesh, "Ex", "e") @ e, axis=-1)
        ey = np.sum(_getP(self, mesh, "Ey", "e") @ e, axis=-1)

        hx = np.sum(_getP(self, mesh, "Fx", "h") @ h, axis=-1)
        hy = np.sum(_getP(self, mesh, "Fy", "h") @ h, axis=-1)
        hz = np.sum(_getP(self, mesh, "Fz", "h") @ h, axis=-1)

        top = np.abs(hx) ** 2 + np.abs(hy) ** 2 + np.abs(hz) ** 2
        bot = np.abs(ex) ** 2 + np.abs(ey) ** 2

        return (2 * np.pi * src.frequency * mu_0) * top / bot

    def _eval_mobilemt_deriv(self, src, mesh, f, du_dm_v=None, v=None, adjoint=False):
        if mesh.dim < 3:
            raise NotImplementedError(
                "Admittance receiver not implemented for dim < 3."
            )

        # Compute admittances
        e = f[src, "e"]
        h = f[src, "h"]

        Pex = _getP(self, mesh, "Ex", "e")
        Pey = _getP(self, mesh, "Ey", "e")
        Phx = _getP(self, mesh, "Fx", "h")
        Phy = _getP(self, mesh, "Fy", "h")
        Phz = _getP(self, mesh, "Fz", "h")

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
            # J_T * v = d_top_T * a_v + d_bot_T * b
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

    def eval(self, src, mesh, f, return_complex=False):  # noqa: A003
        """Compute receiver data from the discrete field solution.

        Parameters
        ----------
        src : .frequency_domain.sources.BaseFDEMSrc
            NSEM source.
        mesh : discretize.TensorMesh
            Mesh on which the discretize solution is obtained.
        f : .frequency_domain.fields.FieldsFDEM
            NSEM fields object of the source.

        Returns
        -------
        numpy.ndarray
            Evaluated data for the receiver.
        """
        return self._eval_mobilemt(src, mesh, f)

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
        f : .frequency_domain.fields.FieldsFDEM
            NSEM fields object for the source.
        du_dm_v : None,
            Supply pre-computed derivative?
        v : numpy.ndarray
            Vector of size
        adjoint : bool
            Whether to compute the ajoint operation.

        Returns
        -------
        numpy.ndarray
            Calculated derivative (n_data,) if `adjoint` is ``False`` and (n_param, 2)
            if `adjoint` is ``True`` for both polarizations.
        """
        return self._eval_mobilemt_deriv(
            src, mesh, f, du_dm_v=du_dm_v, v=v, adjoint=adjoint
        )


class Impedance(MobileMT):
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
    locations : (n_loc, n_dim) numpy.ndarray
        Locations where the horizontal electric and magnetic fields are measured.
        If electric and magnetic fields are measured in different locations, please
        use `locations_e` and `locations_h` when instantiating.
    orientation : {'xx', 'xy', 'yx', 'yy'}
        MT receiver orientation. Specifies whether the receiver's data correspond to
        the :math:`Z_{xx}`, :math:`Z_{xy}`, :math:`Z_{yx}` or :math:`Z_{yy}` impedance.
        The data type is specified by the `component` input argument.
    component : {'real', 'imag', 'apparent_resistivity', 'phase'}
        MT data type. For the impedance element :math:`Z_{ij}` specified by the `orientation`
        input argument, the receiver can be set to compute the following:
        - 'real': Real component of the impedance (V/A)
        - 'imag': Imaginary component of the impedance (V/A)
        - 'rho': Apparent resistivity (:math:`\Omega m`)
        - 'phase': Phase angle (degrees)
    locations_e : (n_loc, n_dim) numpy.ndarray
        Locations where the horizontal electric fields are measured.
        Must be same size as `locations_h`.
    locations_h : (n_loc, n_dim) numpy.ndarray
        Locations where the horizontal magnetic fields are measured.
        Must be same size as `locations_e`.
    """

    def __init__(
        self,
        locations=None,
        orientation="xy",
        component="real",
        locations_e=None,
        locations_h=None,
    ):
        self.orientation = orientation
        self.component = component
        super().__init__(
            locations=locations, locations_e=locations_e, locations_h=locations_h
        )

    @property
    def component(self):
        r"""MT data type; i.e. "real", "imag", "apparent_resistivity", "phase"

        For the impedance element :math:`Z_{ij}`, the `component` property specifies
        whether the data are:
        - 'real': Real component of the impedance (V/A)
        - 'imag': Imaginary component of the impedance (V/A)
        - 'rho': Apparent resistivity (:math:`\Omega m`)
        - 'phase': Phase angle (degrees)

        Returns
        -------
        str
            MT data type; i.e. "real", "imag", "apparent_resistivity", "phase"
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
            ],
        )

    @property
    def orientation(self):
        """MT receiver orientation.

        Specifies whether the receiver's data correspond to
        the :math:`Z_{xx}`, :math:`Z_{xy}`, :math:`Z_{yx}` or :math:`Z_{yy}` impedance.
        The data type is specified by the `component` input argument.

        Returns
        -------
        str
            MT receiver orientation. One of {'xx', 'xy', 'yx', 'yy'}
        """
        return self._orientation

    @orientation.setter
    def orientation(self, var):
        self._orientation = validate_string(
            "orientation", var, string_list=("xx", "xy", "yx", "yy")
        )

    def _eval_impedance(self, src, mesh, f):
        if mesh.dim < 3 and self.orientation in ["xx", "yy"]:
            return 0.0
        e = f[src, "e"]
        h = f[src, "h"]
        if mesh.dim == 3:
            if self.orientation[0] == "x":
                e = _getP(self, mesh, "Ex", "e") @ e
            else:
                e = _getP(self, mesh, "Ey", "e") @ e

            hx = _getP(self, mesh, "Fx", "h") @ h
            hy = _getP(self, mesh, "Fy", "h") @ h
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
                PE = _getP(self, mesh, e_loc)
                PH = _getP(self, mesh, h_loc)
            elif mesh.dim == 2:
                if self.orientation == "xy":
                    PE = _getP(self, mesh, "Ex")
                    PH = _getP(self, mesh, "CC")
                elif self.orientation == "yx":
                    PE = _getP(self, mesh, "CC")
                    PH = _getP(self, mesh, "Ex")
            top = PE @ e[:, 0]
            bot = PH @ h[:, 0]

            # need to negate if 'yx' and fields are xy
            # and as well if 'xy' and fields are 'yx'
            if mesh.dim == 1 and self.orientation != f.field_directions:
                bot = -bot
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
                Pe = _getP(self, mesh, "Ex", "e")
                e = Pe @ e
            else:
                Pe = _getP(self, mesh, "Ey", "e")
                e = Pe @ e

            Phx = _getP(self, mesh, "Fx", "h")
            Phy = _getP(self, mesh, "Fy", "h")
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
                PE = _getP(self, mesh, e_loc)
                PH = _getP(self, mesh, h_loc)
            elif mesh.dim == 2:
                if self.orientation == "xy":
                    PE = _getP(self, mesh, "Ex")
                    PH = _getP(self, mesh, "CC")
                elif self.orientation == "yx":
                    PE = _getP(self, mesh, "CC")
                    PH = _getP(self, mesh, "Ex")

            top = PE @ e[:, 0]
            bot = PH @ h[:, 0]

            if mesh.dim == 1 and self.orientation != f.field_directions:
                bot = -bot

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
                dh_v = -dh_v

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

    def eval(self, src, mesh, f, return_complex=False):  # noqa: A003
        """Compute receiver data from the discrete field solution.

        Parameters
        ----------
        src : .frequency_domain.sources.BaseFDEMSrc
            NSEM source.
        mesh : discretize.TensorMesh
            Mesh on which the discretize solution is obtained.
        f : .frequency_domain.fields.FieldsFDEM
            NSEM fields object of the source.
        return_complex : bool, optional
            Flag for returning the complex evaluation.

        Returns
        -------
        numpy.ndarray
            Evaluated data for the receiver.
        """

        imp = self._eval_impedance(src, mesh, f)
        if return_complex:
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
        f : .frequency_domain.fields.FieldsFDEM
            NSEM fields object for the source.
        du_dm_v : None,
            Supply pre-computed derivative?
        v : numpy.ndarray
            Vector of size
        adjoint : bool
            Whether to compute the ajoint operation.

        Returns
        -------
        numpy.ndarray
            Calculated derivative (nD,) (adjoint=False) and (nP,2) (adjoint=True) for both polarizations
        """
        return self._eval_impedance_deriv(
            src, mesh, f, du_dm_v=du_dm_v, v=v, adjoint=adjoint
        )


class Admittance(Impedance):
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
    locations : (n_loc, n_dim) numpy.ndarray
        Locations where the electric and magnetic fields are measured.
        If electric and magnetic fields are measured in different locations, please
        use `locations_e` and `locations_h` when instantiating.
    orientation : {'xx', 'xy', 'yx', 'yy', 'zx', 'zy'}
        Admittance receiver orientation. Specifies the admittance tensor element
        :math:`Y_{ij}` corresponding to the data. The data type is specified by
        the `component` input argument.
    component : {'real', 'imag'}
        Admittance data type. For the admittance element :math:`Y_{ij}` specified by the
        `orientation` input argument, the receiver can be set to compute the following:
        - 'real': Real component of the admittance (A/V)
        - 'imag': Imaginary component of the admittance (A/V)
    locations_e : (n_loc, n_dim) numpy.ndarray
        Locations where the horizontal electric fields are measured.
        Must be same size as `locations_h`.
    locations_h : (n_loc, n_dim) numpy.ndarray
        Locations where the magnetic fields are measured.
        Must be same size as `locations_e`.
    """

    def __init__(
        self,
        locations=None,
        orientation="xy",
        component="real",
        locations_e=None,
        locations_h=None,
    ):
        super().__init__(
            locations=locations,
            orientation=orientation,
            component=component,
            locations_e=locations_e,
            locations_h=locations_h,
        )

    @property
    def orientation(self):
        """Admittance receiver orientation.

        Specifies the admittance tensor element :math:`Y_{ij}` corresponding to the data.
        The data type is specified by the `component` property.

        Returns
        -------
        str
            Admittance receiver orientation. One of {'xx', 'xy', 'yx', 'yy', 'zx', 'zy'}.
        """
        return self._orientation

    @orientation.setter
    def orientation(self, var):
        self._orientation = validate_string(
            "orientation", var, string_list=("xx", "xy", "yx", "yy", "zx", "zy")
        )

    @property
    def component(self):
        r"""Data type; i.e. "real" or "imag".

        For the admittance element :math:`Y_{ij}`, the `component` property specifies
        whether the data are:
        - 'real': Real component of the admittance (A/V)
        - 'imag': Imaginary component of the admittance (A/V)

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
            ],
        )

    def _eval_admittance(self, src, mesh, f):
        if mesh.dim < 3:
            raise NotImplementedError(
                "Admittance receiver not implemented for dim < 3."
            )

        e = f[src, "e"]
        h = f[src, "h"]

        ex = _getP(self, mesh, "Ex", "e") @ e
        ey = _getP(self, mesh, "Ey", "e") @ e

        h = _getP(self, mesh, "F" + self.orientation[0], "h") @ h

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

        Pex = _getP(self, mesh, "Ex", "e")
        Pey = _getP(self, mesh, "Ey", "e")
        Ph = _getP(self, mesh, "F" + self.orientation[0], "h")

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

    def eval(self, src, mesh, f, return_complex=False):  # noqa: A003
        # Docstring inherited from parent class (Impedance).
        adm = self._eval_admittance(src, mesh, f)
        if return_complex:
            return adm
        # elif self.component == "apparent_resistivity":
        #     return _alpha(src) * (adm.real**2 + adm.imag**2)
        # elif self.component == "phase":
        #     return 180 / np.pi * (np.arctan2(adm.imag, adm.real))
        else:
            return getattr(adm, self.component)

    def evalDeriv(self, src, mesh, f, du_dm_v=None, v=None, adjoint=False):
        # Docstring inherited from parent class (Impedance).
        return self._eval_admittance_deriv(
            src, mesh, f, du_dm_v=du_dm_v, v=v, adjoint=adjoint
        )


class Tipper(BaseRx):
    r"""Receiver class for tipper data (3D problems only).

    This class can be used to simulate AFMag tipper data, defined according to:

    .. math::
        \begin{bmatrix} T_{yy} & T_{zy} \end{bmatrix} =
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
    locations : (n_loc, n_dim) numpy.ndarray
        Mobile receiver locations.
    orientation : {'xx', 'yx', 'zx', 'zy', 'yy', 'zy'}
        Specifies the tipper element :math:`T_{ij}` corresponding to the data.
    component : {'real', 'imag'}
        Tipper data type. For the tipper element :math:`T_{ij}` specified by the `orientation`
        input argument, the receiver can be set to compute the following:
        - 'real': Real component of the tipper (unitless)
        - 'imag': Imaginary component of the tipper (unitless)
    locations_bs : (n_loc, n_dim) numpy.ndarray, optional
        Locations for remote horizontal magnetic field measurements (e.g. base station).
        Must be same shape as `locations`.
    locations_e : (n_loc, n_dim) numpy.ndarray, optional
        Deprecated property. To be removed in simpeg 0.20.0.
    locations_h : (n_loc, n_dim) numpy.ndarray, optional
        Deprecated property. To be removed in simpeg 0.20.0.
    """

    def __init__(
        self,
        locations,
        orientation="zx",
        component="real",
        locations_bs=None,
        locations_e=None,
        locations_h=None,
    ):
        self.orientation = orientation
        self.component = component

        # check if locations and locations_bs have been provided
        if locations_bs is not None:
            # check that locations are same size
            if locations_bs.size == locations.size:
                self._locations_bs = locations_bs
            else:
                raise Exception("location_bs needs to be same size as locations")

        else:
            # check shape of locations
            if isinstance(locations, list):
                if len(locations) == 1:
                    pass
                elif len(locations) == 2:
                    if locations[0].size != locations[1].size:
                        raise Exception(
                            "location_bs needs to be same size as locations"
                        )
                else:
                    raise Exception("incorrect size of list, must be length of 1 or 2")

                self._locations_bs = locations[0]
                locations = locations[-1]
            elif isinstance(locations, np.ndarray):
                self._locations_bs = locations
            else:
                raise Exception("locations need to be either a list or numpy array")

        super().__init__(locations)

        if (locations_e is not None) or (locations_h is not None):
            warnings.warn(
                (
                    "'locations_e' and 'locations_h' are deprecated properties that are unused by the Tipper class.",
                    "Receiver locations are set using 'locations' (and 'locations_bs').",
                    "These property will be removed in simpeg v.0.20.0.",
                ),
                DeprecationWarning,
                stacklevel=2,
            )

    @property
    def locations_bs(self):
        """Locations for remote horizontal magnetic field measurements.

        Must be same shape as the `locations` property.

        Returns
        -------
        (n_loc, n_dim) np.ndarray
            Locations for remote horizontal magnetic field measurements.
        """
        return self._locations_bs

    @locations_bs.setter
    def locations_bs(self, locs):
        self._locations_bs = validate_ndarray_with_shape(
            "locations_bs", locs, shape=("*", "*"), dtype=float
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

        if self.locations_bs is not None:
            is_tipper_bs = True
        else:
            is_tipper_bs = False

        hx = _getP(self, mesh, "Fx", "h", is_tipper_bs) @ h
        hy = _getP(self, mesh, "Fy", "h", is_tipper_bs) @ h
        hz = _getP(self, mesh, "F" + self.orientation[0], "h") @ h

        if self.orientation[1] == "x":
            h = -hy
        else:
            h = hx

        top = h[:, 0] * hz[:, 1] - h[:, 1] * hz[:, 0]
        bot = hx[:, 0] * hy[:, 1] - hx[:, 1] * hy[:, 0]
        return top / bot

    def _eval_tipper_deriv(self, src, mesh, f, du_dm_v=None, v=None, adjoint=False):
        # will grab both primary and secondary and sum them!
        h = f[src, "h"]

        if self.locations_bs is not None:
            is_tipper_bs = True
        else:
            is_tipper_bs = False

        Phx = _getP(self, mesh, "Fx", "h", is_tipper_bs)
        Phy = _getP(self, mesh, "Fy", "h", is_tipper_bs)
        Phz = _getP(self, mesh, "F" + self.orientation[0], "h")
        hx = Phx @ h
        hy = Phy @ h
        hz = Phz @ h

        if self.orientation[1] == "x":
            h = -hy
        else:
            h = hx

        top = h[:, 0] * hz[:, 1] - h[:, 1] * hz[:, 0]
        bot = hx[:, 0] * hy[:, 1] - hx[:, 1] * hy[:, 0]
        tip = top / bot

        if adjoint:
            # Work backwards!
            gtop_v = (v / bot)[..., None]
            gbot_v = (-tip * v / bot)[..., None]
            n_d = self.nD

            ghx_v = np.c_[hy[:, 1], -hy[:, 0]] * gbot_v
            ghy_v = np.c_[-hx[:, 1], hx[:, 0]] * gbot_v
            ghz_v = np.c_[-h[:, 1], h[:, 0]] * gtop_v
            gh_v = np.c_[hz[:, 1], -hz[:, 0]] * gtop_v

            if self.orientation[1] == "x":
                ghy_v -= gh_v
            else:
                ghx_v += gh_v

            if v.ndim == 2:
                # collapse into a long list of n_d vectors
                ghx_v = ghx_v.reshape((n_d, -1))
                ghy_v = ghy_v.reshape((n_d, -1))
                ghz_v = ghz_v.reshape((n_d, -1))

            gh_v = Phx.T @ ghx_v + Phy.T @ ghy_v + Phz.T @ ghz_v
            return f._hDeriv(src, None, gh_v, adjoint=True)

        dh_v = f._hDeriv(src, du_dm_v, v, adjoint=False)
        dhx_v = Phx @ dh_v
        dhy_v = Phy @ dh_v
        dhz_v = Phz @ dh_v
        if self.orientation[1] == "x":
            dh_v = -dhy_v
        else:
            dh_v = dhx_v

        dtop_v = (
            h[:, 0] * dhz_v[:, 1]
            + dh_v[:, 0] * hz[:, 1]
            - h[:, 1] * dhz_v[:, 0]
            - dh_v[:, 1] * hz[:, 0]
        )
        dbot_v = (
            hx[:, 0] * dhy_v[:, 1]
            + dhx_v[:, 0] * hy[:, 1]
            - hx[:, 1] * dhy_v[:, 0]
            - dhx_v[:, 1] * hy[:, 0]
        )

        return (bot * dtop_v - top * dbot_v) / (bot * bot)

    def eval(self, src, mesh, f, return_complex=False):  # noqa: A003
        # Docstring inherited from parent class (Impedance).
        rx_eval_complex = self._eval_tipper(src, mesh, f)
        return getattr(rx_eval_complex, self.component)

    def evalDeriv(self, src, mesh, f, du_dm_v=None, v=None, adjoint=False):
        # Docstring inherited from parent class (Impedance).
        if adjoint:
            if self.component == "imag":
                v = -1j * v
        imp_deriv = self._eval_tipper_deriv(
            src, mesh, f, du_dm_v=du_dm_v, v=v, adjoint=adjoint
        )
        if adjoint:
            return imp_deriv
        return getattr(imp_deriv, self.component)


@deprecate_class(removal_version="0.20.0", error=False)
class PointNaturalSource(Impedance):
    """This class is deprecated and will be removed in simpeg v0.20.0.
    Please use :class:`.Impedance`."""

    def __init__(
        self,
        locations=None,
        orientation="xy",
        component="real",
        locations_e=None,
        locations_h=None,
    ):
        super().__init__(
            locations=locations,
            orientation=orientation,
            component=component,
            locations_e=locations_e,
            locations_h=locations_h,
        )


@deprecate_class(removal_version="0.20.0", error=False)
class Point3DTipper(Impedance):
    """This class is deprecated and will be removed in simpeg v0.20.0.
    Please use :class:`.Tipper`."""

    def __init__(
        self,
        locations,
        orientation="zx",
        component="real",
        locations_bs=None,
        locations_e=None,
        locations_h=None,
    ):
        super().__init__(
            locations=locations,
            orientation=orientation,
            component=component,
            locations_bs=locations_bs,
            locations_e=locations_e,
            locations_h=locations_h,
        )
