from ...utils.code_utils import deprecate_class, validate_string

import numpy as np
from scipy.constants import mu_0

# import properties

from ...survey import BaseRx


def _alpha(src):
    return 1 / (2 * np.pi * mu_0 * src.frequency)


class PointNaturalSource(BaseRx):
    """Point receiver class for magnetotelluric simulations.

    Assumes that the data locations are standard xyz coordinates;
    i.e. (x,y,z) is (Easting, Northing, up).

    Parameters
    ----------
    locations : (n_loc, n_dim) numpy.ndarray
        Receiver locations.
    orientation : {'xx', 'xy', 'yx', 'yy'}
        MT receiver orientation.
    component : {'real', 'imag', 'apparent_resistivity', 'phase'}
        MT data type.
    """

    # component = properties.StringChoice(
    #     "component of the field (real, imag, apparent_resistivity, or phase)",
    #     {
    #         "real": ["re", "in-phase", "in phase"],
    #         "imag": ["imaginary", "im", "out-of-phase", "out of phase"],
    #         "apparent_resistivity": [
    #             "apparent resistivity",
    #             "apparent-resistivity",
    #             "app_rho",
    #             "app_res",
    #         ],
    #         "phase": ["phi"],
    #     },
    # )

    # orientation = properties.StringChoice(
    #     "orientation of the receiver. Must currently be 'xy', 'yx'",
    #     ["xx", "xy", "yx", "yy"],
    # )

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
    def component(self):
        """Data type; i.e. "real", "imag", "apparent_resistivity", "phase"

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
            ],
        )

    @property
    def orientation(self):
        """Orientation of the receiver.

        Returns
        -------
        str
            Orientation of the receiver. One of {'xx', 'xy', 'yx', 'yy'}
        """
        return self._orientation

    @orientation.setter
    def orientation(self, var):
        self._orientation = validate_string(
            "orientation", var, string_list=("xx", "xy", "yx", "yy")
        )

    @property
    def locations_e(self):
        """Electric field measurement locations

        Returns
        -------
        numpy.ndarray
            Location where the electric field is measured for all receiver data
        """
        return self._locations_e

    @property
    def locations_h(self):
        """Magnetic field measurement locations

        Returns
        -------
        numpy.ndarray
            Location where the magnetic field is measured for all receiver data
        """
        return self._locations_h

    def getP(self, mesh, projected_grid, field="e"):
        """Projection matrices for all components collected by the receivers

        Note projection matrices are stored as a dictionary listed by meshes.

        Parameters
        ----------
        mesh : discretize.base.BaseMesh
            The mesh on which the discrete set of equations is solved
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

        field : str, default = "e"
            Whether to project electric or magnetic fields from mesh.
            Choose "e" or "h"
        """
        if mesh.dim < 3:
            return super().getP(mesh, projected_grid)

        if (mesh, projected_grid) in self._Ps:
            return self._Ps[(mesh, projected_grid, field)]

        if field == "e":
            locs = self.locations_e
        else:
            locs = self.locations_h
        P = mesh.get_interpolation_matrix(locs, projected_grid)
        if self.storeProjections:
            self._Ps[(mesh, projected_grid, field)] = P
        return P

    def _eval_impedance(self, src, mesh, f):
        if mesh.dim < 3 and self.orientation in ["xx", "yy"]:
            return 0.0
        e = f[src, "e"]
        h = f[src, "h"]
        if mesh.dim == 3:
            if self.orientation[0] == "x":
                e = self.getP(mesh, "Ex", "e") @ e
            else:
                e = self.getP(mesh, "Ey", "e") @ e

            hx = self.getP(mesh, "Fx", "h") @ h
            hy = self.getP(mesh, "Fy", "h") @ h
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
                Pe = self.getP(mesh, "Ex", "e")
                e = Pe @ e
            else:
                Pe = self.getP(mesh, "Ey", "e")
                e = Pe @ e

            Phx = self.getP(mesh, "Fx", "h")
            Phy = self.getP(mesh, "Fy", "h")
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
                bot = -bot

            imp = top / bot

        if adjoint:
            if self.component == "phase":
                # gradient of arctan2(y, x) is (-y/(x**2 + y**2), x/(x**2 + y**2))
                v = 180 / np.pi * imp / (imp.real ** 2 + imp.imag ** 2) * v
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

            if mesh.dim == 3:
                ghx_v = np.c_[hy[:, 1], -hy[:, 0]] * gbot_v[:, None]
                ghy_v = np.c_[-hx[:, 1], hx[:, 0]] * gbot_v[:, None]
                ge_v = np.c_[h[:, 1], -h[:, 0]] * gtop_v[:, None]
                gh_v = np.c_[-e[:, 1], e[:, 0]] * gtop_v[:, None]

                if self.orientation[1] == "x":
                    ghy_v += gh_v
                else:
                    ghx_v -= gh_v

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
            amp2 = imp.imag ** 2 + imp.real ** 2
            deriv_re = -imp.imag / amp2 * imp_deriv.real
            deriv_im = imp.real / amp2 * imp_deriv.imag

            rx_deriv = (180 / np.pi) * (deriv_re + deriv_im)
        else:
            rx_deriv = getattr(imp_deriv, self.component)
        return rx_deriv

    def eval(self, src, mesh, f, return_complex=False):
        """
        Project the fields to natural source data.

        Parameters
        ----------
        src : SimPEG.electromagnetics.frequency_domain.sources.BaseFDEMSrc
            NSEM source
        mesh : discretize.TensorMesh mesh
            Mesh on which the discretize solution is obtained
        f : SimPEG.electromagnetics.frequency_domain.fields.FieldsFDEM
            NSEM fields object of the source
        return_complex : bool (optional)
            Flag for return the complex evaluation

        Returns
        -------
        numpy.ndarray
            Evaluated data for the receiver
        """

        imp = self._eval_impedance(src, mesh, f)
        if return_complex:
            return imp
        elif self.component == "apparent_resistivity":
            return _alpha(src) * (imp.real ** 2 + imp.imag ** 2)
        elif self.component == "phase":
            return 180 / np.pi * (np.arctan2(imp.imag, imp.real))
        else:
            return getattr(imp, self.component)

    def evalDeriv(self, src, mesh, f, du_dm_v=None, v=None, adjoint=False):
        """Derivative of projection with respect to the fields

        Parameters
        ----------
        str : SimPEG.electromagnetics.frequency_domain.sources.BaseFDEMSrc
            NSEM source
        mesh : discretize.TensorMesh
            Mesh on which the discretize solution is obtained
        f : SimPEG.electromagnetics.frequency_domain.fields.FieldsFDEM
            NSEM fields object of the source
        du_dm_v : None,
            Supply pre-computed derivative?
        v : numpy.ndarray
            Vector of size
        adjoint : bool, default = ``False``
            If ``True``, compute the adjoint operation

        Returns
        -------
        numpy.ndarray
            Calculated derivative (nD,) (adjoint=False) and (nP,2) (adjoint=True) for both polarizations
        """
        return self._eval_impedance_deriv(
            src, mesh, f, du_dm_v=du_dm_v, v=v, adjoint=adjoint
        )


class Point3DTipper(PointNaturalSource):
    """Point receiver class for Z-axis tipper simulations.

    Assumes that the data locations are standard xyz coordinates;
    i.e. (x,y,z) is (Easting, Northing, up).

    Parameters
    ----------
    locations : (n_loc, n_dim) numpy.ndarray
        Receiver locations.
    orientation : str, default = 'zx'
        NSEM receiver orientation. Must be one of {'zx', 'zy'}
    component : str, default = 'real'
        NSEM data type. Choose one of {'real', 'imag', 'apparent_resistivity', 'phase'}
    """

    # orientation = properties.StringChoice(
    #     "orientation of the receiver. Must currently be 'zx', 'zy'", ["zx", "zy"]
    # )

    def __init__(
        self,
        locations,
        orientation="zx",
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
        """Orientation of the receiver.

        Returns
        -------
        str
            Orientation of the receiver. One of {'zx', 'zy'}
        """
        return self._orientation

    @orientation.setter
    def orientation(self, var):
        self._orientation = validate_string(
            "orientation", var, string_list=("zx", "zy")
        )

    def _eval_tipper(self, src, mesh, f):
        # will grab both primary and secondary and sum them!
        h = f[src, "h"]

        hx = self.getP(mesh, "Fx", "h") @ h
        hy = self.getP(mesh, "Fy", "h") @ h
        hz = self.getP(mesh, "Fz", "h") @ h

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

        Phx = self.getP(mesh, "Fx", "h")
        Phy = self.getP(mesh, "Fy", "h")
        Phz = self.getP(mesh, "Fz", "h")
        hx = Phx @ h
        hy = Phy @ h
        hz = Phz @ h

        if self.orientation[1] == "x":
            h = -hy
        else:
            h = hx

        top = h[:, 0] * hz[:, 1] - h[:, 1] * hz[:, 0]
        bot = hx[:, 0] * hy[:, 1] - hx[:, 1] * hy[:, 0]
        imp = top / bot

        if adjoint:
            # Work backwards!
            gtop_v = (v / bot)[:, None]
            gbot_v = (-imp * v / bot)[:, None]

            ghx_v = np.c_[hy[:, 1], -hy[:, 0]] * gbot_v
            ghy_v = np.c_[-hx[:, 1], hx[:, 0]] * gbot_v
            ghz_v = np.c_[-h[:, 1], h[:, 0]] * gtop_v
            gh_v = np.c_[hz[:, 1], -hz[:, 0]] * gtop_v

            if self.orientation[1] == "x":
                ghy_v -= gh_v
            else:
                ghx_v += gh_v

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

    def eval(self, src, mesh, f, return_complex=False):
        """
        Project the fields to natural source data.

        Parameters
        ----------
        src : SimPEG.electromagnetics.frequency_domain.sources.BaseFDEMSrc
            NSEM source
        mesh : discretize.TensorMesh mesh
            Mesh on which the discretize solution is obtained
        f : SimPEG.electromagnetics.frequency_domain.fields.FieldsFDEM
            NSEM fields object of the source
        return_complex : bool (optional)
            Flag for return the complex evaluation

        Returns
        -------
        numpy.ndarray
            Evaluated data for the receiver
        """

        rx_eval_complex = self._eval_tipper(src, mesh, f)

        return getattr(rx_eval_complex, self.component)

    def evalDeriv(self, src, mesh, f, du_dm_v=None, v=None, adjoint=False):
        """Derivative of projection with respect to the fields

        Parameters
        ----------
        str : SimPEG.electromagnetics.frequency_domain.sources.BaseFDEMSrc
            NSEM source
        mesh : discretize.TensorMesh
            Mesh on which the discretize solution is obtained
        f : SimPEG.electromagnetics.frequency_domain.fields.FieldsFDEM
            NSEM fields object of the source
        du_dm_v : None,
            Supply pre-computed derivative?
        v : numpy.ndarray
            Vector of size
        adjoint : bool, default = ``False``
            If ``True``, compute the adjoint operation

        Returns
        -------
        numpy.ndarray
            Calculated derivative (nD,) (adjoint=False) and (nP,2) (adjoint=True) for both polarizations
        """

        if adjoint:
            if self.component == "imag":
                v = -1j * v
        imp_deriv = self._eval_tipper_deriv(
            src, mesh, f, du_dm_v=du_dm_v, v=v, adjoint=adjoint
        )
        if adjoint:
            return imp_deriv
        return getattr(imp_deriv, self.component)


############
# Deprecated
############


@deprecate_class(removal_version="0.19.0", error=True)
class Point_impedance1D(PointNaturalSource):
    pass


@deprecate_class(removal_version="0.19.0", error=True)
class Point_impedance3D(PointNaturalSource):
    pass


@deprecate_class(removal_version="0.19.0", error=True)
class Point_tipper3D(Point3DTipper):
    pass
