""" Module RxNSEM.py

Receivers for the NSEM problem

"""
from ...utils.code_utils import deprecate_class

import numpy as np
from scipy.constants import mu_0
import properties

from ...survey import BaseRx


def _alpha(src):
    return 1 / (2 * np.pi * mu_0 * src.frequency)


class PointNaturalSource(BaseRx):
    """
    Natural source receiver base class.

    Assumes that the data locations are xyz coordinates.

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    :param string component: real or imaginary component 'real' or 'imag'
    """

    component = properties.StringChoice(
        "component of the field (real, imag, apparent_resistivity, or phase)",
        {
            "real": ["re", "in-phase", "in phase"],
            "imag": ["imaginary", "im", "out-of-phase", "out of phase"],
            "apparent_resistivity": [
                "apparent resistivity",
                "apparent-resistivity",
                "app_rho",
                "app_res",
            ],
            "phase": ["phi"],
        },
    )

    orientation = properties.StringChoice(
        "orientation of the receiver. Must currently be 'xy', 'yx'", ["xx", "xy", "yx", "yy"],
    )

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
    def locations_e(self):
        return self._locations_e

    @property
    def locations_h(self):
        return self._locations_h

    def getP(self, mesh, projGLoc=None, field="e"):
        """
            Returns the projection matrices as a
            list for all components collected by
            the receivers.

            .. note::

                Projection matrices are stored as a dictionary listed by meshes.
        """
        if mesh.dim < 3:
            return super().getP(mesh, projGloc=projGLoc)
        if projGLoc is None:
            projGLoc = self.projGLoc

        if (mesh, projGLoc) in self._Ps:
            return self._Ps[(mesh, projGLoc, field)]

        if field == "e":
            locs = self.locations_e
        else:
            locs = self.locations_h
        P = mesh.getInterpolationMat(locs, projGLoc)
        if self.storeProjections:
            self._Ps[(mesh, projGLoc, field)] = P
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

        :param SimPEG.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: NSEM source
        :param discretize.TensorMesh mesh: Mesh defining the topology of the problem
        :param SimPEG.electromagnetics.frequency_domain.fields.FieldsFDEM f: NSEM fields object of the source
        :param bool (optional) return_complex: Flag for return the complex evaluation
        :rtype: numpy.ndarray
        :return: Evaluated data for the receiver
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
        """method evalDeriv

        The derivative of the projection wrt u

        :param SimPEG.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: NSEM source
        :param discretize.TensorMesh mesh: Mesh defining the topology of the problem
        :param SimPEG.electromagnetics.frequency_domain.fields.FieldsFDEM f: NSEM fields object of the source
        :param numpy.ndarray v: vector of size (nU,) (adjoint=False) and size (nD,) (adjoint=True)
        :rtype: numpy.ndarray
        :return: Calculated derivative (nD,) (adjoint=False) and (nP,2) (adjoint=True) for both polarizations
        """
        return self._eval_impedance_deriv(
            src, mesh, f, du_dm_v=du_dm_v, v=v, adjoint=adjoint
        )


class Point3DTipper(PointNaturalSource):
    """
    Natural source 3D tipper receiver base class

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    :param string component: real or imaginary component 'real' or 'imag'
    """

    orientation = properties.StringChoice(
        "orientation of the receiver. Must currently be 'zx', 'zy'", ["zx", "zy"]
    )

    def __init__(
        self,
        locations=None,
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

        :param SimPEG.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: The source of the fields to project
        :param discretize.TensorMesh mesh: Mesh defining the topology of the problem
        :param SimPEG.electromagnetics.frequency_domain.fields.FieldsFDEM f: Natural source fields object to project
        :rtype: numpy.ndarray
        :return: Evaluated component of the impedance data
        """

        rx_eval_complex = self._eval_tipper(src, mesh, f)

        return getattr(rx_eval_complex, self.component)

    def evalDeriv(self, src, mesh, f, du_dm_v=None, v=None, adjoint=False):
        """
        The derivative of the projection wrt u

        :param SimPEG.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: NSEM source
        :param discretize.TensorMesh mesh: Mesh defining the topology of the problem
        :param SimPEG.electromagnetics.frequency_domain.fields.FieldsFDEM f: NSEM fields object of the source
        :param numpy.ndarray v: Random vector of size
        :rtype: numpy.ndarray
        :return: Calculated derivative (nD,) (adjoint=False) and (nP,2) (adjoint=True)
            for both polarizations
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
