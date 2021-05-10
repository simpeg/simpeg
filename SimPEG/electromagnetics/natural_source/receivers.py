""" Module RxNSEM.py

Receivers for the NSEM problem

"""
from ...utils.code_utils import deprecate_class

import numpy as np
import scipy as sp
from scipy.constants import mu_0
import properties

from ...utils import sdiag, mkvc
from ...utils import spzeros
from ...survey import BaseRx


class BaseRxNSEM_Point(BaseRx):
    """
    Natural source receiver base class.

    Assumes that the data locations are xyz coordinates.

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    :param string component: real or imaginary component 'real' or 'imag'
    """

    component = properties.StringChoice(
        "component of the field (real or imag)",
        {
            "real": ["re", "in-phase", "in phase"],
            "imag": ["imaginary", "im", "out-of-phase", "out of phase"],
        },
    )

    def __init__(
        self,
        locations=None,
        orientation=None,
        component=None,
        locations_e=None,
        locations_h=None
    ):
        self.orientation = orientation
        self.component = component
        self.reference_locations = None

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
                    self._locations_e = locations[:, 0]
                    self._locations_h = locations[:, 1]
                elif len(locations) == 1:
                    self._locations_e = locations
                    self._locations_h = locations
                else:
                    raise Exception("incorrect size of list, must be length of 1 or 2")
            elif isinstance(locations, np.ndarray):
                self._locations_e = locations
                self._locations_h = locations
            else:
                raise Exception("locations need to be either a list or numpy array")
        else:
            raise Exception(
                "locations need to be either E & H coincident or seperate items"
            )

        BaseRx.__init__(self, locations)

    def locations_e(self):
        return self._locations_e

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
        if projGLoc is None:
            projGLoc = self.projGLoc

        if (mesh, projGLoc) in self._Ps:
            return self._Ps[(mesh, projGLoc, field)]

        if field == "e":
            locs = self.locations_e()
        else:
            if self.reference_locations is not None:
                if ('x' in projGLoc) or ('y' in projGLoc):
                    # if self.ref_locations != None:
                    locs = self.reference_locations
                else:
                    locs = self.locations_h()
                    # else:
                    #     raise NotImplementedError("please set a ref location if using ztem")
            else:
                locs = self.locations_h()
        P = mesh.getInterpolationMat(locs, projGLoc)
        if self.storeProjections:
            self._Ps[(mesh, projGLoc, field)] = P
        return P

    def eval(self, src, mesh, f, return_complex=False):
        """
        Function to evaluate datum for this receiver
        """
        raise NotImplementedError("SimPEG.EM.NSEM receiver has to have an eval method")

    def evalDeriv(self, src, mesh, f, v, adjoint=False):
        """
        Function to evaluate datum for this receiver
        """
        raise NotImplementedError(
            "SimPEG.EM.NSEM receiver has to have an evalDeriv method"
        )


class Point1DImpedance(BaseRx):
    """
    Natural source 1D impedance receiver class

    :param string component: real or imaginary component 'real' or 'imag'
    """

    component = properties.StringChoice(
        "component of the field (real or imag)",
        {
            "real": ["re", "in-phase", "in phase"],
            "imag": ["imaginary", "im", "out-of-phase", "out of phase"],
        },
    )

    orientation = "yx"

    def __init__(self, locs, component="real"):
        self.component = component
        BaseRx.__init__(self, locs)

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        if value is getattr(self, "_mesh", None):
            pass
        else:
            self._mesh = value

    @property
    def src(self):
        return self._src

    @src.setter
    def src(self, value):
        self._src = value

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, value):
        self._f = value

    def _eval_impedance(self, src, mesh, f):
        # NOTE: Maybe set this as a property
        PEx = self.getP(mesh, "Fx")
        PHy = self.getP(mesh, "Ex")

        e = f[src, "e"]
        h = f[src, "h"]

        e_top = PEx @ e[:, 0]
        h_bot = PHy @ -h[:, 0]

        return e_top / h_bot

    def _eval_impedance_deriv(self, src, mesh, f, du_dm_v=None, v=None, adjoint=False):
        PEx = self.getP(mesh, "Fx")
        PHy = self.getP(mesh, "Ex")

        e = f[src, "e"]
        h = f[src, "h"]

        top = PEx @ e[:, 0]
        bot = PHy @ h[:, 0]

        imp = top / -bot

        if adjoint:
            # Work backwards!
            gtop_v = (v / bot)
            gbot_v = (-imp * v / bot)

            gh_v = PHy.T @ gbot_v
            ge_v = PEx.T @ gtop_v

            gfu_h_v, gfm_h_v = f._hDeriv(src, None, gh_v, adjoint=True)
            gfu_e_v, gfm_e_v = f._eDeriv(src, None, -ge_v, adjoint=True)

            return gfu_h_v + gfu_e_v, gfm_h_v + gfm_e_v

        de_v = PEx @ f._eDeriv(src, du_dm_v, v, adjoint=False)
        dh_v = PHy @ f._hDeriv(src, du_dm_v, v, adjoint=False)

        return (1 / bot) * (-de_v - imp * dh_v)

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

        if adjoint:
            if self.component == "imag":
                v = -1j * v

        imp_deriv = self._eval_impedance_deriv(
            src, mesh, f, du_dm_v=du_dm_v, v=v, adjoint=adjoint
        )

        if adjoint:
            return imp_deriv

        return getattr(imp_deriv, self.component)


class Point3DImpedance(BaseRxNSEM_Point):
    """
    Natural source 3D impedance receiver class

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string orientation: receiver orientation 'xx', 'xy', 'yx' or 'yy'
    :param string component: real or imaginary component 'real' or 'imag'
    """

    orientation = properties.StringChoice(
        "orientation of the receiver. Must currently be 'xx', 'xy', 'yx', 'yy'",
        ["xx", "xy", "yx", "yy"],
    )

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

    def _eval_impedance(self, src, mesh, f):
        e = f[src, "e"]  # will grab both primary and secondary and sum them!
        h = f[src, "h"]

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
        return top / bot

    def _eval_impedance_deriv(self, src, mesh, f, du_dm_v=None, v=None, adjoint=False):
        e = f[src, "e"]
        h = f[src, "h"]

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

        if adjoint:
            
            # Work backwards!
            gtop_v = np.c_[v] / bot[:, None]
            gbot_v = -imp[:, None] * np.c_[v] / bot[:, None]

            ghx_v = np.einsum('ij,ik->ijk', gbot_v, np.c_[hy[:, 1], -hy[:, 0]]).reshape((hy.shape[0], -1))
            ghy_v = np.einsum('ij,ik->ijk', gbot_v, np.c_[-hx[:, 1], hx[:, 0]]).reshape((hx.shape[0], -1))
            ge_v = np.einsum('ij,ik->ijk', gtop_v, np.c_[h[:, 1], -h[:, 0]]).reshape((h.shape[0], -1))
            gh_v = np.einsum('ij,ik->ijk', gtop_v, np.c_[-e[:, 1], e[:, 0]]).reshape((e.shape[0], -1))

            if self.orientation[1] == "x":
                ghy_v += gh_v
            else:
                ghx_v -= gh_v

            gh_v = Phx.T @ ghx_v + Phy.T @ ghy_v
            ge_v = Pe.T @ ge_v

            gfu_h_v, gfm_h_v = f._hDeriv(src, None, gh_v, adjoint=True)
            gfu_e_v, gfm_e_v = f._eDeriv(src, None, ge_v, adjoint=True)

            return gfu_h_v + gfu_e_v, gfm_h_v + gfm_e_v

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

        return (bot * dtop_v - top * dbot_v) / (bot * bot)

    def eval(self, src, mesh, f, return_complex=False):
        """
        Project the fields to natural source data.

        :param SimPEG.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: The source of the fields to project
        :param discretize.TensorMesh mesh: topological mesh corresponding to the fields
        :param SimPEG.electromagnetics.frequency_domain.fields.FieldsFDEM f: Natural source fields object to project
        :rtype: numpy.ndarray
        :return: component of the impedance evaluation
        """

        imp = self._eval_impedance(src, mesh, f)
        if return_complex:
            return imp
        else:
            return getattr(imp, self.component)

    def evalDeriv(self, src, mesh, f, du_dm_v=None, v=None, adjoint=False):
        """
        The derivative of the projection wrt u

        :param SimPEG.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: NSEM source
        :param discretize.TensorMesh mesh: Mesh defining the topology of the problem
        :param SimPEG.electromagnetics.frequency_domain.fields.FieldsFDEM f: NSEM fields object of the source
        :param numpy.ndarray v: vector of size (nU, 2) (adjoint=False) and size (nD,) (adjoint=True)
        :rtype: numpy.ndarray
        :return: Calculated derivative (nD,) (adjoint=False) and (nP,2) (adjoint=True) for both polarizations
        """

        if adjoint:
            if self.component == "imag":
                v = -1j * v
        imp_deriv = self._eval_impedance_deriv(
            src, mesh, f, du_dm_v=du_dm_v, v=v, adjoint=adjoint
        )
        if adjoint:
            return imp_deriv
        return getattr(imp_deriv, self.component)


class Point3DComplexResistivity(Point3DImpedance):
    """
    Natural source 3D impedance receiver class
    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string orientation: receiver orientation 'xx', 'xy', 'yx' or 'yy'
    :param string component: real or imaginary component 'real' or 'imag'
    """

    orientation = properties.StringChoice(
        "orientation of the receiver. Must currently be 'xx', 'xy', 'yx', 'yy'",
        ["xx", "xy", "yx", "yy"],
    )

    component = properties.StringChoice(
        "component of the field (real or imag)",
        {
            "apparent resistivity": [
                "apparent_resistivity",
                "apparent-resistivity",
                "app_rho",
            ],
            "phase": ["phi"],
        },
    )

    def __init__(
        self,
        locations=None,
        orientation="xy",
        component="apparent_resistivity",
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

    def _alpha(self, src):
        return 1 / (2 * np.pi * mu_0 * src.frequency)

    def _eval_rx_deriv(self, src, mesh, f, du_dm_v=None, v=None, adjoint=False):
        alpha = self._alpha(src)

        e = f[src, "e"]  # will grab both primary and secondary and sum them!
        h = f[src, "h"]

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

        if adjoint:
            if self.component == "phase":
                # gradient of arctan2(y, x) is (-y/(x**2 + y**2), x/(x**2 + y**2))
                v = 180 / np.pi * imp / (imp.real ** 2 + imp.imag ** 2) * v
                # switch real and imaginary, and negate real part of output
                v = -v.imag - 1j * v.real
                # imaginary part gets extra (-) due to conjugate transpose
            else:
                v = 2 * alpha * imp * v
                v = v.real - 1j * v.imag
                # imaginary part gets extra (-) due to conjugate transpose

            # the multipliers
            gtop_v = np.c_[v] / bot[:, None]
            gbot_v = -imp[:, None] * np.c_[v] / bot[:, None]

            # multiply the fields by v
            ghx_v = np.einsum('ij,ik->ijk', gbot_v, np.c_[hy[:, 1], -hy[:, 0]]).reshape((hy.shape[0], -1))
            ghy_v = np.einsum('ij,ik->ijk', gbot_v, np.c_[-hx[:, 1], hx[:, 0]]).reshape((hx.shape[0], -1))
            ge_v = np.einsum('ij,ik->ijk', gtop_v, np.c_[h[:, 1], -h[:, 0]]).reshape((h.shape[0], -1))
            gh_v = np.einsum('ij,ik->ijk', gtop_v, np.c_[-e[:, 1], e[:, 0]]).reshape((e.shape[0], -1))

            if self.orientation[1] == "x":
                ghy_v += gh_v
            else:
                ghx_v -= gh_v

            gh_v_imag = Phx.T @ ghx_v + Phy.T @ ghy_v
            ge_v_imag = Pe.T @ ge_v

            # take derivative
            gfu_h_v, gfm_h_v = f._hDeriv(src, None, gh_v_imag, adjoint=True)
            gfu_e_v, gfm_e_v = f._eDeriv(src, None, ge_v_imag, adjoint=True)

            return gfu_h_v + gfu_e_v, gfm_h_v + gfm_e_v
        else:
            de_v = Pe @ f._eDeriv(src, du_dm_v, v, adjoint=False)
            dh_v = f._hDeriv(src, du_dm_v, v, adjoint=False)
            dhx_v = Phx @ dh_v
            dhy_v = Phy @ dh_v
            if self.orientation[1] == "x":
                dh_v = dhy_v
            else:
                dh_v = -dhx_v

            top_dm_v = (
                e[:, 0] * dh_v[:, 1]
                + de_v[:, 0] * h[:, 1]
                - e[:, 1] * dh_v[:, 0]
                - de_v[:, 1] * h[:, 0]
            )
            bot_dm_v = (
                hx[:, 0] * dhy_v[:, 1]
                + dhx_v[:, 0] * hy[:, 1]
                - hx[:, 1] * dhy_v[:, 0]
                - dhx_v[:, 1] * hy[:, 0]
            )

            imp_deriv = (bot * top_dm_v - top * bot_dm_v) / (bot * bot)
            if self.component == "apparent resistivity":
                rx_deriv = (
                    2 * alpha * (imp.real * imp_deriv.real + imp.imag * imp_deriv.imag)
                )
            elif self.component == "phase":
                amp2 = imp.imag ** 2 + imp.real ** 2
                deriv_re = -imp.imag / amp2 * imp_deriv.real
                deriv_im = imp.real / amp2 * imp_deriv.imag

                rx_deriv = (180 / np.pi) * (deriv_re + deriv_im)
            return rx_deriv

    def eval(self, src, mesh, f, return_complex=False):
        """
        Project the fields to natural source data.
        :param SimPEG.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: The source of the fields to project
        :param discretize.TensorMesh mesh: topological mesh corresponding to the fields
        :param SimPEG.electromagnetics.frequency_domain.fields.FieldsFDEM f: Natural source fields object to project
        :rtype: numpy.ndarray
        :return: component of the impedance evaluation
        """

        alpha = self._alpha(src)

        # Calculate the complex value
        rx_eval_complex = self._eval_impedance(src, mesh, f)

        if self.component == "apparent resistivity":
            return alpha * (rx_eval_complex.real ** 2 + rx_eval_complex.imag ** 2)
        elif self.component == "phase":
            return (
                180 / np.pi * (np.arctan2(rx_eval_complex.imag, rx_eval_complex.real))
            )

    def evalDeriv(self, src, mesh, f, du_dm_v=None, v=None, adjoint=False):
        """
        The derivative of the projection wrt u
        :param SimPEG.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: NSEM source
        :param discretize.TensorMesh mesh: Mesh defining the topology of the problem
        :param SimPEG.electromagnetics.frequency_domain.fields.FieldsFDEM f: NSEM fields object of the source
        :param numpy.ndarray v: vector of size (nU,) (adjoint=False) and size (nD,) (adjoint=True)
        :rtype: numpy.ndarray
        :return: Calculated derivative (nD,) (adjoint=False) and (nP,2) (adjoint=True) for both polarizations
        """

        return self._eval_rx_deriv(src, mesh, f, du_dm_v=du_dm_v, v=v, adjoint=adjoint)


class Point3DTipper(BaseRxNSEM_Point):
    """
    Natural source 3D tipper receiver base class

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    :param string component: real or imaginary component 'real' or 'imag'
    """

    orientation = properties.StringChoice(
        "orientation of the receiver. Must currently be 'zx', 'zy'", ["zx", "zy"]
    )

    def __init__(self, locs, orientation="zx", component="real"):

        super().__init__(locs, orientation=orientation, component=component)

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


##################################################
# Receiver for 1D Analytic


class AnalyticReceiver1D(BaseRx):
    """
    Receiver class for the 1D and pseudo-3D problems. For the 1D problem,
    locations are not necessary. For the 3D problem, xyz positions are required.

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string component: 'real'|'imag'|'app_res'
    """

    component = properties.StringChoice(
        "component of the field (real, imag or app_res)",
        {
            "real": ["re", "in-phase", "in phase"],
            "imag": ["imaginary", "im", "out-of-phase", "out of phase"],
            "apparent resistivity": [
                "apparent_resistivity",
                "apparent-resistivity",
                "app_res",
            ],
            "phase": ["phi"],
        },
    )

    def __init__(self, locations=None, component=None):
        self.component = component

        BaseRx.__init__(self, locations)

    @property
    def nD(self):
        """Number of data in the receiver."""

        if self.locations == None:
            return 1
        else:
            return self.locations.shape[0]


############
# Deprecated
############


@deprecate_class(removal_version="0.15.0")
class Point_impedance1D(Point1DImpedance):
    pass


@deprecate_class(removal_version="0.15.0")
class Point_impedance3D(Point3DImpedance):
    pass


@deprecate_class(removal_version="0.15.0")
class Point_tipper3D(Point3DTipper):
    pass
