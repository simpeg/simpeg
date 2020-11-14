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
        orientation=None,
        component=None,
        locations_e=None,
        locations_b=None,
    ):
        self.orientation = orientation
        self.component = component
        self.locations_e = locations_e
        self.locations_b = locations_b

        if locations is None:
            locations = np.hstack([locations_e, locations_b])

        BaseRx.__init__(self, locations)

    def _locs_e(self):
        if self.locations_e is None:
            if self.locations.ndim == 3:
                loc = self.locations[:, :, 0]
            else:
                loc = self.locations
            return loc
        else:
            return self.locations_e

    def _locs_h(self):
        if self.locations_b is None:
            if self.locations.ndim == 3:
                loc = self.locations[:, :, 1]
            else:
                loc = self.locations
            return loc
        else:
            return self.locations_b

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
            locs = self._locs_e()
        else:
            locs = self._locs_h()
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

    @property
    def Pex(self):
        if getattr(self, "_Pex", None) is None:
            self._Pex = self._mesh.getInterpolationMat(self.locations[:, -1], "Fx")
        return self._Pex

    @property
    def Pbx(self):
        if getattr(self, "_Pbx", None) is None:
            self._Pbx = self._mesh.getInterpolationMat(self.locations[:, -1], "Ex")
        return self._Pbx

    @property
    def _ex(self):
        return self.Pex * mkvc(self.f[self.src, "eSolution"], 2)

    @property
    def _hx(self):
        return self.Pbx * mkvc(self.f[self.src, "b_1d"], 2) / mu_0

    def _ex_u(self, v, adjoint=False):
        if adjoint:
            return self.f._eDeriv_u(self.src, self.Pex.T * v, adjoint=True)
        return self.Pex * self.f._eDeriv_u(self.src, v)

    def _hx_u(self, v, adjoint=False):
        if adjoint:
            return self.f._bDeriv_u(self.src, self.Pbx.T * v, adjoint=True) / mu_0
        return self.Pbx * self.f._bDeriv_u(self.src, v) / mu_0

    @property
    def _Hd(self):
        return sdiag(1.0 / self._hx)

    def _Hd_deriv_u(self, v, adjoint=False):
        if adjoint:
            return -self._hx_u(sdiag(1.0 / self._hx ** 2) * v, adjoint=True)
        return -sdiag(1.0 / self._hx ** 2) * self._hx_u(v)

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
        # NOTE: Maybe set this as a property
        PEx = self.getP(mesh, "Fx")
        PBy = self.getP(mesh, "Ex")

        self.src = src
        self.mesh = mesh
        self.f = f

        e = f[src, "e"]
        h = f[src, "h"]
        rx_eval_complex = -self._Hd * self._ex
        # Return the full impedance
        # if return_complex:
        #     return rx_eval_complex
        return getattr(rx_eval_complex, self.component)

    def evalDeriv(self, src, mesh, f, v, adjoint=False):
        """method evalDeriv

        The derivative of the projection wrt u

        :param SimPEG.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: NSEM source
        :param discretize.TensorMesh mesh: Mesh defining the topology of the problem
        :param SimPEG.electromagnetics.frequency_domain.fields.FieldsFDEM f: NSEM fields object of the source
        :param numpy.ndarray v: vector of size (nU,) (adjoint=False) and size (nD,) (adjoint=True)
        :rtype: numpy.ndarray
        :return: Calculated derivative (nD,) (adjoint=False) and (nP,2) (adjoint=True) for both polarizations
        """
        self.src = src
        self.mesh = mesh
        self.f = f

        if adjoint:
            rx_deriv = -self._ex_u(self._Hd * v, adjoint=True) - self._Hd_deriv_u(
                sdiag(self._ex) * v, adjoint=True
            )
            if self.component == "imag":
                rx_deriv_component = 1j * rx_deriv
            elif self.component == "real":
                rx_deriv_component = rx_deriv.astype(complex)

        else:
            rx_deriv = -self._Hd * self._ex_u(v) - sdiag(self._ex) * self._Hd_deriv_u(v)
            rx_deriv_component = np.array(getattr(rx_deriv, self.component))
        return rx_deriv_component


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
        locations_b=None,
    ):

        super().__init__(
            locations=locations,
            orientation=orientation,
            component=component,
            locations_e=locations_e,
            locations_b=locations_b,
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
        out = top / bot

        if adjoint:
            # Work backwards!
            gtop_v = (v / bot)[:, None]
            gbot_v = (-out * v / bot)[:, None]

            ghx_v = np.c_[hy[:, 1], -hy[:, 0]] * gbot_v
            ghy_v = np.c_[-hx[:, 1], hx[:, 0]] * gbot_v
            ge_v = np.c_[h[:, 1], -h[:, 0]] * gtop_v
            gh_v = np.c_[-e[:, 1], e[:, 0]] * gtop_v

            if self.orientation[1] == "x":
                ghy_v += gh_v
            else:
                ghx_v -= gh_v

            gh_v = Phx.T @ ghx_v + Phy.T @ ghy_v
            ge_v = Pe.T @ ge_v

            gfu_h_v, gfm_h_v = f._hDeriv(src, None, gh_v, adjoint=True)
            gfu_e_v, gfm_e_v = f._eDeriv(src, None, ge_v, adjoint=True)

            return gfu_h_v + gfu_e_v, gfm_h_v + gfm_e_v

        de_dm_v = Pe @ f._eDeriv(src, du_dm_v, v, adjoint=False)
        dh_dm_v = f._hDeriv(src, du_dm_v, v, adjoint=False)
        dhx_dm_v = Phx @ dh_dm_v
        dhy_dm_v = Phy @ dh_dm_v
        if self.orientation[1] == "x":
            dh_dm_v = dhy_dm_v
        else:
            dh_dm_v = -dhx_dm_v

        top_dm_v = (
            e[:, 0] * dh_dm_v[:, 1]
            + de_dm_v[:, 0] * h[:, 1]
            - e[:, 1] * dh_dm_v[:, 0]
            - de_dm_v[:, 1] * h[:, 0]
        )
        bot_dm_v = (
            hx[:, 0] * dhy_dm_v[:, 1]
            + dhx_dm_v[:, 0] * hy[:, 1]
            - hx[:, 1] * dhy_dm_v[:, 0]
            - dhx_dm_v[:, 1] * hy[:, 0]
        )

        return (bot * top_dm_v - top * bot_dm_v) / (bot * bot)

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
                # TODO add minus sign here
                v = -1j * v
            else:
                v = v + 0j
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

    def __init__(
        self,
        locations=None,
        orientation="xy",
        component="apparent_resistivity",
        locations_e=None,
        locations_b=None,
    ):

        super().__init__(
            locations=locations,
            orientation=orientation,
            component=component,
            locations_e=locations_e,
            locations_b=locations_b,
        )

    def _alpha(self, src):
        return 1 / (2 * np.pi * mu_0 * src.frequency)

    def eval(self, src, mesh, f, return_complex=False):
        """
        Project the fields to natural source data.
        :param SimPEG.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: The source of the fields to project
        :param discretize.TensorMesh mesh: topological mesh corresponding to the fields
        :param SimPEG.electromagnetics.frequency_domain.fields.FieldsFDEM f: Natural source fields object to project
        :rtype: numpy.ndarray
        :return: component of the impedance evaluation
        """
        # NOTE: Maybe set this as a property
        self.src = src
        self.mesh = mesh
        self.f = f

        alpha = self._alpha(src)
        # Zij = self._eval_impedance(src, mesh, f)

        # Calculate the complex value
        rx_eval_complex = self._eval_impedance(src, mesh, f)

        # # Return the full impedance
        # if return_complex:
        #     return rx_eval_complex
        # else:
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
        self.src = src
        self.mesh = mesh
        self.f = f

        if adjoint is True:
            alpha = self._alpha(src)
            Zij = self._eval_impedance(src, mesh, f, numerator=True)
            HdZij = self._eval_impedance(
                src, mesh, f
            )  # self._Hd @ Zij  # since Hd is diagonal, don't need transpose

            def rx_derivT(v):
                return self._deriv_impedance(
                    src, mesh, f, self._Hd * v, adjoint=adjoint
                ) + self._deriv_Hd_uV(
                    sdiag(Zij) * v, adjoint=adjoint
                )  # _deriv_Hd_uV(v)

            if self.component == "apparent resistivity":
                rx_deriv_component = (
                    2
                    * alpha
                    * (
                        rx_derivT(sdiag(HdZij.real) * v)  # +  #real) #+
                        - rx_derivT(1j * sdiag(HdZij.imag) * v)
                    ).reshape((mesh.nE, 2), order="F")
                )

            elif self.component == "phase":
                HdZij_re = HdZij.real
                HdZij_im = HdZij.imag

                rx_deriv_component = (180 / np.pi) * (
                    rx_derivT(sdiag(-HdZij_im / (HdZij_im ** 2 + HdZij_re ** 2)) * v)
                    + -rx_derivT(
                        1j * sdiag(HdZij_re / (HdZij_im ** 2 + HdZij_re ** 2)) * v
                    )
                ).reshape((mesh.nE, 2), order="F")

        else:
            alpha = self._alpha(src)
            imp_deriv = self._eval_impedance_deriv(
                src, mesh, f, du_dm_v=du_dm_v, v=v, adjoint=adjoint
            )
            imp = self._eval_impedance(src, mesh, f)  # self._Hd @ Zij

            if self.component == "apparent resistivity":
                rx_deriv_component = (
                    2
                    * alpha
                    * (
                        sdiag(imp.real) * imp_deriv.real
                        + sdiag(imp.imag) * imp_deriv.imag
                    )
                )
            elif self.component == "phase":
                deriv_re = (
                    sdiag(-imp.imag / (imp.imag ** 2 + imp.real ** 2)) * imp_deriv.real
                )
                deriv_im = (
                    sdiag(imp.real / (imp.imag ** 2 + imp.real ** 2)) * imp_deriv.imag
                )

                rx_deriv_component = (180 / np.pi) * (deriv_re + deriv_im)

        return rx_deriv_component


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

    def _eval_Tij_numerator(self, src, mesh, f, numerator=False):
        # will grab both primary and secondary and sum them!
        h = f[src, "h"]

        Pbx = self.getP(mesh, "Fx", "b")
        hx_px = Pbx * h[:, 0]
        hx_py = Pbx * h[:, 1]
        Pby = self.getP(mesh, "Fy", "b")
        hy_px = Pby * h[:, 0]
        hy_py = Pby * h[:, 1]
        Pbz = self.getP(mesh, "Fz", "b")
        hz_px = Pbz * h[:, 0]
        hz_py = Pbz * h[:, 1]

        if self.orientation[1] == "x":
            h_px = -hy_px
            h_py = -hy_py
        else:
            h_px = hx_px
            h_py = hx_py

        top = h_px * hz_py - h_py * hz_px
        bot = hx_px * hy_py - hx_py * hy_px
        if numerator:
            return top
        return top / bot

    def _deriv_Tij_numerator(self, v, adjoint=False):
        if "zx" in self.orientation:
            if not adjoint:
                TijN_uV = (
                    -sdiag(self._hy_px) * self._hz_py_u(v)
                    - sdiag(self._hz_py) * self._hy_px_u(v)
                    + sdiag(self._hy_py) * self._hz_px_u(v)
                    + sdiag(self._hz_px) * self._hy_py_u(v)
                )
            elif adjoint:
                TijN_uV = (
                    -self._hz_py_u(sdiag(self._hy_px) * v, adjoint=True)
                    - self._hy_px_u(sdiag(self._hz_py) * v, adjoint=True)
                    + self._hz_px_u(sdiag(self._hy_py) * v, adjoint=True)
                    + self._hy_py_u(sdiag(self._hz_px) * v, adjoint=True)
                )
        elif "zy" in self.orientation:
            if not adjoint:
                TijN_uV = (
                    sdiag(self._hz_py) * self._hx_px_u(v)
                    + sdiag(self._hx_px) * self._hz_py_u(v)
                    - sdiag(self._hx_py) * self._hz_px_u(v)
                    - sdiag(self._hz_px) * self._hx_py_u(v)
                )
            elif adjoint:
                TijN_uV = (
                    self._hx_px_u(sdiag(self._hz_py) * v, adjoint=True)
                    + self._hz_py_u(sdiag(self._hx_px) * v, adjoint=True)
                    - self._hz_px_u(sdiag(self._hx_py) * v, adjoint=True)
                    - self._hx_py_u(sdiag(self._hz_px) * v, adjoint=True)
                )
        return TijN_uV

    def eval(self, src, mesh, f, return_complex=False):
        """
        Project the fields to natural source data.

        :param SimPEG.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: The source of the fields to project
        :param discretize.TensorMesh mesh: Mesh defining the topology of the problem
        :param SimPEG.electromagnetics.frequency_domain.fields.FieldsFDEM f: Natural source fields object to project
        :rtype: numpy.ndarray
        :return: Evaluated component of the impedance data
        """
        # # NOTE: Maybe set this as a property
        # self.src = src
        # self.mesh = mesh
        # self.f = f

        # Tij_numerator = self._eval_Tij_numerator(src, mesh, f)
        rx_eval_complex = self._eval_Tij_numerator(
            src, mesh, f
        )  # self._Hd * Tij_numerator

        return getattr(rx_eval_complex, self.component)

    def evalDeriv(self, src, mesh, f, v, adjoint=False):
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
        self.src = src
        self.mesh = mesh
        self.f = f

        Tij_numerator = self._eval_Tij_numerator(src, mesh, f, numerator=True)

        if adjoint:

            # Calculate the complex derivative
            rx_deriv = self._Hd_deriv_u(
                sdiag(Tij_numerator) * v, adjoint=True
            ) + self._deriv_Tij_numerator(self._Hd * v, adjoint=True)
            # NOTE: Need to reshape the output to go from 2*nU array to a (nU,2) matrix for each polarization
            # rx_deriv_real = np.hstack((mkvc(rx_deriv_real[:len(rx_deriv_real)/2],2),mkvc(rx_deriv_real[len(rx_deriv_real)/2::],2)))
            rx_deriv = rx_deriv.reshape((2, self.mesh.nE)).T
            # Extract the data
            if self.component == "imag":
                rx_deriv_component = 1j * rx_deriv
            elif self.component == "real":
                rx_deriv_component = rx_deriv.astype(complex)
        else:
            rx_deriv_complex = sdiag(Tij_numerator) * self._Hd_deriv_u(
                v
            ) + self._Hd * self._deriv_Tij_numerator(v)
            rx_deriv_component = np.array(getattr(rx_deriv_complex, self.component))

        return rx_deriv_component


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
