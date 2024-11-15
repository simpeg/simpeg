import numpy as np
from scipy.constants import epsilon_0
from ...fields import Fields
from ...utils import Identity, Zero, mkvc
from ..utils import omega


class FieldsFDEM(Fields):
    r"""Base class for storing FDEM fields.

    FDEM fields classes are used to store the discrete solution of the fields for a
    corresponding FDEM simulation; see :class:`.BaseFDEMSimulation`.
    Only one field type (e.g. ``'e'``, ``'j'``, ``'h'``, or ``'b'``) is stored, but certain field types
    can be rapidly computed and returned on the fly. The field type that is stored and the
    field types that can be returned depend on the formulation used by the associated simulation class.
    Once a field object has been created, the individual fields can be accessed; see the example below.

    Parameters
    ----------
    simulation : .BaseFDEMSimulation
        The FDEM simulation object used to compute the discrete field solution.

    Example
    -------
    We want to access the fields for a discrete solution with :math:`\mathbf{e}` discretized
    to edges and :math:`\mathbf{b}` discretized to faces. To extract the fields for all sources:

    .. code-block:: python

        f = simulation.fields(m)
        e = f[:,'e']
        b = f[:,'b']

    The array ``e`` returned will have shape (`n_edges`, `n_sources`). And the array ``b``
    returned will have shape (`n_faces`, `n_sources`). We can also extract the fields for
    a subset of the source list used for the simulation as follows:

    .. code-block:: python

        f = simulation.fields(m)
        e = f[source_list,'e']
        b = f[source_list,'b']

    """

    def __init__(self, simulation):
        dtype = complex
        super().__init__(simulation=simulation, dtype=dtype)

    def _GLoc(self, fieldType):
        """Return grid locations of the fieldType.

        Parameters
        ----------
        fieldType : str
            The field type.

        Returns
        -------
        str
            The grid locations. One of {``'CC'``, ``'N'``, ``'E'``, ``'F'``}.
        """
        return self.aliasFields[fieldType][1]

    def _e(self, solution, source_list):
        """
        Total electric field is sum of primary and secondary

        :param numpy.ndarray solution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: total electric field
        """
        if (
            getattr(self, "_ePrimary", None) is None
            or getattr(self, "_eSecondary", None) is None
        ):
            raise NotImplementedError(
                "Getting e from {0!s} is not implemented".format(
                    self.knownFields.keys()[0]
                )
            )

        return self._ePrimary(solution, source_list) + self._eSecondary(
            solution, source_list
        )

    def _b(self, solution, source_list):
        """
        Total magnetic flux density is sum of primary and secondary

        :param numpy.ndarray solution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: total magnetic flux density
        """
        if (
            getattr(self, "_bPrimary", None) is None
            or getattr(self, "_bSecondary", None) is None
        ):
            raise NotImplementedError(
                "Getting b from {0!s} is not implemented".format(
                    self.knownFields.keys()[0]
                )
            )

        return self._bPrimary(solution, source_list) + self._bSecondary(
            solution, source_list
        )

    def _bSecondary(self, solution, source_list):
        """
        Total magnetic flux density is sum of primary and secondary

        :param numpy.ndarray solution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: total magnetic flux density
        """
        if getattr(self, "_bSecondary", None) is None:
            raise NotImplementedError(
                "Getting b from {} is not implemented".format(
                    self.knownFields.keys()[0]
                )
            )

        return self._bSecondary(solution, source_list)

    def _h(self, solution, source_list):
        """
        Total magnetic field is sum of primary and secondary

        :param numpy.ndarray solution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: total magnetic field
        """
        if (
            getattr(self, "_hPrimary", None) is None
            or getattr(self, "_hSecondary", None) is None
        ):
            raise NotImplementedError(
                "Getting h from {0!s} is not implemented".format(
                    self.knownFields.keys()[0]
                )
            )

        return self._hPrimary(solution, source_list) + self._hSecondary(
            solution, source_list
        )

    def _j(self, solution, source_list):
        """
        Total current density is sum of primary and secondary

        :param numpy.ndarray solution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: total current density
        """
        if (
            getattr(self, "_jPrimary", None) is None
            or getattr(self, "_jSecondary", None) is None
        ):
            raise NotImplementedError(
                "Getting j from {0!s} is not implemented".format(
                    self.knownFields.keys()[0]
                )
            )

        return self._jPrimary(solution, source_list) + self._jSecondary(
            solution, source_list
        )

    def _eDeriv(self, src, du_dm_v, v, adjoint=False):
        r"""
        Total derivative of e with respect to the inversion model. Returns
        :math:`d\mathbf{e}/d\mathbf{m}` for forward and
        (:math:`d\mathbf{e}/d\mathbf{u}`, :math:`d\mathb{u}/d\mathbf{m}`)
        for the adjoint

        :param simpeg.electromagnetics.frequency_domain.Src.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: derivative of the solution vector with
            respect to the model times a vector (is None for adjoint)
        :param numpy.ndarray v: vector to take sensitivity product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: derivative times a vector (or tuple for adjoint)
        """
        if (
            getattr(self, "_eDeriv_u", None) is None
            or getattr(self, "_eDeriv_m", None) is None
        ):
            raise NotImplementedError(
                "Getting eDerivs from {0!s} is not implemented".format(
                    self.knownFields.keys()[0]
                )
            )

        if adjoint:
            return (self._eDeriv_u(src, v, adjoint), self._eDeriv_m(src, v, adjoint))
        return np.array(
            self._eDeriv_u(src, du_dm_v, adjoint) + self._eDeriv_m(src, v, adjoint),
            dtype=complex,
        )

    def _bDeriv(self, src, du_dm_v, v, adjoint=False):
        r"""
        Total derivative of b with respect to the inversion model. Returns
        :math:`d\mathbf{b}/d\mathbf{m}` for forward and
        (:math:`d\mathbf{b}/d\mathbf{u}`, :math:`d\mathb{u}/d\mathbf{m}`) for
        the adjoint

        :param simpeg.electromagnetics.frequency_domain.Src.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: derivative of the solution vector with
            respect to the model times a vector (is None for adjoint)
        :param numpy.ndarray v: vector to take sensitivity product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: derivative times a vector (or tuple for adjoint)
        """
        if (
            getattr(self, "_bDeriv_u", None) is None
            or getattr(self, "_bDeriv_m", None) is None
        ):
            raise NotImplementedError(
                "Getting bDerivs from {0!s} is not implemented".format(
                    self.knownFields.keys()[0]
                )
            )

        if adjoint:
            return (self._bDeriv_u(src, v, adjoint), self._bDeriv_m(src, v, adjoint))
        return np.array(
            self._bDeriv_u(src, du_dm_v, adjoint) + self._bDeriv_m(src, v, adjoint),
            dtype=complex,
        )

    def _bSecondaryDeriv(self, src, du_dm_v, v, adjoint=False):
        r"""
        Total derivative of b with respect to the inversion model. Returns
        :math:`d\mathbf{b}/d\mathbf{m}` for forward and
        (:math:`d\mathbf{b}/d\mathbf{u}`, :math:`d\mathb{u}/d\mathbf{m}`) for
        the adjoint

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: sorce
        :param numpy.ndarray du_dm_v: derivative of the solution vector with
            respect to the model times a vector (is None for adjoint)
        :param numpy.ndarray v: vector to take sensitivity product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: derivative times a vector (or tuple for adjoint)
        """
        # TODO: modify when primary field is dependent on m

        return self._bDeriv(src, du_dm_v, v, adjoint=adjoint)

    def _hDeriv(self, src, du_dm_v, v, adjoint=False):
        r"""
        Total derivative of h with respect to the inversion model. Returns
        :math:`d\mathbf{h}/d\mathbf{m}` for forward and
        (:math:`d\mathbf{h}/d\mathbf{u}`, :math:`d\mathb{u}/d\mathbf{m}`)
        for the adjoint

        :param simpeg.electromagnetics.frequency_domain.Src.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: derivative of the solution vector with
            respect to the model times a vector (is None for adjoint)
        :param numpy.ndarray v: vector to take sensitivity product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: derivative times a vector (or tuple for adjoint)
        """
        if (
            getattr(self, "_hDeriv_u", None) is None
            or getattr(self, "_hDeriv_m", None) is None
        ):
            raise NotImplementedError(
                "Getting hDerivs from {0!s} is not implemented".format(
                    self.knownFields.keys()[0]
                )
            )

        if adjoint:
            return (self._hDeriv_u(src, v, adjoint), self._hDeriv_m(src, v, adjoint))
        return np.array(
            self._hDeriv_u(src, du_dm_v, adjoint) + self._hDeriv_m(src, v, adjoint),
            dtype=complex,
        )

    def _jDeriv(self, src, du_dm_v, v, adjoint=False):
        r"""
        Total derivative of j with respect to the inversion model. Returns
        :math:`d\mathbf{j}/d\mathbf{m}` for forward and
        (:math:`d\mathbf{j}/d\mathbf{u}`, :math:`d\mathb{u}/d\mathbf{m}`) for
        the adjoint

        :param simpeg.electromagnetics.frequency_domain.Src.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: derivative of the solution vector with
            respect to the model times a vector (is None for adjoint)
        :param numpy.ndarray v: vector to take sensitivity product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: derivative times a vector (or tuple for adjoint)
        """
        if (
            getattr(self, "_jDeriv_u", None) is None
            or getattr(self, "_jDeriv_m", None) is None
        ):
            raise NotImplementedError(
                "Getting jDerivs from {0!s} is not implemented".format(
                    self.knownFields.keys()[0]
                )
            )

        if adjoint:
            return (self._jDeriv_u(src, v, adjoint), self._jDeriv_m(src, v, adjoint))
        return np.array(
            self._jDeriv_u(src, du_dm_v, adjoint) + self._jDeriv_m(src, v, adjoint),
            dtype=complex,
        )


class Fields3DElectricField(FieldsFDEM):
    r"""Fields class for storing 3D total electric field solutions.

    This class stores the total electric field solution computed using a
    :class:`.frequency_domain.Simulation3DElectricField`
    simulation object. This class can be used to extract the following quantities:

    * ``'e'``, ``'ePrimary'``, ``'eSecondary'`` and ``'j'`` on mesh edges.
    * ``'h'``, ``'b'``, ``'bPrimary'`` and ``'bSecondary'`` on mesh faces.
    * ``'charge'`` on mesh nodes.
    * ``'charge_density'`` at cell centers.

    See the example below to learn how fields can be extracted from a
    ``Fields3DElectricField`` object.

    Parameters
    ----------
    simulation : .frequency_domain.Simulation3DElectricField
        The FDEM simulation object associated with the fields.

    Example
    -------
    The ``Fields3DElectricField`` object stores the total electric field solution
    on mesh edges. To extract the discrete electric fields and magnetic flux
    densities for all sources:

    .. code-block:: python

        f = simulation.fields(m)
        e = f[:, 'e']
        b = f[:, 'b']

    The array ``e`` returned will have shape (`n_edges`, `n_sources`). And the array ``b``
    returned will have shape (`n_faces`, `n_sources`). We can also extract the fields for
    a subset of the source list used for the simulation as follows:

    .. code-block:: python

        f = simulation.fields(m)
        e = f[source_list,'e']
        b = f[source_list,'b']
    """

    def __init__(self, simulation):
        super().__init__(simulation=simulation)
        self._knownFields = {"eSolution": "E"}
        self._aliasFields = {
            "e": ["eSolution", "E", "_e"],
            "ePrimary": ["eSolution", "E", "_ePrimary"],
            "eSecondary": ["eSolution", "E", "_eSecondary"],
            "b": ["eSolution", "F", "_b"],
            "bPrimary": ["eSolution", "F", "_bPrimary"],
            "bSecondary": ["eSolution", "F", "_bSecondary"],
            "j": ["eSolution", "E", "_j"],
            "h": ["eSolution", "F", "_h"],
            "charge": ["eSolution", "N", "_charge"],
            "charge_density": ["eSolution", "CC", "_charge_density"],
        }

    def startup(self):
        # Docstring inherited from parent.
        self._edgeCurl = self.simulation.mesh.edge_curl
        self._aveE2CCV = self.simulation.mesh.aveE2CCV
        self._aveF2CCV = self.simulation.mesh.aveF2CCV
        self._nC = self.simulation.mesh.nC
        self._Me_conductivity = self.simulation._Me_conductivity
        self._Me_conductivity_deriv = self.simulation._Me_conductivity_deriv
        self._Mf__perm_inv = self.simulation._Mf__perm_inv
        self._Mf__perm_inv_deriv = self.simulation._Mf__perm_inv_deriv
        self._inv_Me = self.simulation._inv_Me
        self._inv_Mf = self.simulation._inv_Mf

    def _GLoc(self, fieldType):
        if fieldType in ["e", "eSecondary", "ePrimary", "j"]:
            return "E"
        elif fieldType in ["b", "bSecondary", "bPrimary", "h"]:
            return "F"
        else:
            raise Exception("Field type must be e, b, h, j")

    def _ePrimary(self, eSolution, source_list):
        """
        Primary electric field from source

        :param numpy.ndarray eSolution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: primary electric field as defined by the sources
        """

        n_fields = sum(src._fields_per_source for src in source_list)
        ePrimary = np.zeros([self.simulation.mesh.nE, n_fields], dtype=complex)
        i = 0
        for src in source_list:
            ii = i + src._fields_per_source
            ep = src.ePrimary(self.simulation)
            if not isinstance(ep, Zero) and ep.ndim == 1:
                ep = ep[:, None]
            ePrimary[:, i:ii] = ePrimary[:, i:ii] + ep
            i = ii

        return ePrimary

    def _eSecondary(self, eSolution, source_list):
        """
        Secondary electric field is the thing we solved for

        :param numpy.ndarray eSolution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: secondary electric field
        """
        return eSolution

    def _eDeriv_u(self, src, v, adjoint=False):
        """
        Partial derivative of the total electric field with respect to the
        thing we solved for.

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the electric field with respect
            to the field we solved for with a vector
        """

        return Identity() * v

    def _eDeriv_m(self, src, v, adjoint=False):
        """
        Partial derivative of the total electric field with respect to the
        inversion model. Here, we assume that the primary does not depend on
        the model. Note that this also includes derivative contributions from
        the sources.

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: simpeg.utils.Zero
        :return: product of the electric field derivative with respect to the
            inversion model with a vector
        """

        return src.ePrimaryDeriv(self.simulation, v, adjoint)

    def _bPrimary(self, eSolution, source_list):
        """
        Primary magnetic flux density from source

        :param numpy.ndarray eSolution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: primary magnetic flux density as defined by the sources
        """

        bPrimary = np.zeros(
            [self._edgeCurl.shape[0], eSolution.shape[1]], dtype=complex
        )

        i = 0
        for src in source_list:
            ii = i + src._fields_per_source
            bp = src.bPrimary(self.simulation)
            if not isinstance(bp, Zero) and bp.ndim == 1:
                bp = bp[:, None]
            bPrimary[:, i:ii] = bPrimary[:, i:ii] + bp
            i = ii
        return bPrimary

    def _bSecondary(self, eSolution, source_list):
        """
        Secondary magnetic flux density from eSolution

        :param numpy.ndarray eSolution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: secondary magnetic flux density
        """

        C = self._edgeCurl
        b = C * eSolution
        i = 0
        for src in source_list:
            ii = i + src._fields_per_source
            b[:, i:ii] *= -1.0 / (
                1j * omega(src.frequency)
            )  # freq depends on the source
            s_m = src.s_m(self.simulation)
            if not isinstance(s_m, Zero) and s_m.ndim == 1:
                s_m = s_m[:, None]
            b[:, i:ii] = b[:, i:ii] + 1.0 / (1j * omega(src.frequency)) * s_m
        return b

    def _bDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Derivative of the magnetic flux density with respect to the thing we
        solved for

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic flux density with
            respect to the field we solved for with a vector
        """

        C = self._edgeCurl
        if adjoint:
            return -1.0 / (1j * omega(src.frequency)) * (C.T * du_dm_v)
        return -1.0 / (1j * omega(src.frequency)) * (C * du_dm_v)

    def _bDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the magnetic flux density with respect to the inversion
        model.

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the magnetic flux density derivative with respect
            to the inversion model with a vector
        """

        return self._bDeriv_src(src, v, adjoint=adjoint)

    def _bDeriv_src(self, src, v, adjoint=False):
        s_mDeriv = src.s_mDeriv(self.simulation, v, adjoint)
        return 1.0 / (1j * omega(src.frequency)) * s_mDeriv + src.bPrimaryDeriv(
            self.simulation, v, adjoint
        )

    def _j(self, eSolution, source_list):
        """
        Current density from eSolution

        :param numpy.ndarray eSolution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: current density
        """
        return self._inv_Me * (self._Me_conductivity * self._e(eSolution, source_list))

    def _jDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Derivative of the current density with respect to the thing we solved
        for

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the current density with respect
            to the field we solved for with a vector
        """
        if adjoint:
            return self._eDeriv_u(
                src,
                self._Me_conductivity.T * (self._inv_Me.T * du_dm_v),
                adjoint=adjoint,
            )
        return self._inv_Me * (
            self._Me_conductivity * (self._eDeriv_u(src, du_dm_v, adjoint=adjoint))
        )

    def _jDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the current density with respect to the inversion model.

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the current density derivative with respect to the
            inversion model with a vector
        """
        e = self[src, "e"]

        if adjoint:
            return (
                self._Me_conductivity_deriv(e, (self._inv_Me.T * v), adjoint=adjoint)
                + self._eDeriv_m(src, (self._inv_Me.T * v), adjoint=adjoint)
            ) + src.jPrimaryDeriv(self.simulation, v, adjoint)
        return (
            self._inv_Me
            * (
                self._eDeriv_m(src, v, adjoint=adjoint)
                + self._Me_conductivity_deriv(e, v, adjoint=adjoint)
            )
        ) + src.jPrimaryDeriv(self.simulation, v, adjoint)

    def _h(self, eSolution, source_list):
        """
        Magnetic field from eSolution

        :param numpy.ndarray eSolution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: magnetic field
        """

        return self._inv_Mf * (self._Mf__perm_inv * self._b(eSolution, source_list))

    def _hDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Derivative of the magnetic field with respect to the thing we solved
        for

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic field with respect
            to the field we solved for with a vector
        """
        if adjoint:
            v = self._Mf__perm_inv.T * (self._inv_Mf.T * du_dm_v)
            return self._bDeriv_u(src, v, adjoint=adjoint)
        return self._inv_Mf * (
            self._Mf__perm_inv * self._bDeriv_u(src, du_dm_v, adjoint=adjoint)
        )

    def _hDeriv__perm_inv(self, src, v, adjoint=False):
        # n = int(self._aveF2CCV.shape[0] / self._nC)  # Number of Components
        # VI = sdiag(np.kron(np.ones(n), 1./self.simulation.mesh.cell_volumes))

        if adjoint is True:
            return self._Mf__perm_inv_deriv(
                self[src, "b"], (self._inv_Mf.T * v), adjoint
            )

        return self._inv_Mf * (self._Mf__perm_inv_deriv(self[src, "b"], v))

    def _hDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the magnetic field with respect to the inversion model.

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the magnetic field derivative with respect to the
            inversion model with a vector
        """
        # n = int(self._aveF2CCV.shape[0] / self._nC)  # Number of Components
        # VI = sdiag(np.kron(np.ones(n), 1./self.simulation.mesh.cell_volumes))
        if adjoint:
            return self._bDeriv_m(
                src, self._Mf__perm_inv.T * (self._inv_Mf.T * v), adjoint=adjoint
            ) + self._hDeriv__perm_inv(src, v, adjoint=adjoint)
        return (
            self._inv_Mf
            * (self._Mf__perm_inv * self._bDeriv_m(src, v, adjoint=adjoint))
        ) + self._hDeriv__perm_inv(src, v, adjoint=adjoint)

    def _charge(self, eSolution, source_list):
        r"""
        .. math::
            \int \nabla \codt \vec{e} =  \int \frac{\rho_v }{\epsillon_0}
        """
        return -epsilon_0 * (
            self.mesh.nodal_gradient.T
            * self.mesh.get_edge_inner_product()
            * self._e(eSolution, source_list)
        )

    def _charge_density(self, eSolution, source_list):
        return (
            self.mesh.aveN2CC * self._charge(eSolution, source_list)
        ) / self.mesh.cell_volumes[:, None]


class Fields3DMagneticFluxDensity(FieldsFDEM):
    r"""Fields class for storing 3D total magnetic flux density solutions.

    This class stores the total magnetic flux density solution computed using a
    :class:`.frequency_domain.Simulation3DMagneticFluxDensity`
    simulation object. This class can be used to extract the following quantities:

    * ``'b'``, ``'bPrimary'``, ``'bSecondary'`` and ``'h'`` on mesh faces.
    * ``'e'``, ``'ePrimary'``, ``'eSecondary'`` and ``'j'`` on mesh edges.
    * ``'charge'`` on mesh nodes.
    * ``'charge_density'`` at cell centers.

    See the example below to learn how fields can be extracted from a
    ``Fields3DMagneticFluxDensity`` object.

    Parameters
    ----------
    simulation : .frequency_domain.Simulation3DMagneticFluxDensity
        The FDEM simulation object associated with the fields.

    Example
    -------
    The ``Fields3DMagneticFluxDensity`` object stores the total magnetic flux density solution
    on mesh faces. To extract the discrete electric fields and magnetic flux
    densities for all sources:

    .. code-block:: python

        f = simulation.fields(m)
        e = f[:, 'e']
        b = f[:, 'b']

    The array ``e`` returned will have shape (`n_edges`, `n_sources`). And the array ``b``
    returned will have shape (`n_faces`, `n_sources`). We can also extract the fields for
    a subset of the source list used for the simulation as follows:

    .. code-block:: python

        f = simulation.fields(m)
        e = f[source_list, 'e']
        b = f[source_list, 'b']

    """

    def __init__(self, simulation):
        super().__init__(simulation=simulation)
        self._knownFields = {"bSolution": "F"}
        self._aliasFields = {
            "b": ["bSolution", "F", "_b"],
            "bPrimary": ["bSolution", "F", "_bPrimary"],
            "bSecondary": ["bSolution", "F", "_bSecondary"],
            "e": ["bSolution", "E", "_e"],
            "ePrimary": ["bSolution", "E", "_ePrimary"],
            "eSecondary": ["bSolution", "E", "_eSecondary"],
            "j": ["bSolution", "E", "_j"],
            "h": ["bSolution", "F", "_h"],
            "charge": ["bSolution", "N", "_charge"],
            "charge_density": ["bSolution", "CC", "_charge_density"],
        }

    def startup(self):
        # Docstring inherited from parent.
        self._edgeCurl = self.simulation.mesh.edge_curl
        self._Me_conductivity = self.simulation._Me_conductivity
        self._inv_Me_conductivity = self.simulation._inv_Me_conductivity
        self._Mf__perm_inv = self.simulation._Mf__perm_inv
        self._Mf__perm_inv_deriv = self.simulation._Mf__perm_inv_deriv
        self._Me_conductivity_deriv = self.simulation._Me_conductivity_deriv
        self._inv_Me_conductivity_deriv = self.simulation._inv_Me_conductivity_deriv
        self._Me = self.simulation._Me
        self._aveF2CCV = self.simulation.mesh.aveF2CCV
        self._aveE2CCV = self.simulation.mesh.aveE2CCV
        self._conductivity = self.simulation.conductivity
        self.__perm_inv = self.simulation._perm_inv
        self._nC = self.simulation.mesh.nC
        self._inv_Me = self.simulation._inv_Me
        self._inv_Mf = self.simulation._inv_Mf

    def _GLoc(self, fieldType):
        if fieldType in ["e", "eSecondary", "ePrimary", "j"]:
            return "E"
        elif fieldType in ["b", "bSecondary", "bPrimary", "h"]:
            return "F"
        else:
            raise Exception("Field type must be e, b, h, j")

    def _bPrimary(self, bSolution, source_list):
        """
        Primary magnetic flux density from source

        :param numpy.ndarray bSolution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: primary electric field as defined by the sources
        """

        bPrimary = np.zeros([self.simulation.mesh.nF, len(source_list)], dtype=complex)
        for i, src in enumerate(source_list):
            bp = src.bPrimary(self.simulation)
            bPrimary[:, i] = bPrimary[:, i] + bp
        return bPrimary

    def _bSecondary(self, bSolution, source_list):
        """
        Secondary magnetic flux density is the thing we solved for

        :param numpy.ndarray bSolution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: secondary magnetic flux density
        """

        return bSolution

    def _bDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Partial derivative of the total magnetic flux density with respect to
        the thing we solved for.

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic flux density with
            respect to the field we solved for with a vector
        """

        return Identity() * du_dm_v

    def _bDeriv_m(self, src, v, adjoint=False):
        """
        Partial derivative of the total magnetic flux density with respect to
        the inversion model. Here, we assume that the primary does not depend
        on the model. Note that this also includes derivative contributions
        from the sources.

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: simpeg.utils.Zero
        :return: product of the magnetic flux density derivative with respect
            to the inversion model with a vector
        """

        # assuming primary does not depend on the model
        return Zero()

    def _ePrimary(self, bSolution, source_list):
        """
        Primary electric field from source

        :param numpy.ndarray bSolution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: primary electric field as defined by the sources
        """

        ePrimary = np.zeros(
            [self._edgeCurl.shape[1], bSolution.shape[1]], dtype=complex
        )
        for i, src in enumerate(source_list):
            ep = src.ePrimary(self.simulation)
            ePrimary[:, i] = ePrimary[:, i] + ep
        return ePrimary

    def _eSecondary(self, bSolution, source_list):
        """
        Secondary electric field from bSolution

        :param numpy.ndarray bSolution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: secondary electric field
        """

        e = self._edgeCurl.T * (self._Mf__perm_inv * bSolution)
        for i, src in enumerate(source_list):
            s_e = src.s_e(self.simulation)
            e[:, i] = e[:, i] + -s_e

            if self.simulation.permittivity is not None:
                MeyhatI = self.simulation._get_edge_admittivity_property_matrix(
                    src.frequency, invert_matrix=True
                )
                e[:, i] = MeyhatI * e[:, i]

        if self.simulation.permittivity is None:
            return self._inv_Me_conductivity * e
        else:
            return e

    def _eDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Derivative of the electric field with respect to the thing we solved
        for

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the electric field with respect
            to the field we solved for with a vector
        """

        if not adjoint:
            return self._inv_Me_conductivity * (
                self._edgeCurl.T * (self._Mf__perm_inv * du_dm_v)
            )
        return self._Mf__perm_inv.T * (
            self._edgeCurl * (self._inv_Me_conductivity.T * du_dm_v)
        )

    def _eDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the electric field with respect to the inversion model

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the electric field with respect
            to the model with a vector
        """

        bSolution = mkvc(self[src, "bSolution"])
        s_e = src.s_e(self.simulation)

        w = -s_e + self._edgeCurl.T * (self._Mf__perm_inv * bSolution)

        if adjoint:
            s_eDeriv = src.s_eDeriv(
                self.simulation, self._inv_Me_conductivity.T * v, adjoint
            )
            return (
                self._inv_Me_conductivity_deriv(w, v, adjoint)
                + self._Mf__perm_inv_deriv(
                    bSolution,
                    self._edgeCurl * (self._inv_Me_conductivity.T * v),
                    adjoint,
                )
                - s_eDeriv
                + src.ePrimaryDeriv(self.simulation, v, adjoint)
            )
        s_eDeriv = src.s_eDeriv(self.simulation, v, adjoint)
        return (
            self._inv_Me_conductivity_deriv(w, v)
            + self._inv_Me_conductivity
            * (self._edgeCurl.T * self._Mf__perm_inv_deriv(bSolution, v))
            - self._inv_Me_conductivity * s_eDeriv
            + src.ePrimaryDeriv(self.simulation, v, adjoint)
        )

    def _j(self, bSolution, source_list):
        """
        Secondary current density from bSolution

        :param numpy.ndarray bSolution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: primary current density
        """

        if self.simulation.permittivity is None:
            j = self._edgeCurl.T * (self._Mf__perm_inv * bSolution)

            for i, src in enumerate(source_list):
                s_e = src.s_e(self.simulation)
                j[:, i] = j[:, i] - s_e

            return self._inv_Me * j
        else:
            return self._inv_Me * (
                self._Me_conductivity * self._e(bSolution, source_list)
            )

    def _jDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Partial derivative of the current density with respect to the thing we
        solved for.

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the current density with respect
            to the field we solved for with a vector
        """
        if adjoint:
            return self._Mf__perm_inv.T * (self._edgeCurl * (self._inv_Me.T * du_dm_v))
        return self._inv_Me * (self._edgeCurl.T * (self._Mf__perm_inv * du_dm_v))
        # forgetting the source term here

    def _jDeriv__perm_inv(self, src, v, adjoint=False):
        if adjoint:
            return self._Mf__perm_inv_deriv(
                self[src, "b"], (self._edgeCurl * (self._inv_Me.T * v)), adjoint
            )

        return self._inv_Me * (
            self._edgeCurl.T * self._Mf__perm_inv_deriv(self[src, "b"], v)
        )

    def _jDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the current density with respect to the inversion model

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the current density with respect
            to the model with a vector
        """

        return self._jDeriv__perm_inv(src, v, adjoint)

    def _h(self, bSolution, source_list):
        """
        Magnetic field from bSolution

        :param numpy.ndarray bSolution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: magnetic field
        """
        return self._inv_Mf * (self._Mf__perm_inv * self._b(bSolution, source_list))

    def _hDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Partial derivative of the magnetic field with respect to the thing we
        solved for.

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic field with respect
            to the field we solved for with a vector
        """
        if adjoint:
            return self._Mf__perm_inv.T * (self._inv_Mf.T * du_dm_v)
        return self._inv_Mf * (self._Mf__perm_inv * du_dm_v)

    def _hDeriv__perm_inv(self, src, v, adjoint=False):
        b = self[src, "b"]
        if adjoint:
            return self._Mf__perm_inv_deriv(b, self._inv_Mf.T * v, adjoint)
        return self._inv_Mf * self._Mf__perm_inv_deriv(b, v)

    def _hDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the magnetic field with respect to the inversion model

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic field with respect
            to the model with a vector
        """
        return src.hPrimaryDeriv(self.simulation, v, adjoint) + self._hDeriv__perm_inv(
            src, v, adjoint
        )

    def _charge(self, bSolution, source_list):
        r"""
        .. math::

            \int \nabla \codt \vec{e} =  \int \frac{\rho_v }{\epsillon_0}
        """
        return -epsilon_0 * (
            self.mesh.nodal_gradient.T
            * self.mesh.get_edge_inner_product()
            * self._e(bSolution, source_list)
        )

    def _charge_density(self, bSolution, source_list):
        return (
            self.mesh.aveN2CC * self._charge(bSolution, source_list)
        ) / self.mesh.cell_volumes[:, None]


class Fields3DCurrentDensity(FieldsFDEM):
    r"""Fields class for storing 3D current density solutions.

    This class stores the total current density solution computed using a
    :class:`.frequency_domain.Simulation3DCurrentDensity`
    simulation object. This class can be used to extract the following quantities:

    * ``'j'``, ``'jPrimary'``, ``'jSecondary'`` and ``'e'`` on mesh faces.
    * ``'h'``, ``'hPrimary'``, ``'hSecondary'`` and ``'b'`` on mesh edges.
    * ``'charge'`` and ``'charge_density'`` at cell centers.

    See the example below to learn how fields can be extracted from a
    ``Fields3DCurrentDensity`` object.

    Parameters
    ----------
    simulation : .frequency_domain.Simulation3DCurrentDensity
        The FDEM simulation object associated with the fields.

    Example
    -------
    The ``Fields3DCurrentDensity`` object stores the total current density solution
    on mesh faces. To extract the discrete current density and magnetic field:

    .. code-block:: python

        f = simulation.fields(m)
        j = f[:, 'j']
        h = f[:, 'h']

    The array ``j`` returned will have shape (`n_faces`, `n_sources`). And the array ``h``
    returned will have shape (`n_edges`, `n_sources`). We can also extract the fields for
    a subset of the source list used for the simulation as follows:

    .. code-block:: python

        f = simulation.fields(m)
        j = f[source_list, 'j']
        h = f[source_list, 'h']

    """

    def __init__(self, simulation):
        super().__init__(simulation=simulation)
        self._knownFields = {"jSolution": "F"}
        self._aliasFields = {
            "j": ["jSolution", "F", "_j"],
            "jPrimary": ["jSolution", "F", "_jPrimary"],
            "jSecondary": ["jSolution", "F", "_jSecondary"],
            "h": ["jSolution", "E", "_h"],
            "hPrimary": ["jSolution", "E", "_hPrimary"],
            "hSecondary": ["jSolution", "E", "_hSecondary"],
            "e": ["jSolution", "F", "_e"],
            "b": ["jSolution", "E", "_b"],
            "charge": ["jSolution", "CC", "_charge"],
            "charge_density": ["jSolution", "CC", "_charge_density"],
        }

    def startup(self):
        # Docstring inherited from parent.
        self._edgeCurl = self.simulation.mesh.edge_curl
        self._Me_permeability = self.simulation._Me_permeability
        self._inv_Me_permeability = self.simulation._inv_Me_permeability
        self._inv_Me_permeability_deriv = self.simulation._inv_Me_permeability_deriv
        self._Mf_resistivity = self.simulation._Mf_resistivity
        self._Mf_resistivity_deriv = self.simulation._Mf_resistivity_deriv
        self._resistivity = self.simulation.resistivity
        self._perm_inv = self.simulation._perm_inv
        self._aveF2CCV = self.simulation.mesh.aveF2CCV
        self._aveE2CCV = self.simulation.mesh.aveE2CCV
        self._nC = self.simulation.mesh.nC
        self._inv_Me = self.simulation._inv_Me
        self._inv_Mf = self.simulation._inv_Mf

    def _GLoc(self, fieldType):
        if fieldType in ["h", "hSecondary", "hPrimary", "b"]:
            return "E"
        elif fieldType in ["j", "jSecondary", "jPrimary", "e"]:
            return "F"
        else:
            raise Exception("Field type must be e, b, h, j")

    def _jPrimary(self, jSolution, source_list):
        """
        Primary current density from source

        :param numpy.ndarray jSolution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: primary current density as defined by the sources
        """

        jPrimary = np.zeros_like(jSolution, dtype=complex)
        for i, src in enumerate(source_list):
            jp = src.jPrimary(self.simulation)
            jPrimary[:, i] = jPrimary[:, i] + jp
        return jPrimary

    def _jSecondary(self, jSolution, source_list):
        """
        Secondary current density is the thing we solved for

        :param numpy.ndarray jSolution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: secondary current density
        """

        return jSolution

    def _j(self, jSolution, source_list):
        """
        Total current density is sum of primary and secondary

        :param numpy.ndarray jSolution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: total current density
        """

        return self._jPrimary(jSolution, source_list) + self._jSecondary(
            jSolution, source_list
        )

    def _jDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Partial derivative of the total current density with respect to the
        thing we solved for.

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the current density with respect
            to the field we solved for with a vector
        """

        return Identity() * du_dm_v

    def _jDeriv_m(self, src, v, adjoint=False):
        """
        Partial derivative of the total current density with respect to the
        inversion model. Here, we assume that the primary does not depend on
        the model. Note that this also includes derivative contributions from
        the sources.

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: simpeg.utils.Zero
        :return: product of the current density derivative with respect to the
            inversion model with a vector
        """
        # assuming primary does not depend on the model
        return src.jPrimaryDeriv(self.simulation, v, adjoint)

    def _hPrimary(self, jSolution, source_list):
        """
        Primary magnetic field from source

        :param numpy.ndarray hSolution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: primary magnetic field as defined by the sources
        """

        hPrimary = np.zeros(
            [self._edgeCurl.shape[1], jSolution.shape[1]], dtype=complex
        )
        for i, src in enumerate(source_list):
            hp = src.hPrimary(self.simulation)
            hPrimary[:, i] = hPrimary[:, i] + hp
        return hPrimary

    def _hSecondary(self, jSolution, source_list):
        """
        Secondary magnetic field from bSolution

        :param numpy.ndarray jSolution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: secondary magnetic field
        """

        if self.simulation.permittivity is not None:
            h = np.zeros((self.mesh.n_edges, len(source_list)), dtype=complex)
        else:
            h = self._edgeCurl.T * (self._Mf_resistivity * jSolution)

        for i, src in enumerate(source_list):
            if self.simulation.permittivity is not None:
                h[:, i] = self._edgeCurl.T * (
                    self.simulation._get_face_admittivity_property_matrix(
                        src.frequency, invert_model=True
                    )
                    * jSolution[:, i]
                )

            h[:, i] *= -1.0 / (1j * omega(src.frequency))
            s_m = src.s_m(self.simulation)
            h[:, i] = h[:, i] + 1.0 / (1j * omega(src.frequency)) * (s_m)
        return self._inv_Me_permeability * h

    def _hDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Derivative of the magnetic field with respect to the thing we solved
        for

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic field with respect
            to the field we solved for with a vector
        """

        if adjoint:
            return (
                -1.0
                / (1j * omega(src.frequency))
                * self._Mf_resistivity.T
                * (self._edgeCurl * (self._inv_Me_permeability.T * du_dm_v))
            )
        return (
            -1.0
            / (1j * omega(src.frequency))
            * self._inv_Me_permeability
            * (self._edgeCurl.T * (self._Mf_resistivity * du_dm_v))
        )

    def _hDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the magnetic field with respect to the inversion model

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic field with respect
            to the model with a vector
        """

        jSolution = mkvc(self[[src], "jSolution"])
        MeMuI = self._inv_Me_permeability
        MeMuIDeriv = self._inv_Me_permeability_deriv
        C = self._edgeCurl
        MfRho = self._Mf_resistivity
        MfRhoDeriv = self._Mf_resistivity_deriv

        s_m = src.s_m(self.simulation)

        def s_mDeriv(v):
            return src.s_mDeriv(self.simulation, v, adjoint=adjoint)

        if not adjoint:
            hDeriv_m = (
                1.0
                / (1j * omega(src.frequency))
                * (
                    -1.0
                    * (
                        MeMuI * (C.T * MfRhoDeriv(jSolution, v, adjoint))
                        + MeMuIDeriv(C.T * (MfRho * jSolution)) * v
                    )
                    + MeMuI * s_mDeriv(v)
                    + MeMuIDeriv(s_m) * v
                )
            )

        elif adjoint:
            hDeriv_m = (
                1.0
                / (1j * omega(src.frequency))
                * (
                    (
                        -1.0
                        * (
                            MfRhoDeriv(jSolution).T * (C * (MeMuI.T * v))
                            + MeMuIDeriv(C.T * (MfRho * jSolution)).T * v
                        )
                    )
                    + s_mDeriv(MeMuI.T * v)
                    + MeMuIDeriv(s_m).T * v
                )
            )

        return hDeriv_m + src.hPrimaryDeriv(self.simulation, v, adjoint)

    def _e(self, jSolution, source_list):
        """
        Electric field from jSolution

        :param numpy.ndarray hSolution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: electric field
        """
        # if self.simulation.permittivity is None:
        return self._inv_Mf * (self._Mf_resistivity * self._j(jSolution, source_list))

        # e = np.zeros((self.mesh.n_faces, len(source_list)), dtype=complex)
        # for i, source in enumerate(source_list):
        #     Mfyhati = self.simulation._get_face_admittivity_property_matrix(
        #         source.frequency, invert_model=True
        #     )
        #     e[:, i] = Mfyhati * mkvc(self._j(jSolution, [source]))
        # return self._inv_Mf * e

    def _eDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Derivative of the electric field with respect to the thing we solved
        for

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the electric field with respect
            to the field we solved for with a vector
        """
        if adjoint:
            return self._Mf_resistivity.T * (self._inv_Mf.T * du_dm_v)
        return self._inv_Mf * (self._Mf_resistivity * du_dm_v)

    def _eDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the electric field with respect to the inversion model

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the electric field with respect
            to the model with a vector
        """
        jSolution = mkvc(self[src, "jSolution"])
        if adjoint:
            return self._Mf_resistivity_deriv(jSolution).T * (
                self._inv_Mf.T * v
            ) + src.ePrimaryDeriv(self.simulation, v, adjoint)
        return self._inv_Mf * (
            self._Mf_resistivity_deriv(jSolution) * v
        ) + src.ePrimaryDeriv(self.simulation, v, adjoint)

    def _b(self, jSolution, source_list):
        """
        Secondary magnetic flux density from jSolution

        :param numpy.ndarray hSolution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: secondary magnetic flux density
        """

        return self._inv_Me * (self._Me_permeability * self._h(jSolution, source_list))

    def _bDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Derivative of the magnetic flux density with respect to the thing we
        solved for

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic flux density with
            respect to the field we solved for with a vector
        """
        if adjoint:
            return (
                -1.0
                / (1j * omega(src.frequency))
                * self._Mf_resistivity.T
                * (self._edgeCurl * (self._inv_Me.T * du_dm_v))
            )

        return (
            -1.0
            / (1j * omega(src.frequency))
            * (self._inv_Me * (self._edgeCurl.T * (self._Mf_resistivity * du_dm_v)))
        )

    def _bDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the magnetic flux density with respect to the inversion
        model

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic flux density with
            respect to the model with a vector
        """
        jSolution = self[src, "jSolution"]

        def s_mDeriv(v):
            return src.s_mDeriv(self.simulation, v, adjoint=adjoint)

        if adjoint:
            v = self._inv_Me.T * v
            return 1.0 / (1j * omega(src.frequency)) * (
                s_mDeriv(v)
                - self._Mf_resistivity_deriv(jSolution, self._edgeCurl * v, adjoint)
            ) + src.bPrimaryDeriv(self.simulation, v, adjoint)
        return 1.0 / (1j * omega(src.frequency)) * self._inv_Me * (
            s_mDeriv(v)
            - self._edgeCurl.T * self._Mf_resistivity_deriv(jSolution, v, adjoint)
        ) + src.bPrimaryDeriv(self.simulation, v, adjoint)

    def _charge(self, jSolution, source_list):
        r"""
        .. math::

            \int \nabla \codt \vec{e} =  \int \frac{\rho_v }{\epsillon_0}
        """
        return self.mesh.cell_volumes[:, None] * self._charge_density(
            jSolution, source_list
        )

    def _charge_density(self, jSolution, source_list):
        r"""
        .. math::

            \frac{1}{V}\int \nabla \codt \vec{e} =
            \frac{1}{V}\int \frac{\rho_v }{\epsillon_0}
        """
        return epsilon_0 * (self._faceDiv * self._e(jSolution, source_list))


class Fields3DMagneticField(FieldsFDEM):
    r"""Fields class for storing 3D magnetic field solutions.

    This class stores the total magnetic field solution computed using a
    :class:`.frequency_domain.Simulation3DMagneticField`
    simulation object. This class can be used to extract the following quantities:

    * ``'h'``, ``'hPrimary'``, ``'hSecondary'`` and ``'b'`` on mesh edges.
    * ``'j'``, ``'jPrimary'``, ``'jSecondary'`` and ``'e'`` on mesh faces.
    * ``'charge'`` and ``'charge_density'`` at cell centers.

    See the example below to learn how fields can be extracted from a
    ``Fields3DMagneticField`` object.

    Parameters
    ----------
    simulation : .frequency_domain.Simulation3DMagneticField
        The FDEM simulation object associated with the fields.

    Example
    -------
    The ``Fields3DMagneticField`` object stores the total magnetic field solution
    on mesh edges. To extract the discrete current density and magnetic field:

    .. code-block:: python

        f = simulation.fields(m)
        j = f[:, 'j']
        h = f[:, 'h']

    The array ``j`` returned will have shape (`n_faces`, `n_sources`). And the array ``h``
    returned will have shape (`n_edges`, `n_sources`). We can also extract the fields for
    a subset of the source list used for the simulation as follows:

    .. code-block:: python

        f = simulation.fields(m)
        j = f[source_list, 'j']
        h = f[source_list, 'h']

    """

    def __init__(self, simulation):
        super().__init__(simulation=simulation)
        self._knownFields = {"hSolution": "E"}
        self._aliasFields = {
            "h": ["hSolution", "E", "_h"],
            "hPrimary": ["hSolution", "E", "_hPrimary"],
            "hSecondary": ["hSolution", "E", "_hSecondary"],
            "j": ["hSolution", "F", "_j"],
            "jPrimary": ["hSolution", "F", "_jPrimary"],
            "jSecondary": ["hSolution", "F", "_jSecondary"],
            "e": ["hSolution", "CCV", "_e"],
            "b": ["hSolution", "CCV", "_b"],
            "charge": ["hSolution", "CC", "_charge"],
            "charge_density": ["hSolution", "CC", "_charge_density"],
        }

    def startup(self):
        # Docstring inherited from parent.
        self._edgeCurl = self.simulation.mesh.edge_curl
        self._Me_permeability = self.simulation._Me_permeability
        self._Me_permeability_deriv = self.simulation._Me_permeability_deriv
        # self._inv_Me_permeability = self.simulation._inv_Me_permeability
        self._Mf_resistivity = self.simulation._Mf_resistivity
        self._Mf_resistivity_deriv = self.simulation._Mf_resistivity_deriv
        self._resistivity = self.simulation.resistivity
        self._perm_inv = self.simulation._perm_inv
        self._aveF2CCV = self.simulation.mesh.aveF2CCV
        self._aveE2CCV = self.simulation.mesh.aveE2CCV
        self._nC = self.simulation.mesh.nC
        self._inv_Mf = self.simulation._inv_Mf
        self._inv_Me = self.simulation._inv_Me

    def _GLoc(self, fieldType):
        if fieldType in ["h", "hSecondary", "hPrimary", "b"]:
            return "E"
        elif fieldType in ["j", "jSecondary", "jPrimary", "e"]:
            return "F"
        else:
            raise Exception("Field type must be e, b, h, j")

    def _hPrimary(self, hSolution, source_list):
        """
        Primary magnetic field from source

        :param numpy.ndarray eSolution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: primary magnetic field as defined by the sources
        """

        hPrimary = np.zeros_like(hSolution, dtype=complex)
        for i, src in enumerate(source_list):
            hp = src.hPrimary(self.simulation)
            hPrimary[:, i] = hPrimary[:, i] + hp
        return hPrimary

    def _hSecondary(self, hSolution, source_list):
        """
        Secondary magnetic field is the thing we solved for

        :param numpy.ndarray hSolution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: secondary magnetic field
        """

        return hSolution

    def _hDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Partial derivative of the total magnetic field with respect to the
        thing we solved for.

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic field with respect
            to the field we solved for with a vector
        """

        return Identity() * du_dm_v

    def _hDeriv_m(self, src, v, adjoint=False):
        """
        Partial derivative of the total magnetic field with respect to the
            inversion model. Here, we assume that the primary does not depend
            on the model. Note that this also includes derivative contributions
            from the sources.

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: simpeg.utils.Zero
        :return: product of the magnetic field derivative with respect to the
            inversion model with a vector
        """

        return src.hPrimaryDeriv(self.simulation, v, adjoint)

    def _jPrimary(self, hSolution, source_list):
        """
        Primary current density from source

        :param numpy.ndarray hSolution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: primary current density as defined by the sources
        """

        jPrimary = np.zeros(
            [self._edgeCurl.shape[0], hSolution.shape[1]], dtype=complex
        )
        for i, src in enumerate(source_list):
            jp = src.jPrimary(self.simulation)
            jPrimary[:, i] = jPrimary[:, i] + jp
        return jPrimary

    def _jSecondary(self, hSolution, source_list):
        """
        Secondary current density from hSolution

        :param numpy.ndarray hSolution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: secondary current density
        """

        j = self._edgeCurl * hSolution
        for i, src in enumerate(source_list):
            s_e = src.s_e(self.simulation)
            j[:, i] = j[:, i] + -s_e
        return j

    def _jDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Derivative of the current density with respect to the thing we solved
        for

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the current density with respect
            to the field we solved for with a vector
        """

        if not adjoint:
            return self._edgeCurl * du_dm_v
        elif adjoint:
            return self._edgeCurl.T * du_dm_v

    def _jDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the current density with respect to the inversion model.

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the current density derivative with respect to the
            inversion model with a vector
        """

        return -src.s_eDeriv(self.simulation, v, adjoint) + src.jPrimaryDeriv(
            self.simulation, v, adjoint
        )

    def _e(self, hSolution, source_list):
        """
        Electric field from hSolution

        :param numpy.ndarray hSolution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: electric field
        """

        return self._inv_Mf * (self._Mf_resistivity * self._j(hSolution, source_list))

    def _eDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Derivative of the electric field with respect to the thing we solved
        for

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the electric field with respect
            to the field we solved for with a vector
        """
        if adjoint:
            return self._edgeCurl.T * (
                self._Mf_resistivity.T * (self._inv_Mf * du_dm_v)
            )
        return self._inv_Mf * (self._Mf_resistivity * self._edgeCurl * du_dm_v)

    def _eDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the electric field with respect to the inversion model.

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the electric field derivative with respect to the
            inversion model with a vector
        """
        hSolution = mkvc(self[src, "hSolution"])
        s_e = src.s_e(self.simulation)

        if adjoint:
            w = self._inv_Mf.T * v
            return (
                self._Mf_resistivity_deriv(self._edgeCurl * hSolution, w, adjoint)
                - self._Mf_resistivity_deriv(s_e, w, adjoint)
                + src.ePrimaryDeriv(self.simulation, v, adjoint)
            )
        return self._inv_Mf * (
            self._Mf_resistivity_deriv(self._edgeCurl * hSolution, v)
            - self._Mf_resistivity_deriv(s_e, v)
        ) + src.ePrimaryDeriv(self.simulation, v, adjoint)

    def _b(self, hSolution, source_list):
        """
        Magnetic flux density from hSolution

        :param numpy.ndarray hSolution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: magnetic flux density
        """
        h = self._h(hSolution, source_list)
        return self._inv_Me * (self._Me_permeability * h)

    def _bDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Derivative of the magnetic flux density with respect to the thing we
        solved for

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic flux density with
            respect to the field we solved for with a vector
        """
        if adjoint:
            return self._Me_permeability.T * (self._inv_Me.T * du_dm_v)
        return self._inv_Me * (self._Me_permeability * du_dm_v)

    def _bDeriv_permeability(self, src, v, adjoint=False):
        h = self[src, "h"]

        if adjoint:
            return self._Me_permeability_deriv(h, self._inv_Me.T * v, adjoint)
        return self._inv_Me * self._Me_permeability_deriv(h, v)

    def _bDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the magnetic flux density with respect to the inversion
        model.

        :param simpeg.electromagnetics.frequency_domain.sources.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the magnetic flux density derivative with respect
            to the inversion model with a vector
        """
        return src.bPrimaryDeriv(
            self.simulation, v, adjoint
        ) + self._bDeriv_permeability(src, v, adjoint)

    def _charge(self, hSolution, source_list):
        r"""
        .. math::

            \int \nabla \codt \vec{e} =  \int \frac{\rho_v }{\epsillon_0}
        """
        return self.mesh.cell_volumes[:, None] * self._charge_density(
            hSolution, source_list
        )

    def _charge_density(self, hSolution, source_list):
        r"""
        .. math::

            \frac{1}{V}\int \nabla \codt \vec{e} =
            \frac{1}{V}\int \frac{\rho_v }{\epsillon_0}
        """
        return epsilon_0 * (self._faceDiv * self._e(hSolution, source_list))
