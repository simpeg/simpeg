import numpy as np
from scipy.constants import epsilon_0

from ...fields import TimeFields
from ...utils import mkvc, sdiag, Zero


class FieldsTDEM(TimeFields):
    r"""Base class for storing TDEM fields.

    TDEM fields classes are used to store the discrete solution of the fields for a
    corresponding TDEM simulation; see :class:`.time_domain.BaseTDEMSimulation`.
    Only one field type (e.g. ``'e'``, ``'j'``, ``'h'``, ``'b'``) is stored, but certain field types
    can be rapidly computed and returned on the fly. The field type that is stored and the
    field types that can be returned depend on the formulation used by the associated simulation class.
    Once a field object has been created, the individual fields can be accessed; see the example below.

    Parameters
    ----------
    simulation : .time_domain.BaseTDEMSimulation
        The TDEM simulation object used to compute the discrete field solution.

    Example
    -------
    We want to access the fields for a discrete solution with :math:`\mathbf{e}` discretized
    to edges and :math:`\mathbf{b}` discretized to faces. To extract the fields for all sources
    and all time steps:

    .. code-block:: python

        f = simulation.fields(m)
        e = f[:, 'e', :]
        b = f[:, 'b', :]

    The array ``e`` returned will have shape (`n_edges`, `n_sources`, `n_steps`).
    And the array ``b`` returned will have shape (`n_faces`, `n_sources`, `n_steps`).
    We can also extract the fields for
    a subset of the source list used for the simulation and/or a subset of the time steps as follows:

    .. code-block:: python

        f = simulation.fields(m)
        e = f[source_list, 'e', t_inds]
        b = f[source_list, 'b', t_inds]

    """

    def __init__(self, simulation):
        dtype = float
        super().__init__(simulation=simulation, dtype=dtype)

    def _GLoc(self, fieldType):
        """Grid location of the fieldType"""
        return self.aliasFields[fieldType][1]

    def _eDeriv(self, tInd, src, dun_dm_v, v, adjoint=False):
        if adjoint is True:
            return (
                self._eDeriv_u(tInd, src, v, adjoint),
                self._eDeriv_m(tInd, src, v, adjoint),
            )
        return self._eDeriv_u(tInd, src, dun_dm_v) + self._eDeriv_m(tInd, src, v)

    def _bDeriv(self, tInd, src, dun_dm_v, v, adjoint=False):
        if adjoint is True:
            return (
                self._bDeriv_u(tInd, src, v, adjoint),
                self._bDeriv_m(tInd, src, v, adjoint),
            )
        return self._bDeriv_u(tInd, src, dun_dm_v) + self._bDeriv_m(tInd, src, v)

    def _dbdtDeriv(self, tInd, src, dun_dm_v, v, adjoint=False):
        if adjoint is True:
            return (
                self._dbdtDeriv_u(tInd, src, v, adjoint),
                self._dbdtDeriv_m(tInd, src, v, adjoint),
            )
        return self._dbdtDeriv_u(tInd, src, dun_dm_v) + self._dbdtDeriv_m(tInd, src, v)

    def _hDeriv(self, tInd, src, dun_dm_v, v, adjoint=False):
        if adjoint is True:
            return (
                self._hDeriv_u(tInd, src, v, adjoint),
                self._hDeriv_m(tInd, src, v, adjoint),
            )
        return self._hDeriv_u(tInd, src, dun_dm_v) + self._hDeriv_m(tInd, src, v)

    def _dhdtDeriv(self, tInd, src, dun_dm_v, v, adjoint=False):
        if adjoint is True:
            return (
                self._dhdtDeriv_u(tInd, src, v, adjoint),
                self._dhdtDeriv_m(tInd, src, v, adjoint),
            )
        return self._dhdtDeriv_u(tInd, src, dun_dm_v) + self._dhdtDeriv_m(tInd, src, v)

    def _jDeriv(self, tInd, src, dun_dm_v, v, adjoint=False):
        if adjoint is True:
            return (
                self._jDeriv_u(tInd, src, v, adjoint),
                self._jDeriv_m(tInd, src, v, adjoint),
            )
        return self._jDeriv_u(tInd, src, dun_dm_v) + self._jDeriv_m(tInd, src, v)


class FieldsDerivativesEB(FieldsTDEM):
    r"""Field class for stashing derivatives for EB formulations.

    Parameters
    ----------
    simulation : .time_domain.BaseTDEMSimulation
        The TDEM simulation object associated with the fields.

    """

    def __init__(self, simulation):
        super().__init__(simulation=simulation)
        self._knownFields = {
            "bDeriv": "F",
            "eDeriv": "E",
            "hDeriv": "F",
            "jDeriv": "E",
            "dbdtDeriv": "F",
            "dhdtDeriv": "F",
        }


class FieldsDerivativesHJ(FieldsTDEM):
    r"""Field class for stashing derivatives for HJ formulations.

    Parameters
    ----------
    simulation : .time_domain.BaseTDEMSimulation
        The TDEM simulation object associated with the fields.

    """

    def __init__(self, simulation):
        super().__init__(simulation=simulation)
        self._knownFields = {
            "bDeriv": "E",
            "eDeriv": "F",
            "hDeriv": "E",
            "jDeriv": "F",
            "dbdtDeriv": "E",
            "dhdtDeriv": "E",
        }


class Fields3DMagneticFluxDensity(FieldsTDEM):
    r"""Fields class for storing 3D total magnetic flux density solutions.

    This class stores the total magnetic flux density solution computed using a
    :class:`.time_domain.Simulation3DMagneticFluxDensity`
    simulation object. This class can be used to extract the following quantities:

    * ``'b'``, ``'h'``, ``'dbdt'`` and ``'dhdt'`` on mesh faces.
    * ``'e'`` and ``'j'`` on mesh edges.

    See the example below to learn how fields can be extracted from a
    ``Fields3DMagneticFluxDensity`` object.

    Parameters
    ----------
    simulation : .time_domain.Simulation3DMagneticFluxDensity
        The TDEM simulation object associated with the fields.

    Example
    -------
    The ``Fields3DMagneticFluxDensity`` object stores the total magnetic flux density solution
    on mesh faces. To extract the discrete electric fields and magnetic flux
    densities for all sources and time-steps:

    .. code-block:: python

        f = simulation.fields(m)
        e = f[:, 'e', :]
        b = f[:, 'b', :]

    The array ``e`` returned will have shape (`n_edges`, `n_sources`, `n_steps`).
    And the array ``b`` returned will have shape (`n_faces`, `n_sources`, `n_steps`).
    We can also extract the fields for a subset of the sources and time-steps as follows:

    .. code-block:: python

        f = simulation.fields(m)
        e = f[source_list, 'e', t_inds]
        b = f[source_list, 'b', t_inds]

    """

    def __init__(self, simulation):
        super().__init__(simulation=simulation)
        self._knownFields = {"bSolution": "F"}
        self._aliasFields = {
            "b": ["bSolution", "F", "_b"],
            "h": ["bSolution", "F", "_h"],
            "e": ["bSolution", "E", "_e"],
            "j": ["bSolution", "E", "_j"],
            "dbdt": ["bSolution", "F", "_dbdt"],
            "dhdt": ["bSolution", "F", "_dhdt"],
        }

    def startup(self):
        # Docstring inherited from parent.
        self._times = self.simulation.times
        self._Me_conductivity = self.simulation._Me_conductivity
        self._inv_Me_conductivity = self.simulation._inv_Me_conductivity
        self._Me_conductivity_deriv = self.simulation._Me_conductivity_deriv
        self._inv_Me_conductivity_deriv = self.simulation._inv_Me_conductivity_deriv
        self._edgeCurl = self.simulation.mesh.edge_curl
        self._Mf__perm_inv = self.simulation._Mf__perm_inv
        self._timeMesh = self.simulation.time_mesh

    def _TLoc(self, fieldType):
        return "N"

    def _b(self, bSolution, source_list, tInd):
        return bSolution

    def _bDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        return dun_dm_v

    def _bDeriv_m(self, tInd, src, v, adjoint=False):
        return Zero()

    def _dbdt(self, bSolution, source_list, tInd):
        # self._timeMesh.face_divergence
        dbdt = -self._edgeCurl * self._e(bSolution, source_list, tInd)
        for i, src in enumerate(source_list):
            s_m = src.s_m(self.simulation, self._times[tInd])
            dbdt[:, i] = dbdt[:, i] + s_m
        return dbdt

    def _dbdtDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint is True:
            return -self._eDeriv_u(tInd, src, self._edgeCurl.T * dun_dm_v, adjoint)
        return -(self._edgeCurl * self._eDeriv_u(tInd, src, dun_dm_v))

    def _dbdtDeriv_m(self, tInd, src, v, adjoint=False):
        if adjoint is True:
            return -(self._eDeriv_m(tInd, src, self._edgeCurl.T * v, adjoint))
        return -(
            self._edgeCurl * self._eDeriv_m(tInd, src, v)
        )  # + src.s_mDeriv() assuming src doesn't have deriv for now

    def _e(self, bSolution, source_list, tInd):
        e = self._inv_Me_conductivity * (
            self._edgeCurl.T * (self._Mf__perm_inv * bSolution)
        )
        for i, src in enumerate(source_list):
            s_e = src.s_e(self.simulation, self._times[tInd])
            e[:, i] = e[:, i] - self._inv_Me_conductivity * s_e
        return e

    def _eDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint is True:
            return self._Mf__perm_inv.T * (
                self._edgeCurl * (self._inv_Me_conductivity.T * dun_dm_v)
            )
        return self._inv_Me_conductivity * (
            self._edgeCurl.T * (self._Mf__perm_inv * dun_dm_v)
        )

    def _eDeriv_m(self, tInd, src, v, adjoint=False):
        _, s_e = src.eval(self.simulation, self._times[tInd])
        bSolution = self[[src], "bSolution", tInd].flatten()

        _, s_eDeriv = src.evalDeriv(self._times[tInd], self, adjoint=adjoint)

        if adjoint is True:
            return self._inv_Me_conductivity_deriv(
                -s_e + self._edgeCurl.T * (self._Mf__perm_inv * bSolution), v, adjoint
            ) - s_eDeriv(self._inv_Me_conductivity.T * v)

        return self._inv_Me_conductivity_deriv(
            -s_e + self._edgeCurl.T * (self._Mf__perm_inv * bSolution), v, adjoint
        ) - self._inv_Me_conductivity * s_eDeriv(v)

    def _j(self, hSolution, source_list, tInd):
        return self.simulation.MeI * (
            self._Me_conductivity * self._e(hSolution, source_list, tInd)
        )

    def _jDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return self._eDeriv_u(
                tInd,
                src,
                self._Me_conductivity.T * (self.simulation.MeI.T * dun_dm_v),
                adjoint=True,
            )
        return self.simulation.MeI * (
            self._Me_conductivity * self._eDeriv_u(tInd, src, dun_dm_v)
        )

    def _jDeriv_m(self, tInd, src, v, adjoint=False):
        e = self[src, "e", tInd]
        if adjoint:
            w = self.simulation.MeI.T * v
            return self._Me_conductivity_deriv(e).T * w + self._eDeriv_m(
                tInd, src, self._Me_conductivity.T * w, adjoint=True
            )
        return self.simulation.MeI * (
            self._Me_conductivity_deriv(e) * v
            + self._Me_conductivity * self._eDeriv_m(tInd, src, v)
        )

    def _h(self, hSolution, source_list, tInd):
        return self.simulation.MfI * (
            self._Mf__perm_inv * self._b(hSolution, source_list, tInd)
        )

    def _hDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return self._bDeriv_u(
                tInd,
                src,
                self._Mf__perm_inv.T * (self.simulation.MfI.T * dun_dm_v),
                adjoint=True,
            )
        return self.simulation.MfI * (
            self._Mf__perm_inv * self._bDeriv_u(tInd, src, dun_dm_v)
        )

    def _hDeriv_m(self, tInd, src, v, adjoint=False):
        if adjoint:
            return self._bDeriv_m(
                tInd,
                src,
                self._Mf__perm_inv.T * (self.simulation.MfI.T * v),
                adjoint=True,
            )
        return self.simulation.MfI * (self._Mf__perm_inv * self._bDeriv_m(tInd, src, v))

    def _dhdt(self, hSolution, source_list, tInd):
        return self.simulation.MfI * (
            self._Mf__perm_inv * self._dbdt(hSolution, source_list, tInd)
        )

    def _dhdtDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return self._dbdtDeriv_u(
                tInd,
                src,
                self._Mf__perm_inv.T * (self.simulation.MfI.T * dun_dm_v),
                adjoint=True,
            )
        return self.simulation.MfI * (
            self._Mf__perm_inv * self._dbdtDeriv_u(tInd, src, dun_dm_v)
        )

    def _dhdtDeriv_m(self, tInd, src, v, adjoint=False):
        if adjoint:
            return self._dbdtDeriv_m(
                tInd,
                src,
                self._Mf__perm_inv.T * (self.simulation.MfI.T * v),
                adjoint=True,
            )
        return self.simulation.MfI * (
            self._Mf__perm_inv * self._dbdtDeriv_m(tInd, src, v)
        )


class Fields3DElectricField(FieldsTDEM):
    r"""Fields class for storing 3D total electric field solutions.

    This class stores the total electric field solution computed using a
    :class:`.time_domain.Simulation3DElectricField`
    simulation object. This class can be used to extract the following quantities:

    * ``'e'`` and ``'j'`` on mesh edges.
    * ``'b'``, ``'dbdt'`` and ``'dhdt'`` on mesh faces.

    See the example below to learn how fields can be extracted from a
    ``Fields3DElectricField`` object.

    Parameters
    ----------
    simulation : .time_domain.Simulation3DElectricField
        The TDEM simulation object associated with the fields.

    Example
    -------
    The ``Fields3DElectricField`` object stores the total electric field solution
    on mesh edges. To extract the discrete electric fields and db/dt
    for all sources and time-steps:

    .. code-block:: python

        f = simulation.fields(m)
        e = f[:, 'e', :]
        dbdt = f[:, 'dbdt', :]

    The array ``e`` returned will have shape (`n_edges`, `n_sources`, `n_steps`).
    And the array ``dbdt`` returned will have shape (`n_faces`, `n_sources`, `n_steps`).
    We can also extract the fields for a subset of the sources and time-steps as follows:

    .. code-block:: python

        f = simulation.fields(m)
        e = f[source_list, 'e', t_inds]
        dbdt = f[source_list, 'dbdt', t_inds]

    """

    def __init__(self, simulation):
        super().__init__(simulation=simulation)
        self._knownFields = {"eSolution": "E"}
        self._aliasFields = {
            "e": ["eSolution", "E", "_e"],
            "j": ["eSolution", "E", "_j"],
            "b": ["eSolution", "F", "_b"],
            # 'h': ['eSolution', 'F', '_h'],
            "dbdt": ["eSolution", "F", "_dbdt"],
            "dhdt": ["eSolution", "F", "_dhdt"],
        }

    def startup(self):
        # Docstring inherited from parent.
        self._times = self.simulation.times
        self._Me_conductivity = self.simulation._Me_conductivity
        self._inv_Me_conductivity = self.simulation._inv_Me_conductivity
        self._Me_conductivity_deriv = self.simulation._Me_conductivity_deriv
        self._inv_Me_conductivity_deriv = self.simulation._inv_Me_conductivity_deriv
        self._edgeCurl = self.simulation.mesh.edge_curl
        self._Mf__perm_inv = self.simulation._Mf__perm_inv

    def _TLoc(self, fieldType):
        return "N"

    def _e(self, eSolution, source_list, tInd):
        return eSolution

    def _eDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        return dun_dm_v

    def _eDeriv_m(self, tInd, src, v, adjoint=False):
        return Zero()

    def _dbdt(self, eSolution, source_list, tInd):
        s_m = np.zeros((self.mesh.nF, len(source_list)))
        for i, src in enumerate(source_list):
            s_m_src = src.s_m(self.simulation, self._times[tInd])
            s_m[:, i] = s_m[:, i] + s_m_src
        return s_m - self._edgeCurl * eSolution

    def _dbdtDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return -self._edgeCurl.T * dun_dm_v
        return -self._edgeCurl * dun_dm_v

    def _dbdtDeriv_m(self, tInd, src, v, adjoint=False):
        # s_mDeriv = src.s_mDeriv(
        #     self._times[tInd], self, adjoint=adjoint
        # )
        return Zero()  # assumes source doesn't depend on model

    def _b(self, eSolution, source_list, tInd):
        """
        Integrate _db_dt using rectangles
        """
        raise NotImplementedError(
            "To obtain b-fields, please use Simulation3DMagneticFluxDensity"
        )
        # dbdt = self._dbdt(eSolution, source_list, tInd)
        # dt = self.simulation.time_mesh.h[0]
        # # assume widths of "ghost cells" same on either end
        # dtn = np.hstack([dt[0], 0.5*(dt[1:] + dt[:-1]), dt[-1]])
        # return dtn[tInd] * dbdt
        # # raise NotImplementedError

    def _j(self, eSolution, source_list, tInd):
        return self.simulation.MeI * (
            self._Me_conductivity * self._e(eSolution, source_list, tInd)
        )

    def _jDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return self._eDeriv_u(
                tInd,
                src,
                self._Me_conductivity.T * (self.simulation.MeI.T * dun_dm_v),
                adjoint=True,
            )
        return self.simulation.MeI * (
            self._Me_conductivity * self._eDeriv_u(tInd, src, dun_dm_v)
        )

    def _jDeriv_m(self, tInd, src, v, adjoint=False):
        e = self[src, "e", tInd]
        if adjoint:
            w = self.simulation.MeI.T * v
            return self._Me_conductivity_deriv(e).T * w + self._eDeriv_m(
                tInd, src, self._Me_conductivity.T * w, adjoint=True
            )
        return self.simulation.MeI * (
            self._Me_conductivity_deriv(e) * v
            + self._Me_conductivity * self._eDeriv_m(tInd, src, v)
        )

    def _dhdt(self, eSolution, source_list, tInd):
        return self.simulation.MfI * (
            self._Mf__perm_inv * self._dbdt(eSolution, source_list, tInd)
        )

    def _dhdtDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return self._dbdtDeriv_u(
                tInd,
                src,
                self._Mf__perm_inv.T * (self.simulation.MfI.T * dun_dm_v),
                adjoint=True,
            )
        return self.simulation.MfI * (
            self._Mf__perm_inv * self._dbdtDeriv_u(tInd, src, dun_dm_v)
        )

    def _dhdtDeriv_m(self, tInd, src, v, adjoint=False):
        if adjoint:
            return self._dbdtDeriv_m(
                tInd, src, self._Mf__perm_inv.T * (self.simulation.MfI.T * v)
            )
        return self.simulation.MfI * (
            self._Mf__perm_inv * self._dbdtDeriv_m(tInd, src, v)
        )


class Fields3DMagneticField(FieldsTDEM):
    r"""Fields class for storing 3D total magnetic field solutions.

    This class stores the total magnetic field solution computed using a
    :class:`.time_domain.Simulation3DElectricField`
    simulation object. This class can be used to extract the following quantities:

    * ``'h'``, ``'b'``, ``'dbdt'`` and ``'dbdt'`` on mesh edges.
    * ``'j'`` and ``'e'`` on mesh faces.
    * ``'charge'`` at cell centers.

    See the example below to learn how fields can be extracted from a
    ``Fields3DMagneticField`` object.

    Parameters
    ----------
    simulation : .time_domain.Simulation3DMagneticField
        The TDEM simulation object associated with the fields.

    Example
    -------
    The ``Fields3DMagneticField`` object stores the total magnetic field solution
    on mesh edges. To extract the discrete magnetic fields and current density
    for all sources and time-steps:

    .. code-block:: python

        f = simulation.fields(m)
        h = f[:, 'h', :]
        j = f[:, 'j', :]

    The array ``h`` returned will have shape (`n_edges`, `n_sources`, `n_steps`).
    And the array ``j`` returned will have shape (`n_faces`, `n_sources`, `n_steps`).
    We can also extract the fields for a subset of the sources and time-steps as follows:

    .. code-block:: python

        f = simulation.fields(m)
        h = f[source_list, 'e', t_inds]
        j = f[source_list, 'j', t_inds]

    """

    def __init__(self, simulation):
        super().__init__(simulation=simulation)
        self._knownFields = {"hSolution": "E"}
        self._aliasFields = {
            "h": ["hSolution", "E", "_h"],
            "b": ["hSolution", "E", "_b"],
            "dhdt": ["hSolution", "E", "_dhdt"],
            "dbdt": ["hSolution", "E", "_dbdt"],
            "j": ["hSolution", "F", "_j"],
            "e": ["hSolution", "F", "_e"],
            "charge": ["hSolution", "CC", "_charge"],
        }

    def startup(self):
        # Docstring inherited from parent.
        self._times = self.simulation.times
        self._edgeCurl = self.simulation.mesh.edge_curl
        self._MeMuI = self.simulation.MeMuI
        self._MeMu = self.simulation.MeMu
        self._Mf_resistivity = self.simulation._Mf_resistivity
        self._Mf_resistivity_deriv = self.simulation._Mf_resistivity_deriv

    def _TLoc(self, fieldType):
        # if fieldType in ['h', 'j']:
        return "N"
        # else:
        #     raise NotImplementedError

    def _h(self, hSolution, source_list, tInd):
        return hSolution

    def _hDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        return dun_dm_v

    def _hDeriv_m(self, tInd, src, v, adjoint=False):
        return Zero()

    def _dhdt(self, hSolution, source_list, tInd):
        C = self._edgeCurl
        MeMuI = self._MeMuI
        MfRho = self._Mf_resistivity

        dhdt = -MeMuI * (C.T * (MfRho * (C * hSolution)))

        for i, src in enumerate(source_list):
            s_m, s_e = src.eval(self.simulation, self._times[tInd])
            dhdt[:, i] = MeMuI * (C.T * MfRho * s_e + s_m) + dhdt[:, i]
        return dhdt

    def _dhdtDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        C = self._edgeCurl
        MeMuI = self._MeMuI
        MfRho = self._Mf_resistivity

        if adjoint:
            return -C.T * (MfRho.T * (C * (MeMuI * dun_dm_v)))
        return -MeMuI * (C.T * (MfRho * (C * dun_dm_v)))

    def _dhdtDeriv_m(self, tInd, src, v, adjoint=False):
        C = self._edgeCurl
        MeMuI = self._MeMuI
        MfRhoDeriv = self._Mf_resistivity_deriv

        hSolution = self[[src], "hSolution", tInd].flatten()
        s_e = src.s_e(self.simulation, self._times[tInd])

        if adjoint:
            return -MfRhoDeriv(C * hSolution - s_e, (C * (MeMuI * v)), adjoint)
        return -MeMuI * (C.T * (MfRhoDeriv(C * hSolution - s_e, v, adjoint)))

    def _j(self, hSolution, source_list, tInd):
        s_e = np.zeros((self.mesh.nF, len(source_list)))
        for i, src in enumerate(source_list):
            s_e_src = src.s_e(self.simulation, self._times[tInd])
            s_e[:, i] = s_e[:, i] + s_e_src

        return self._edgeCurl * hSolution - s_e

    def _jDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return self._edgeCurl.T * dun_dm_v
        return self._edgeCurl * dun_dm_v

    def _jDeriv_m(self, tInd, src, v, adjoint=False):
        return Zero()  # assumes the source doesn't depend on the model

    def _b(self, hSolution, source_list, tInd):
        h = self._h(hSolution, source_list, tInd)
        return self.simulation.MeI * (self._MeMu * h)

    def _bDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return self._hDeriv_u(
                tInd,
                src,
                self._MeMu.T * (self.simulation.MeI.T * dun_dm_v),
                adjoint=adjoint,
            )
        return self.simulation.MeI * (self._MeMu * self._hDeriv_u(tInd, src, dun_dm_v))

    def _bDeriv_m(self, tInd, src, v, adjoint=False):
        if adjoint:
            return self._hDeriv_m(
                tInd, src, self._MeMu.T * (self.simulation.MeI.T * v), adjoint=adjoint
            )
        return self.simulation.MeI * (self._MeMu * self._hDeriv_m(tInd, src, v))

    def _dbdt(self, hSolution, source_list, tInd):
        dhdt = self._dhdt(hSolution, source_list, tInd)
        return self.simulation.MeI * (self._MeMu * dhdt)

    def _dbdtDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return self._dhdtDeriv_u(
                tInd,
                src,
                self._MeMu.T * (self.simulation.MeI.T * dun_dm_v),
                adjoint=adjoint,
            )
        return self.simulation.MeI * (
            self._MeMu * self._dhdtDeriv_u(tInd, src, dun_dm_v)
        )

    def _dbdtDeriv_m(self, tInd, src, v, adjoint=False):
        if adjoint:
            return self._dhdtDeriv_m(
                tInd, src, self._MeMu.T * (self.simulation.MeI.T * v), adjoint=adjoint
            )
        return self.simulation.MeI * (self._MeMu * self._dhdtDeriv_m(tInd, src, v))

    def _e(self, hSolution, source_list, tInd):
        return self.simulation.MfI * (
            self._Mf_resistivity * self._j(hSolution, source_list, tInd)
        )

    def _eDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return self._jDeriv_u(
                tInd,
                src,
                self._Mf_resistivity.T * (self.simulation.MfI.T * dun_dm_v),
                adjoint=True,
            )
        return self.simulation.MfI * (
            self._Mf_resistivity * self._jDeriv_u(tInd, src, dun_dm_v)
        )

    def _eDeriv_m(self, tInd, src, v, adjoint=False):
        j = mkvc(self[src, "j", tInd])
        if adjoint is True:
            return self._Mf_resistivity_deriv(
                j, self.simulation.MfI.T * v, adjoint
            ) + self._jDeriv_m(tInd, src, self._Mf_resistivity * v)
        return self.simulation.MfI * (
            self._Mf_resistivity_deriv(j, v)
            + self._Mf_resistivity * self._jDeriv_m(tInd, src, v)
        )

    def _charge(self, hSolution, source_list, tInd):
        vol = sdiag(self.simulation.mesh.cell_volumes)
        return (
            epsilon_0
            * vol
            * (
                self.simulation.mesh.face_divergence
                * self._e(hSolution, source_list, tInd)
            )
        )


class Fields3DCurrentDensity(FieldsTDEM):
    r"""Fields class for storing 3D current density solutions.

    This class stores the total current density solution computed using a
    :class:`.time_domain.Simulation3DCurrentDensity`
    simulation object. This class can be used to extract the following quantities:

    * ``'j'`` and ``'e'`` on mesh faces.
    * ``'dbdt'`` and ``'dhdt'`` on mesh edges.
    * ``'charge'`` and ``'charge_density'`` at cell centers.

    See the example below to learn how fields can be extracted from a
    ``Fields3DCurrentDensity`` object.

    Parameters
    ----------
    simulation : .time_domain.Simulation3DCurrentDensity
        The TDEM simulation object associated with the fields.

    Example
    -------
    The ``Fields3DCurrentDensity`` object stores the total current density solution
    on mesh faces. To extract the discrete current densities and magnetic fields
    for all sources and time-steps:

    .. code-block:: python

        f = simulation.fields(m)
        j = f[:, 'j', :]
        h = f[:, 'h', :]

    The array ``j`` returned will have shape (`n_faces`, `n_sources`, `n_steps`).
    And the array ``h`` returned will have shape (`n_edges`, `n_sources`, `n_steps`).
    We can also extract the fields for a subset of the sources and time-steps as follows:

    .. code-block:: python

        f = simulation.fields(m)
        j = f[source_list, 'j', t_inds]
        h = f[source_list, 'h', t_inds]

    """

    def __init__(self, simulation):
        super().__init__(simulation=simulation)
        self._knownFields = {"jSolution": "F"}
        self._aliasFields = {
            "dhdt": ["jSolution", "E", "_dhdt"],
            "dbdt": ["jSolution", "E", "_dbdt"],
            "j": ["jSolution", "F", "_j"],
            "e": ["jSolution", "F", "_e"],
            "charge": ["jSolution", "CC", "_charge"],
            "charge_density": ["jSolution", "CC", "_charge_density"],
        }

    def startup(self):
        # Docstring inherited from parent.
        self._times = self.simulation.times
        self._edgeCurl = self.simulation.mesh.edge_curl
        self._MeMuI = self.simulation.MeMuI
        self._Mf_resistivity = self.simulation._Mf_resistivity
        self._Mf_resistivity_deriv = self.simulation._Mf_resistivity_deriv

    def _TLoc(self, fieldType):
        # if fieldType in ['h', 'j']:
        return "N"

    def _j(self, jSolution, source_list, tInd):
        return jSolution

    def _jDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        return dun_dm_v

    def _jDeriv_m(self, tInd, src, v, adjoint=False):
        return Zero()

    def _h(self, jSolution, source_list, tInd):
        raise NotImplementedError(
            "Please use Simulation3DMagneticField to get h-fields"
        )

    def _dhdt(self, jSolution, source_list, tInd):
        C = self._edgeCurl
        MfRho = self._Mf_resistivity
        MeMuI = self._MeMuI

        dhdt = -MeMuI * (C.T * (MfRho * jSolution))
        for i, src in enumerate(source_list):
            s_m = src.s_m(self.simulation, self.simulation.times[tInd])
            dhdt[:, i] = MeMuI * s_m + dhdt[:, i]
        return dhdt

    def _dhdtDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        C = self._edgeCurl
        MfRho = self._Mf_resistivity
        MeMuI = self._MeMuI

        if adjoint is True:
            return -MfRho.T * (C * (MeMuI.T * dun_dm_v))
        return -MeMuI * (C.T * (MfRho * dun_dm_v))

    def _dhdtDeriv_m(self, tInd, src, v, adjoint=False):
        jSolution = self[[src], "jSolution", tInd].flatten()
        C = self._edgeCurl
        MeMuI = self._MeMuI

        if adjoint is True:
            return -self._Mf_resistivity_deriv(jSolution, C * (MeMuI * v), adjoint)
        return -MeMuI * (C.T * (self._Mf_resistivity_deriv(jSolution, v)))

    def _e(self, jSolution, source_list, tInd):
        return self.simulation.MfI * (
            self._Mf_resistivity * self._j(jSolution, source_list, tInd)
        )

    def _eDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint is True:
            return self._Mf_resistivity.T * (self.simulation.MfI.T * dun_dm_v)
        return self.simulation.MfI * (self._Mf_resistivity * dun_dm_v)

    def _eDeriv_m(self, tInd, src, v, adjoint=False):
        jSolution = mkvc(self[src, "jSolution", tInd])
        if adjoint:
            return self._Mf_resistivity_deriv(
                jSolution, self.simulation.MfI.T * v, adjoint
            )
        return self.simulation.MfI * self._Mf_resistivity_deriv(jSolution, v)

    def _charge(self, jSolution, source_list, tInd):
        vol = sdiag(self.simulation.mesh.cell_volumes)
        return vol * self._charge_density(jSolution, source_list, tInd)

    def _charge_density(self, jSolution, source_list, tInd):
        return epsilon_0 * (
            self.simulation.mesh.face_divergence * self._e(jSolution, source_list, tInd)
        )

    def _dbdt(self, jSolution, source_list, tInd):
        dhdt = mkvc(self._dhdt(jSolution, source_list, tInd))
        return self.simulation.MeI * (self.simulation.MeMu * dhdt)

    def _dbdtDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        # dhdt = mkvc(self[src, 'dhdt', tInd])
        if adjoint:
            return self._dhdtDeriv_u(
                tInd,
                src,
                self.simulation.MeMu.T * (self.simulation.MeI.T * dun_dm_v),
                adjoint,
            )
        return self.simulation.MeI * (
            self.simulation.MeMu * self._dhdtDeriv_u(tInd, src, dun_dm_v)
        )

    def _dbdtDeriv_m(self, tInd, src, v, adjoint=False):
        if adjoint:
            return self._dhdtDeriv_m(
                tInd, src, self.simulation.MeMu.T * (self.simulation.MeI.T * v), adjoint
            )
        return self.simulation.MeI * (
            self.simulation.MeMu * self._dhdtDeriv_m(tInd, src, v)
        )
