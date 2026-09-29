import numpy as np
import scipy.sparse as sp
from discretize import TensorMesh, TreeMesh
from discretize.utils import Zero

from ...base.pde_simulation import _inner_mat_mul_op
from ...utils import mkvc
from ... import maps
from ..frequency_domain.simulation import BaseFDEMSimulation, Simulation3DElectricField
from ..frequency_domain.survey import Survey
from ..utils import omega
from .sources import Planewave
from .receivers import Impedance, Tipper, Admittance
from .fields import (
    Fields1DPrimarySecondary,
    Fields1DElectricField,
    Fields1DMagneticField,
    Fields2DElectricField,
    Fields2DMagneticField,
)


def _centers_to_widths(centers):
    """Cell widths from cell centers measured from the start of the first cell."""
    centers = np.asarray(centers)
    d = np.empty_like(centers)
    n = centers.shape[-1]
    d[..., 0] = 2 * centers[..., 0]
    for i in range(1, n):
        d[..., i] = 2 * (centers[..., i] - centers[..., i - 1]) - d[..., i - 1]
    return d


###################################
# 1D problems
###################################


def _bottom_robin_matrix(n, value):
    """Sparse (n, n) matrix with ``value`` in the entry for the bottom node."""
    return sp.csr_matrix(([value], ([0], [0])), shape=(n, n))


def _bottom_robin_deriv(prop_deriv, d_value, u, v, adjoint):
    """Derivative of the bottom Robin term times ``u`` with respect to the model.

    The term only depends on the property of the bottom cell (cell 0), through
    ``d_value``, and only acts on the bottom node (node 0).
    """
    # squeeze like _inner_mat_mul_op, so the shapes of the two parts match
    u = np.squeeze(np.asarray(u))
    v = np.squeeze(np.asarray(v))
    e0 = np.zeros(prop_deriv.shape[0])
    e0[0] = 1.0
    deriv_row = prop_deriv.T @ e0
    if adjoint:
        coef = d_value * u[0] * v[0]
        if u.ndim > 1:
            # sum over the fields stored in the columns of u
            coef = np.sum(coef)
        return np.multiply.outer(deriv_row, coef) if np.ndim(coef) else deriv_row * coef
    d_prop = deriv_row @ v
    out = np.zeros((u.shape[0],) + np.broadcast(u[0], d_prop).shape, dtype=complex)
    out[0] = d_value * u[0] * d_prop
    return out


class Simulation1DElectricField(BaseFDEMSimulation):
    r"""
    1D finite volume simulation for the natural source electromagnetic problem.

    This corresponds to the TE mode 2D simulation where the electric field is
    located on nodes and the magnetic field at cell centers.

    We are solving the discrete version of

    .. math::

        \partial_z E_y = i \omega \mu_0 H_x = 0

        sigma E_y = \partial_z H_x

    with default boundary conditions that $H_x[z_max] = 1$ (a plane wave source at
    the top of the domain), and $H_x[z_min] = 0$.

    When we discretize, we obtain:

    where the electric field is defined on nodes, and the magnetic field is
    defined on cell centers.
    """

    _solutionType = "eSolution"
    _formulation = "EB"  # electric-field component is on nodes
    fieldsPair = Fields1DElectricField

    def __init__(self, mesh, **kwargs):
        if mesh.dim > 1:
            raise ValueError(
                f"The mesh must be a 1D mesh. The provided mesh has dimension {mesh.dim}"
            )

        super().__init__(mesh, **kwargs)

        self._rhs = mesh.boundary_node_vector_integral * [0 + 0j, 1 + 0j]

    def getA(self, freq):
        r"""
        System matrix

        .. math::

            \mathbf{A} =
                \mathbf{G}^\top \mathbf{M}^e_{\mu^{-1}} \mathbf{G}
                + 1\omega \mathbf{M}^f_\sigma
        """

        G = self.mesh.nodal_gradient
        MeMui = self.MeMui
        MfSigma = self.MfSigma

        A = G.T.tocsr() @ MeMui @ G + 1j * omega(freq) * MfSigma
        return A + _bottom_robin_matrix(A.shape[0], self._bottom_robin(freq)[0])

    def _bottom_robin(self, freq):
        r"""Robin (downgoing plane wave) term at the bottom node.

        :math:`i k / \mu` with :math:`k = \sqrt{-i \omega \mu \sigma}` for the
        bottom cell, and its derivatives with respect to the bottom cell's
        conductivity and inverse permeability.
        """
        sigma = np.atleast_1d(self.sigma)[0]
        mui = np.atleast_1d(self.mui)[0]
        value = 1j * np.sqrt(-1j * omega(freq) * sigma * mui)
        return value, value / (2 * sigma), value / (2 * mui)

    def getADeriv_sigma(self, freq, u, v, adjoint=False):
        dA_v = 1j * omega(freq) * self.MfSigmaDeriv(u, v, adjoint=adjoint)
        if self.sigmaMap is None:
            return dA_v
        d_value = self._bottom_robin(freq)[1]
        return dA_v + _bottom_robin_deriv(self.sigmaDeriv, d_value, u, v, adjoint)

    def getADeriv_mui(self, freq, u, v, adjoint=False):
        G = self.mesh.nodal_gradient
        if adjoint:
            dA_v = self.MeMuiDeriv(G * u, G * v, adjoint)
        else:
            dA_v = G.T * self.MeMuiDeriv(G * u, v, adjoint)
        if self.muiMap is None:
            return dA_v
        d_value = self._bottom_robin(freq)[2]
        return dA_v + _bottom_robin_deriv(self.muiDeriv, d_value, u, v, adjoint)

    def getRHS(self, freq):
        """
        Right hand side constructed using Dirichlet boundary conditions
        """
        return 1j * omega(freq) * self._rhs

    def getRHSDeriv(self, freq, src, v, adjoint=False):
        return Zero()

    def getADeriv(self, freq, u, v, adjoint=False):
        return self.getADeriv_sigma(freq, u, v, adjoint) + self.getADeriv_mui(
            freq, u, v, adjoint
        )

    def getJ(self, m, f=None):
        r"""Generate the full sensitivity matrix.

        .. important::

            This method hasn't been implemented yet for this class.

        Raises
        ------
        NotImplementedError
        """
        msg = (
            "The getJ method hasn't been implemented for the "
            f"{type(self).__name__} yet."
        )
        raise NotImplementedError(msg)


class Simulation1DMagneticField(BaseFDEMSimulation):
    """
    1D finite volume simulation for the natural source electromagnetic problem.

    This corresponds to the TM mode 2D simulation where the magnetic field is
    located at faces (nodes) and the electric field is on edges (cell_centers).
    """

    _solutionType = "hSolution"
    _formulation = "HJ"
    fieldsPair = Fields1DMagneticField

    def __init__(self, mesh, **kwargs):
        if mesh.dim > 1:
            raise ValueError(
                f"The mesh must be a 1D mesh. The provided mesh has dimension {mesh.dim}"
            )

        super().__init__(mesh, **kwargs)

        # corresponds to a dirichlet boundaries at the top (= 1 ) and bottom(=0)
        # for the y component of electric field
        self._rhs = -mesh.boundary_node_vector_integral * [0 + 0j, 1 + 0j]

    def getA(self, freq):
        """
        system matrix
        """
        G = self.mesh.nodal_gradient
        MeRho = self.MeRho
        MnMu = self.MnMu

        A = G.T.tocsr() @ MeRho @ G + 1j * omega(freq) * MnMu
        return A + _bottom_robin_matrix(A.shape[0], self._bottom_robin(freq)[0])

    def _bottom_robin(self, freq):
        r"""Robin (downgoing plane wave) term at the bottom node.

        :math:`i k \rho` with :math:`k = \sqrt{-i \omega \mu \sigma}` for the
        bottom cell, and its derivatives with respect to the bottom cell's
        resistivity and permeability.
        """
        rho = np.atleast_1d(self.rho)[0]
        mu = np.atleast_1d(self.mu)[0]
        value = 1j * np.sqrt(-1j * omega(freq) * mu * rho)
        return value, value / (2 * rho), value / (2 * mu)

    def getADeriv_rho(self, freq, u, v, adjoint=False):
        G = self.mesh.nodal_gradient
        if adjoint:
            dA_v = self.MeRhoDeriv(G * u, G * v, adjoint)
        else:
            dA_v = G.T * self.MeRhoDeriv(G * u, v, adjoint)
        if self.rhoMap is None:
            return dA_v
        d_value = self._bottom_robin(freq)[1]
        return dA_v + _bottom_robin_deriv(self.rhoDeriv, d_value, u, v, adjoint)

    def getADeriv_mu(self, freq, u, v, adjoint=False):
        MnMuDeriv = self.MnMuDeriv(u)
        if adjoint is True:
            dA_v = 1j * omega(freq) * (MnMuDeriv.T * v)
        else:
            dA_v = 1j * omega(freq) * (MnMuDeriv * v)
        if self.muMap is None:
            return dA_v
        d_value = self._bottom_robin(freq)[2]
        return dA_v + _bottom_robin_deriv(self.muDeriv, d_value, u, v, adjoint)

    def getRHS(self, freq):
        """
        right hand side
        """
        return self._rhs

    def getRHSDeriv(self, freq, src, v, adjoint=False):
        return Zero()

    def getADeriv(self, freq, u, v, adjoint=False):
        return self.getADeriv_rho(freq, u, v, adjoint) + self.getADeriv_mu(
            freq, u, v, adjoint
        )

    def getJ(self, m, f=None):
        r"""Generate the full sensitivity matrix.

        .. important::

            This method hasn't been implemented yet for this class.

        Raises
        ------
        NotImplementedError
        """
        msg = (
            "The getJ method hasn't been implemented for the "
            f"{type(self).__name__} yet."
        )
        raise NotImplementedError(msg)


class Simulation1DPrimarySecondary(Simulation1DElectricField):
    r"""
    A NSEM problem solving a e formulation and primary/secondary fields decomposition.

    By eliminating the magnetic flux density using

    .. math ::

        \mathbf{b} = \frac{1}{i \omega} \left(-\mathbf{C} \mathbf{e} \right)


    we can write Maxwell's equations as a second order system in
    :math:`\mathbf{e}` only:

    .. math ::

        \left[
            \mathbf{C}^{\top} \mathbf{M_{\mu^{-1}}^e } \mathbf{C}
            + i \omega \mathbf{M_{\sigma}^f}
        \right]
        \mathbf{e}_{s}
        = i \omega \mathbf{M_{\sigma_{s}}^f } \mathbf{e}_{p}

    which we solve for :math:`\mathbf{e_s}`.
    The total field :math:`\mathbf{e} = \mathbf{e_p} + \mathbf{e_s}`.

    The primary field is estimated from a background model (commonly half space ).
    """

    fieldsPair = Fields1DPrimarySecondary

    def __init__(self, mesh, survey=None, sigmaPrimary=None, **kwargs):
        super().__init__(mesh=mesh, survey=survey, **kwargs)
        self.sigmaPrimary = sigmaPrimary

    @property
    def sigmaPrimary(self):
        """
        A background model, use for the calculation of the primary fields.

        """
        return self._sigmaPrimary

    @sigmaPrimary.setter
    def sigmaPrimary(self, val):
        # Note: TODO add logic for val, make sure it is the correct size.
        self._sigmaPrimary = val

    def getADeriv(self, freq, u, v, adjoint=False):
        """
        The derivative of A wrt sigma
        """
        # Only select the yx polarization
        return super().getADeriv(freq, u[:, 1], v, adjoint=adjoint)

    def getRHS(self, freq):
        """
        Function to return the right hand side for the system.

        :param float freq: Frequency
        :rtype: numpy.ndarray
        :return: RHS for 1 polarizations, primary fields (nF, 1)
        """

        # Get sources for the frequncy(polarizations)
        src = self.survey.get_sources_by_frequency(freq)[0]
        # Only select the yx polarization
        S_e = mkvc(src.s_e(self)[:, 1], 2)
        return -1j * omega(freq) * S_e

    def getRHSDeriv(self, freq, src, v, adjoint=False):
        """
        The derivative of the RHS wrt sigma
        """
        S_eDeriv = src.s_eDeriv_m(self, v, adjoint)
        return -1j * omega(freq) * S_eDeriv


###################################
# 2D problems
###################################
def _bottom_robin_ops_2d(mesh):
    """Operators for a Robin term on the bottom boundary edges of a 2D mesh.

    Returns
    -------
    P : (n_bottom_edges, n_edges) scipy.sparse.csr_matrix
        Selects the edges on the bottom boundary.
    K : (n_bottom_edges, n_cells) scipy.sparse.csr_matrix
        The length of each bottom edge, at the cell above it.
    """
    edges = np.r_[mesh.edges_x, mesh.edges_y]
    bottom = np.where(
        np.isclose(edges[:, 1], mesh.nodes_y[0])
        & (np.arange(len(edges)) < mesh.n_edges_x)
    )[0]
    n_bot = len(bottom)
    P = sp.csr_matrix(
        (np.ones(n_bot), (np.arange(n_bot), bottom)), shape=(n_bot, mesh.n_edges)
    )
    eps = 1e-3 * mesh.h[1].min()
    cells = np.atleast_1d(mesh.point2index(edges[bottom] + np.r_[0.0, eps]))
    K = sp.csr_matrix(
        (mesh.edge_lengths[bottom], (np.arange(n_bot), cells)),
        shape=(n_bot, mesh.n_cells),
    )
    return P, K


def _bottom_robin_deriv_2d(ops, scale, dg_dm, u, v, adjoint):
    """Model derivative of ``scale * P.T @ diag(K @ g) @ P @ u``."""
    P, K = ops
    dKg_dm = K @ dg_dm
    if adjoint:
        return _inner_mat_mul_op(dKg_dm, P @ u, P @ (scale * v), adjoint=True)
    return scale * (P.T @ _inner_mat_mul_op(dKg_dm, P @ u, v))


class Simulation2DElectricField(BaseFDEMSimulation):
    """
    A
    """

    _solutionType = "eSolution"
    _formulation = "EB"
    fieldsPair = Fields2DElectricField

    def __init__(self, mesh, h_bc=None, **kwargs):
        if mesh.dim != 2:
            raise ValueError(
                f"The mesh must be a 2D mesh. The provided mesh has dimension {mesh.dim}"
            )

        super().__init__(mesh, **kwargs)

        for src in self.survey.source_list:
            for rx in src.receiver_list:
                if not (
                    (rx.orientation == "xy" and isinstance(rx, Impedance))
                    or (rx.orientation == "yx" and isinstance(rx, Admittance))
                ):
                    raise TypeError(
                        "natural_source.Simulation2DElectricField only supports Impedance for"
                        " an xy receiver orientation OR Admittance for a yx receiver"
                        " orientation. Please provide a survey with valid receivers."
                    )

        if h_bc is None:
            if isinstance(mesh, (TensorMesh, TreeMesh)):
                b_e = mesh.boundary_edges
                top = np.where(b_e[:, 1] == mesh.nodes_y[-1])
                bot = np.where(b_e[:, 1] == mesh.nodes_y[0])
                left = np.where(b_e[:, 0] == mesh.nodes_x[0])
                right = np.where(b_e[:, 0] == mesh.nodes_x[-1])

                if isinstance(mesh, TensorMesh):
                    h_l = h_r = mesh.h[1]
                    is_b = np.zeros(mesh.shape_cells, dtype=bool)
                    is_b[0, :] = True
                    P_l = maps.Projection(mesh.n_cells, is_b.reshape(-1, order="F"))
                    is_b[0, :] = False
                    is_b[-1, :] = True
                    P_r = maps.Projection(mesh.n_cells, is_b.reshape(-1, order="F"))
                else:
                    h_l = _centers_to_widths(b_e[left][:, 1] - mesh.nodes_y[0])
                    h_r = _centers_to_widths(b_e[right][:, 1] - mesh.nodes_y[0])
                    b_l, b_r, _, __ = mesh.cell_boundary_indices
                    P_l = maps.Projection(mesh.n_cells, b_l)
                    P_r = maps.Projection(mesh.n_cells, b_r)

                self._b_inds = (left, right, bot, top)
                self._P_l = P_l
                self._P_r = P_r

                map_l_kwargs = {}
                map_r_kwargs = {}
                if self.sigmaMap is not None:
                    map_l_kwargs["sigmaMap"] = P_l * self.sigmaMap
                    map_r_kwargs["sigmaMap"] = P_r * self.sigmaMap
                if self.muiMap is not None:
                    map_l_kwargs["muiMap"] = P_l * self.muiMap
                    map_r_kwargs["muiMap"] = P_r * self.muiMap

                # create a survey with 1 source per frequency (no receivers)
                frequencies = self.survey.frequencies
                survey = Survey([Planewave([], freq) for freq in frequencies])
                self._sim_left = Simulation1DElectricField(
                    TensorMesh((h_l,), (mesh.nodes_y[0],)),
                    survey=survey,
                    solver=self.solver,
                    **map_l_kwargs,
                )
                self._sim_right = Simulation1DElectricField(
                    TensorMesh((h_r,), (mesh.nodes_y[0],)),
                    survey=survey,
                    solver=self.solver,
                    **map_r_kwargs,
                )
            else:
                raise NotImplementedError(
                    f"Unable to infer 1D mesh from {type(mesh)}. You must supply custom"
                    " boundary conditions for the electric field."
                )
            self._h_bc = None
        else:
            n_be = mesh.boundary_edges.shape[0]
            for freq in self.survey.frequencies:
                try:
                    h = h_bc[freq]
                    if len(h) != n_be:
                        raise ValueError(
                            f"Boundary condition item for frequency {freq} is incorrect length."
                            f" Should be the same length as number of boundary_edges, {n_be}, "
                            f" saw a length of {len(h)}"
                        )
                except TypeError:
                    raise TypeError(
                        "h_bc must be a dictionary of numpy arrays indexed by frequency."
                    )
                except IndexError:
                    raise TypeError(
                        "h_bc must be a dictionary of numpy arrays indexed by frequency. Did not"
                        f" find key {freq}."
                    )
                except KeyError:
                    raise KeyError(
                        "h_bc must be a dictionary of numpy arrays indexed by frequency. Did not"
                        f" find key {freq}."
                    )
            self._h_bc = h_bc
        self._M_bc = mesh.boundary_edge_vector_integral

    def getA(self, freq):
        r"""
        System matrix

        .. math::

            \mathbf{A} =
                \mathbf{C}^\top \mathbf{M}^{cc}_{\mu} \mathbf{C}
                + 1\omega \mathbf{M}^e_\sigma

        """
        C = self.mesh.edge_curl
        Mcc_mui = self.MccMui
        Me_sigma = self.MeSigma

        A = C.T.tocsr() @ Mcc_mui @ C + 1j * omega(freq) * Me_sigma
        if self._h_bc is None:
            A = A + self._bottom_robin_matrix(freq)
        return A

    def _bottom_robin_values(self):
        sigma = np.broadcast_to(self.sigma, (self.mesh.n_cells,))
        mui = np.broadcast_to(self.mui, (self.mesh.n_cells,))
        return sigma, mui, np.sqrt(sigma * mui)

    def _bottom_robin_matrix(self, freq):
        r"""Robin (downgoing plane wave) term on the bottom boundary edges.

        :math:`\sqrt{i \omega \sigma / \mu}` of the bottom cells, integrated
        along the bottom boundary.
        """
        P, K = self._bottom_robin_ops
        scale = np.sqrt(1j * omega(freq))
        return scale * (P.T @ sp.diags(K @ self._bottom_robin_values()[2]) @ P)

    @property
    def _bottom_robin_ops(self):
        if getattr(self, "_bottom_robin_ops_cache", None) is None:
            self._bottom_robin_ops_cache = _bottom_robin_ops_2d(self.mesh)
        return self._bottom_robin_ops_cache

    def getRHS(self, freq):
        """
        Right hand side constructed using Dirichlet boundary conditions
        """
        M_bc = self._M_bc
        if self._h_bc is None:
            # left and right have the same 1D survey
            src = self._sim_left.survey.get_sources_by_frequency(freq)[0]
            f_left, f_right = self.boundary_fields()
            h_bc = np.zeros(M_bc.shape[1], dtype=complex)
            left, right, bot, top = self._b_inds
            h_bc[top] = 1.0
            h_bc[left] = f_left[src, "h"][:, 0]
            h_bc[right] = f_right[src, "h"][:, 0]
        else:
            h_bc = self._h_bc[freq]
        return 1j * omega(freq) * (M_bc @ h_bc)

    def getADeriv_sigma(self, freq, u, v, adjoint=False):
        dA_v = 1j * omega(freq) * self.MeSigmaDeriv(u, v, adjoint=adjoint)
        if self._h_bc is not None or self.sigmaMap is None:
            return dA_v
        sigma, _, g = self._bottom_robin_values()
        dg_dm = sp.diags(0.5 * g / sigma) @ self.sigmaDeriv
        scale = np.sqrt(1j * omega(freq))
        return dA_v + _bottom_robin_deriv_2d(
            self._bottom_robin_ops, scale, dg_dm, u, v, adjoint
        )

    def getADeriv_mui(self, freq, u, v, adjoint=False):
        C = self.mesh.edge_curl
        if adjoint:
            dA_v = self.MccMuiDeriv(C * u, C * v, adjoint)
        else:
            dA_v = C.T * self.MccMuiDeriv(C * u, v, adjoint)
        if self._h_bc is not None or self.muiMap is None:
            return dA_v
        _, mui, g = self._bottom_robin_values()
        dg_dm = sp.diags(0.5 * g / mui) @ self.muiDeriv
        scale = np.sqrt(1j * omega(freq))
        return dA_v + _bottom_robin_deriv_2d(
            self._bottom_robin_ops, scale, dg_dm, u, v, adjoint
        )

    def getADeriv(self, freq, u, v, adjoint=False):
        return self.getADeriv_sigma(freq, u, v, adjoint) + self.getADeriv_mui(
            freq, u, v, adjoint
        )

    def getRHSDeriv(self, freq, src, v, adjoint=False):
        if self._h_bc is not None:
            return Zero()
        M_bc = self._M_bc
        f_left, f_right = self.boundary_fields()
        left, right, _, __ = self._b_inds
        src_1d = self._sim_left.survey.get_sources_by_frequency(freq)[0]

        # derivatives from the Jv func of the 1D sim
        if not adjoint:
            h_bc_dm_v = np.zeros(M_bc.shape[1], dtype=complex)
            h_bc_dm_v[left] = f_left.field_deriv_m("h", freq, src_1d, v, adjoint=False)
            h_bc_dm_v[right] = f_right.field_deriv_m(
                "h", freq, src_1d, v, adjoint=False
            )

            return 1j * omega(freq) * (M_bc @ h_bc_dm_v)
        else:
            v_dm = M_bc.T @ v
            v_left, v_right = v_dm[left], v_dm[right]
            df_dmT = f_left.field_deriv_m("h", freq, src_1d, v_left, adjoint=True)
            df_dmT += f_right.field_deriv_m("h", freq, src_1d, v_right, adjoint=True)

            return 1j * omega(freq) * df_dmT

    def boundary_fields(self, model=None):
        "Returns the 1D field objects at the boundaries"
        if getattr(self, "_boundary_fields", None) is None:
            if model is None:
                model = self.model
            sim = self._sim_left
            if self.muiMap is None:
                try:
                    sim.mui = self._P_l @ self.mui
                except Exception:
                    sim.mui = self.mui
            if self.sigmaMap is None:
                try:
                    sim.sigma = self._P_l @ self.sigma
                except Exception:
                    sim.sigma = self.sigma
            f_left = sim.fields(model)

            sim = self._sim_right
            if self.muiMap is None:
                try:
                    sim.mui = self._P_r @ self.mui
                except Exception:
                    sim.mui = self.mui
            if self.sigmaMap is None:
                try:
                    sim.sigma = self._P_r @ self.sigma
                except Exception:
                    sim.sigma = self.sigma
            f_right = sim.fields(model)

            self._boundary_fields = (f_left, f_right)
        return self._boundary_fields

    @property
    def _delete_on_model_update(self):
        items = super()._delete_on_model_update
        items.append("_boundary_fields")
        return items


class Simulation2DMagneticField(BaseFDEMSimulation):
    """
    A
    """

    _solutionType = "hSolution"
    _formulation = "HJ"
    fieldsPair = Fields2DMagneticField

    def __init__(self, mesh, e_bc=None, **kwargs):
        if mesh.dim != 2:
            raise ValueError(
                f"The mesh must be a 2D mesh. The provided mesh has dimension {mesh.dim}"
            )

        super().__init__(mesh, **kwargs)

        for src in self.survey.source_list:
            for rx in src.receiver_list:
                if not (
                    (rx.orientation == "yx" and isinstance(rx, Impedance))
                    or (rx.orientation == "xy" and isinstance(rx, Admittance))
                    or (rx.orientation == "zx" and isinstance(rx, Tipper))
                ):
                    raise TypeError(
                        "natural_source.Simulation2DMagneticField supports Impedance and"
                        " Admittance receivers for an yx orientation, and Tipper receivers"
                        " for a zx orientation. Please provide a survey with valid receivers."
                    )

        if e_bc is None:
            if isinstance(mesh, (TensorMesh, TreeMesh)):
                b_e = mesh.boundary_edges
                top = np.where(b_e[:, 1] == mesh.nodes_y[-1])
                bot = np.where(b_e[:, 1] == mesh.nodes_y[0])
                left = np.where(b_e[:, 0] == mesh.nodes_x[0])
                right = np.where(b_e[:, 0] == mesh.nodes_x[-1])

                if isinstance(mesh, TensorMesh):
                    h_l = h_r = mesh.h[1]
                    is_b = np.zeros(mesh.shape_cells, dtype=bool)
                    is_b[0, :] = True
                    P_l = maps.Projection(mesh.n_cells, is_b.reshape(-1, order="F"))
                    is_b[0, :] = False
                    is_b[-1, :] = True
                    P_r = maps.Projection(mesh.n_cells, is_b.reshape(-1, order="F"))
                else:
                    h_l = _centers_to_widths(b_e[left][:, 1] - mesh.nodes_y[0])
                    h_r = _centers_to_widths(b_e[right][:, 1] - mesh.nodes_y[0])
                    b_l, b_r, _, __ = mesh.cell_boundary_indices
                    P_l = maps.Projection(mesh.n_cells, b_l)
                    P_r = maps.Projection(mesh.n_cells, b_r)

                self._b_inds = (left, right, bot, top)
                self._P_l = P_l
                self._P_r = P_r

                map_l_kwargs = {}
                map_r_kwargs = {}
                if self.rhoMap is not None:
                    map_l_kwargs["rhoMap"] = P_l * self.rhoMap
                    map_r_kwargs["rhoMap"] = P_r * self.rhoMap
                if self.muMap is not None:
                    map_l_kwargs["muMap"] = P_l * self.muMap
                    map_r_kwargs["muMap"] = P_r * self.muMap

                # create a survey with 1 source per frequency (no receivers)
                frequencies = self.survey.frequencies
                survey = Survey([Planewave([], freq) for freq in frequencies])
                self._sim_left = Simulation1DMagneticField(
                    TensorMesh((h_l,), (mesh.nodes_y[0],)),
                    survey=survey,
                    solver=self.solver,
                    **map_l_kwargs,
                )
                self._sim_right = Simulation1DMagneticField(
                    TensorMesh((h_r,), (mesh.nodes_y[0],)),
                    survey=survey,
                    solver=self.solver,
                    **map_r_kwargs,
                )
            else:
                raise NotImplementedError(
                    f"Unable to infer 1D mesh from {type(mesh)}. You must supply custom"
                    " boundary conditions for the electric field."
                )
            self._e_bc = None
        else:
            n_be = mesh.boundary_edges.shape[0]
            for freq in self.survey.frequencies:
                try:
                    e = e_bc[freq]
                    if len(e) != n_be:
                        raise ValueError(
                            f"Boundary condition item for frequency {freq} is incorrect length."
                            f" Should be the same length as number of boundary_edges, {n_be}, "
                            f" saw a length of {len(e)}"
                        )
                except TypeError:
                    raise TypeError(
                        "e_bc must be a dictionary of numpy arrays indexed by frequency."
                    )
                except IndexError:
                    raise TypeError(
                        "e_bc must be a dictionary of numpy arrays indexed by frequency."
                    )
                except KeyError:
                    raise KeyError(
                        "e_bc must be a dictionary of numpy arrays indexed by frequency. Did not"
                        f" find key {freq}."
                    )
            self._e_bc = e_bc
        self._M_bc = mesh.boundary_edge_vector_integral

    def getA(self, freq):
        r"""
        System matrix

        .. math::

            \mathbf{A} =
                \mathbf{C}^\top \mathbf{M}^{cc}_{\rho} \mathbf{C}
                + 1\omega \mathbf{M}^e_\mu
        """
        C = self.mesh.edge_curl
        Mcc_rho = self.MccRho
        Me_mu = self.MeMu

        A = C.T.tocsr() @ Mcc_rho @ C + 1j * omega(freq) * Me_mu
        if self._e_bc is None:
            A = A + self._bottom_robin_matrix(freq)
        return A

    def _bottom_robin_values(self):
        rho = np.broadcast_to(self.rho, (self.mesh.n_cells,))
        mu = np.broadcast_to(self.mu, (self.mesh.n_cells,))
        return rho, mu, np.sqrt(rho * mu)

    def _bottom_robin_matrix(self, freq):
        r"""Robin (downgoing plane wave) term on the bottom boundary edges.

        :math:`i \sqrt{-i \omega \mu \rho}` of the bottom cells, integrated
        along the bottom boundary.
        """
        P, K = self._bottom_robin_ops
        scale = 1j * np.sqrt(-1j * omega(freq))
        return scale * (P.T @ sp.diags(K @ self._bottom_robin_values()[2]) @ P)

    @property
    def _bottom_robin_ops(self):
        if getattr(self, "_bottom_robin_ops_cache", None) is None:
            self._bottom_robin_ops_cache = _bottom_robin_ops_2d(self.mesh)
        return self._bottom_robin_ops_cache

    def getRHS(self, freq):
        """
        Right hand side constructed using Dirichlet boundary conditions
        """
        M_bc = self._M_bc
        if self._e_bc is None:
            # left and right have the same 1D survey
            src = self._sim_left.survey.get_sources_by_frequency(freq)[0]
            f_left, f_right = self.boundary_fields()
            e_bc = np.zeros(M_bc.shape[1], dtype=complex)
            left, right, bot, top = self._b_inds
            e_bc[top] = 1.0
            e_bc[left] = f_left[src, "e"][:, 0]
            e_bc[right] = f_right[src, "e"][:, 0]
        else:
            e_bc = self._e_bc[freq]
        return -M_bc @ e_bc

    def getADeriv_rho(self, freq, u, v, adjoint=False):
        C = self.mesh.edge_curl
        if adjoint:
            dA_v = self.MccRhoDeriv(C * u, C * v, adjoint)
        else:
            dA_v = C.T * self.MccRhoDeriv(C * u, v, adjoint)
        if self._e_bc is not None or self.rhoMap is None:
            return dA_v
        rho, _, g = self._bottom_robin_values()
        dg_dm = sp.diags(0.5 * g / rho) @ self.rhoDeriv
        scale = 1j * np.sqrt(-1j * omega(freq))
        return dA_v + _bottom_robin_deriv_2d(
            self._bottom_robin_ops, scale, dg_dm, u, v, adjoint
        )

    def getADeriv_mu(self, freq, u, v, adjoint=False):
        dA_v = 1j * omega(freq) * self.MeMuDeriv(u, v, adjoint=adjoint)
        if self._e_bc is not None or self.muMap is None:
            return dA_v
        _, mu, g = self._bottom_robin_values()
        dg_dm = sp.diags(0.5 * g / mu) @ self.muDeriv
        scale = 1j * np.sqrt(-1j * omega(freq))
        return dA_v + _bottom_robin_deriv_2d(
            self._bottom_robin_ops, scale, dg_dm, u, v, adjoint
        )

    def getADeriv(self, freq, u, v, adjoint=False):
        return self.getADeriv_rho(freq, u, v, adjoint) + self.getADeriv_mu(
            freq, u, v, adjoint
        )

    def getRHSDeriv(self, freq, src, v, adjoint=False):
        if self._e_bc is not None:
            return Zero()
        M_bc = self._M_bc
        f_left, f_right = self.boundary_fields()
        left, right, _, __ = self._b_inds
        src_1d = self._sim_left.survey.get_sources_by_frequency(freq)[0]

        # derivatives from the Jv func of the 1D sim
        if not adjoint:
            e_bc_dm_v = np.zeros(M_bc.shape[1], dtype=complex)
            e_bc_dm_v[left] = f_left.field_deriv_m("e", freq, src_1d, v, adjoint=False)
            e_bc_dm_v[right] = f_right.field_deriv_m(
                "e", freq, src_1d, v, adjoint=False
            )
            return -(M_bc @ e_bc_dm_v)
        else:
            v_dm = -(M_bc.T @ v)
            v_left, v_right = v_dm[left], v_dm[right]
            df_dmT = f_left.field_deriv_m("e", freq, src_1d, v_left, adjoint=True)
            df_dmT += f_right.field_deriv_m("e", freq, src_1d, v_right, adjoint=True)
            return df_dmT

    def boundary_fields(self, model=None):
        "Returns the 1D field objects at the boundaries"
        if getattr(self, "_boundary_fields", None) is None:
            if model is None:
                model = self.model
            sim = self._sim_left
            if self.muMap is None:
                try:
                    sim.mu = self._P_l @ self.mu
                except Exception:
                    sim.mu = self.mu
            if self.rhoMap is None:
                try:
                    sim.rho = self._P_l @ self.rho
                except Exception:
                    sim.rho = self.rho
            f_left = sim.fields(model)

            sim = self._sim_right
            if self.muMap is None:
                try:
                    sim.mu = self._P_r @ self.mu
                except Exception:
                    sim.mu = self.mu
            if self.rhoMap is None:
                try:
                    sim.rho = self._P_r @ self.rho
                except Exception:
                    sim.rho = self.rho
            f_right = sim.fields(model)

            self._boundary_fields = (f_left, f_right)
        return self._boundary_fields

    @property
    def _delete_on_model_update(self):
        items = super()._delete_on_model_update
        items.append("_boundary_fields")
        return items


###################################
# 3D problems
###################################


class Simulation3DPrimarySecondary(Simulation3DElectricField):
    r"""
    A NSEM problem solving a e formulation and a primary/secondary fields decomposition.

    By eliminating the magnetic flux density using

    .. math ::

        \mathbf{b} = \frac{1}{i \omega} \left(-\mathbf{C} \mathbf{e} \right)


    we can write Maxwell's equations as a second order system in
    :math:`\mathbf{e}` only:

    .. math ::

        \left[
            \mathbf{C}^{\top} \mathbf{M_{\mu^{-1}}^f} \mathbf{C}
            + i \omega \mathbf{M_{\sigma}^e}
        \right]
        \mathbf{e}_{s}
        = i \omega \mathbf{M_{\sigma_{p}}^e} \mathbf{e}_{p}

    which we solve for :math:`\mathbf{e_s}`.
    The total field :math:`\mathbf{e} = \mathbf{e_p} + \mathbf{e_s}`.

    The primary field is estimated from a background model (commonly as a 1D model).
    """

    def __init__(self, mesh, survey=None, sigmaPrimary=None, **kwargs):
        super().__init__(mesh=mesh, survey=survey, **kwargs)
        self.sigmaPrimary = sigmaPrimary

    # fieldsPair = Fields3DPrimarySecondary

    @property
    def sigmaPrimary(self):
        """
        A background model, use for the calculation of the primary fields.

        """
        return self._sigmaPrimary

    @sigmaPrimary.setter
    def sigmaPrimary(self, val):
        # Note: TODO add logic for val, make sure it is the correct size.
        self._sigmaPrimary = val
