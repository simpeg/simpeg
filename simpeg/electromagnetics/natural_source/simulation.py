import numpy as np
import scipy.sparse as sp
from scipy.constants import mu_0
from discretize import TensorMesh, TreeMesh
from discretize.utils import Zero

from ...base.pde_simulation import _inner_mat_mul_op
from ...utils import mkvc, validate_string
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

    with a unit magnetic field at the top of the domain (the plane wave source) and
    a Robin (downgoing plane wave) condition at the bottom.

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
                + i\omega \mathbf{M}^f_\sigma + \mathbf{R}

        where :math:`\mathbf{R}` is the Robin term at the bottom node.
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
        Right hand side for a unit magnetic field at the top of the domain.
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
    The electric field is one at the top of the domain (the plane wave source),
    and the bottom has a Robin (downgoing plane wave) condition.
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

        # a unit electric field at the top (the bottom is the robin term in getA)
        self._rhs = -mesh.boundary_node_vector_integral * [0 + 0j, 1 + 0j]

    def getA(self, freq):
        r"""
        System matrix

        .. math::

            \mathbf{A} =
                \mathbf{G}^\top \mathbf{M}^e_{\rho} \mathbf{G}
                + i\omega \mathbf{M}^n_\mu + \mathbf{R}

        where :math:`\mathbf{R}` is the Robin term at the bottom node.
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
def _clear_factorization(sim):
    """Drop factorizations stored by ``fields`` after the system matrix changed."""
    sim.__dict__.pop("Ainv", None)


def _robin_ops_2d(mesh, sides=True):
    """Operators for a Robin term on the bottom (and side) boundary edges of a 2D mesh.

    Returns
    -------
    P : (n_robin_edges, n_edges) scipy.sparse.csr_matrix
        Selects the edges on the bottom boundary, and on the two sides when
        ``sides`` is true.
    K : (n_robin_edges, n_cells) scipy.sparse.csr_matrix
        The length of each of those edges, at the cell inside it.
    """
    edges = np.r_[mesh.edges_x, mesh.edges_y]
    is_x = np.arange(len(edges)) < mesh.n_edges_x
    # exact comparisons, as the boundary edges sit exactly on the outer nodes
    selected = [np.where((edges[:, 1] == mesh.nodes_y[0]) & is_x)[0]]
    inward = [np.r_[0.0, 1.0]]
    if sides:
        for x, direction in ((mesh.nodes_x[0], 1.0), (mesh.nodes_x[-1], -1.0)):
            selected.append(np.where((edges[:, 0] == x) & ~is_x)[0])
            inward.append(np.r_[direction, 0.0])
    eps = 1e-3 * min(mesh.h[0].min(), mesh.h[1].min())
    inside = np.vstack(
        [edges[sel] + eps * normal for sel, normal in zip(selected, inward)]
    )
    selected = np.concatenate(selected)
    n = len(selected)
    P = sp.csr_matrix((np.ones(n), (np.arange(n), selected)), shape=(n, mesh.n_edges))
    cells = np.atleast_1d(mesh.point2index(inside))
    K = sp.csr_matrix(
        (mesh.edge_lengths[selected], (np.arange(n), cells)),
        shape=(n, mesh.n_cells),
    )
    return P, K


def _robin_deriv_2d(ops, scale, dg_dm, u, v, adjoint):
    """Model derivative of ``scale * P.T @ diag(K @ g) @ P @ u``."""
    P, K = ops
    dKg_dm = K @ dg_dm
    if adjoint:
        return _inner_mat_mul_op(dKg_dm, P @ u, P @ (scale * v), adjoint=True)
    return scale * (P.T @ _inner_mat_mul_op(dKg_dm, P @ u, v))


class Simulation2DElectricField(BaseFDEMSimulation):
    """2D natural source simulation solving for the electric field.

    The electric field in the plane is on edges and the magnetic field along
    strike on cell centers. The top has a unit magnetic field (exact for this
    mode, since the magnetic field is constant through the air), the sides take
    their magnetic field from 1D columns of the side cells, and the bottom has
    a Robin (downgoing plane wave) condition.

    Parameters
    ----------
    mesh : discretize.TensorMesh or discretize.TreeMesh
        A 2D mesh.
    h_bc : dict, optional
        Tangential magnetic field on every boundary edge, indexed by frequency,
        to use instead of the default boundary conditions. This also turns off
        the robin terms on the bottom and sides (``side_boundary`` is ignored).
    side_boundary : {"robin", "neumann"}, default: "robin"
        Condition on the sides of the mesh (see :attr:`side_boundary`).
    """

    _solutionType = "eSolution"
    _formulation = "EB"
    fieldsPair = Fields2DElectricField

    def __init__(self, mesh, h_bc=None, side_boundary="robin", **kwargs):
        if mesh.dim != 2:
            raise ValueError(
                f"The mesh must be a 2D mesh. The provided mesh has dimension {mesh.dim}"
            )

        super().__init__(mesh, **kwargs)
        self.side_boundary = side_boundary

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

    @property
    def side_boundary(self):
        """Boundary condition on the two sides of the mesh.

        Both options take the tangential field along each side from a 1D
        column of the side cells. ``"robin"`` also lets the field scattered by
        2D structure leave through the sides, with the same outgoing plane wave
        term as the bottom, which reduces reflections from structure near the
        sides. ``"neumann"`` prescribes the 1D column's field on the sides.

        Returns
        -------
        {"robin", "neumann"}
        """
        return self._side_boundary

    @side_boundary.setter
    def side_boundary(self, value):
        self._side_boundary = validate_string(
            "side_boundary", value, ["robin", "neumann"]
        )
        self._robin_ops_cache = None
        _clear_factorization(self)

    def getA(self, freq):
        r"""
        System matrix

        .. math::

            \mathbf{A} =
                \mathbf{C}^\top \mathbf{M}^{cc}_{\mu} \mathbf{C}
                + i\omega \mathbf{M}^e_\sigma + \mathbf{R}

        where :math:`\mathbf{R}` is the Robin term on the bottom (and side)
        boundary edges.
        """
        C = self.mesh.edge_curl
        Mcc_mui = self.MccMui
        Me_sigma = self.MeSigma

        A = C.T.tocsr() @ Mcc_mui @ C + 1j * omega(freq) * Me_sigma
        if self._h_bc is None:
            A = A + self._robin_matrix(freq)
        return A

    def _robin_values(self):
        sigma = np.broadcast_to(self.sigma, (self.mesh.n_cells,))
        mui = np.broadcast_to(self.mui, (self.mesh.n_cells,))
        return sigma, mui, np.sqrt(sigma * mui)

    def _robin_matrix(self, freq):
        r"""Robin (outgoing plane wave) term on the bottom and side boundary edges.

        :math:`\sqrt{i \omega \sigma / \mu}` of the cells inside the
        boundary, integrated along it.
        """
        P, K = self._robin_ops
        scale = np.sqrt(1j * omega(freq))
        return scale * (P.T @ sp.diags(K @ self._robin_values()[2]) @ P)

    @property
    def _robin_ops(self):
        if getattr(self, "_robin_ops_cache", None) is None:
            self._robin_ops_cache = _robin_ops_2d(
                self.mesh, sides=self.side_boundary == "robin"
            )
        return self._robin_ops_cache

    def getRHS(self, freq):
        """
        Right hand side from the boundary data: a unit magnetic field along the
        top, and the 1D columns along the sides.
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
        sigma, _, g = self._robin_values()
        dg_dm = sp.diags(0.5 * g / sigma) @ self.sigmaDeriv
        scale = np.sqrt(1j * omega(freq))
        return dA_v + _robin_deriv_2d(self._robin_ops, scale, dg_dm, u, v, adjoint)

    def getADeriv_mui(self, freq, u, v, adjoint=False):
        C = self.mesh.edge_curl
        if adjoint:
            dA_v = self.MccMuiDeriv(C * u, C * v, adjoint)
        else:
            dA_v = C.T * self.MccMuiDeriv(C * u, v, adjoint)
        if self._h_bc is not None or self.muiMap is None:
            return dA_v
        _, mui, g = self._robin_values()
        dg_dm = sp.diags(0.5 * g / mui) @ self.muiDeriv
        scale = np.sqrt(1j * omega(freq))
        return dA_v + _robin_deriv_2d(self._robin_ops, scale, dg_dm, u, v, adjoint)

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


def _log_integral(x, a, b):
    """Integral of ``ln|x - t|`` over ``t`` in ``[a, b]``, for each ``x`` and interval."""

    def antiderivative(u):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(u == 0, 0.0, u * np.log(np.abs(u)) - u)

    return antiderivative(b[:, None] - x[None, :]) - antiderivative(
        a[:, None] - x[None, :]
    )


def _upward_continuation_ops(x, length, x_left, x_right, freq):
    r"""Anomalous magnetic field along the top from the electric field there.

    Above the top of the mesh the anomalous field decays upward, so the
    vertical magnetic field along the top, :math:`H_z = -\partial_x E_y / (i
    \omega \mu_0)`, and the anomalous horizontal field,
    :math:`H_x - H_{inc}`, are a Hilbert transform pair. Beyond the sides the
    electric field is taken as constant along the top (the 1D side columns).

    Parameters
    ----------
    x, length : (n,) numpy.ndarray
        Centers and lengths of the top edges, sorted by ``x``.
    x_left, x_right : float
        Ends of the top boundary.
    freq : float
        Frequency in Hz.

    Returns
    -------
    D : (n, n) numpy.ndarray
    d_left, d_right : (n,) numpy.ndarray
        The anomalous magnetic field averaged over each top edge is
        ``D @ e + d_left * e_left + d_right * e_right`` for an electric field
        ``e`` on the top edges and ``e_left``, ``e_right`` at the two ends.
    """
    n = len(x)
    a, b = x - length / 2, x + length / 2
    nodes = np.r_[x_left, x, x_right]
    # slopes of the electric field between consecutive points
    dx = np.diff(nodes)
    slope = np.zeros((n + 1, n + 2))
    slope[np.arange(n + 1), np.arange(n + 1)] = -1 / dx
    slope[np.arange(n + 1), np.arange(1, n + 2)] = 1 / dx
    # (1 / pi) * int over each interval of 1 / (x - t), averaged over each top edge
    hilbert = (_log_integral(nodes[:-1], a, b) - _log_integral(nodes[1:], a, b)) / (
        np.pi * length[:, None]
    )
    ops = hilbert @ (slope * (-1.0 / (1j * omega(freq) * mu_0)))
    return ops[:, 1:-1], ops[:, 0], ops[:, -1]


class Simulation2DMagneticField(BaseFDEMSimulation):
    """2D natural source simulation solving for the magnetic field.

    The magnetic field in the plane is on edges and the electric field along
    strike on cell centers. The sides take their electric field from 1D
    columns of the side cells, and the bottom has a Robin (downgoing plane
    wave) condition.

    Parameters
    ----------
    mesh : discretize.TensorMesh or discretize.TreeMesh
        A 2D mesh.
    e_bc : dict, optional
        Tangential electric field on every boundary edge, indexed by frequency,
        to use instead of the default boundary conditions. This also turns off
        the robin terms on the top, bottom and sides (``top_boundary`` and
        ``side_boundary`` are ignored).
    top_boundary : {"robin", "uniform"}, default: "robin"
        Condition on the top of the mesh (see :attr:`top_boundary`).
    side_boundary : {"robin", "neumann"}, default: "robin"
        Condition on the sides of the mesh (see :attr:`side_boundary`).
    """

    _solutionType = "hSolution"
    _formulation = "HJ"
    fieldsPair = Fields2DMagneticField

    def __init__(
        self, mesh, e_bc=None, top_boundary="robin", side_boundary="robin", **kwargs
    ):
        if mesh.dim != 2:
            raise ValueError(
                f"The mesh must be a 2D mesh. The provided mesh has dimension {mesh.dim}"
            )

        super().__init__(mesh, **kwargs)
        self.top_boundary = top_boundary
        self.side_boundary = side_boundary

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

    @property
    def side_boundary(self):
        """Boundary condition on the two sides of the mesh.

        Both options take the tangential field along each side from a 1D
        column of the side cells. ``"robin"`` also lets the field scattered by
        2D structure leave through the sides, with the same outgoing plane wave
        term as the bottom, which reduces reflections from structure near the
        sides. ``"neumann"`` prescribes the 1D column's field on the sides.

        Returns
        -------
        {"robin", "neumann"}
        """
        return self._side_boundary

    @side_boundary.setter
    def side_boundary(self, value):
        self._side_boundary = validate_string(
            "side_boundary", value, ["robin", "neumann"]
        )
        self._robin_ops_cache = None
        _clear_factorization(self)

    @property
    def top_boundary(self):
        """Boundary condition on the top of the mesh.

        ``"robin"`` relates the electric field along the top to the magnetic
        field there by upward continuation: above the mesh the anomalous field
        decays upward, so the electric field is not fixed along the top. A unit
        incident magnetic field drives it, and the side columns are normalized
        to the same incident field, giving the same incident wave on every
        boundary. This lets the top of the mesh sit close to the surface.

        ``"uniform"`` sets a unit electric field along the top, with each side
        column normalized to a unit electric field at its top. It needs the top
        far above the surface.

        Returns
        -------
        {"robin", "uniform"}
        """
        return self._top_boundary

    @top_boundary.setter
    def top_boundary(self, value):
        self._top_boundary = validate_string(
            "top_boundary", value, ["robin", "uniform"]
        )
        self._top_robin_cache = None
        _clear_factorization(self)

    @property
    def _uses_top_robin(self):
        return self._e_bc is None and self.top_boundary == "robin"

    def _top_robin_ops(self, freq):
        """Top edges and the top robin operators.

        Along the top, ``H_x - 1 = D e + d_left e_left + d_right e_right`` (see
        :func:`_upward_continuation_ops`). Taking out the mean of the corner
        values, ``e = (e_left + e_right) / 2 + D^-1 (P_top h - 1 - s)`` with
        ``s = (d_left - d_right) (e_left - e_right) / 2``. The boundary integral
        of ``D^-1`` is symmetrized: the continuous operator (a logarithmic
        convolution) is symmetric, and the system matrix must be symmetric for
        the adjoint solves to reuse the forward factorization. Taking out the
        mean first keeps a laterally uniform model exact.

        Returns
        -------
        cache : dict
            The top edges and their operators, for all frequencies.
        ops : tuple of (B, b_0, u_left, u_right)
            ``B`` is the symmetrized ``M_top D^-1`` on the top edges, and the
            right hand side from the top is ``b_0 + u_left e_left + u_right
            e_right``.
        """
        if getattr(self, "_top_robin_cache", None) is None:
            mesh = self.mesh
            top = np.asarray(self._b_inds[3]).ravel()
            top = top[np.argsort(mesh.boundary_edges[top, 0])]
            P_top = mesh.project_edge_to_boundary_edge.tocsr()[top]
            self._top_robin_cache = {
                "top": top,
                "P_top": P_top,
                "M_top": self._M_bc[:, top].tocsr(),
                "x": mesh.boundary_edges[top, 0],
                "length": mesh.edge_lengths[P_top.indices],
                "ops": {},
            }
        cache = self._top_robin_cache
        if freq not in cache["ops"]:
            D, d_left, d_right = _upward_continuation_ops(
                cache["x"],
                cache["length"],
                self.mesh.nodes_x[0],
                self.mesh.nodes_x[-1],
                freq,
            )
            P_top, M_top = cache["P_top"], cache["M_top"]
            # M_top D^-1 on the top edges, with M_top diagonal there
            B = (P_top @ M_top).diagonal()[:, None] * np.linalg.inv(D)
            B = 0.5 * (B + B.T)
            mean_part = -0.5 * (M_top @ np.ones(M_top.shape[1]))
            step_part = 0.5 * (P_top.T @ (B @ (d_left - d_right)))
            cache["ops"][freq] = (
                B,
                P_top.T @ (B @ np.ones(len(d_left))),
                mean_part + step_part,
                mean_part - step_part,
            )
        return cache, cache["ops"][freq]

    def _side_columns(self, freq):
        """Electric field of the side columns, and the scaling of each.

        Returns, for the left and right columns, the electric field on the side
        edges, the electric field at the column top, and the incident magnetic
        field the column was normalized to.
        """
        src = self._sim_left.survey.get_sources_by_frequency(freq)[0]
        out = []
        for f in self.boundary_fields():
            e = f[src, "e"][:, 0]
            if self._uses_top_robin:
                # the 1D simulation's h is -H_x of the 2D simulation
                h_top = -f[src, "h"][-1, 0]
                out.append((e / h_top, 1 / h_top, h_top))
            else:
                out.append((e, 1.0, 1.0))
        return out

    def getA(self, freq):
        r"""
        System matrix

        .. math::

            \mathbf{A} =
                \mathbf{C}^\top \mathbf{M}^{cc}_{\rho} \mathbf{C}
                + i\omega \mathbf{M}^e_\mu + \mathbf{R} + \mathbf{T}

        where :math:`\mathbf{R}` is the Robin term on the bottom (and side)
        boundary edges, and :math:`\mathbf{T}` the robin top (a dense block on
        the top edges, see :attr:`top_boundary`).
        """
        C = self.mesh.edge_curl
        Mcc_rho = self.MccRho
        Me_mu = self.MeMu

        A = C.T.tocsr() @ Mcc_rho @ C + 1j * omega(freq) * Me_mu
        if self._e_bc is None:
            A = A + self._robin_matrix(freq)
        if self._uses_top_robin:
            # the electric field along the top is D^-1 (P_top h - ...) (see getRHS)
            cache, (B, _, _, _) = self._top_robin_ops(freq)
            P_top = cache["P_top"]
            A = A + P_top.T @ sp.csr_matrix(B) @ P_top
        return A

    def _robin_values(self):
        rho = np.broadcast_to(self.rho, (self.mesh.n_cells,))
        mu = np.broadcast_to(self.mu, (self.mesh.n_cells,))
        return rho, mu, np.sqrt(rho * mu)

    def _robin_matrix(self, freq):
        r"""Robin (outgoing plane wave) term on the bottom and side boundary edges.

        :math:`i \sqrt{-i \omega \mu \rho}` of the cells inside the
        boundary, integrated along it.
        """
        P, K = self._robin_ops
        scale = 1j * np.sqrt(-1j * omega(freq))
        return scale * (P.T @ sp.diags(K @ self._robin_values()[2]) @ P)

    @property
    def _robin_ops(self):
        if getattr(self, "_robin_ops_cache", None) is None:
            self._robin_ops_cache = _robin_ops_2d(
                self.mesh, sides=self.side_boundary == "robin"
            )
        return self._robin_ops_cache

    def getRHS(self, freq):
        """
        Right hand side from the boundary data: the 1D columns along the sides,
        and the incident field along the top.
        """
        M_bc = self._M_bc
        if self._e_bc is not None:
            return -M_bc @ self._e_bc[freq]
        (e_left, top_left, _), (e_right, top_right, _) = self._side_columns(freq)
        e_bc = np.zeros(M_bc.shape[1], dtype=complex)
        left, right, bot, top = self._b_inds
        e_bc[left] = e_left
        e_bc[right] = e_right
        if not self._uses_top_robin:
            e_bc[top] = 1.0
            return -M_bc @ e_bc
        # along the top, D e = (H_x - 1) - d_left e_left - d_right e_right for a
        # unit incident magnetic field, and the H_x part is in the system matrix
        _, (_, b_0, u_left, u_right) = self._top_robin_ops(freq)
        return -M_bc @ e_bc + b_0 + u_left * top_left + u_right * top_right

    def getADeriv_rho(self, freq, u, v, adjoint=False):
        C = self.mesh.edge_curl
        if adjoint:
            dA_v = self.MccRhoDeriv(C * u, C * v, adjoint)
        else:
            dA_v = C.T * self.MccRhoDeriv(C * u, v, adjoint)
        if self._e_bc is not None or self.rhoMap is None:
            return dA_v
        rho, _, g = self._robin_values()
        dg_dm = sp.diags(0.5 * g / rho) @ self.rhoDeriv
        scale = 1j * np.sqrt(-1j * omega(freq))
        return dA_v + _robin_deriv_2d(self._robin_ops, scale, dg_dm, u, v, adjoint)

    def getADeriv_mu(self, freq, u, v, adjoint=False):
        dA_v = 1j * omega(freq) * self.MeMuDeriv(u, v, adjoint=adjoint)
        if self._e_bc is not None or self.muMap is None:
            return dA_v
        _, mu, g = self._robin_values()
        dg_dm = sp.diags(0.5 * g / mu) @ self.muDeriv
        scale = 1j * np.sqrt(-1j * omega(freq))
        return dA_v + _robin_deriv_2d(self._robin_ops, scale, dg_dm, u, v, adjoint)

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

        if not self._uses_top_robin:
            # derivatives from the Jv func of the 1D sim
            if not adjoint:
                e_bc_dm_v = np.zeros(M_bc.shape[1], dtype=complex)
                e_bc_dm_v[left] = f_left.field_deriv_m(
                    "e", freq, src_1d, v, adjoint=False
                )
                e_bc_dm_v[right] = f_right.field_deriv_m(
                    "e", freq, src_1d, v, adjoint=False
                )
                return -(M_bc @ e_bc_dm_v)
            v_dm = -(M_bc.T @ v)
            v_left, v_right = v_dm[left], v_dm[right]
            df_dmT = f_left.field_deriv_m("e", freq, src_1d, v_left, adjoint=True)
            df_dmT += f_right.field_deriv_m("e", freq, src_1d, v_right, adjoint=True)
            return df_dmT

        # with the robin top, each column is scaled by its top magnetic field H,
        # e -> e / H and its top electric field 1 / H, with H = -h[-1] of the 1D
        # field, and the top contributes u_left / H_left + u_right / H_right
        _, (_, _, u_left, u_right) = self._top_robin_ops(freq)
        columns = self._side_columns(freq)
        sides = ((f_left, left, u_left), (f_right, right, u_right))
        if not adjoint:
            e_bc_dm_v = np.zeros(M_bc.shape[1], dtype=complex)
            d_top = np.zeros(M_bc.shape[0], dtype=complex)
            for (f, inds, u_side), (e_s, _, h_top) in zip(sides, columns):
                de = np.ravel(f.field_deriv_m("e", freq, src_1d, v, adjoint=False))
                dh = -np.ravel(f.field_deriv_m("h", freq, src_1d, v, adjoint=False))[-1]
                e_bc_dm_v[inds] = de / h_top - e_s * dh / h_top
                d_top += u_side * (-dh / h_top**2)
            return -(M_bc @ e_bc_dm_v) + d_top
        v = np.ravel(v)
        v_dm = -(M_bc.T @ v)
        df_dmT = 0
        for (f, inds, u_side), (e_s, _, h_top) in zip(sides, columns):
            w = v_dm[inds]
            # coefficient of the column's top magnetic field h[-1] (= -H)
            c_h = np.sum(w * e_s) / h_top + (u_side @ v) / h_top**2
            v_h = np.zeros(f[src_1d, "h"].shape[0], dtype=complex)
            v_h[-1] = c_h
            df_dmT = df_dmT + np.ravel(
                f.field_deriv_m("e", freq, src_1d, w / h_top, adjoint=True)
            )
            df_dmT = df_dmT + np.ravel(
                f.field_deriv_m("h", freq, src_1d, v_h, adjoint=True)
            )
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
