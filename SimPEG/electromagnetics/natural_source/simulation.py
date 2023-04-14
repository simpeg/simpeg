import numpy as np
from discretize import TensorMesh, TreeMesh
from discretize.utils import Zero

from ...utils import mkvc
from ... import maps
from ..frequency_domain.simulation import BaseFDEMSimulation, Simulation3DElectricField
from ..frequency_domain.survey import Survey
from ..utils import omega
from .sources import Planewave
from .fields import (
    Fields1DPrimarySecondary,
    Fields1DElectricField,
    Fields1DMagneticField,
    Fields2DElectricField,
    Fields2DMagneticField,
)


def _centers_to_widths(centers):
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


class Simulation1DElectricField(BaseFDEMSimulation):
    r"""
    1D finite volume simulation for the natural source electromagnetic problem.

    This corresponds to the TE mode 2D simulation where the electric field is
    located at cell centers and the magnetic flux is on edges.

    We are solving the discrete version of

    .. math::

        \partial_z E_y = i \omega \mu_0 H_x = 0

        sigma E_y = \partial_z H_x

    with default boundary conditions that $H_x[z_max] = 1$ (a plane wave source at
    the top of the domain), and $H_x[z_min] = 0$.

    When we discretize, we obtain:

    where the Magnetic field is defined on edges, and the electric field is
    defined on cell centers.
    """

    _solutionType = "eSolution"
    _formulation = "EB"  # electric-field component is on cell-centers
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

        return G.T.tocsr() @ MeMui @ G + 1j * omega(freq) * MfSigma

    def getADeriv_sigma(self, freq, u, v, adjoint=False):
        return 1j * omega(freq) * self.MfSigmaDeriv(u, v, adjoint=adjoint)

    def getADeriv_mui(self, freq, u, v, adjoint=False):
        G = self.mesh.nodal_gradient
        if adjoint:
            return self.MeMuiDeriv(G * u, G * v, adjoint)
        return G.T * self.MeMuiDeriv(G * u, v, adjoint)

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

        return G.T.tocsr() @ MeRho @ G + 1j * omega(freq) * MnMu

    def getADeriv_rho(self, freq, u, v, adjoint=False):
        G = self.mesh.nodal_gradient
        if adjoint:
            return self.MeRhoDeriv(G * u, G * v, adjoint)
        return G.T * self.MeRhoDeriv(G * u, v, adjoint)

    def getADeriv_mu(self, freq, u, v, adjoint=False):
        MnMuDeriv = self.MnMuDeriv(u)
        if adjoint is True:
            return 1j * omega(freq) * (MnMuDeriv.T * v)

        return 1j * omega(freq) * (MnMuDeriv * v)

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
                if rx.orientation != "xy":
                    raise TypeError(
                        "natural_source.Simulation2DElectricField only supports xy oriented"
                        " receivers. Please use the Simulation2DMagneticField class for"
                        " those receivers."
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
                    h_l = _centers_to_widths(b_e[left][:, 1])
                    h_r = _centers_to_widths(b_e[right][:, 1])
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

        return C.T.tocsr() @ Mcc_mui @ C + 1j * omega(freq) * Me_sigma

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
        return 1j * omega(freq) * self.MeSigmaDeriv(u, v, adjoint=adjoint)

    def getADeriv_mui(self, freq, u, v, adjoint=False):
        C = self.mesh.edge_curl
        if adjoint:
            return self.MccMuiDeriv(C * u, C * v, adjoint)
        return C.T * self.MccMuiDeriv(C * u, v, adjoint)

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
    def deleteTheseOnModelUpdate(self):
        items = super().deleteTheseOnModelUpdate
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
                if rx.orientation != "yx":
                    raise TypeError(
                        "natural_source.Simulation2DMagneticField only supports yx oriented"
                        " receivers. Please use the Simulation2DElectricField class for"
                        " those receivers."
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
                    h_l = _centers_to_widths(b_e[left][:, 1])
                    h_r = _centers_to_widths(b_e[right][:, 1])
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

        return C.T.tocsr() @ Mcc_rho @ C + 1j * omega(freq) * Me_mu

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
            return self.MccRhoDeriv(C * u, C * v, adjoint)
        return C.T * self.MccRhoDeriv(C * u, v, adjoint)

    def getADeriv_mu(self, freq, u, v, adjoint=False):
        return 1j * omega(freq) * self.MeMuDeriv(u, v, adjoint=adjoint)

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
    def deleteTheseOnModelUpdate(self):
        items = super().deleteTheseOnModelUpdate
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
