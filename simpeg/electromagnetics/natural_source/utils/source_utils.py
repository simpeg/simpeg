"""
Utility functions for NSEM sources.
"""

import numpy as np
from warnings import warn
from scipy.constants import mu_0
import scipy.sparse as sp
from discretize import TensorMesh

from ...utils import omega
from ....utils import get_default_solver
from .analytic_1d import getEHfields


def primary_e_1d_solution(
    mesh,
    sigma_1d,
    freq,
    top_bc="dirichlet",
    bot_bc="robin",
    n_skin_depths=3,
    n_pad=None,
):
    r"""Compute 1D electric field solution on nodes.

    Parameters
    ----------
    mesh : discretize.base.BaseTensorMesh
        A 1d, 2d or 3d tensor mesh or tree mesh.
    sigma_1d : array
        1D conductivity model defined along the vertical discretization.
        Conductivities are defined from the bottom cell upwards.
    freq : float
        Operating frequency in Hz.
    top_bc : string {"dirichlet", "neumann"}
        Use "dirichlet" for a solution where the electric field is equal
        to 1 at the top of the mesh. Use "neumann" to implement a boundary
        condition such that the magnetic field is equal to 1 at the top of
        the mesh.
    bot_bc : string {"dirichlet", "robin"}
        Assumes only a downgoing wave at the bottom boundary. Use "dirichlet"
        to set the value directly from the semi-analytic propagator matrix
        solution. Use "robin" to set the boundary condition discretely.
    n_skin_depths : float
        Number of additional skin depths added to the bottom of the 1D mesh
        where the discrete solution is solved.
    n_pad : int
        Number of padding cells added to the bottom of discrete 1D solution.
        This ensures accuracy of the 1D solution at the bottom of the mesh
        supplied by the user.

    Returns
    -------
    numpy.ndarray (n_edges,)
        Total electric field solution on the nodes of the 1D vertical
        discretization.

    Notes
    -----
    For the 1D electric field solution, Maxwell's equations take the form:

    .. math::
        \begin{align}
        &\frac{\partial e_x}{\partial z} + i\omega b_y = 0 \\
        &\frac{\partial (\mu^{-1}b_y}{\partial z} + \sigma e_x = 0
        \end{align}

    We take the inner product of Faraday's law with a test function $\phi$,
    and the inner product of Ampere's law with a test function $\psi$.

    .. math::
        \begin{align}
        &\langle \phi , \partial_z e_x \rangle
        + i\omega \langle \phi , b_y \rangle = 0 \\
        &\langle \psi , \partial_z (\mu^{-1}b_y) \rangle
        + \langle \psi, \sigma e \rangle = 0
        \end{align}

    We multiply Ampere's law by :math:`i\omega` and integrate by parts. Using
    Faraday's law to express the surface integral in terms of the electric field
    we obtain:

    .. math::
        - i\omega \langle \partial_z \psi , \mu^{-1}b_y \rangle
        - \int_{\partial \Omega }\psi \mu^{-1} \partial_z \, e_x \, da
        + i\omega \langle \psi, \sigma e_x \rangle = 0

    In discrete form, we obtain the following equations:

    .. math::
        \begin{align}
        & \boldsymbol{\phi}^T \mathbf{M_f G_n e}
        = - i\omega \phi^T \mathbf{M_f b} \\
        &-i \omega \boldsymbol{\psi}^T \mathbf{G_n^T M_{\frac{1}{\mu}} b}
        + i \omega \boldsymbol{\psi}^T \mathbf{M_\sigma e}
        - \boldsymbol{\psi}^T (\mathbf{B \, e + q}) = 0
        \end{align}

    Combining these equations, we obtain the following linear system for
    computing the electric fields on mesh nodes:

    .. math::
        \big [ \mathbf{G_n^T M_{\frac{1}{\mu}} G_n}
        + i \omega \mathbf{M_\sigma - B }\big ] \mathbf{e}
        = \mathbf{q}

    where matrix :math:`\mathbf{B}` and vector :math:`\mathbf{q}` are used to
    impose boundary conditions.
    """
    # Extract vertical discretization
    hz = mesh.h[-1]

    if len(hz) != len(sigma_1d):
        raise ValueError(
            "Number of cells in vertical direction must match length of "
            f"'sigma_1d'. Here hz has length {len(hz)} and sigma_1d has "
            f"length {len(sigma_1d)}"
        )

    # Generate extended 1D mesh and conductivity model to solve 1D problem
    if n_pad is not None:
        raise warn(
            "Number of padding cells no longer set directly with 'n_pad'. "
            "Use 'n_skin_depths' to set extent of padding for 1D mesh."
        )
    else:
        skin_depth = np.sqrt(2 / (omega(freq) * mu_0 * sigma_1d[0]))
        n_pad = int(np.ceil(n_skin_depths * skin_depth / hz[0]))

    hz_ext = np.pad(hz, (n_pad, 0), mode="edge")
    mesh_ext = TensorMesh([hz_ext], origin=[mesh.origin[-1] - hz[0] * n_pad])
    sigma_1d_ext = np.pad(sigma_1d, (n_pad, 0), mode="edge")

    # Generate the system matrix
    G = mesh_ext.nodal_gradient
    M_e_mui = mesh_ext.get_edge_inner_product(mu_0, invert_model=True)
    M_f_sigma = mesh_ext.get_face_inner_product(sigma_1d_ext)

    w = 2 * np.pi * freq
    A = G.T @ M_e_mui @ G + 1j * w * M_f_sigma

    fixed_nodes = np.zeros(mesh_ext.n_nodes, dtype=bool)
    e_fixed = []
    q = np.zeros(mesh_ext.n_nodes, dtype=np.complex128)

    # Bottom BC
    if bot_bc == "dirichlet":
        e_d, e_u, h_d, h_u = getEHfields(mesh_ext, sigma_1d_ext, freq, mesh.nodes_x)
        e_tot = e_d + e_u
        e_tot /= e_tot[-1]
        fixed_nodes[0] = True
        e_fixed.append(e_tot[0])
    elif bot_bc == "robin":
        k_bot = np.sqrt(-1.0j * w * mu_0 * sigma_1d_ext[0])
        A[0, 0] += 1j * k_bot / mu_0
    else:
        msg = (
            f"Invalid 'bot_bc' equal to '{bot_bc}'. "
            "It must be one of {'dirichlet', 'robin'}."
        )
        raise ValueError(msg)

    # Top BC
    if top_bc == "dirichlet":
        fixed_nodes[-1] = True
        e_fixed.append(1.0)
    elif top_bc == "neumann":
        q[-1] = -1j * w
    else:
        msg = (
            f"Invalid 'top_bc' equal to '{top_bc}'. "
            "It must be one of {'dirichlet', 'neumann'}."
        )
        raise ValueError(msg)

    P_fixed = sp.eye(mesh_ext.n_nodes, format="csc")[:, fixed_nodes]
    P_free = sp.eye(mesh_ext.n_nodes, format="csc")[:, ~fixed_nodes]
    q_free = P_free.T @ (q - A @ (P_fixed @ e_fixed))
    A_free = P_free.T @ A @ P_free

    Ainv = get_default_solver()(A_free)
    e_1d = P_free @ (Ainv @ q_free) + P_fixed @ e_fixed

    if n_pad != 0:
        e_1d = e_1d[n_pad:]
    return e_1d


def primary_h_1d_solution(
    mesh,
    sigma_1d,
    freq,
    top_bc="dirichlet",
    bot_bc="robin",
    n_skin_depths=3,
    n_pad=None,
):
    r"""Compute 1D magnetic field solution on nodes.

    Parameters
    ----------
    mesh : discretize.base.BaseTensorMesh
        A 1d, 2d or 3d tensor mesh or tree mesh.
    sigma_1d : array
        1D conductivity model defined along the vertical discretization.
        Conductivities are defined from the bottom cell upwards.
    freq : float
        Operating frequency in Hz.
    top_bc : string {"dirichlet", "neumann"}
        Use "dirichlet" for a solution where the magnetic field is equal
        to 1 at the top of the mesh. Use "neumann" to implement a boundary
        condition such that the electric field is equal to 1 at the top of
        the mesh.
    bot_bc : string {"dirichlet", "robin"}
        Assumes only a downgoing wave at the bottom boundary. Use "dirichlet"
        to set the value directly from the semi-analytic propagator matrix
        solution. Use "robin" to set the boundary condition discretely.
    n_skin_depths : float
        Number of additional skin depths added to the bottom of the 1D mesh
        where the discrete solution is solved.
    n_pad : int
        Number of padding cells added to the bottom of discrete 1D solution.
        This ensures accuracy of the 1D solution at the bottom of the mesh
        supplied by the user.

    Returns
    -------
    numpy.ndarray (n_edges,)
        Total magnetic field solution on the nodes of the 1D vertical
        discretization.

    Notes
    -----
    For the 1D magnetic field solution, Maxwell's equations take the form:

    .. math::
        \begin{align}
        &\frac{\partial (\rho j_y)}{\partial z} - i\omega \mu h_x = 0 \\
        &\frac{\partial h_x}{\partial z} - j_y = 0
        \end{align}

    We take the inner product of Faraday's law with a test function $\phi$,
    and the inner product of Ampere's law with a test function $\psi$.

    .. math::
        \begin{align}
        &\langle \phi , \partial_z (\rho j_y) \rangle
        - i\omega \langle \phi , \mu h_x \rangle = 0 \\
        &\langle \psi , \partial_z h_x \rangle
        - \langle \psi, j_y \rangle = 0
        \end{align}

    We integrate Ampere's law by parts, the use Faraday's law to represent the
    surface integral in terms of the magnetic field:

    .. math::
        -\langle \partial_z \phi , \rho \, j_y \rangle
        + \int_{\partial \Omega} \, \phi \rho \partial_z h_x \, da
        - i\omega \langle \phi , \mu h_x \rangle = 0

    In discrete form, we obtain the following equations:

    .. math::
        \begin{align}
        & \mathbf{\phi^T G_n^T \, M_\rho \, j}
        + i\omega \phi^T \mathbf{M_\mu \, h}
        - \phi^T (\mathbf{B \, h + q}) = 0 \\
        & \mathbf{\psi^T M_f G_n \, h} = \mathbf{\psi^T M_f j}
        \end{align}

    Combining these equations, we obtain the following linear system for
    computing the electric fields on mesh nodes:

    .. math::
        \big [ \mathbf{G_n^T M_\rho G_n}
        + i \omega\mathbf{M_\mu}
        - \mathbf{B} \big ] \mathbf{h}
        = \mathbf{q}

    where matrix :math:`\mathbf{B}` and vector :math:`\mathbf{q}` are used to
    impose boundary conditions.
    """
    # Extract vertical discretization
    hz = mesh.h[-1]

    if len(hz) != len(sigma_1d):
        raise ValueError(
            "Number of cells in vertical direction must match length of "
            f"'sigma_1d'. Here hz has length {len(hz)} and sigma_1d has "
            f"length {len(sigma_1d)}"
        )

    # Generate extended 1D mesh and resistivity model to solve 1D problem
    if n_pad is not None:
        raise warn(
            "Number of padding cells no longer set directly with 'n_pad'. "
            "Use 'n_skin_depths' to set extent of padding for 1D mesh."
        )
    else:
        skin_depth = np.sqrt(2 / (omega(freq) * mu_0 * sigma_1d[0]))
        n_pad = int(np.ceil(n_skin_depths * skin_depth / hz[0]))

    hz_ext = np.pad(hz, (n_pad, 0), mode="edge")
    mesh_ext = TensorMesh([hz_ext], origin=[mesh.origin[-1] - hz[0] * n_pad])
    sigma_1d_ext = np.pad(sigma_1d, (n_pad, 0), mode="edge")

    # Generate the system matrix
    G = mesh_ext.nodal_gradient
    M_e_rho = mesh_ext.get_edge_inner_product(sigma_1d_ext, invert_model=True)
    M_f_mu = mesh_ext.get_face_inner_product(mu_0)

    w = 2 * np.pi * freq
    A = G.T @ M_e_rho @ G + 1j * w * M_f_mu

    fixed_nodes = np.zeros(mesh_ext.n_nodes, dtype=bool)
    h_fixed = []
    q = np.zeros(mesh_ext.n_nodes, dtype=np.complex128)

    # Bottom BC
    if bot_bc == "dirichlet":
        e_d, e_u, h_d, h_u = getEHfields(mesh_ext, sigma_1d_ext, freq, mesh.nodes_x)
        h_tot = h_d + h_u
        h_tot /= h_tot[-1]
        fixed_nodes[0] = True
        h_fixed.append(h_tot[0])
    elif bot_bc == "robin":
        k_bot = np.sqrt(-1.0j * w * mu_0 * sigma_1d_ext[0])
        A[0, 0] += 1j * k_bot / sigma_1d[0]
    else:
        msg = (
            f"Invalid 'bot_bc' equal to '{bot_bc}'. "
            "It must be one of {'dirichlet', 'robin'}."
        )
        raise ValueError(msg)

    # Top BC
    if top_bc == "dirichlet":
        fixed_nodes[-1] = True
        h_fixed.append(1.0)
    elif top_bc == "neumann":
        q[-1] = 1.0
    else:
        msg = (
            f"Invalid 'top_bc' equal to '{top_bc}'. "
            "It must be one of {'dirichlet', 'neumann'}."
        )
        raise ValueError(msg)

    P_fixed = sp.eye(mesh_ext.n_nodes, format="csc")[:, fixed_nodes]
    P_free = sp.eye(mesh_ext.n_nodes, format="csc")[:, ~fixed_nodes]
    q_free = P_free.T @ (q - A @ (P_fixed @ h_fixed))
    A_free = P_free.T @ A @ P_free

    Ainv = get_default_solver()(A_free)
    h_1d = P_free @ (Ainv @ q_free) + P_fixed @ h_fixed

    if n_pad != 0:
        h_1d = h_1d[n_pad:]
    return h_1d


def project_1d_fields_to_mesh_edges(mesh, u_1d):
    """Project 1D nodal field solution to edges of a mesh.

    Parameters
    ----------
    mesh : discretize.base.BaseTensorMesh
        A 1d, 2d or 3d tensor mesh or tree mesh.
    u_1d : np.ndarray
        1D field solution along the vertical discretization of the input mesh.

    Returns
    -------
    numpy.ndarray (n_edges, n_polarization)
        Fields on the edges of the mesh for each polarization.
    """
    if len(u_1d) != (expected := len(mesh.h[-1]) + 1):
        msg = (
            f"Found invalid 'u_1d' with '{len(u_1d)}' elements. "
            f"It must match the number of vertical edges in the mesh ({expected})."
        )
        raise ValueError(msg)

    if mesh.dim == 1:
        return u_1d

    # Vertical 1D discretization
    hz = mesh.h[-1]
    mesh_1d = TensorMesh([hz], origin=[mesh.origin[-1]])

    # Field polarized along x-direction
    u_x = (
        mesh_1d.get_interpolation_matrix(mesh.edges_x[:, -1], location_type="nodes")
        @ u_1d
    )

    if mesh.dim == 2:
        return np.r_[u_x, np.zeros(mesh.n_edges_y)]

    else:

        # Field polarized along y-direction
        u_y = (
            mesh_1d.get_interpolation_matrix(mesh.edges_y[:, 2], location_type="nodes")
            @ u_1d
        )

        u_x = np.r_[u_x, np.zeros(mesh.n_edges_y + mesh.n_edges_z)]
        u_y = np.r_[np.zeros(mesh.n_edges_x), u_y, np.zeros(mesh.n_edges_z)]

        return np.c_[u_x, u_y]
