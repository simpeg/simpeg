# noqa: D100
import numpy as np
from scipy.constants import mu_0
from discretize import TensorMesh

from ....utils import mkvc, get_default_solver
from .solutions_1d import get1DEfields
from .analytic_1d import getEHfields


def homo1DModelSource(mesh, freq, sigma_1d):
    """Function that calculates and return background fields.

    Parameters
    ----------
    mesh : discretize.base.BaseTensorMesh
        A 1d, 2d or 3d tensor mesh or tree mesh.
    freq : float
        Operating frequency in Hz.
    sigma_1d :
        1D conductivity model defined along the vertical discretization.
        Conductivities are defined from the bottom cell upwards.

    Returns
    -------
    numpy.ndarray (n_edges, 2)
        E fields for the background model at both polarizations with shape.
    """

    # Get a 1d solution for a halfspace background
    if mesh.dim == 1:
        mesh1d = mesh
    else:
        mesh1d = TensorMesh([mesh.h[-1]], [mesh.x0[-1]])

    # Note: Everything is using e^iwt
    e0_1d = get1DEfields(mesh1d, sigma_1d, freq)
    if mesh.dim == 1:
        eBG_px = mkvc(e0_1d, 2)
        eBG_py = -mkvc(
            e0_1d, 2
        )  # added a minus to make the results in the correct quadrents.
    elif mesh.dim == 2:
        ex_px = np.zeros(mesh.vnEx, dtype=complex)
        ey_px = np.zeros((mesh.nEy, 1), dtype=complex)
        for i in np.arange(mesh.vnEx[0]):
            ex_px[i, :] = -e0_1d
        eBG_px = np.vstack((mkvc(ex_px, 2), ey_px))
        # Setup y (north) polarization (_py)
        ex_py = np.zeros((mesh.nEx, 1), dtype="complex128")
        ey_py = np.zeros(mesh.vnEy, dtype="complex128")
        # Assign the source to ey_py
        for i in np.arange(mesh.vnEy[0]):
            ey_py[i, :] = e0_1d
        # ey_py[1:-1, 1:-1, 1:-1] = 0
        eBG_py = np.vstack((ex_py, mkvc(ey_py, 2)))
    elif mesh.dim == 3:
        # us the z component of ex_grid as lookup for solution
        edges_u, inv_edges = np.unique(mesh.gridEx[:, -1], return_inverse=True)
        map_to_edge_u = np.where(np.isclose(mesh1d.gridN, edges_u[:, None], atol=0.0))[
            1
        ]
        ex_px = -e0_1d[map_to_edge_u][inv_edges]
        ey_px = np.zeros(mesh.nEy, dtype=complex)
        ez_px = np.zeros(mesh.nEz, dtype=complex)
        eBG_px = np.r_[ex_px, ey_px, ez_px][:, None]

        edges_u, inv_edges = np.unique(mesh.gridEy[:, -1], return_inverse=True)
        map_to_edge_u = np.where(np.isclose(mesh1d.gridN, edges_u[:, None], atol=0.0))[
            1
        ]
        ex_py = np.zeros(mesh.nEx, dtype=complex)
        ey_py = e0_1d[map_to_edge_u][inv_edges]
        ez_py = np.zeros(mesh.nEz, dtype=complex)
        eBG_py = np.r_[ex_py, ey_py, ez_py][:, None]

    # Return the electric fields
    eBG_bp = np.hstack((eBG_px, eBG_py))
    return eBG_bp


def analytic1DModelSource(mesh, freq, sigma_1d):
    """Function that calculates and return background fields.

    Parameters
    ----------
    mesh : discretize.base.BaseTensorMesh
        A 1d, 2d or 3d tensor mesh or tree mesh.
    freq : float
        Operating frequency in Hz.
    sigma_1d :
        1D conductivity model defined along the vertical discretization.
        Conductivities are defined from the bottom cell upwards.

    Returns
    -------
    numpy.ndarray (n_edges, 2)
        E fields for the background model at both polarizations with shape.
    """
    from simpeg.NSEM.Utils import getEHfields

    # Get a 1d solution for a halfspace background
    if mesh.dim == 1:
        mesh1d = mesh
    elif mesh.dim == 2:
        mesh1d = TensorMesh([mesh.h[1]], np.array([mesh.x0[1]]))
    elif mesh.dim == 3:
        mesh1d = TensorMesh([mesh.h[2]], np.array([mesh.x0[2]]))

    # # Note: Everything is using e^iwt
    Eu, Ed, _, _ = getEHfields(mesh1d, sigma_1d, freq, mesh.nodes_z)
    # Make the fields into a dictionary of location and the fields
    e0_1d = Eu + Ed
    E1dFieldDict = dict(zip(mesh.nodes_z, e0_1d))
    if mesh.dim == 1:
        eBG_px = mkvc(e0_1d, 2)
        eBG_py = -mkvc(
            e0_1d, 2
        )  # added a minus to make the results in the correct quadrents.
    elif mesh.dim == 2:
        ex_px = np.zeros(mesh.vnEx, dtype=complex)
        ey_px = np.zeros((mesh.nEy, 1), dtype=complex)
        for i in np.arange(mesh.vnEx[0]):
            ex_px[i, :] = -e0_1d
        eBG_px = np.vstack((mkvc(ex_px, 2), ey_px))
        # Setup y (north) polarization (_py)
        ex_py = np.zeros((mesh.nEx, 1), dtype="complex128")
        ey_py = np.zeros(mesh.vnEy, dtype="complex128")
        # Assign the source to ey_py
        for i in np.arange(mesh.vnEy[0]):
            ey_py[i, :] = e0_1d
        # ey_py[1:-1, 1:-1, 1:-1] = 0
        eBG_py = np.vstack((ex_py, mkvc(ey_py, 2)))
    elif mesh.dim == 3:
        # Setup x (east) polarization (_x)
        ex_px = -np.array([E1dFieldDict[i] for i in mesh.gridEx[:, 2]]).reshape(-1, 1)
        ey_px = np.zeros((mesh.nEy, 1), dtype=complex)
        ez_px = np.zeros((mesh.nEz, 1), dtype=complex)
        # Construct the full fields
        eBG_px = np.vstack((ex_px, ey_px, ez_px))
        # Setup y (north) polarization (_py)
        ex_py = np.zeros((mesh.nEx, 1), dtype="complex128")
        ey_py = np.array([E1dFieldDict[i] for i in mesh.gridEy[:, 2]]).reshape(-1, 1)
        ez_py = np.zeros((mesh.nEz, 1), dtype="complex128")
        # Construct the full fields
        eBG_py = np.vstack((ex_py, mkvc(ey_py, 2), ez_py))

    # Return the electric fields
    eBG_bp = np.hstack((eBG_px, eBG_py))
    return eBG_bp


# def homo3DModelSource(mesh, model, freq):
#     """
#     Function that estimates 1D analytic background fields from a 3D model.

#     Parameters
#     ----------
#     mesh : discretize.base.BaseTensorMesh
#         A 1d, 2d or 3d tensor mesh or tree mesh.
#     model :
#         1D conductivity model defined along the vertical discretization.
#         Conductivities are defined from the bottom cell upwards.
#     freq : float
#         Operating frequency in Hz.

#     Returns
#     -------
#     numpy.ndarray (n_edges, 2)
#         E fields for the background model at both polarizations.
#     """

#     if mesh.dim < 3:
#         raise IOError("Input mesh has to have 3 dimensions.")

#     # Get the locations
#     a = mesh.gridCC[:, 0:2].copy()
#     unixy = np.unique(
#         a.view(a.dtype.descr * a.shape[1])
#     ).view(float).reshape(-1, 2)
#     uniz = np.unique(mesh.gridCC[:, 2])
#     # # Note: Everything is using e^iwt
#     # Need to loop thourgh the xy locations, assess the model and
#     # calculate the fields at the phusdo cell centers.
#     # Then interpolate the cc fields to the edges.

#     e0_1d = get1DEfields(mesh1d, sigma_1d, freq)

#     # Setup x (east) polarization (_x)
#     ex_px = np.zeros(mesh.vnEx, dtype=complex)
#     ey_px = np.zeros((mesh.nEy, 1), dtype=complex)
#     ez_px = np.zeros((mesh.nEz, 1), dtype=complex)
#     # Assign the source to ex_x
#     for i in np.arange(mesh.vnEx[0]):
#         for j in np.arange(mesh.vnEx[1]):
#             ex_px[i, j, :] = -e0_1d
#     eBG_px = np.vstack((mkvc(ex_px, 2), ey_px, ez_px))
#     # Setup y (north) polarization (_py)
#     ex_py = np.zeros((mesh.nEx, 1), dtype="complex128")
#     ey_py = np.zeros(mesh.vnEy, dtype="complex128")
#     ez_py = np.zeros((mesh.nEz, 1), dtype="complex128")
#     # Assign the source to ey_py
#     for i in np.arange(mesh.vnEy[0]):
#         for j in np.arange(mesh.vnEy[1]):
#             ey_py[i, j, :] = e0_1d
#     # ey_py[1:-1, 1:-1, 1:-1] = 0
#     eBG_py = np.vstack((ex_py, mkvc(ey_py, 2), ez_py))

#     # Return the electric fields
#     eBG_bp = np.hstack((eBG_px, eBG_py))
#     return eBG_bp


def primary_e_1d_solution(
    mesh, sigma_1d, freq, top_bc="dirichlet", bot_bc="dirichlet", n_pad=500
):
    r"""Compute 1D electric field solution on nodes.

    Parameters
    ----------
    mesh : discretize.base.BaseTensorMesh
        A 1d, 2d or 3d tensor mesh or tree mesh.
    sigma_1d :
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
        &\frac{\partial h_y}{\partial z} + \sigma e_x = 0
        \end{align}

    **Discrete system:**

    We take the inner product of Faraday's law with a test function $\phi$,
    and the inner product of Ampere's law with a test function $\psi$.

    .. math::
        \begin{align}
        &\langle \phi , \partial_z e_x \rangle
        + i\omega \langle \phi , b \rangle = 0 \\
        &\langle \psi , \partial_z h_y \rangle
        + \langle \psi, \sigma e \rangle = 0
        \end{align}

    The inner-product with Ampere's law is Integrated by parts,
    and since $b_y = \mu h_y$:

    .. math::
        \begin{align}
        &\langle \phi , \partial_z e_x \rangle
        + i\omega \langle \phi , b_y \rangle = 0 \\
        - \langle \partial_z \psi , \mu^{-1} b_y \rangle
        &+ \psi h_y \bigg |_{bot}^{top} + \langle \psi, \sigma e_x \rangle = 0
        \end{align}

    In discrete form, the above equations are approximated by:

    .. math::
        \begin{align}
        &\mathbf{G_n e} = - i\omega \mathbf{b} \\
        &-i \omega \mathbf{G_n^T M_{\frac{1}{\mu}} b}
        + i \omega\mathbf{M_\sigma e} + i\omega h_y \bigg |_{bot}^{top} = 0
        \end{align}

    Combining these equations, we obtain the following system:

    .. math::
        \big [ \mathbf{G_n^T M_{\frac{1}{\mu}} G_n}
        + i \omega\mathbf{M_\sigma} \big ] \mathbf{e}
        + i\omega h_y \bigg |_{bot}^{top} = 0

    **Boundary Conditions:**

    At the top boundary, we can set either a Dirichlet or Neumann boundary
    condition on the electric field solution. For the Dirichlet condition,
    we set:

    .. math::
        e_x^{(top)} = 1

    For the Neumann condition, we set:

    .. math::
        \frac{\partial e_x^{(top)}}{\partial z} = - i\omega\mu

    which corresponds to $h_y^{(top)} = 1$. At the bottom boundary, there
    is only a downgoing wave of the form:

    .. math::
        e_x = E^- \exp (ikz)

    where

    .. math::
        k = \sqrt{-i\omega\mu_0\sigma}
        = (1 - i)\sqrt{\frac{\omega \mu_0\sigma}{2}}

    So if $\Delta z$ is negative, the downgoing wave decays. Thus we have the
    following condition:

    .. math::
        \begin{align}
        & ik E^- \exp (ikz) + i\omega b = 0 \\
        \implies & ik e_x + i \omega b = 0 \\
        \implies & k e_x + \omega b = 0
        \end{align}

    **Implementing the top boundary condition:**

    For the Dirichlet condition, let $\mathbf{A}$ represent the systems matrix
    and let $\mathbf{q}$ represent the right-hand side. To set $e_n=1$ at the
    top of the mesh, we replace $A_{n, n-1}=0$ and set $q_n = A_{n, n}$.

    The Neumann condition corresponds to setting the magnetic field
    to 1 at the top node of the mesh. At the top of the mesh,
    we know that:

    .. math::
        \frac{-e_{n-1} + 2 e_n - e_{n+1}}{(\Delta z)^2}
        + i \omega \, \mu \, \sigma e_n
        = \bigg ( \frac{-e_{n-1} + e_n}{(\Delta z)^2} \bigg )
        + \bigg ( \frac{e_n - e_{n+1}}{(\Delta z)^2} \bigg )
        + i \omega \mu \sigma e_n = 0

    where $\Delta z$ is the width of the last mesh cell, and $\sigma$
    and $\mu$ are homogeneous within the region. To set the Neumann condition,
    we use Taylor expansion:

    .. math::
        e_{n+1} - e_n = \frac{\partial e_n}{\partial z} \Delta z
        = - i\omega \mu \Delta z

    We then combine the above two expressions and multiply by
    $\Delta z/\mu$ to obtain:

    .. math::
        \frac{1}{\mu} \bigg ( \frac{-e_{n-1} + e_n}{\Delta z} \bigg )
        + \frac{i\omega}
        + i \omega \sigma \Delta z \, e_n = 0

    The boundary condition is implemented by setting $q_n = - i \omega$.

    **Implementing the bottom boundary condition**

    At the bottom of the mesh, we assume that there is only a downgoing
    planewave. We know that:

    .. math::
        \frac{-e_{-1} + 2 e_0 - e_{1}}{(\Delta z)^2} + i \omega \mu \sigma e_0
        = \frac{-e_{-1} + e_0}{(\Delta z)^2} + \frac{e_0 - e_{1}}{(\Delta z)^2}
        + i \omega \mu \sigma e_0 = 0

    where $Delta z$ is the width of the first mesh cell, and $\sigma$ and $\mu$
    are homogeneous within the local region. From Taylor expansion:

    .. math::
        e_{-1} - e_0 = -\frac{\partial e_0}{\partial z} \Delta z
        = i\omega b_0 \Delta z

    where $b_0$ is the magnetic flux density at node $0$. Thus:

    .. math::
        \frac{e_0 - e_{1}}{(\Delta z)^2} + i \omega \mu \sigma e_0
        - \frac{i \omega b_0}{\Delta z} = 0

    Given a downgoing planewave has the form $E = E_0 \, e^{ik(z-z_0)}$,
    where $k = \sqrt{-i \omega \mu \sigma}$, we know from Faraday's law that:

    .. math::
        k e_0 + \omega b_0 = 0

    By combining and multiplying through by $\Delta z / \mu$ we obtain:

    .. math::
        \frac{1}{\mu} \bigg ( \frac{e_0 - e_{1}}{\Delta z} \bigg )
        + i \omega \sigma \Delta z \, e_0 + \frac{ik}{\mu} e_0 = 0

    Boundary conditions are implemented by replacing the entries of the
    system matrix $\mathbf{A}$.

    """
    # Extract vertical discretization
    hz = mesh.h[-1]

    if len(hz) != len(sigma_1d):
        raise ValueError(
            "Number of cells in vertical direction must match length of "
            "'sigma_1d'. Here hz has length {} and sigma_1d has length "
            "{}".format(len(hz), len(sigma_1d))
        )

    # Generate extended 1D mesh and conductivity model to solve 1D problem
    hz_ext = np.pad(hz, (n_pad, 0), mode="edge")
    mesh_ext = TensorMesh([hz_ext], origin=[mesh.origin[-1] - hz[0] * n_pad])
    sigma_1d_ext = np.pad(sigma_1d, (n_pad, 0), mode="edge")

    # Generate the system matrix
    G = mesh_ext.nodal_gradient
    M_e_mui = mesh_ext.get_edge_inner_product(mu_0, invert_model=True)
    M_f_sigma = mesh_ext.get_face_inner_product(sigma_1d_ext)

    w = 2 * np.pi * freq
    A = G.T @ M_e_mui @ G + 1j * w * M_f_sigma

    # Approach 1

    # Impose boundary conditions
    q = np.zeros(mesh_ext.n_faces, dtype=np.complex128)

    if bot_bc == "dirichlet":
        e_d, e_u, h_d, h_u = getEHfields(mesh_ext, sigma_1d_ext, freq, mesh.nodes_x)
        e_tot = e_d + e_u
        q[0] = A[0, 0] * e_tot[0]
        A[0, 1] = 0.0
    elif bot_bc == "robin":
        k = np.sqrt(-1.0j * w * mu_0 * sigma_1d[0])
        A[0, 0] = (
            1.0 / (mu_0 * hz[0]) + 1.0j * k / mu_0 + 1.0j * w * hz[0] * sigma_1d[0]
        )
        A[0, 1] = -1.0 / (mu_0 * hz[0])
    else:
        raise ValueError("'bot_bc' must be one of {'dirichlet', 'robin'}.")

    if top_bc == "dirichlet":
        A[-1, -2] = 0
        q[-1] = A[-1, -1]
    elif top_bc == "neumann":
        q[-1] = -1.0j * w
    else:
        raise ValueError("'top_bc' must be one of {'dirichlet', 'neumann'}.")

    # Solve and return along original discretization
    Ainv = get_default_solver()(A)
    e_1d = Ainv @ q

    # Approach 2
    #
    # fixed_nodes = np.zeros(mesh_ext.n_nodes, dtype=bool)
    # e_fixed = []
    # if bot_bc == "dirichlet":
    #     Ed, Eu, Hd, Hu = getEHfields(mesh_ext, sigma_1d_ext, freq, mesh.nodes_x)
    #     e_tot = Ed + Eu
    #     e_tot /= e_tot[-1]
    #     fixed_nodes[0] = True
    #     e_fixed.append(e_tot[0])
    # else:
    #     # for bottom robin boundary condition
    #     k_bot = np.sqrt(-1.0j * omega * mu_0 * sigma_1d_ext[0])
    #     A[0, 0] += 1j * k_bot/mu_0

    # q = np.zeros(mesh_ext.n_nodes, dtype=np.complex128)
    # if top_bc == "dirichlet":
    #     fixed_nodes[-1] = True
    #     e_fixed.append(1.0)
    # else:
    #     q[-1] = -1j * omega

    # P_fixed = sp.eye(mesh_ext.n_nodes, format='csc')[:, fixed_nodes]
    # P_free = sp.eye(mesh_ext.n_nodes, format='csc')[:, ~fixed_nodes]
    # q_free = P_free.T @ (q - A @ (P_fixed @ e_fixed))
    # A_free = P_free.T @ A @ P_free

    # Ainv = get_default_solver()(A_free)
    # e_1d = P_free @ (Ainv @ q_free) + P_fixed @ e_fixed

    if n_pad != 0:
        e_1d = e_1d[n_pad:]
    return e_1d


def primary_h_1d_solution(
    mesh, sigma_1d, freq, top_bc="dirichlet", bot_bc="dirichlet", n_pad=500
):
    r"""Compute 1D magnetic field solution on nodes.

    Parameters
    ----------
    mesh : discretize.base.BaseTensorMesh
        A 1d, 2d or 3d tensor mesh or tree mesh.
    sigma_1d :
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
        &\frac{\partial e_y}{\partial z} - i\omega \mu h_x = 0 \\
        &\frac{\partial h_x}{\partial z} - j_y = 0
        \end{align}

    **Discrete system:**

    We take the inner product of Faraday's law with a test function $\phi$,
    and the inner product of Ampere's law with a test function $\psi$.

    .. math::
        \begin{align}
        &\langle \phi , \partial_z e_y \rangle
        - i\omega \langle \phi , \mu h_x \rangle = 0 \\
        &\langle \psi , \partial_z h_x \rangle
        - \langle \psi, j_y \rangle = 0
        \end{align}

    The inner-product with Ampere's law is Integrated by parts,
    and since $e_y = \rho j_y$:

    .. math::
        \begin{align}
        &-\langle \partial_z \phi , \rho \, j_y \rangle
        + \phi e_y \bigg |_{bot}^{top}
        - i\omega \langle \phi , \mu h_x \rangle = 0 \\
        &\langle \psi , \partial_z h_x \rangle
        - \langle \psi, j_y \rangle = 0
        \end{align}

    In discrete form, the above equations are approximated by:

    .. math::
        \begin{align}
        &\mathbf{G_n^T M_\rho \, j} + i\omega \mathbf{M_\mu \, h}
        = e_y \bigg |_{bot}^{top} \\
        &\mathbf{G_n \n h} = \mathbf{j}
        \end{align}

    Combining these equations, we obtain the following system:

    .. math::
        \big [ \mathbf{G_n^T M_\rho G_n}
        + i \omega\mathbf{M_\mu} \big ] \mathbf{h}
        = e_y \bigg |_{bot}^{top}


    **Boundary Conditions:**

    At the top boundary, we can set either a Dirichlet or Neumann boundary
    condition on the magnetic field solution. For the Dirichlet condition,
    we set:

    .. math::
        h_x^{(top)} = 1

    For the Neumann condition, we set:

    .. math::
        \rho \frac{\partial h_x^{(top)}}{\partial z} = 1

    which corresponds to $e_y^{(top)} = 1$. At the bottom boundary, there is
    only a downgoing wave of the form:

    .. math::
        h_x = H^- \exp (ikz)

    where

    .. math::
        k = \sqrt{\frac{-i\omega\mu}{\rho}}
        = (1 - i)\sqrt{\frac{\omega \mu}{2 \rho}}

    So if $\Delta z$ is negative, the downgoing wave decays. From Ampere's law:

    .. math::
        \begin{align}
        & ik H^- \exp (ikz) - \frac{e_y}{\rho} = 0 \\
        \implies & ik h_x - \frac{e_y}{\rho} = 0\\
        \implies & \frac{\partial h_x}{\partial z} - i k h_x = 0
        \end{align}

    **Implementing the top boundary condition:**

    For the Dirichlet condition, let $\mathbf{A}$ represent the systems matrix
    and let $\mathbf{q}$ represent the right-hand side. To set $h_n=1$ at the
    top of the mesh, we replace $A_{n, n-1}=0$ and set $q_n = A_{n, n}$.

    The Neumann condition corresponds to setting the electric field
    to 1 at the top node of the mesh. At the top of the mesh,
    we know that:

    .. math::
        \rho \bigg ( \frac{-h_{n-1} + 2 h_n - h_{n+1}}{(\Delta z)^2}\bigg )
        + i \omega \mu h_n =
        \rho \bigg ( \frac{-h_{n-1} + h_n}{(\Delta z)^2} \bigg )
        + \rho \bigg ( \frac{h_n - h_{n+1}}{(\Delta z)^2} \bigg )
        + i \omega \mu h_n = 0

    where $\Delta z$ is the width of the last mesh cell, and $\rho$
    and $\mu$ are homogeneous within the region. To set the Neumann condition,
    we use Taylor expansion:

    .. math::
        h_{n+1} - h_n = \frac{\partial h_n}{\partial z} h = \frac{1}{\rho}

    We then combine the above two expressions and multiply by
    $\rho \Delta z$ to obtain:

    .. math::
        \rho \bigg ( \frac{h_{n-1} + h_n}{\Delta z} \bigg )
        + i \omega \mu \Delta z \, h_n = 1

    The boundary condition is implemented by setting $q_n = 1$.

    **Implementing the bottom boundary condition**

    At the bottom of the mesh, we assume that there is only a downgoing
    planewave. We know that:

    .. math::
        \bigg ( \frac{-h_{-1} + 2 h_0 - h_{1}}{(\Delta z)^2} \bigg )
        + \frac{i \omega \mu}{\rho} h_0
        = \bigg ( \frac{-h_{-1} + h_0}{(\Delta z)^2} \bigg )
        + \bigg ( \frac{h_0 - h_{1}}{(\Delta z)^2} \bigg )
        + \frac{i \omega \mu}{\rho} h_0 = 0

    where $Delta z$ is the width of the first mesh cell, and $\rho$ and $\mu$
    are homogeneous within the local region. From Taylor expansion:

    .. math::
        h_{-1} - h_0 = -\frac{\partial h_0}{\partial z} \Delta z
        = -\Delta z \, j_0

    where $j_0$ is the electric current density at node $0$. Thus:

    .. math::
        \frac{j_0}{\Delta z}
        + \bigg ( \frac{h_0 - h_1}{(\Delta z)^2} \bigg )
        + \frac{i \omega \mu}{\rho} h_0 = 0

    Given a downgoing planewave has the form $h = H^- \, e^{ik(z-z_0)}$,
    where $k = \sqrt{-i \omega \mu /\rho}$, we know from Ampere's law that:

    .. math::
        i k h_0 = j_0

    By combining and multiplying through by $\rho \Delta z$ we obtain:

    .. math::
        ik \rho h_0 + i \omega \mu \Delta z \, h_0 +
        \rho \bigg ( \frac{h_0 - h_1}{\Delta z} \bigg ) = 0

    Boundary conditions are implemented by replacing the entries of the
    system matrix $\mathbf{A}$.

    """
    # Extract vertical discretization
    hz = mesh.h[-1]

    if len(hz) != len(sigma_1d):
        raise ValueError(
            "Number of cells in vertical direction must match length of "
            "'sigma_1d'. Here hz has length {} and sigma_1d has length "
            "{}".format(len(hz), len(sigma_1d))
        )

    # Generate extended 1D mesh and resistivity model to solve 1D problem
    hz_ext = np.pad(hz, (n_pad, 0), mode="edge")
    mesh_ext = TensorMesh([hz_ext], origin=[mesh.origin[-1] - hz[0] * n_pad])
    rho_1d_ext = np.pad(1.0 / sigma_1d, (n_pad, 0), mode="edge")

    # Generate the system matrix
    G = mesh_ext.nodal_gradient
    M_e_rho = mesh_ext.get_edge_inner_product(rho_1d_ext)
    M_f_mu = mesh_ext.get_face_inner_product(mu_0)

    w = 2 * np.pi * freq
    A = G.T @ M_e_rho @ G + 1j * w * M_f_mu

    # Impose boundary conditions
    q = np.zeros(mesh_ext.n_faces, dtype=np.complex128)

    if bot_bc == "dirichlet":
        e_d, e_u, h_d, h_u = getEHfields(mesh_ext, 1 / rho_1d_ext, freq, mesh.nodes_x)
        h_tot = h_d + h_u
        q[0] = A[0, 0] * h_tot[0]
        A[0, 1] = 0.0
    elif bot_bc == "robin":
        k = np.sqrt(-1.0j * w * mu_0 / rho_1d_ext[0])
        A[0, 0] = rho_1d_ext[0] * (1.0j * k + 1 / hz[0]) + 1.0j * w * mu_0 * hz[0]
        A[0, 1] = -rho_1d_ext[0] / hz[0]
    else:
        raise ValueError("'bot_bc' must be one of {'dirichlet', 'robin'}.")

    if top_bc == "dirichlet":
        A[-1, -2] = 0
        q[-1] = A[-1, -1]
    elif top_bc == "neumann":
        q[-1] = 1
    else:
        raise ValueError("'top_bc' must be one of {'dirichlet', 'neumann'}.")

    # Solve and return along original discretization
    Ainv = get_default_solver()(A)
    h_1d = Ainv @ q
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
    if len(u_1d) != mesh.n_edges_per_direction[mesh.dim - 1]:
        raise ValueError("Length of u_1d must match number of vertical edges in mesh.")

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
