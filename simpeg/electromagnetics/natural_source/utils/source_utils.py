import numpy as np
from scipy.constants import mu_0
from discretize import TensorMesh
from discretize.utils import sdiag

from ....utils import mkvc
from .solutions_1d import get1DEfields
from pymatsolver import Solver


def homo1DModelSource(mesh, freq, sigma_1d):
    """
    Function that calculates and return background fields

    :param discretize.base.BaseMesh mesh: Holds information on the discretization
    :param float freq: The frequency to solve at
    :param numpy.ndarray sigma_1d: Background model of conductivity to base the calculations on, 1d model.
    :rtype: numpy.ndarray
    :return: eBG_bp, E fields for the background model at both polarizations with shape (mesh.nE, 2).

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
    """
    Function that calculates and return background fields

    :param discretize.base.BaseMesh mesh: Holds information on the discretization
    :param float freq: The frequency to solve at
    :param numpy.ndarray sigma_1d: Background model of conductivity to base the calculations on, 1d model.
    :rtype: numpy.ndarray
    :return: eBG_bp, E fields for the background model at both polarizations with shape (mesh.nE, 2).

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
#         Function that estimates 1D analytic background fields from a 3D model.

#         :param Simpeg mesh object mesh: Holds information on the discretization
#         :param float freq: The frequency to solve at
#         :param np.array sigma_1d: Background model of conductivity to base the calculations on, 1d model.
#         :rtype: numpy.ndarray (mesh.nE, 2)
#         :return: eBG_bp, E fields for the background model at both polarizations.

#     """

#     if mesh.dim < 3:
#         raise IOError('Input mesh has to have 3 dimensions.')


#     # Get the locations
#     a = mesh.gridCC[:, 0:2].copy()
#     unixy = np.unique(a.view(a.dtype.descr * a.shape[1])).view(float).reshape(-1, 2)
#     uniz = np.unique(mesh.gridCC[:, 2])
#     # # Note: Everything is using e^iwt
#     # Need to loop thourgh the xy locations, assess the model and
#     # calculate the fields at the phusdo cell centers.
#     # Then interpolate the cc fields to the edges.

#     e0_1d = get1DEfields(mesh1d, sigma_1d, freq)

#     elif mesh.dim == 3:
#         # Setup x (east) polarization (_x)
#         ex_px = np.zeros(mesh.vnEx, dtype=complex)
#         ey_px = np.zeros((mesh.nEy, 1), dtype=complex)
#         ez_px = np.zeros((mesh.nEz, 1), dtype=complex)
#         # Assign the source to ex_x
#         for i in np.arange(mesh.vnEx[0]):
#             for j in np.arange(mesh.vnEx[1]):
#                 ex_px[i, j, :] = -e0_1d
#         eBG_px = np.vstack((mkvc(ex_px, 2), ey_px, ez_px))
#         # Setup y (north) polarization (_py)
#         ex_py = np.zeros((mesh.nEx, 1), dtype='complex128')
#         ey_py = np.zeros(mesh.vnEy, dtype='complex128')
#         ez_py = np.zeros((mesh.nEz, 1), dtype='complex128')
#         # Assign the source to ey_py
#         for i in np.arange(mesh.vnEy[0]):
#             for j in np.arange(mesh.vnEy[1]):
#                 ey_py[i, j, :] = e0_1d
#         # ey_py[1:-1, 1:-1, 1:-1] = 0
#         eBG_py = np.vstack((ex_py, mkvc(ey_py, 2), ez_py))

#     # Return the electric fields
#     eBG_bp = np.hstack((eBG_px, eBG_py))
#     return eBG_bp


def primary_e_1d_solution(mesh, sigma_1d, freq, top_bc="dirichlet", n_pad=500):
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
    if n_pad == 0:
        hz_ext = hz
    else:
        hz_ext = np.r_[hz[0] * np.ones(n_pad), hz]
    mesh_1d_ext = TensorMesh([hz_ext], origin=[mesh.origin[-1] - hz[0] * n_pad])

    sigma_1d_ext = np.r_[
        sigma_1d[0] * np.ones(n_pad),
        sigma_1d,
    ]
    sigma_1d_ext = mesh_1d_ext.average_face_to_cell.T * (hz_ext * sigma_1d_ext)
    sigma_1d_ext[-1] = hz[-1] * sigma_1d[-1]  # Needed for top BC

    # Could add background susceptibility in future.
    mui_1d_ext = hz_ext / mu_0

    # Generate system matrix
    w = 2 * np.pi * freq
    A = mesh_1d_ext.nodal_gradient.T @ (
        sdiag(mui_1d_ext) @ mesh_1d_ext.nodal_gradient
    ) + 1j * w * sdiag(sigma_1d_ext)

    # Bottom boundary condition
    k = np.sqrt(-1.0j * w * mu_0 * sigma_1d_ext[0])
    A[0, 0] = 1.0 / (mu_0 * hz[0]) + 1j * k / mu_0 + 1j * w * hz[0] * sigma_1d[0]
    A[0, 1] = -1 / (mu_0 * hz[0])

    # Top boundary condition
    q = np.zeros(mesh_1d_ext.n_faces, dtype=np.complex128)
    if top_bc == "neumann":
        q[-1] = -1j * w
    else:
        A[-1, -2] = 0
        q[-1] = A[-1, -1]

    # Solve and return solution on original vertical discretization
    Ainv = Solver(A)
    e_1d = Ainv * q
    return e_1d[n_pad:]


def primary_h_1d_solution(mesh, sigma_1d, freq, top_bc="dirichlet", n_pad=500):
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

    # Generate extended 1D mesh and conductivity model to solve 1D problem
    if n_pad == 0:
        hz_ext = hz
    else:
        hz_ext = np.r_[hz[0] * np.ones(n_pad), hz]
    mesh_1d_ext = TensorMesh([hz_ext], origin=[mesh.origin[-1] - hz[0] * n_pad])

    rho_1d = sigma_1d**-1
    rho_1d_ext = np.r_[rho_1d[0] * np.ones(n_pad), rho_1d]
    rho_1d_ext = hz_ext * rho_1d_ext

    # Could add background susceptibility in future.
    mu_1d_ext = mesh_1d_ext.average_face_to_cell.T * (hz_ext * mu_0)
    mu_1d_ext[-1] = hz[-1] * mu_0

    # Generate system matrix
    w = 2 * np.pi * freq
    A = mesh_1d_ext.nodal_gradient.T @ (
        sdiag(rho_1d_ext) @ mesh_1d_ext.nodal_gradient
    ) + 1j * w * sdiag(mu_1d_ext)

    # Bottom boundary condition
    k = np.sqrt(-1.0j * w * mu_0 / rho_1d_ext[0])
    A[0, 0] = rho_1d_ext[0] * (1.0j * k + 1 / hz[0]) + 1.0j * w * mu_0 * hz[0]
    A[0, 1] = -rho_1d_ext[0] / hz[0]

    # Top boundary condition
    q = np.zeros(mesh_1d_ext.n_faces, dtype=np.complex128)
    if top_bc == "neumann":
        q[-1] = 1
    else:
        A[-1, -2] = 0
        q[-1] = A[-1, -1]

    # Solve and return solution on original vertical discretization
    Ainv = Solver(A)
    e_1d = Ainv * q
    return e_1d[n_pad:]


# def project_1d_primary_to_mesh(mesh, u_1d):
def project_e_1d_to_e_primary(mesh, e_1d):
    """Project 1D electric field solution on nodes to edges of a mesh.

    Parameters
    ----------
    mesh : discretize.base.BaseTensorMesh
        A 1d, 2d or 3d tensor mesh or tree mesh.
    e_1d : np.ndarray
        1D electric field solution along the vertical discretization.

    Returns
    -------
    numpy.ndarray (n_edges, n_polarization)
        Electric fields on the edges of the mesh for each polarization.
    """
    if mesh.dim == 1:
        return e_1d

    hz = mesh.h[-1]
    mesh_1d = TensorMesh([hz], origin=[mesh.origin[-1]])

    # Incident E-field polarized along x-direction
    ep_x = (
        mesh_1d.get_interpolation_matrix(mesh.edges_x[:, -1], location_type="nodes")
        @ e_1d
    )

    if mesh.dim == 2:
        return np.r_[ep_x, np.zeros(mesh.n_edges_y)]

    elif mesh.dim == 3:

        ep_x = np.r_[ep_x, np.zeros(mesh.n_edges_y + mesh.n_edges_z)]

        # Incident E-field polarized along y-direction
        ep_y = (
            mesh_1d.get_interpolation_matrix(
                mesh.edges_y[:, 2], location_type="nodes"
            ) @ e_1d
        )
        ep_y = np.r_[np.zeros(mesh.n_edges_x), ep_y, np.zeros(mesh.n_edges_z)]

        return np.c_[ep_x, ep_y]
