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
#     # Need to loop thourgh the xy locations, assess the model and calculate the fields at the phusdo cell centers.
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



def primary_e_1d_solution(mesh, sigma_1d, freq, n_pad=2000):
    r"""Compute 1D electric field solution. 

    Parameters
    ----------
    mesh : discretize.base.BaseTensorMesh
        A 1d, 2d or 3d tensor mesh or tree mesh.
    sigma_1d :
        1D conductivity model defined along the vertical discretization. Conductivities are
        defined from the bottom cell upwards.
    freq : float
        Operating frequency in Hz.
    n_pad : int
        Number of padding cells added to top and bottom of discrete 1D solution
        to ensure boundary conditions implemented accurately.

    Returns
    -------
    numpy.ndarray (n_edges,)
        Total electric field solution on the nodes of the 1D vertical discretization.

    Notes
    -----
    For the 1D electric field solution, Maxwell's equations take the form:

    .. math::
        \begin{align}
        &\frac{\partial e_x}{\partial z} + i\omega b_y = 0 \\
        &\frac{\partial h_y}{\partial z} + \sigma e_x = 0
        \end{align}

    **Boundary Conditions:**

    At the top boundary, we set $h_y^{(top)} = 1$. Therefore from Faraday's law:

    .. math::
        \frac{\partial e_x^{(top)}}{\partial z} = - i\omega\mu_0

    At the bottom boundary, there is only a downgoing wave of the form:

    .. math::
        e_x = E^- \exp (ikz)

    where

    .. math::
        k = \sqrt{-i\omega\mu_0\sigma} = (1 - i)\sqrt{\frac{\omega \mu_0\sigma}{2}}

    So if $\Delta z$ is negative, the downgoing wave decays. From Faraday's law:

    .. math::
        \begin{align}
        & ik E^- \exp (ikz) + i\omega b = 0 \\
        \implies & ik e_x + i \omega b = 0 \\
        \implies & k e_x + \omega b = 0
        \end{align}

    **Discrete system:**

    We take the inner product of Faraday's law with a test function $\mathbf{u}$, and the
    inner product of Ampere's law with a test function $\mathbf{f}$.
    .. math::
        \begin{align}
        &\langle u , \partial_z e_x \rangle + i\omega \langle u , b \rangle = 0 \\
        &\langle f , \partial_z h_y \rangle + \langle f, \sigma e \rangle = 0
        \end{align}

    The inner-product with Ampere's law is Integrated by parts:
    
    .. math::
        \begin{align}
        &\langle u , \partial_z e_x \rangle + i\omega \langle u , b_y \rangle = 0 \\
        &- \langle \partial_z f , h_y \rangle + f h_y \bigg |_{bot}^{top} + \langle f, \sigma e_x \rangle = 0
        \end{align}

    In discrete form, the above equations are approximated by:
    .. math::
        \begin{align}
        &\mathbf{G_n e} = - i\omega \mathbf{b} \\
        &-i \omega \mathbf{G_n^T M_{\mu} b} + i \omega\mathbf{M_\sigma e} + i\omega h_y \bigg |_{bot}^{top} = 0
        \end{align}

    Combining these equations, we obtain the following system:
    
    .. math::
        \big [ \mathbf{G_n^T M_{\mu} G_n} + i \omega\mathbf{M_\sigma} \big ] \mathbf{e} + i\omega h_y \bigg |_{bot}^{top} = 0

    When $\mu = \mu_0$, we can multiply through an obtain:

    .. math::
        \big [ \mathbf{G_n^T G_n} + i \omega \mu_0 \, diag (\sigma) \big ] \mathbf{e} + i\omega \mu_0 h_y \bigg |_{bot}^{top} = 0
    
    **Implementing Discrete Boundary Conditions:**

    At the top node of the mesn, we know that:

    .. math::
    \frac{-e_{n-1} + 2 e_n - e_{n+1}}{h^2} + i \omega \mu_0 \sigma_n e_n
    = \bigg ( \frac{-e_{n-1} + e_n}{h^2} \bigg ) + \bigg ( \frac{e_n - e_{n+1}}{h^2} \bigg ) + i \omega \mu_0 \sigma_n e_n = 0

    Using Taylor expansion:

    .. math::
        e_{n+1} - e_n = \frac{\partial e_n}{\partial z} h = - i\omega\mu_0 h

    Thus:

    .. math::
        \frac{-e_{n-1} + 2 e_n - e_{n+1}}{h^2} + i \omega \mu_0 \sigma_n e_n
        = \bigg ( \frac{-e_{n-1} + e_n}{h^2} \bigg ) + \bigg ( \frac{i\omega \mu_0}{h} \bigg ) + i \omega \mu_0 \sigma_n e_n = 0

    And we set
    
    .. math::
        q_n = - \frac{i \omega \mu_0}{h}

    At the bottom of the mesh, we know that:

    .. math::
    \frac{-e_{-1} + 2 e_0 - e_{1}}{h^2} + i \omega \mu_0 \sigma e_0
    = \frac{-e_{-1} + e_0}{h^2} + \frac{e_0 - e_{1}}{h^2} + i \omega \mu_0 \sigma e_0 = 0

    But we know at the bottom, from Taylor expansion:

    .. math::
    e_{-1} - e_0 = -\frac{\partial e_0}{\partial z} h = i\omega b_0 h

    Thus:

    .. math::
        \frac{e_0 - e_{1}}{h^2} + i \omega \mu_0 \sigma e_0 - \frac{i \omega b_0}{h} = 0

    But

    .. math::
        k e_0 + \omega b_0 = 0

    So

    .. math::
        \frac{e_0 - e_{1}}{h^2} + i \omega \mu_0 \sigma e_0 + \frac{ik}{h} e_0 = 0

    """

    # Extract vertical discretization
    if mesh.dim == 1:
        hz = mesh.h
    else:
        hz = mesh.h[-1]

    if len(hz) != len(sigma_1d):
        raise ValueError(
            "Number of cells in vertical direction must match length of 'sigma_1d'. Here hz has length {} and sigma_1d has length {}".format(len(hz), len(sigma_1d)))

    # Generate extended 1D mesh and model to solve 1D problem
    hz_ext = np.r_[hz[0] * np.ones(n_pad), hz, hz[-1] * np.ones(n_pad)]
    mesh_1d_ext = TensorMesh([hz_ext], origin=[mesh.origin[-1] - hz[0] * n_pad])
    
    sigma_1d_ext = np.r_[sigma_1d[0] * np.ones(n_pad), sigma_1d, sigma_1d[-1] * np.ones(n_pad)]
    sigma_1d_ext = mesh_1d_ext.average_face_to_cell.T * sigma_1d_ext
    sigma_1d_ext[0] = sigma_1d[1]
    sigma_1d_ext[-1] = sigma_1d[-2]

    # Solve the 1D problem for electric fields on nodes
    w = 2*np.pi*freq
    k = np.sqrt(-1.j * w * mu_0 * sigma_1d_ext[0])

    A = mesh_1d_ext.nodal_gradient.T @ mesh_1d_ext.nodal_gradient + 1j*w*mu_0 * sdiag(sigma_1d_ext)
    A[0, 0] = (1. + 1j*k*hz[0]) / hz[0]**2 + 1j*w*mu_0*sigma_1d[0]
    A[0, 1] = -1 / hz[0]**2

    q = np.zeros(mesh_1d_ext.n_faces, dtype=np.complex128)
    q[-1] = -1j*w*mu_0 / hz[-1]

    Ainv = Solver(A)
    e_1d = Ainv * q

    # Return solution along original vertical discretization
    return e_1d[n_pad:-n_pad]


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
        mesh_1d.get_interpolation_matrix(
            mesh.edges_x[:, -1], location_type="nodes"
        ) @ e_1d
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














