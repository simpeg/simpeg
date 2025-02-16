import numpy as np
from scipy.constants import mu_0

from ... import maps
from ..frequency_domain.sources import BaseFDEMSrc
from ..utils import omega
from .utils.source_utils import homo1DModelSource
from .utils.solutions_1d import get1DEfields
import discretize
from discretize.utils import volume_average
from pymatsolver import Solver


#################
#    Sources    #
#################


class Planewave(BaseFDEMSrc):
    """
    Source class for the 1D and pseudo-3D problems.

    Parameters
    ----------
    receiver_list : list of simpeg.electromagnetics.frequency_domain.receivers.BaseRx
        A list of NSEM receivers
    frequency : float
        Source frequency
    """

    # This class is only provided to have a uniquely identifiable "Planewave" class
    pass


# Need to implement such that it works for all dims.
# Rename to be more descriptive
class PlanewaveXYPrimary(Planewave):
    """
    NSEM planewave source for both polarizations (x and y)
    estimated from a single 1D primary models.

    Parameters
    ----------
    receiver_list : list of simpeg.electromagnetics.frequency_domain.receivers.BaseRx
        A list of NSEM receivers
    frequency : float
        Source frequency
    sigma_primary : float, default: ``None``
        Wholespace conductivity for primary field
    """

    _fields_per_source = 2

    def __init__(self, receiver_list, frequency, sigma_primary=None):
        # assert mkvc(self.mesh.h[2].shape,1) == mkvc(sigma1d.shape,1),'The number of values in the 1D background model does not match the number of vertical cells (hz).'
        self.sigma1d = None
        self._sigma_primary = sigma_primary
        super(PlanewaveXYPrimary, self).__init__(receiver_list, frequency)

    def _get_sigmas(self, simulation):
        try:
            return self._sigma1d, self._sigma_p
        except AttributeError:
            # set _sigma 1D
            if self._sigma_primary is None:
                self._sigma_primary = simulation.sigmaPrimary
            # Create 3d_1d mesh like me...
            if simulation.mesh.dim == 3:
                mesh3d = simulation.mesh
                x0 = mesh3d.x0
                hs = [
                    [mesh3d.nodes_x[-1] - x0[0]],
                    [mesh3d.nodes_y[-1] - x0[1]],
                    mesh3d.h[-1],
                ]
                mesh1d = discretize.TensorMesh(hs, x0=x0)
                if len(self._sigma_primary) == mesh3d.nC:
                    # volume average down to 1D mesh
                    self._sigma1d = np.exp(
                        volume_average(mesh3d, mesh1d, np.log(self._sigma_primary))
                    )
                elif len(self._sigma_primary) == mesh1d.nC:
                    self._sigma1d = self._sigma_primary
                else:
                    self._sigma1d = np.ones(mesh1d.nC) * self._sigma_primary
                self._sigma_p = np.exp(
                    volume_average(mesh1d, mesh3d, np.log(self._sigma1d))
                )
            else:
                self._sigma1d = simulation.mesh.reshape(
                    simulation._sigmaPrimary, "CC", "CC", "M"
                )[:]
                self._sigma_p = None
                self.sigma1d = self._sigma1d
            return self._sigma1d, self._sigma_p

    def ePrimary(self, simulation):
        """Primary electric field

        Parameters
        ----------
        simulation : simpeg.electromagnetics.natural_source_simulation.BaseNSEMSimulation
            A NSEM simulation

        Returns
        -------
        numpy.ndarray
            Primary electric field
        """
        if self._ePrimary is None:
            sigma_1d, _ = self._get_sigmas(simulation)
            self._ePrimary = homo1DModelSource(
                simulation.mesh, self.frequency, sigma_1d
            )
        return self._ePrimary

    def bPrimary(self, simulation):
        """Primary magnetic field

        Parameters
        ----------
        simulation : simpeg.electromagnetics.frequency_domain.simulation.BaseFDEMSimulation
            A NSEM simulation

        Returns
        -------
        numpy.ndarray
            Primary magnetic field
        """
        # Project ePrimary to bPrimary
        # Satisfies the primary(background) field conditions
        if simulation.mesh.dim == 1:
            C = simulation.mesh.nodal_gradient
        elif simulation.mesh.dim == 3:
            C = simulation.mesh.edge_curl
        bBG_bp = (-C * self.ePrimary(simulation)) * (1 / (1j * omega(self.frequency)))
        return bBG_bp

    def s_e(self, simulation):
        """Electric source term

        Parameters
        ----------
        simulation : simpeg.electromagnetics.frequency_domain.simulation.BaseFDEMSimulation
            A NSEM simulation

        Returns
        -------
        numpy.ndarray
            Electric source term on mesh.
        """
        e_p = self.ePrimary(simulation)
        # Make mass matrix
        # Note: M(sig) - M(sig_p) = M(sig - sig_p)
        # Need to deal with the edge/face discrepencies between 1d/2d/3d
        if simulation.mesh.dim == 1:
            Map_sigma_p = maps.SurjectVertical1D(simulation.mesh)
            sigma_p = Map_sigma_p._transform(self.sigma1d)
            Mesigma = simulation.mesh.get_face_inner_product(simulation.sigma)
            Mesigma_p = simulation.mesh.get_face_inner_product(sigma_p)
        if simulation.mesh.dim == 2:
            pass
        if simulation.mesh.dim == 3:
            _, sigma_p = self._get_sigmas(simulation)
            Mesigma = simulation.MeSigma
            Mesigma_p = simulation.mesh.get_edge_inner_product(sigma_p)
        return Mesigma * e_p - Mesigma_p * e_p

    def s_eDeriv(self, simulation, v, adjoint=False):
        """Derivative of electric source term with respect to model

        Parameters
        ----------
        simulation : simpeg.electromagnetics.frequency_domain.simulation.BaseFDEMSimulation
            A NSEM simulation
        v : numpy.ndarray
            A vector
        adjoint : bool, default: ``False``
            If ``True``, perform the adjoint operation

        Returns
        -------
        numpy.ndarray
            Derivative of electric source term on mesh.
        """

        return self.s_eDeriv_m(simulation, v, adjoint)

    def s_eDeriv_m(self, simulation, v, adjoint=False):
        """Derivative of electric source term with respect to model

        Parameters
        ----------
        simulation : simpeg.electromagnetics.frequency_domain.simulation.BaseFDEMSimulation
            A NSEM simulation
        v : numpy.ndarray
            A vector
        adjoint : bool, default: ``False``
            If ``True``, perform the adjoint operation

        Returns
        -------
        numpy.ndarray
            Derivative of electric source term on mesh.
        """
        # Need to deal with
        if simulation.mesh.dim == 1:
            # Need to use the faceInnerProduct
            ePri = self.ePrimary(simulation)[:, 1]
            return simulation.MfSigmaDeriv(ePri, v, adjoint=adjoint)
        if simulation.mesh.dim == 2:
            raise NotImplementedError("The NSEM 2D simulation is not implemented")
        if simulation.mesh.dim == 3:
            # Need to take the derivative of both u_px and u_py
            # And stack them to be of the correct size
            e_p = self.ePrimary(simulation)
            return simulation.MeSigmaDeriv(e_p, v, adjoint=adjoint)

    S_e = s_e
    S_eDeriv = s_eDeriv


class FictitiousSource3D(BaseFDEMSrc):
    r"""Fictitious source class for 3D natural source EM simulations.

    This class uses the method of fictitious sources to implement the boundary conditions
    required to compute the NSEM fields. The ``FictitiousSource3D`` class is used in
    conjunction with the :class:`.natural_source.Simulation3DFictitiousSource` simulation class.
    See the *Notes* section for a discription of how fictitious sources are generated.

    Parameters
    ----------
    receiver_list : list of .natural_source.receivers.ApparentConductivity
        List of NSEM receivers.
    frequency : float
        Source frequency in Hz.

    Notes
    -----
    Let :math:`\vec{u}_0` represent the known field solution corresponding to a background conductivity
    distribution :math:`\sigma_0`. Where :math:`\mathbf{u_0}` is the known solution discretized to a
    3D mesh and :math:`\mathbf{A}(\sigma_0)` is the system matrix constructed from the background conductivity,
    the fictitious source :math:`\mathbf{s_e}` is obtained by computing:

    .. math::
        \mathbf{s_e} = \frac{1}{i \omega} \mathbf{A}(\sigma_0) \, \mathbf{u_0}

    where :math:`\omega` is the angular frequency. Once the source term is obtained, the unknown discrete
    field solution :math:`\mathbf{u}` for a conductivity distribution :math:`\sigma` can be computed by
    solving:

    .. math::
        mathbf{A}(\sigma) \, \mathbf{u} = i \omega \mathbf{s_e}

    For NSEM simulations, we must obtain a fictitious source for each incident planewave polarization.
    Depending on the background conductivity provided, the discrete background field solution
    :math:`\mathbf{u_0}` is obtained one of two ways.

    **1D Method:**

    In the absence of surface topography, the background conductivity :math:`\sigma_0` is defined within
    the :class:`.natural_source.Simulation3DFictitiousSource` as a 1D layered Earth. The 1D finite volume
    NSEM problem is solved for the background conductivity to obtain a 1D field solution :math:`\mathbf{u_{1D}}`.
    The solution uses a :math:`\mathbf{u_{1D}}=1` boundary condition on the top and :math:`\mathbf{u_{1d}}=0`
    boundary condition at the bottom.

    The known 3D field solution :math:`\mathbf{u_0}` for an incident planewave polarized along the x-direction
    is obtained by projecting the 1D solution :math:`\mathbf{u_{1D}}` to all x-edges on the 3D mesh; fields
    on y and z-edges are zero. Similar for the known 3D field solution for an incident planewave polarized
    along the y-direction.

    **3D Method:**

    This approach is encouraged when surface topography is significant. Here, the background conductivity
    :math:`\sigma_0` is defined on the 3D mesh for the :class:`.natural_source.Simulation3DFictitiousSource`
    simulation.

    We consider the solution for an incident planewave polarized along the x-direction.
    Let :math:`i` define the indeces of all of the edges NOT on the x or z-boundaries of the 3D mesh;
    i.e. internal edges. And let :math:`j` define the indeces of the x-edges on the top boundary of the 3D mesh.
    From the system matrix for the background conductivity :math:`\mathbf{A}(\sigma_0)`,
    we solve a reduced system:

    .. math::
        \mathbf{A_{i,i} \, u_i} = \mathbf{A_{i, j} v}

    where :math:`\mathbf{v}` is a vector of 1s. Once the reduced system is solved, the background
    solution :math:`\mathbf{u_0}` is constructed such that:

    .. math::
        \mathbf{u_0} = \begin{cases}
        u_i \; on \; i \; edges \\
        u_j \; on \; j \; edges \\
        0 \; otherwise
        \end{cases}

    """

    _fields_per_source = 2

    def s_e(self, simulation):
        """Electric source term

        Parameters
        ----------
        simulation : simpeg.electromagnetics.frequency_domain.simulation.BaseFDEMSimulation
            A NSEM simulation

        Returns
        -------
        numpy.ndarray
            Electric source term on mesh.
        """

        if getattr(self, "_s_e", None) is not None:
            return getattr(self, "_s_e")

        # Fictitious source from 1D
        if len(simulation.sigma_background) == len(simulation.mesh.h[2]):

            # Generate 1D mesh and conductivity averaged to nodes on
            mesh_3d = simulation.mesh
            hz = mesh_3d.h[2]
            sigma_1d = simulation.sigma_background

            n_pad = 2000  # arbitrary num of padding cells added
            hz = np.r_[hz[0] * np.ones(n_pad), hz, hz[-1] * np.ones(n_pad)]
            sigma_1d = np.r_[
                sigma_1d[0] * np.ones(n_pad), sigma_1d, sigma_1d[-1] * np.ones(n_pad)
            ]

            mesh_1d = discretize.TensorMesh(
                [hz], origin=[mesh_3d.origin[2] - hz[0] * n_pad]
            )

            sigma_1d = mesh_1d.average_face_to_cell.T * sigma_1d
            sigma_1d[0] = sigma_1d[1]
            sigma_1d[-1] = sigma_1d[-2]

            # Solve the 1D problem for electric fields on nodes
            w = 2*np.pi*self.frequency
            k = np.sqrt(-1.j * w * mu_0 * sigma_1d[0])

            A = mesh_1d.nodal_gradient.T @ mesh_1d.nodal_gradient + 1j*w*mu_0 * sdiag(sigma_1d)
            A[0, 0] = (1. + 1j*k*hz[0]) / hz[0]**2 + 1j*w*mu_0*sigma_1d[0]
            A[0, 1] = -1 / hz[0]**2

            q = np.zeros(mesh_1d.n_faces, dtype=np.complex128)
            q[-1] = -1j*w*mu_0 / hz[-1]

            Ainv = Solver(A)
            u_1 = Ainv * q

            # Project to X and Y edges
            fields_x = (
                mesh_1d.get_interpolation_matrix(
                    mesh_3d.edges_x[:, 2], location_type="nodes"
                )
                @ u_1d
            )
            fields_y = (
                mesh_1d.get_interpolation_matrix(
                    mesh_3d.edges_y[:, 2], location_type="nodes"
                )
                @ u_1d
            )

            fields_x = np.r_[fields_x, np.zeros(mesh_3d.n_edges_y + mesh_3d.n_edges_z)]
            fields_y = np.r_[
                np.zeros(mesh_3d.n_edges_x), fields_y, np.zeros(mesh_3d.n_edges_z)
            ]

            # Generate fictitious sources
            sigma_3d = (
                mesh_1d.get_interpolation_matrix(
                    mesh_3d.cell_centers[:, 2], location_type="cell_centers"
                )
                @ sigma_1d
            )

            C = mesh_3d.edge_curl
            MfMui = mesh_3d.get_face_inner_product(model=mu_0, invert_model=True)
            MeSigma = mesh_3d.get_edge_inner_product(model=sigma_3d)

            A = C.T.tocsr() * MfMui * C + 1j * omega(self.frequency) * MeSigma

            s_e = (A @ np.c_[fields_x, fields_y]) / (1j * omega(self.frequency))

        else:

            mesh_3d = simulation.mesh

            # Construct operator
            C = mesh_3d.edge_curl
            MfMui = mesh_3d.get_face_inner_product(model=mu_0, invert_model=True)
            MeSigma = mesh_3d.get_edge_inner_product(model=simulation.sigma_background)
            A = C.T.tocsr() * MfMui * C + 1j * omega(self.frequency) * MeSigma

            # x-polarization
            ind_exterior = (
                (mesh_3d.edges[:, 0] == min(mesh_3d.faces_x[:, 0]))
                | (mesh_3d.edges[:, 0] == max(mesh_3d.faces_x[:, 0]))
                | (mesh_3d.edges[:, 2] == min(mesh_3d.faces_z[:, 2]))
                | (mesh_3d.edges[:, 2] == max(mesh_3d.faces_z[:, 2]))
            )

            ind_top = (mesh_3d.edges[:, 2] == max(mesh_3d.faces_z[:, 2])) & (
                mesh_3d.edge_tangents[:, 0] == 1.0
            )

            A_interior = A.copy()[~ind_exterior, :]  # eliminate boundary edge rows
            b = (A_interior.copy()[:, ind_top]) @ np.ones(np.sum(ind_top))
            A_interior = A_interior[:, ~ind_exterior]

            A_inv = simulation.solver(A_interior, **simulation.solver_opts)
            u_interior = A_inv * -b
            A_inv.clean()

            u_x = np.zeros(mesh_3d.n_edges, dtype=complex)
            u_x[~ind_exterior] = u_interior
            u_x[ind_top] = 1.0 + 0.0j

            # y-polarization
            ind_exterior = (
                (mesh_3d.edges[:, 1] == min(mesh_3d.faces_y[:, 1]))
                | (mesh_3d.edges[:, 1] == max(mesh_3d.faces_y[:, 1]))
                | (mesh_3d.edges[:, 2] == min(mesh_3d.faces_z[:, 2]))
                | (mesh_3d.edges[:, 2] == max(mesh_3d.faces_z[:, 2]))
            )

            ind_top = (mesh_3d.edges[:, 2] == max(mesh_3d.faces_z[:, 2])) & (
                mesh_3d.edge_tangents[:, 1] == 1.0
            )

            A_interior = A.copy()[~ind_exterior, :]  # eliminate boundary edge rows
            b = (A_interior.copy()[:, ind_top]) @ np.ones(np.sum(ind_top))
            A_interior = A_interior[:, ~ind_exterior]

            A_inv = simulation.solver(A_interior, **simulation.solver_opts)
            u_interior = A_inv * -b
            A_inv.clean()

            u_y = np.zeros(mesh_3d.n_edges, dtype=complex)
            u_y[~ind_exterior] = u_interior
            u_y[ind_top] = 1.0 + 0.0j

            # Get fictitious sources
            s_e = (A @ np.c_[u_x, u_y]) / (1j * omega(self.frequency))

            print(
                "3D FICTITIOUS SOURCES COMPUTED FOR FREQUENCY {} Hz".format(
                    self.frequency
                )
            )

        # Set and return fictitious sources
        setattr(self, "_s_e", s_e)
        return getattr(self, "_s_e")
