import numpy as np
from scipy.constants import mu_0

from ... import maps
from ..frequency_domain.sources import BaseFDEMSrc
from ..utils import omega
from .utils.source_utils import homo1DModelSource
import discretize
from discretize.utils import volume_average


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

    This class applies boundary conditions for the natural source EM problem by using the method of
    fictitious source to generate a right-hand side. The ``FictitiousSource3D`` class is used in
    conjunction with the ``Simulation3DFictitiousSource`` simulation class.
    
    For a background conductivity distribution :math:`\sigma_0`, let :math:`\mathbf{u_0}` represent the
    known solution of the fields discretized to the mesh. Where :math:`\mathbf{A}(\sigma_0)` is the
    system matrix constructed from the background conductivity, the fictitious source :math:`\mathbf{s_e}`
    is obtained by computing:

    .. math::
        \mathbf{s_e} = \frac{1}{i \omega} \mathbf{A}(\sigma_0) \, \mathbf{u_0}

    where :math:`\omega` is the angular frequency. Once the source term is obtained, the unknown fields
    :math:`\mathbf{u}` for a conductivity distribution :math:`\sigma` can be computed by solving the
    system:

    .. math::
        mathbf{A}(\sigma) \, \mathbf{u} = i \omega \mathbf{s_e}

    Parameters
    ----------
    receiver_list : list of .natural_source.receivers.ApparentConductivity
        List of NSEM receivers.
    frequency : float
        Source frequency in Hz.

    Notes
    -----

    Describe method.

    """

    _fields_per_source = 2

    # def __init__(self, receiver_list, frequency):

    #     super(PlanewaveXYPrimary, self).__init__(receiver_list, frequency)

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
        
        if getattr(self, '_s_e', None) is not None:
            return getattr(self, '_s_e')

        # Fictitious source from 1D 
        if len(simulation.sigma_background) == len(simulation.mesh.h[2]):

            # Generate 1D mesh and conductivity on extended 1D mesh
            mesh_3d = simulation.mesh
            hz = mesh_3d.h[2]
            sigma_1d = simulation.sigma_background

            n_pad = 200  # arbitrary num of padding cells added
            hz = np.r_[hz[0]*np.ones(n_pad), hz, hz[-1]*np.ones(n_pad)]
            sigma_1d = np.r_[sigma_1d[0]*np.ones(n_pad), sigma_1d, sigma_1d[-1]*np.ones(n_pad)]

            mesh_1d = discretize.TensorMesh([hz], origin=[mesh_3d.origin[2]-hz[0]*n_pad])

            # Solve the 1D problem for electric fields on nodes
            G = mesh_1d.nodal_gradient
            MeMui = mesh_1d.get_edge_inner_product(model=mu_0, invert_model=True)
            MfSigma = mesh_1d.get_face_inner_product(model=sigma_1d)

            A = G.T.tocsr() @ MeMui @ G + 1j * omega(self.frequency) * MfSigma

            RHS = 1j * omega(self.frequency) * mesh_1d.boundary_node_vector_integral * [0 + 0j, 1 + 0j]

            Ainv = simulation.solver(A, **simulation.solver_opts)
            u_1d = Ainv * RHS

            # Project to X and Y edges
            fields_x = mesh_1d.get_interpolation_matrix(mesh_3d.edges_x[:, 2], location_type="nodes") @ u_1d
            fields_y = mesh_1d.get_interpolation_matrix(mesh_3d.edges_y[:, 2], location_type="nodes") @ u_1d

            fields_x = np.r_[fields_x, np.zeros(mesh_3d.n_edges_y + mesh_3d.n_edges_z)]
            fields_y = np.r_[np.zeros(mesh_3d.n_edges_x), fields_y, np.zeros(mesh_3d.n_edges_z)]

            # Generate fictitious sources
            sigma_3d = mesh_1d.get_interpolation_matrix(mesh_3d.cell_centers[:, 2], location_type="cell_centers") @ sigma_1d

            C = mesh_3d.edge_curl
            MfMui = mesh_3d.get_face_inner_product(model=mu_0, invert_model=True)
            MeSigma = mesh_3d.get_edge_inner_product(model=sigma_3d)
            
            A = C.T.tocsr() * MfMui * C + 1j * omega(self.frequency) * MeSigma

            s_e = 1j * omega(self.frequency) * (A @ np.c_[fields_x, fields_y])

        else:
            
            mesh_3d = simulation.mesh
            
            # UTILITY FCNS IN DISCRETIZE COULD BE ADDED TO MAKE THIS CLEANER.
            # Indices of all boundary edges, top boundary x-edges and top boundary y-edges.
            ind_all = (
                (mesh_3d.edges[:, 0] == min(mesh_3d.faces_x[:, 0])) |
                (mesh_3d.edges[:, 0] == max(mesh_3d.faces_x[:, 0])) |
                (mesh_3d.edges[:, 1] == min(mesh_3d.faces_y[:, 1])) |
                (mesh_3d.edges[:, 1] == max(mesh_3d.faces_y[:, 1])) |
                (mesh_3d.edges[:, 2] == min(mesh_3d.faces_z[:, 2])) |
                (mesh_3d.edges[:, 2] == max(mesh_3d.faces_z[:, 2]))
            )
            
            ind_top_x = (mesh_3d.edges[:, 2] == max(mesh_3d.faces_z[:, 2])) & (mesh_3d.edge_tangents[:, 0] == 1.)
            ind_top_y = (mesh_3d.edges[:, 2] == max(mesh_3d.faces_z[:, 2])) & (mesh_3d.edge_tangents[:, 1] == 1.)
            
            # Construct operator
            C = mesh_3d.edge_curl
            MfMui = mesh_3d.get_face_inner_product(model=mu_0, invert_model=True)
            MeSigma = mesh_3d.get_edge_inner_product(model=simulation.sigma_background)
            A = C.T.tocsr() * MfMui * C + 1j * omega(self.frequency) * MeSigma
            
            # ELIMINATING ROWS AND COLUMNS LIKE THIS IS NOT EFFICIENT.
            # Construct and solve system for x and y-polarization
            Ainterior = A.copy()[~ind_all, :]  # eliminate boundary edge rows
            bx = (Ainterior.copy()[:, ind_top_x]) @ -np.ones(np.sum(ind_top_x))
            by = (Ainterior.copy()[:, ind_top_y]) @ -np.ones(np.sum(ind_top_y))
            Ainterior = Ainterior[:, ~ind_all]
            
            Ainv = simulation.solver(Ainterior, **simulation.solver_opts)
            u_interior = Ainv * np.c_[bx, by]
            
            # Compute fictitious source
            u_full = np.ones((mesh_3d.n_edges, 2), dtype=complex)
            u_full[~ind_all, :] = u_interior
            u_full[ind_top_x, 0] = 1. + 0.j
            u_full[ind_top_y, 1] = 1. + 0.j
            
            s_e = 1j * omega(self.frequency) * (A @ u_full)
            
            print('3D FICTITIOUS SOURCES COMPUTED')

        # Set and return fictitious sources
        setattr(self, '_s_e', s_e)
        return getattr(self, '_s_e')


