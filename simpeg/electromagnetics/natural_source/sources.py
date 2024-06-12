import numpy as np

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


class FictitiousSource(BaseFDEMSrc)

    _fields_per_source = 2

    def __init__(self, receiver_list, frequency):

        self.sigma1d = None
        self._sigma_primary = sigma_primary
        super(PlanewaveXYPrimary, self).__init__(receiver_list, frequency)

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
        
        # Model-independent. Return if already computed.
        if getattr(self, '_s_e', None) is not None:
            return getattr(self, '_s_e')

        # Fictitious source from 1D 
        if len(simulation.sigma_background) == len(simulation.mesh.h[2]):

            # Generate 1D mesh and conductivity on extended 1D mesh
            hz = simulation.mesh.h[2]
            sigma_1d = simulation.background_conductivity

            n_pad = 200  # arbitrary number of padding cells added
            hz = np.r_[hz[0]*np.ones(n_pad), hz, hz[-1]*np.ones(n_pad)]
            sigma_1d = np.r_[sigma_1d[0]*np.ones(n_pad), sigma_1d, sigma_1d[-1]*np.ones(n_pad)]

            mesh_1d = discretize.TensorMesh([hz], origin=simulation.mesh.origin[2]-hz[0]*n_pad)

            # Solve the 1D problem for electric fields on nodes
            G = mesh_1d.nodal_gradient
            MeMui = mesh_1d.get_edge_inner_product(model=mu_0, invert_model=True)
            MfSigma = mesh_1d.get_face_inner_product(model=sigma_1d)

            A = G.T.tocsr() @ MeMui @ G + 1j * omega(self.frequency) * MfSigma

            RHS = 1j * omega(self.frequency) * mesh_1d.boundary_node_vector_integral * [0 + 0j, 1 + 0j]

            Ainv = simulation.solver(A, **simulation.solver_opts)
            u_1d = Ainv * RHS

            # Project to X and Y edges
            fields_x = mesh_1d.get_interpolation_matrix(mesh.edges_x[:, 2], location_type="nodes") @ u_1d
            fields_y = mesh_1d.get_interpolation_matrix(mesh.edges_y[:, 2], location_type="nodes") @ u_1d

            fields_x = np.r_[fields_x, np.zeros(mesh.n_edges_y + mesh.n_edges_z)]
            fields_y = np.r_[np.zeros(mesh.n_edges_x), fields_y, np.zeros(mesh.n_edges_z)]

            # Generate fictitious sources
            sigma_3d = mesh_1d.get_interpolation_matrix(mesh.cell_centers[:, 2], location_type="cell_centers") @ sigma_1d

            C = simulation.mesh.edge_curl
            MfMui = simulation.mesh.get_face_inner_product(model=mu_0, invert_model=True)
            MeSigma = simulation.mesh.get_edge_inner_product(model=sigma_3d)
            
            A = C.T.tocsr() * MfMui * C + 1j * omega(self.frequency) * MeSigma

            s_e = 1j * omega(self.frequency) * (A @ np.c_[fields_x, fields_y])

        # Fictitious source from 3D
        else:
            raise NotImplementedError("Fictitious source not implemented for 3D background model.")

        # Set and return fictitious sources
        setattr(self, '_s_e', s_e)
        return getattr(self, '_s_e')


