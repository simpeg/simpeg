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
    conductivity_primary : float, default: ``None``
        Wholespace conductivity for primary field
    """

    _fields_per_source = 2

    def __init__(self, receiver_list, frequency, conductivity_primary=None):
        # assert mkvc(self.mesh.h[2].shape,1) == mkvc(conductivity1d.shape,1),'The number of values in the 1D background model does not match the number of vertical cells (hz).'
        self.conductivity1d = None
        self._conductivity_primary = conductivity_primary
        super(PlanewaveXYPrimary, self).__init__(receiver_list, frequency)

    def _get_conductivities(self, simulation):
        try:
            return self._conductivity1d, self._conductivity_p
        except AttributeError:
            # set _conductivity 1D
            if self._conductivity_primary is None:
                self._conductivity_primary = simulation.conductivityPrimary
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
                if len(self._conductivity_primary) == mesh3d.nC:
                    # volume average down to 1D mesh
                    self._conductivity1d = np.exp(
                        volume_average(
                            mesh3d, mesh1d, np.log(self._conductivity_primary)
                        )
                    )
                elif len(self._conductivity_primary) == mesh1d.nC:
                    self._conductivity1d = self._conductivity_primary
                else:
                    self._conductivity1d = (
                        np.ones(mesh1d.nC) * self._conductivity_primary
                    )
                self._conductivity_p = np.exp(
                    volume_average(mesh1d, mesh3d, np.log(self._conductivity1d))
                )
            else:
                self._conductivity1d = simulation.mesh.reshape(
                    simulation._conductivityPrimary, "CC", "CC", "M"
                )[:]
                self._conductivity_p = None
                self.conductivity1d = self._conductivity1d
            return self._conductivity1d, self._conductivity_p

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
            conductivity_1d, _ = self._get_conductivities(simulation)
            self._ePrimary = homo1DModelSource(
                simulation.mesh, self.frequency, conductivity_1d
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
            Map_conductivity_p = maps.SurjectVertical1D(simulation.mesh)
            conductivity_p = Map_conductivity_p._transform(self.conductivity1d)
            Meconductivity = simulation.mesh.get_face_inner_product(
                simulation.conductivity
            )
            Meconductivity_p = simulation.mesh.get_face_inner_product(conductivity_p)
        if simulation.mesh.dim == 2:
            pass
        if simulation.mesh.dim == 3:
            _, conductivity_p = self._get_conductivities(simulation)
            Meconductivity = simulation._Me_conductivity
            Meconductivity_p = simulation.mesh.get_edge_inner_product(conductivity_p)
        return Meconductivity * e_p - Meconductivity_p * e_p

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
            return simulation._Mf_conductivity_deriv(ePri, v, adjoint=adjoint)
        if simulation.mesh.dim == 2:
            raise NotImplementedError("The NSEM 2D simulation is not implemented")
        if simulation.mesh.dim == 3:
            # Need to take the derivative of both u_px and u_py
            # And stack them to be of the correct size
            e_p = self.ePrimary(simulation)
            return simulation._Me_conductivity_deriv(e_p, v, adjoint=adjoint)

    S_e = s_e
    S_eDeriv = s_eDeriv
