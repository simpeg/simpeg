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
    receiver_list : list of SimPEG.electromagnetics.frequency_domain.receivers.BaseRx
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
    receiver_list : list of SimPEG.electromagnetics.frequency_domain.receivers.BaseRx
        A list of NSEM receivers
    frequency : float
        Source frequency
    sigma_primary : float, default: ``None``
        Wholespace conductivity for primary field
    """

    _fields_per_source = 2

    def __init__(self, receiver_list, frequency, sigma_primary=None):
        # assert mkvc(self.mesh.hz.shape,1) == mkvc(sigma1d.shape,1),'The number of values in the 1D background model does not match the number of vertical cells (hz).'
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
                    [mesh3d.vectorNx[-1] - x0[0]],
                    [mesh3d.vectorNy[-1] - x0[1]],
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
                self._sigma1d = simulation.mesh.r(
                    simulation._sigmaPrimary, "CC", "CC", "M"
                )[:]
                self._sigma_p = None
                self.sigma1d = self._sigma1d
            return self._sigma1d, self._sigma_p

    def ePrimary(self, simulation):
        """Primary electric field

        Parameters
        ----------
        simulation : SimPEG.electromagnetics.natural_source_simulation.BaseNSEMSimulation
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
        simulation : SimPEG.electromagnetics.frequency_domain.simulation.BaseFDEMSimulation
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
        simulation : SimPEG.electromagnetics.frequency_domain.simulation.BaseFDEMSimulation
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
        simulation : SimPEG.electromagnetics.frequency_domain.simulation.BaseFDEMSimulation
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
        simulation : SimPEG.electromagnetics.frequency_domain.simulation.BaseFDEMSimulation
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
