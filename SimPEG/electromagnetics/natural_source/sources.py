import numpy as np

from ... import maps
from ...utils.code_utils import deprecate_class
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

    :param list receiver_list: List of SimPEG.electromagnetics.natural_sources.receivers.PointNaturalSource
    :param float frequency: frequency for the source
    """
    # This class is only provided to have a uniquely identifiable "Planewave" class
    pass


# Need to implement such that it works for all dims.
# Rename to be more descriptive
class PlanewaveXYPrimary(Planewave):
    """
    NSEM planewave source for both polarizations (x and y)
    estimated from a single 1D primary models.


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
        if self._ePrimary is None:
            sigma_1d, _ = self._get_sigmas(simulation)
            self._ePrimary = homo1DModelSource(
                simulation.mesh, self.frequency, sigma_1d
            )
        return self._ePrimary

    def bPrimary(self, simulation):
        # Project ePrimary to bPrimary
        # Satisfies the primary(background) field conditions
        if simulation.mesh.dim == 1:
            C = simulation.mesh.nodalGrad
        elif simulation.mesh.dim == 3:
            C = simulation.mesh.edgeCurl
        bBG_bp = (-C * self.ePrimary(simulation)) * (1 / (1j * omega(self.frequency)))
        return bBG_bp

    def s_e(self, simulation):
        """
        Get the electrical field source
        """
        e_p = self.ePrimary(simulation)
        # Make mass matrix
        # Note: M(sig) - M(sig_p) = M(sig - sig_p)
        # Need to deal with the edge/face discrepencies between 1d/2d/3d
        if simulation.mesh.dim == 1:
            Map_sigma_p = maps.SurjectVertical1D(simulation.mesh)
            sigma_p = Map_sigma_p._transform(self.sigma1d)
            Mesigma = simulation.mesh.getFaceInnerProduct(simulation.sigma)
            Mesigma_p = simulation.mesh.getFaceInnerProduct(sigma_p)
        if simulation.mesh.dim == 2:
            pass
        if simulation.mesh.dim == 3:
            _, sigma_p = self._get_sigmas(simulation)
            Mesigma = simulation.MeSigma
            Mesigma_p = simulation.mesh.getEdgeInnerProduct(sigma_p)
        return Mesigma * e_p - Mesigma_p * e_p

    def s_eDeriv(self, simulation, v, adjoint=False):
        """
        The derivative of S_e with respect to
        """

        return self.s_eDeriv_m(simulation, v, adjoint)

    def s_eDeriv_m(self, simulation, v, adjoint=False):
        """
        Get the derivative of S_e wrt to sigma (m)
        """
        # Need to deal with
        if simulation.mesh.dim == 1:
            # Need to use the faceInnerProduct
            ePri = self.ePrimary(simulation)[:, 1]
            MsigmaDeriv = (
                simulation.mesh.getFaceInnerProductDeriv(simulation.sigma)(ePri)
                * simulation.sigmaDeriv
            )
            # MsigmaDeriv = ( MsigmaDeriv * MsigmaDeriv.T)**2

            if adjoint:
                #
                return MsigmaDeriv.T * v
            else:
                # v should be nC size
                return MsigmaDeriv * v
        if simulation.mesh.dim == 2:
            raise NotImplementedError("The NSEM 2D simulation is not implemented")
        if simulation.mesh.dim == 3:
            # Need to take the derivative of both u_px and u_py
            # And stack them to be of the correct size
            e_p = self.ePrimary(simulation)
            if adjoint:
                return simulation.MeSigmaDeriv(e_p, v, adjoint=adjoint)
                # return simulation.MeSigmaDeriv(
                #     e_p[:, 0], v[: int(v.shape[0] / 2)], adjoint
                # ) + simulation.MeSigmaDeriv(
                #     e_p[:, 1], v[int(v.shape[0] / 2) :], adjoint
                # )
            return simulation.MeSigmaDeriv(e_p, v, adjoint)
            # return np.hstack(
            #     (
            #         mkvc(simulation.MeSigmaDeriv(e_p[:, 0], v, adjoint), 2),
            #         mkvc(simulation.MeSigmaDeriv(e_p[:, 1], v, adjoint), 2),
            #     )
            # )

    S_e = s_e
    S_eDeriv = s_eDeriv


############
# Deprecated
############
@deprecate_class(removal_version="0.15.0")
class Planewave_xy_1Dprimary(PlanewaveXYPrimary):
    pass


@deprecate_class(removal_version="0.15.0")
class Planewave_xy_1DhomotD(PlanewaveXYPrimary):
    pass
