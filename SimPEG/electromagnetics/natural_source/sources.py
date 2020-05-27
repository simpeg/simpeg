import numpy as np
import scipy.sparse as sp

from ... import maps
from ...utils import mkvc
from ..frequency_domain.sources import BaseFDEMSrc
from ..utils import omega
from .utils.source_utils import homo1DModelSource

#################
###   Sources ###
#################

# class BaseFDEMSrc(BaseFDEMSrc):
#     '''
#     Sources for the NSEM simulation.
#     Use the SimPEG BaseSrc, since the source fields share properties with the transmitters.

#     :param float frequency: The frequencyuency of the source
#     :param list receiver_list: A list of receivers associated with the source
#     '''

#     frequency = None #: Frequency (float)


#     def __init__(self, receiver_list, frequency):

#         self.frequency = float(frequency)
#         BaseFDEMSrc.__init__(self, receiver_list)

# 1D sources
class Planewave_xy_1DhomotD(BaseFDEMSrc):
    """
    NSEM source for both polarizations (x and y) for the total Domain.

    It calculates fields calculated based on conditions on the boundary of the domain.
    """

    def __init__(self, receiver_list, frequency):
        super(Planewave_xy_1DhomotD, self).__init__(receiver_list, frequency)


# Need to implement such that it works for all dims.
# Rename to be more descriptive
class Planewave_xy_1Dprimary(BaseFDEMSrc):
    """
    NSEM planewave source for both polarizations (x and y)
    estimated from a single 1D primary models.


    """

    def __init__(self, receiver_list, frequency):
        # assert mkvc(self.mesh.hz.shape,1) == mkvc(sigma1d.shape,1),'The number of values in the 1D background model does not match the number of vertical cells (hz).'
        self.sigma1d = None
        super(Planewave_xy_1Dprimary, self).__init__(receiver_list, frequency)

    def ePrimary(self, simulation):
        # Get primary fields for both polarizations
        if self.sigma1d is None:
            # Set the sigma1d as the 1st column in the background model
            if len(simulation._sigmaPrimary) == simulation.mesh.nC:
                if simulation.mesh.dim == 1:
                    self.sigma1d = simulation.mesh.r(
                        simulation._sigmaPrimary, "CC", "CC", "M"
                    )[:]
                elif simulation.mesh.dim == 3:
                    self.sigma1d = simulation.mesh.r(
                        simulation._sigmaPrimary, "CC", "CC", "M"
                    )[0, 0, :]
            # Or as the 1D model that matches the vertical cell number
            elif len(simulation._sigmaPrimary) == simulation.mesh.nCz:
                self.sigma1d = simulation._sigmaPrimary

        if self._ePrimary is None:
            self._ePrimary = homo1DModelSource(
                simulation.mesh, self.frequency, self.sigma1d
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

    def S_e(self, simulation):
        """
        Get the electrical field source
        """
        e_p = self.ePrimary(simulation)
        Map_sigma_p = maps.SurjectVertical1D(simulation.mesh)
        sigma_p = Map_sigma_p._transform(self.sigma1d)
        # Make mass matrix
        # Note: M(sig) - M(sig_p) = M(sig - sig_p)
        # Need to deal with the edge/face discrepencies between 1d/2d/3d
        if simulation.mesh.dim == 1:
            Mesigma = simulation.mesh.getFaceInnerProduct(simulation.sigma)
            Mesigma_p = simulation.mesh.getFaceInnerProduct(sigma_p)
        if simulation.mesh.dim == 2:
            pass
        if simulation.mesh.dim == 3:
            Mesigma = simulation.MeSigma
            Mesigma_p = simulation.mesh.getEdgeInnerProduct(sigma_p)
        return (Mesigma - Mesigma_p) * e_p

    def S_eDeriv(self, simulation, v, adjoint=False):
        """
        The derivative of S_e with respect to
        """

        return self.S_eDeriv_m(simulation, v, adjoint)

    def S_eDeriv_m(self, simulation, v, adjoint=False):
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
                return simulation.MeSigmaDeriv(
                    e_p[:, 0], v[: int(v.shape[0] / 2)], adjoint
                ) + simulation.MeSigmaDeriv(
                    e_p[:, 1], v[int(v.shape[0] / 2) :], adjoint
                )
                # return sp.hstack((
                #     simulation.MeSigmaDeriv(e_p[:, 0]).T,
                #     simulation.MeSigmaDeriv(e_p[:, 1]).T)) * v
            else:
                return np.hstack(
                    (
                        mkvc(simulation.MeSigmaDeriv(e_p[:, 0], v, adjoint), 2),
                        mkvc(simulation.MeSigmaDeriv(e_p[:, 1], v, adjoint), 2),
                    )
                )


class Planewave_xy_3Dprimary(BaseFDEMSrc):
    """
    NSEM source for both polarizations (x and y) given a 3D primary model.
    It assigns fields calculated from the 1D model
    as fields in the full space of the simulation.
    """

    def __init__(self, receiver_list, frequency):
        # assert mkvc(self.mesh.hz.shape,1) == mkvc(sigma1d.shape,1),'The number of values in the 1D background model does not match the number of vertical cells (hz).'
        self.sigmaPrimary = None
        super(Planewave_xy_3Dprimary, self).__init__(receiver_list, frequency)
        # Hidden property of the ePrimary
        self._ePrimary = None

    def ePrimary(self, simulation):
        # Get primary fields for both polarizations
        self.sigmaPrimary = simulation._sigmaPrimary

        if self._ePrimary is None:
            self._ePrimary = homo3DModelSource(
                simulation.mesh, self.sigmaPrimary, self.frequency
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

    def S_e(self, simulation):
        """
        Get the electrical field source
        """
        e_p = self.ePrimary(simulation)
        Map_sigma_p = maps.SurjectVertical1D(simulation.mesh)
        sigma_p = Map_sigma_p._transform(self.sigma1d)
        # Make mass matrix
        # Note: M(sig) - M(sig_p) = M(sig - sig_p)
        # Need to deal with the edge/face discrepencies between 1d/2d/3d
        if simulation.mesh.dim == 1:
            Mesigma = simulation.mesh.getFaceInnerProduct(simulation.sigma)
            Mesigma_p = simulation.mesh.getFaceInnerProduct(sigma_p)
        if simulation.mesh.dim == 2:
            pass
        if simulation.mesh.dim == 3:
            Mesigma = simulation.MeSigma
            Mesigma_p = simulation.mesh.getEdgeInnerProduct(sigma_p)
        return (Mesigma - Mesigma_p) * e_p

    def S_eDeriv_m(self, simulation, v, adjoint=False):
        """
        Get the derivative of S_e wrt to sigma (m)
        """
        # Need to deal with
        if simulation.mesh.dim == 1:
            # Need to use the faceInnerProduct
            MsigmaDeriv = (
                simulation.mesh.getFaceInnerProductDeriv(simulation.sigma)(
                    self.ePrimary(simulation)[:, 1]
                )
                * simulation.sigmaDeriv
            )
            # MsigmaDeriv = ( MsigmaDeriv * MsigmaDeriv.T)**2
        if simulation.mesh.dim == 2:
            pass
        if simulation.mesh.dim == 3:
            # Need to take the derivative of both u_px and u_py
            ePri = self.ePrimary(simulation)
            if adjoint:
                return simulation.MeSigmaDeriv(
                    ePri[:, 0], v[: int(v.shape[0] / 2)], adjoint
                ) + simulation.MeSigmaDeriv(
                    ePri[:, 1], v[int(v.shape[0] / 2) :], adjoint
                )
                # return sp.hstack((
                #     simulation.MeSigmaDeriv(ePri[:, 0]).T,
                #     simulation.MeSigmaDeriv(ePri[:, 1]).T)) * v
            else:
                return np.hstack(
                    (
                        mkvc(simulation.MeSigmaDeriv(ePri[:, 0], v, adjoint), 2),
                        mkvc(simulation.MeSigmaDeriv(ePri[:, 1], v, adjoint), 2),
                    )
                )
        if adjoint:
            #
            return MsigmaDeriv.T * v
        else:
            # v should be nC size
            return MsigmaDeriv * v
