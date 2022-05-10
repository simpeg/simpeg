import properties

from ..survey import BaseSrc
from ..utils import Zero
from ..base import BaseElectricalPDESimulation, BaseMagneticPDESimulation

__all__ = ["BaseEMSimulation", "BaseEMSrc"]


###############################################################################
#                                                                             #
#                             Base EM Simulation                                 #
#                                                                             #
###############################################################################


class BaseEMSimulation(BaseElectricalPDESimulation, BaseMagneticPDESimulation):

    verbose = False
    storeInnerProduct = True

    ####################################################
    # Make A Symmetric
    ####################################################
    @property
    def _makeASymmetric(self):
        if getattr(self, "__makeASymmetric", None) is None:
            self.__makeASymmetric = True
        return self.__makeASymmetric


###############################################################################
#                                                                             #
#                             Base EM Source                                  #
#                                                                             #
###############################################################################


class BaseEMSrc(BaseSrc):
    """
    Base class for an EM sources
    """

    integrate = properties.Bool("integrate the source term?", default=False)

    def eval(self, simulation):
        """
        - :math:`s_m` : magnetic source term
        - :math:`s_e` : electric source term

        :param BaseFDEMSimulation simulation: FDEM Simulation
        :rtype: tuple
        :return: tuple with magnetic source term and electric source term
        """
        s_m = self.s_m(simulation)
        s_e = self.s_e(simulation)
        return s_m, s_e

    def evalDeriv(self, simulation, v=None, adjoint=False):
        """
        Derivatives of the source terms with respect to the inversion model
        - :code:`s_mDeriv` : derivative of the magnetic source term
        - :code:`s_eDeriv` : derivative of the electric source term

        :param BaseFDEMSimulation simulation: FDEM Simulation
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: tuple
        :return: tuple with magnetic source term and electric source term
            derivatives times a vector

        """
        if v is not None:
            return (
                self.s_mDeriv(simulation, v, adjoint),
                self.s_eDeriv(simulation, v, adjoint),
            )
        else:
            return (
                lambda v: self.s_mDeriv(simulation, v, adjoint),
                lambda v: self.s_eDeriv(simulation, v, adjoint),
            )

    def s_m(self, simulation):
        """
        Magnetic source term

        :param BaseFDEMSimulation simulation: FDEM Simulation
        :rtype: numpy.ndarray
        :return: magnetic source term on mesh
        """
        return Zero()

    def s_e(self, simulation):
        """
        Electric source term

        :param BaseFDEMSimulation simulation: FDEM Simulation
        :rtype: numpy.ndarray
        :return: electric source term on mesh
        """
        return Zero()

    def s_mDeriv(self, simulation, v, adjoint=False):
        """
        Derivative of magnetic source term with respect to the inversion model

        :param BaseFDEMSimulation simulation: FDEM Simulation
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of magnetic source term derivative with a vector
        """

        return Zero()

    def s_eDeriv(self, simulation, v, adjoint=False):
        """
        Derivative of electric source term with respect to the inversion model

        :param BaseFDEMSimulation simulation: FDEM Simulation
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of electric source term derivative with a vector
        """
        return Zero()
