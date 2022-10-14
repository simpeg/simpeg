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
    """Base electromagnetic simulation class"""

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
    """Base class for a electromagnetic sources

    Parameters
    ----------
    location : (n_dim) numpy.ndarray
        Location of the source
    receiver_list : list of SimPEG.survey.BaseRx objects
        Sets the receivers associated with the source
    uid : uuid.UUID
        A universally unique identifier
    integrate : bool
        If ``True``, we integrate the source term
    """

    def __init__(self, receiver_list=None, location=None, integrate=False, **kwargs):

        super(BaseEMSrc, self).__init__(
            receiver_list=receiver_list, location=location, **kwargs
        )
        self.integrate = integrate

    # integrate = properties.Bool("integrate the source term?", default=False)

    @property
    def integrate(self):
        """Integrated source term

        Returns
        -------
        bool
            If ``True``, the source term is integrated
        """
        return self._integrate

    @integrate.setter
    def integrate(self, var):
        if not isinstance(var, bool):
            raise TypeError(f"integrate property is a bool. Got {type(var)}")
        self._integrate = var

    def eval(self, simulation):
        """Return magnetic and electric source terms

        Parameters
        ----------
        simulation : SimPEG.electromagnetics.base.BaseEMSimulation
            An instance of an electromagnetic simulation

        Returns
        -------
        tuple
            A tuple (s_m, s_e), where s_m is the discretized magnetic source term
            and s_e is the discretized electric course term.
        """
        s_m = self.s_m(simulation)
        s_e = self.s_e(simulation)
        return s_m, s_e

    def evalDeriv(self, simulation, v=None, adjoint=False):
        """Return derivative of the magnetic and electric source terms with respect to the model.

        Parameters
        ----------
        simulation : SimPEG.electromagnetics.base.BaseEMSimulation
            An instance of an electromagnetic simulation
        v : np.ndarray
            A vector
        adjoint : bool
            If ``True``, return the adjoint operation

        Returns
        -------
        tuple
            A tuple (s_mDeriv, s_eDerive). If `v` is not ``None``, the method returns
            the derivatives of the magnetic and electric sources times the vector `v`.
            If `v` is ``None``, the method returns the functions for multiplying the
            derivatives with a vector.
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
        """Magnetic source term

        Parameters
        ----------
        simulation : BaseEMSimulation
            An EM Simulation object

        Returns
        -------
        numpy.ndarray
            Magnetic source term defined on the mesh. Defined on faces for EB formulations,
            and defined on edges for HJ formulations.
        """
        return Zero()

    def s_e(self, simulation):
        """Electric source term

        Parameters
        ----------
        simulation : BaseEMSimulation
            An EM Simulation object

        Returns
        -------
        numpy.ndarray
            Electric source term defined on the mesh. Defined on edges for EB formulations,
            and defined on faces for HJ formulations.
        """
        return Zero()

    def s_mDeriv(self, simulation, v, adjoint=False):
        """
        Derivative of magnetic source term with respect to the inversion model

        Parameters
        ----------
        simulation : BaseEMSimulation
            An EM Simulation object
        v : numpy.ndarray
            A vector to take the dot product with
        adjoint : bool, default==Fasel
            If ``True``, return the adjoint operation

        Returns
        -------
        numpy.ndarray
            Product of the derivative of the magnetic source term and a vector
        """

        return Zero()

    def s_eDeriv(self, simulation, v, adjoint=False):
        """
        Derivative of electric source term with respect to the inversion model

        Parameters
        ----------
        simulation : BaseEMSimulation
            An EM Simulation object
        v : numpy.ndarray
            A vector to take the dot product with
        adjoint : bool, default==Fasel
            If ``True``, return the adjoint operation

        Returns
        -------
        numpy.ndarray
            Product of the derivative of the electric source term and a vector
        """
        return Zero()
