import scipy.sparse as sp
import numpy as np
from ...survey import BaseTimeRx
from ...utils import Zero


class Pressure(BaseTimeRx):
    """Richards pressue receiver class"""

    def __call__(self, U, simulation):
        projected_grid = "CC"
        projected_time_grid = "N"
        P = self.getP(
            simulation.mesh, simulation.time_mesh, projected_grid, projected_time_grid
        )
        u = np.concatenate(U)
        return P * u

    def deriv(self, U, simulation, du_dm_v=None, v=None, adjoint=False):
        """Derivative with respect to the model

        Parameters
        ----------
        U : (n_time) list of (n_cells) numpy.ndarray
            Fields computed on the mesh. This is unused for this receiver.
        simulation : SimPEG.flow.richards.simulation.SimulationNDCellCentered
            A Richards flor simulation
        du_dm_v : numpy.ndarray
            Derivative with respect to the model times a vector
        v : numpy.ndarray
            A vector
        adjoint : bool, default = ``False``.
            If ``True``, return the adjoint

        Returns
        -------
        numpy.ndarray
            Derivative with respect to the model times a vector
        """
        projected_grid = "CC"
        projected_time_grid = "N"
        P = self.getP(
            simulation.mesh, simulation.time_mesh, projected_grid, projected_time_grid
        )
        if not adjoint:
            return P * du_dm_v  # + 0 for dRx_dm contribution
        if v is None:
            raise Exception("v must be provided if computing adjoint")
        return P.T * v, Zero()


class Saturation(BaseTimeRx):
    """Richards saturation receiver class"""

    def __call__(self, U, simulation):
        # The water retention curve model should have been updated in the prob
        P = self.getP(simulation.mesh, simulation.time_mesh)
        usat = np.concatenate([simulation.water_retention(ui) for ui in U])
        return P * usat

    def deriv(self, U, simulation, du_dm_v=None, v=None, adjoint=False):
        """Derivative with respect to the model

        Parameters
        ----------
        U : (n_time) list of (n_cells) numpy.ndarray
            Fields computed on the mesh. This is unused for this receiver.
        simulation : SimPEG.flow.richards.simulation.SimulationNDCellCentered
            A Richards flor simulation
        du_dm_v : numpy.ndarray
            Derivative with respect to the model times a vector
        v : numpy.ndarray
            A vector
        adjoint : bool, default = ``False``.
            If ``True``, return the adjoint

        Returns
        -------
        numpy.ndarray
            Derivative with respect to the model times a vector
        """
        P = self.getP(simulation.mesh, simulation.time_mesh)
        dT_du = sp.block_diag([simulation.water_retention.derivU(ui) for ui in U])

        if simulation.water_retention.needs_model:
            dT_dm = sp.vstack([simulation.water_retention.derivM(ui) for ui in U])
        else:
            dT_dm = Zero()

        if v is None and not adjoint:
            # this is called by the fullJ in the problem
            return P * (dT_du * du_dm_v) + P * dT_dm
        if not adjoint:
            return P * (dT_du * du_dm_v) + P * (dT_dm * v)

        # for the adjoint return both parts of the sum separately
        if v is None:
            raise Exception("v must be provided if computing adjoint")
        PTv = P.T * v
        return dT_du.T * PTv, dT_dm.T * PTv
