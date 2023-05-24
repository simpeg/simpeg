import numpy as np


from ...survey import BaseSurvey, BaseRx
from ...utils import validate_list_of_types
from discretize.utils import Zero


class Survey(BaseSurvey):
    """Richards flow surve class"""

    def __init__(self, receiver_list, **kwargs):
        self.receiver_list = receiver_list
        super().__init__(source_list=None, **kwargs)

    @property
    def receiver_list(self):
        """List of receivers associated with the survey

        Returns
        -------
        list of SimPEG.survey.BaseRx
            List of receivers associated with the survey
        """
        return self._receiver_list

    @receiver_list.setter
    def receiver_list(self, new_list):
        self._receiver_list = validate_list_of_types("receiver_list", new_list, BaseRx)

    @property
    def nD(self):
        """Number of data

        Returns
        -------
        int
            Number of data
        """
        return np.array([rx.nD for rx in self.receiver_list]).sum()

    def deriv(self, simulation, f, du_dm_v=None, v=None):
        """The derivative with respect to the model.

        Parameters
        ----------
        simulation : SimPEG.flow.richards.simulation.SimulationNDCellCentered
            A Richards flow simulation class
        f : (n_times) list of numpy.ndarray
            Fields
        du_dm_v : numpy.ndarray, default = ``None``
            Derivative with respect to model times a vector
        v : numpy.ndarray, default = ``Nones``
            A vector

        Returns
        -------
        numpy.ndarray
            Derivative with respect to model times a vector
        """
        dd_dm = [
            rx.deriv(f, simulation, du_dm_v=du_dm_v, v=v, adjoint=False)
            for rx in self.receiver_list
        ]
        return np.concatenate(dd_dm)

    def derivAdjoint(self, simulation, f, v=None):
        """The adjoint derivative with respect to the model.

        Parameters
        ----------
        simulation : SimPEG.flow.richards.simulation.SimulationNDCellCentered
            A Richards flow simulation class
        f : (n_times) list of numpy.ndarray
            Fields.
        v : numpy.ndarray, default = ``Nones``
            A vector

        Returns
        -------
        numpy.ndarray
            Adjoint derivative with respect to model times a vector
        """
        dd_du = []
        dd_dm = []
        cnt = 0
        for ii, rx in enumerate(self.receiver_list):
            du, dm = rx.deriv(f, simulation, v=v[cnt : cnt + rx.nD], adjoint=True)
            if not isinstance(du, Zero):
                dd_du.append(du)
            if not isinstance(dm, Zero):
                dd_dm.append(dm)
            cnt += rx.nD
        return np.sum(dd_du, axis=0), np.sum(dd_dm, axis=0)
