import numpy as np
# import properties

from ...survey import BaseSurvey, BaseRx


class Survey(BaseSurvey):
    """Richards flow surve class"""

    # receiver_list = properties.List(
    #     "list of receivers for flow simulations",
    #     properties.Instance("A receiver instance", BaseRx),
    # )

    def __init__(self, receiver_list, **kwargs):
        self.receiver_list = receiver_list
        BaseSurvey.__init__(self, **kwargs)

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

        if isinstance(new_list, BaseRx):
            new_list = [new_list]
        elif isinstance(new_list, list):
            pass
        else:
            raise TypeError("Receiver list must be a list of SimPEG.survey.BaseRx")

        assert len(set(new_list)) == len(new_list), "The receiver_list must be unique. Cannot re-use receivers"

        self._rxOrder = dict()
        [self._rxOrder.setdefault(rx._uid, ii) for ii, rx in enumerate(new_list)]
        self._receiver_list = new_list

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
        f :
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
        f :
            Fields
        v : numpy.ndarray, default = ``Nones``
            A vector

        Returns
        -------
        numpy.ndarray
            Adjoint derivative with respect to model times a vector
        """
        dd_du = list(range(len(self.receiver_list)))
        dd_dm = list(range(len(self.receiver_list)))
        cnt = 0
        for ii, rx in enumerate(self.receiver_list):
            dd_du[ii], dd_dm[ii] = rx.deriv(
                f, simulation, v=v[cnt : cnt + rx.nD], adjoint=True
            )
            cnt += rx.nD
        return np.sum(dd_du, axis=0), np.sum(dd_dm, axis=0)
