from ..objective_function import ComboObjectiveFunction, BaseObjectiveFunction

import numpy as np
from dask.distributed import Client
from ..data_misfit import L2DataMisfit
from simpeg.meta.dask_sim import _validate_type_or_future_of_type, _reduce

from operator import add


def _calc_fields(objfct, model):
    return objfct.simulation.fields(m=objfct.simulation.model)


def _calc_dpred(objfct, model, field):
    return objfct.simulation.dpred(m=objfct.simulation.model, f=field)


def _calc_residual(objfct, model, field):
    return objfct.W * (
        objfct.data.dobs - objfct.simulation.dpred(m=objfct.simulation.model, f=field)
    )


def _deriv(objfct, multiplier, model, fields):
    if fields is not None and objfct.has_fields:
        return 2 * multiplier * objfct.deriv(objfct.simulation.model, f=fields)
    else:
        return 2 * multiplier * objfct.deriv(objfct.simulation.model)


def _deriv2(objfct, multiplier, model, v, fields):
    if fields is not None and objfct.has_fields:
        return 2 * multiplier * objfct.deriv2(objfct.simulation.model, v, f=fields)
    else:
        return 2 * multiplier * objfct.deriv2(objfct.simulation.model, v)


def _store_model(objfct, model):
    objfct.simulation.model = model


def _get_jtj_diag(objfct, model, field):
    jtj = objfct.simulation.getJtJdiag(objfct.simulation.model, objfct.W, f=field)
    return jtj.flatten()


class DaskComboMisfits(ComboObjectiveFunction):
    """
    A composite objective function for distributed computing.
    """

    def __init__(
        self,
        objfcts: list[BaseObjectiveFunction],
        multipliers=None,
        client: Client | None = None,
        **kwargs,
    ):
        self._model: np.ndarray | None = None
        self.client = client

        super().__init__(objfcts=objfcts, multipliers=multipliers, **kwargs)

    def __call__(self, m, f=None):
        self.model = m
        client = self.client
        m_future = self._m_as_future

        if f is None:
            f = self.fields(m)

        values = []
        for phi, field, worker in zip(self, f, self._workers):
            multiplier, objfct = phi
            if multiplier == 0.0:  # don't evaluate the fct
                continue

            values.append(
                client.submit(
                    _calc_objective, objfct, multiplier, m_future, field, workers=worker
                )
            )

        return _reduce(client, add, values)

    @property
    def client(self):
        """
        Get the dask.distributed.Client instance.
        """
        return self._client

    @client.setter
    def client(self, client):
        if not isinstance(client, Client):
            raise TypeError("client must be a dask.distributed.Client")

        self._client = client

    def deriv(self, m, f=None):
        """
        First derivative of the composite objective function is the sum of the
        derivatives of each objective function in the list, weighted by their
        respective multplier.

        :param numpy.ndarray m: model
        :param SimPEG.Fields f: Fields object (if applicable)
        """
        self.model = m
        client = self.client
        m_future = self._m_as_future

        if f is None:
            f = self.fields(m)

        derivs = []
        for multiplier, objfct, field, worker in zip(
            self.multipliers, self._futures, f, self._workers
        ):
            if multiplier == 0.0:  # don't evaluate the fct
                continue

            derivs.append(
                client.submit(
                    _deriv, objfct, multiplier, m_future, field, workers=worker
                )
            )

        return _reduce(client, add, derivs)

    def deriv2(self, m, v=None, f=None):
        """
        Second derivative of the composite objective function is the sum of the
        second derivatives of each objective function in the list, weighted by
        their respective multplier.

        :param numpy.ndarray m: model
        :param numpy.ndarray v: vector we are multiplying by
        :param SimPEG.Fields f: Fields object (if applicable)
        """
        self.model = m
        client = self.client
        m_future = self._m_as_future
        [v_future] = client.scatter([v], broadcast=True)

        if f is None:
            f = self.fields(m)

        derivs = []
        for multiplier, objfct, field, worker in zip(
            self.multipliers, self._futures, f, self._workers
        ):
            if multiplier == 0.0:  # don't evaluate the fct
                continue

            derivs.append(
                client.submit(
                    _deriv2,
                    objfct,
                    multiplier,
                    m_future,
                    v_future,
                    field,
                    workers=worker,
                )
            )

        return _reduce(client, add, derivs)

    def get_dpred(self, m, f=None):
        self.model = m

        if f is None:
            f = self.fields(m)

        client = self.client
        m_future = self._m_as_future
        dpred = []
        for objfct, worker, field in zip(self._futures, self._workers, f):
            dpred.append(
                client.submit(
                    _calc_dpred,
                    objfct,
                    m_future,
                    field,
                    workers=worker,
                )
            )
        return client.gather(dpred)

    def getJtJdiag(self, m, f=None):
        self.model = m
        m_future = self._m_as_future
        if getattr(self, "_jtjdiag", None) is None:

            jtj_diag = []
            client = self.client
            if f is None:
                f = self.fields(m)
            for objfct, worker, field in zip(self._futures, self._workers, f):
                jtj_diag.append(
                    client.submit(
                        _get_jtj_diag,
                        objfct,
                        m_future,
                        field,
                        workers=worker,
                    )
                )
            self._jtjdiag = _reduce(client, add, jtj_diag)

        return self._jtjdiag

    def fields(self, m):
        self.model = m
        client = self.client
        m_future = self._m_as_future
        if getattr(self, "_stashed_fields", None) is not None:
            return self._stashed_fields
        # The above should pass the model to all the internal simulations.
        f = []
        for objfct, worker in zip(self._futures, self._workers):
            f.append(
                client.submit(
                    _calc_fields,
                    objfct,
                    m_future,
                    workers=worker,
                )
            )
        self._stashed_fields = f
        return f

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        # Only send the model to the internal simulations if it was updated.
        if value is self.model:
            return

        self._stashed_fields = None
        self._jtjdiag = None

        client = self.client
        [self._m_as_future] = client.scatter([value], broadcast=True)

        futures = []
        for objfct, worker in zip(self._futures, self._workers):
            futures.append(
                client.submit(
                    _store_model,
                    objfct,
                    self._m_as_future,
                    workers=worker,
                )
            )
        self.client.gather(futures)  # blocking call to ensure all models were stored

    @property
    def objfcts(self):
        return self._objfcts

    @objfcts.setter
    def objfcts(self, objfcts):
        client = self.client

        futures, workers = _validate_type_or_future_of_type(
            "objfcts", objfcts, L2DataMisfit, client, return_workers=True
        )
        for objfct, future in zip(objfcts, futures):
            if hasattr(objfct, "name"):
                future.name = objfct.name

        self._objfcts = objfcts
        self._futures = futures
        self._workers = workers

    def residuals(self, m, f=None):
        """
        Compute the residual for the data misfit.
        """
        self.model = m
        if f is None:
            f = self.fields(m)
        client = self.client
        m_future = self._m_as_future
        residuals = []
        for objfct, worker, field in zip(self._futures, self._workers, f):
            residuals.append(
                client.submit(
                    _calc_residual,
                    objfct,
                    m_future,
                    field,
                    workers=worker,
                )
            )
        return client.gather(residuals)

    @property
    def workers(self):
        """
        Get the list of dask.distributed.workers associated with the objective functions.
        """
        return self._workers
