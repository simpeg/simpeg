from ..objective_function import ComboObjectiveFunction, BaseObjectiveFunction

import numpy as np
from dask.distributed import Client
from ..data_misfit import L2DataMisfit

from simpeg.utils import validate_list_of_types
from simpeg.objective_function import (
    _validate_multiplier,
    _check_length_objective_funcs_multipliers,
)


def _calc_fields(objfcts, _):
    blocks = []
    for objfct in objfcts:
        blocks.append(objfct.simulation.fields(m=objfct.simulation.model))

    return blocks


def _calc_dpred(objfcts, _):
    blocks = []
    for objfct in objfcts:
        blocks.append(objfct.simulation.dpred(m=objfct.simulation.model))

    return np.hstack(blocks)


def _calc_objective(objfcts, multipliers, model):
    blocks = []
    for multiplier, objfct in zip(multipliers, objfcts):
        if multiplier == 0.0:
            continue

        blocks.append(multiplier * objfct(model))

    return np.sum(blocks)


def _calc_residual(objfcts, _):
    blocks = []
    for objfct in objfcts:
        blocks.append(
            objfct.W
            * (objfct.data.dobs - objfct.simulation.dpred(m=objfct.simulation.model))
        )

    return np.hstack(blocks)


def _deriv(objfcts, multipliers, _):
    blocks = []
    for multiplier, objfct in zip(multipliers, objfcts):
        if multiplier == 0.0:
            continue

        blocks.append(multiplier * objfct.deriv(objfct.simulation.model))

    return np.sum(blocks, axis=0)


def _deriv2(objfcts, multipliers, _, v):
    blocks = []
    for multiplier, objfct in zip(multipliers, objfcts):

        if multiplier == 0.0:
            continue

        blocks.append(multiplier * objfct.deriv2(objfct.simulation.model, v))

    return np.sum(blocks, axis=0)


def _store_model(objfcts, model):
    for objfct in objfcts:
        objfct.simulation.model = model


def _setter_broadcast(objfct, key, value):
    """
    Broadcast a value to all workers.
    """
    if hasattr(objfct, key):
        setattr(objfct, key, value)

    for sim in objfct.simulation.simulations:
        if hasattr(sim, key):
            setattr(sim, key, value)


def _get_jtj_diag(objfcts, _):
    arrays = []
    for objfct in objfcts:
        arrays.append(objfct.simulation.getJtJdiag(objfct.simulation.model, objfct.W))
    return np.sum(arrays, axis=0)


def _validate_type_or_future_of_type(
    property_name,
    objects,
    obj_type,
    client,
    workers: list[str] | None = None,
    return_lookup=False,
):

    if workers is None:
        workers = [
            (worker.worker_address,) for worker in client.cluster.workers.values()
        ]

    objects = validate_list_of_types(
        property_name, objects, obj_type, ensure_unique=True
    )
    funs_split = np.array_split(objects, len(workers))
    workload = {worker: [] for worker in workers}
    lookup = {}
    for objfcts, worker in zip(funs_split, workers):
        for obj in objfcts:
            obj.simulation.simulations[0].worker = worker
            future = client.scatter([obj], workers=worker)[0]

            if hasattr(obj, "name"):
                future.name = obj.name

            workload[worker].append(future)
            lookup[obj] = (future, worker)

    futures = []
    for worker, future_list in workload.items():

        for obj in future_list:
            futures.append(
                client.submit(
                    lambda v: not isinstance(v, obj_type), obj, workers=worker
                )
            )
    is_not_obj = np.array(client.gather(futures))
    if np.any(is_not_obj):
        raise TypeError(f"{property_name} futures must be an instance of {obj_type}")

    if return_lookup:
        return workload, lookup
    else:
        return workload


class DaskComboMisfits(ComboObjectiveFunction):
    """
    A composite objective function for distributed computing.
    """

    def __init__(
        self,
        objfcts: list[BaseObjectiveFunction],
        multipliers=None,
        client: Client | None = None,
        workers: list[str] | None = None,
        **kwargs,
    ):
        self._model: np.ndarray | None = None
        self.client = client
        self.workers = workers

        super().__init__(objfcts=objfcts, multipliers=multipliers, **kwargs)

    def __call__(self, m, f=None):
        self.model = m
        client = self.client
        m_future = self._m_as_future

        future_values = []
        for worker, futures in self._futures.items():
            future_values.append(
                client.submit(
                    _calc_objective,
                    futures,
                    self.multipliers[worker],
                    m_future,
                    workers=worker,
                )
            )
        values = self.client.gather(future_values)
        return np.sum(values)

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

    @property
    def workers(self):
        """
        List of worker addresses
        """
        return self._workers

    @workers.setter
    def workers(self, workers):
        if not isinstance(workers, list | type(None)):
            raise TypeError("workers must be a list of strings")

        self._workers = workers

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
        future_derivs = []
        for worker, futures in self._futures.items():
            future_derivs.append(
                client.submit(
                    _deriv,
                    futures,
                    self.multipliers[worker],
                    m_future,
                    workers=worker,
                )
            )

        derivs = self.client.gather(future_derivs)

        return np.sum(derivs, axis=0)

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
        future_derivs = []
        for worker, futures in self._futures.items():
            future_derivs.append(
                client.submit(
                    _deriv2,
                    futures,
                    self.multipliers[worker],
                    m_future,
                    v_future,
                    # field,
                    workers=worker,
                )
            )

        derivs = self.client.gather(future_derivs)
        return np.sum(derivs, axis=0)

    def get_dpred(self, m, f=None):
        """
        Request calculation of predicted data from all simulations.
        """
        self.model = m

        client = self.client
        m_future = self._m_as_future
        future_preds = []

        for worker, futures in self._futures.items():
            future_preds.append(
                client.submit(
                    _calc_dpred,
                    futures,
                    m_future,
                    workers=worker,
                )
            )
        dpreds = client.gather(future_preds)

        return np.hstack(dpreds)

    def getJtJdiag(self, m, f=None):
        """
        Request calculation of the diagonal of JtJ from all simulations.
        """
        self.model = m
        m_future = self._m_as_future
        if getattr(self, "_jtjdiag", None) is None:

            client = self.client
            work = []
            for worker, futures in self._futures.items():
                work.append(
                    client.submit(
                        _get_jtj_diag,
                        futures,
                        m_future,
                        workers=worker,
                    )
                )

            jtj_diag = client.gather(work)
            self._jtjdiag = np.sum(jtj_diag, axis=0)

        return self._jtjdiag

    def fields(self, m):
        """
        Request calculation of fields from all simulations.

        Store list of futures for fields in self._stashed_fields.
        """
        self.model = m
        client = self.client
        m_future = self._m_as_future
        if getattr(self, "_stashed_fields", None) is not None:
            return self._stashed_fields
        # The above should pass the model to all the internal simulations.
        f = []

        for worker, futures in self._futures.items():
            f.append(
                client.submit(
                    _calc_fields,
                    futures,
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
        if (
            isinstance(value, np.ndarray)
            and isinstance(self.model, np.ndarray)
            and np.allclose(value, self.model)
        ):
            return

        self._stashed_fields = None
        self._jtjdiag = None

        client = self.client
        [self._m_as_future] = client.scatter([value], broadcast=True)

        stores = []
        for worker, futures in self._futures.items():
            stores.append(
                client.submit(
                    _store_model,
                    futures,
                    self._m_as_future,
                    workers=worker,
                )
            )
        self.client.gather(stores)  # blocking call to ensure all models were stored
        self._model = value

    @property
    def objfcts(self):
        """
        List of objective functions associated with the data misfit.
        """
        return self._objfcts

    @objfcts.setter
    def objfcts(self, objfcts):
        client = self.client

        workload, lookup = _validate_type_or_future_of_type(
            "objfcts",
            objfcts,
            L2DataMisfit,
            client,
            workers=self.workers,
            return_lookup=True,
        )

        self._objfcts = objfcts
        self._futures = workload
        self._workers = list(workload)

        self._lookup = {
            misfit.simulation: (future, worker)
            for misfit, (future, worker) in lookup.items()
        }

    def residuals(self, m, f=None):
        """
        Compute the residual for the data misfit.
        """
        self.model = m

        client = self.client
        m_future = self._m_as_future

        future_residuals = []
        for worker, futures in self._futures.items():
            future_residuals.append(
                client.submit(
                    _calc_residual,
                    futures,
                    m_future,
                    workers=worker,
                )
            )
        residuals = client.gather(future_residuals)

        return np.hstack(residuals)

    def broadcast_updates(self, updates: dict):
        """
        Set the attributes of the objective functions and simulations
        """
        stores = []
        client = self.client
        for fun, (key, value) in updates.items():
            if fun not in self._lookup:
                continue

            future, worker = self._lookup[fun]

            stores.append(
                client.submit(
                    _setter_broadcast,
                    future,
                    key,
                    value,
                    workers=worker,
                )
            )
        self.client.gather(stores)  # blocking call to ensure all models were stored

    @property
    def multipliers(self):
        r"""Multipliers for the objective functions.

        For a composite objective function :math:`\phi`, that is, a weighted sum of
        objective functions :math:`\phi_i` with multipliers :math:`c_i` such that

        .. math::
            \phi = \sum_{i = 1}^N c_i \phi_i,

        this method returns the multipliers :math:`c_i` in
        the same order of the ``objfcts``.

        Returns
        -------
        list of int
            Multipliers for the objective functions.
        """

        return {
            worker: multipliers
            for worker, multipliers in zip(
                self._workers, np.array_split(self._multipliers, len(self._workers))
            )
        }

    @multipliers.setter
    def multipliers(self, value):
        """Set multipliers attribute after checking if they are valid."""
        for multiplier in value:
            _validate_multiplier(multiplier)
        _check_length_objective_funcs_multipliers(self.objfcts, value)
        self._multipliers = value
