from ..objective_function import (
    ComboObjectiveFunction,
    BaseObjectiveFunction,
    _validate_multiplier,
    _check_length_objective_funcs_multipliers,
)
from typing import Callable
import numpy as np

from dask.distributed import Client, Future, get_client
from ..data_misfit import L2DataMisfit

from simpeg.utils import validate_list_of_types


def _calc_fields(objfct, _):
    if isinstance(objfct, ComboObjectiveFunction):
        fields = []
        for objfct_ in objfct.objfcts:
            fields.append(_calc_fields(objfct_, _))

        return fields

    return objfct.simulation.fields(m=objfct.simulation.model)


def _calc_dpred(objfct, _, return_residuals=False):
    if isinstance(objfct, ComboObjectiveFunction):
        dpreds = []
        residuals = []
        for objfct_ in objfct.objfcts:

            if return_residuals:
                dpred_, residual_ = _calc_dpred(objfct_, _, return_residuals)
                dpreds.append(dpred_)
                residuals.append(residual_)
            else:
                dpreds.append(_calc_dpred(objfct_, _))

        if return_residuals:
            return np.hstack(dpreds), np.hstack(residuals)
        return np.hstack(dpreds)

    dpred = objfct.simulation.dpred(m=objfct.simulation.model)
    if return_residuals:
        residual = objfct.W * (objfct.data.dobs - dpred)
        return dpred, residual

    return dpred


def _calc_objective(objfct, multiplier, model):
    return multiplier * objfct(model)


def _calc_residual(objfct, _):
    if isinstance(objfct, ComboObjectiveFunction):
        residuals = 0.0
        for objfct_ in objfct.objfcts:
            residuals += _calc_residual(objfct_, _)

        return np.hstack(residuals)

    return objfct.W * (
        objfct.data.dobs - objfct.simulation.dpred(m=objfct.simulation.model)
    )


def _deriv(objfct, multiplier, _):
    if isinstance(objfct, ComboObjectiveFunction):
        deriv = 0.0
        for multiplier_, objfct_ in objfct:
            deriv += _deriv(objfct_, multiplier_, _)
    else:
        deriv = objfct.deriv(objfct.simulation.model)
    return multiplier * deriv


def _deriv2(objfct, multiplier, _, v):

    if isinstance(objfct, ComboObjectiveFunction):
        deriv2 = 0.0
        for multiplier_, objfct_ in objfct:
            deriv2 += _deriv2(objfct_, multiplier_, _, v)
    else:
        deriv2 = objfct.deriv2(objfct.simulation.model, v)
    return multiplier * deriv2


def _store_model(objfct, model):

    if isinstance(objfct, ComboObjectiveFunction):
        for objfct_ in objfct.objfcts:
            _store_model(objfct_, model)
    else:
        objfct.simulation.model = model


def _setter_broadcast(objfct, key, value):
    """
    Broadcast a value to all workers.
    """
    if isinstance(value, Callable):
        value = value(objfct)

    if hasattr(objfct, key):
        setattr(objfct, key, value)

    for sim in objfct.simulation.simulations:
        setattr(sim, key, value)


def _get_jtj_diag(objfct, _):
    if isinstance(objfct, ComboObjectiveFunction):
        jtj = 0.0
        for objfct_ in objfct.objfcts:
            jtj += _get_jtj_diag(objfct_, _)

        return jtj

    jtj = objfct.simulation.getJtJdiag(objfct.simulation.model, objfct.W)
    return jtj.flatten()


def _set_worker(objfct, worker):
    """
    Set the worker for the objective function.
    """
    if isinstance(objfct, ComboObjectiveFunction):
        for objfct_ in objfct.objfcts:
            _set_worker(objfct_, worker)

    else:
        for sim in objfct.simulation.simulations:
            sim.worker = worker


def _validate_type_or_future_of_type(
    property_name,
    objects,
    obj_type,
    client,
    workers: list[str] | None = None,
    return_workers=False,
):

    if workers is None:
        workers = [(worker,) for worker in client.nthreads()]

    objects = validate_list_of_types(
        property_name, objects, obj_type, ensure_unique=True
    )
    workloads = {}
    for worker in workers:
        workloads[worker] = []

    count = 0
    for ii, obj in enumerate(objects):
        count = ii % len(workers)

        if isinstance(obj, Future):
            future = obj
            count = workers.index(client.who_has(obj)[obj.key])
        else:
            future = client.scatter([obj], workers=workers[count])[0]

        workloads[workers[count]].append(future)

    futures = []
    assignments = []
    for worker, work in workloads.items():
        for future in work:
            futures.append(
                client.submit(
                    lambda v: not isinstance(v, obj_type), future, workers=worker
                )
            )
            assignments.append(
                client.submit(_set_worker, future, worker, workers=worker)
            )

    client.gather(assignments)

    is_not_obj = np.array(client.gather(futures))
    if np.any(is_not_obj):
        raise TypeError(f"{property_name} futures must be an instance of {obj_type}")

    # Re-distribute the workload to ensure all workers are equally loaded
    workload = []
    for work in workloads.values():
        for ii, future in enumerate(work):
            if len(workload) <= ii:
                workload.append([])
            workload[ii].append(future)

    if return_workers:
        return workload, workers
    else:
        return workload


class DistributedComboMisfits(ComboObjectiveFunction):
    """
    A composite objective function for distributed computing.
    """

    def __init__(
        self,
        objfcts: list[BaseObjectiveFunction] | list[Future],
        multipliers=None,
        client: Client | None = None,
        workers: list[str] | None = None,
        **kwargs,
    ):
        self._model: np.ndarray | None = None
        self.client = client
        self.workers = workers

        if multipliers is None:
            multipliers = len(objfcts) * [1]

        super().__init__(**kwargs)
        self._nP = None
        self.objfcts = objfcts
        self.multipliers = np.array(multipliers, dtype=float)

    def __call__(self, m, f=None):
        self.model = m
        client = self.client
        m_future = self._m_as_future

        values = []
        count = 0
        for futures in self._workloads:
            for future, worker in zip(futures, self._workers, strict=True):

                if self.multipliers[count] == 0.0:
                    continue

                values.append(
                    client.submit(
                        _calc_objective,
                        future,
                        self.multipliers[count],
                        m_future,
                        workers=worker,
                    )
                )
                count += 1

        values = self.client.gather(values)
        return np.sum(values)

    @property
    def client(self):
        """
        Get the dask.distributed.Client instance.
        """
        return self._client

    @client.setter
    def client(self, client):
        if client is None:
            client = get_client()

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

        available_workers = [(worker,) for worker in self.client.nthreads()]

        if workers is None:
            workers = available_workers

        if not isinstance(workers, list) or not all(
            isinstance(w, tuple) for w in workers
        ):
            raise TypeError("Workers must be a list of tuple[str].")

        invalid_workers = [w for w in workers if w not in available_workers]
        if invalid_workers:
            raise ValueError(
                f"The following workers are not available: {invalid_workers}. "
                f"Available workers are: {available_workers}."
            )

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

        derivs = 0.0
        count = 0

        for futures in self._workloads:
            future_deriv = []
            for future, worker in zip(futures, self._workers, strict=True):
                if self.multipliers[count] == 0.0:  # don't evaluate the fct
                    continue

                future_deriv.append(
                    client.submit(
                        _deriv,
                        future,
                        self.multipliers[count],
                        m_future,
                        workers=worker,
                    )
                )

                count += 1
            future_deriv = client.gather(future_deriv)

            derivs += np.sum(future_deriv, axis=0)

        return derivs

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

        derivs = 0.0
        count = 0

        for futures in self._workloads:

            future_derivs = []
            for future, worker in zip(futures, self._workers, strict=True):
                if self.multipliers[count] == 0.0:  # don't evaluate the fct
                    continue

                future_derivs.append(
                    client.submit(
                        _deriv2,
                        future,
                        self.multipliers[count],
                        m_future,
                        v_future,
                        # field,
                        workers=worker,
                    )
                )
                count += 1

            future_derivs = self.client.gather(future_derivs)
            derivs += np.sum(future_derivs, axis=0)

        return derivs

    def get_dpred(self, m, f=None, return_residuals=False):
        """
        Request calculation of predicted data from all simulations.
        """
        self.model = m

        client = self.client
        m_future = self._m_as_future
        dpred = []
        residuals = []
        for futures in self._workloads:
            future_preds = []
            for future, worker in zip(futures, self._workers, strict=True):
                future_preds.append(
                    client.submit(
                        _calc_dpred,
                        future,
                        m_future,
                        return_residuals,
                        workers=worker,
                    )
                )
            results = client.gather(future_preds)

            for result in results:
                if return_residuals:
                    dpred += [result[0]]
                    residuals += [result[1]]
                else:
                    dpred += [result]

        if return_residuals:
            return dpred, residuals

        return dpred

    def getJtJdiag(self, m, f=None):
        """
        Request calculation of the diagonal of JtJ from all simulations.
        """
        self.model = m
        m_future = self._m_as_future
        if getattr(self, "_jtjdiag", None) is None:

            jtj_diag = 0.0
            client = self.client

            for futures in self._workloads:
                work = []

                for future, worker in zip(futures, self._workers, strict=True):
                    work.append(
                        client.submit(
                            _get_jtj_diag,
                            future,
                            m_future,
                            workers=worker,
                        )
                    )

                work = client.gather(work)
                jtj_diag += np.sum(work, axis=0)

            self._jtjdiag = jtj_diag

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

        for futures in self._workloads:
            f.append([])
            for future, worker in zip(futures, self._workers, strict=True):
                f[-1].append(
                    client.submit(
                        _calc_fields,
                        future,
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
        for futures in self._workloads:
            for future, worker in zip(futures, self._workers, strict=True):
                stores.append(
                    client.submit(
                        _store_model,
                        future,
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

        futures, workers = _validate_type_or_future_of_type(
            "objfcts",
            objfcts,
            (L2DataMisfit, Future, ComboObjectiveFunction),
            client,
            workers=self.workers,
            return_workers=True,
        )

        self._objfcts = objfcts
        self._workloads = futures
        self._workers = workers

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
        return self._multipliers

    @multipliers.setter
    def multipliers(self, value):
        """Set multipliers attribute after checking if they are valid."""
        for multiplier in value:
            _validate_multiplier(multiplier)
        _check_length_objective_funcs_multipliers(self.objfcts, value)
        self._multipliers = value

    def residuals(self, m, f=None):
        """
        Compute the residual for the data misfit.
        """
        self.model = m

        client = self.client
        m_future = self._m_as_future
        residuals = []

        for futures in self._workloads:
            future_residuals = []
            for future, worker in zip(futures, self._workers, strict=True):
                future_residuals.append(
                    client.submit(
                        _calc_residual,
                        future,
                        m_future,
                        workers=worker,
                    )
                )
            residuals += client.gather(future_residuals)

        return residuals

    def broadcast_updates(self, updates: dict):
        """
        Set the attributes of the objective functions and simulations
        """
        stores = []
        client = self.client

        for fun, (key, value) in updates.items():
            worker = client.who_has(fun)[fun.key]
            stores.append(
                client.submit(
                    _setter_broadcast,
                    fun,
                    key,
                    value,
                    workers=worker,
                )
            )

        self.client.gather(stores)  # blocking call to ensure all models were stored
