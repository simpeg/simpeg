from ..objective_function import ComboObjectiveFunction, BaseObjectiveFunction

import numpy as np
from dask.distributed import Client
from ..data_misfit import L2DataMisfit
import os
from simpeg.utils import validate_list_of_types
from time import time

OUTFILE = os.getcwd() + "/update.txt"


def write_message(message, mode="a"):
    with open(OUTFILE, mode) as f:
        f.write(message + "\n")


write_message("Starting", mode="w+")


def _calc_fields(objfct, model):
    return objfct.simulation.fields(m=objfct.simulation.model)


def _calc_dpred(objfct, model):
    return objfct.simulation.dpred(m=objfct.simulation.model)


def _calc_residual(objfct, model):
    return objfct.W * (
        objfct.data.dobs - objfct.simulation.dpred(m=objfct.simulation.model)
    )


def _deriv(objfct, multiplier, model):
    # if fields is not None and objfct.has_fields:
    #     return multiplier * objfct.deriv(objfct.simulation.model)
    # else:
    return multiplier * objfct.deriv(objfct.simulation.model)


def _deriv2(objfct, multiplier, model, v):
    # if fields is not None and objfct.has_fields:
    #     return multiplier * objfct.deriv2(objfct.simulation.model, v)
    # else:
    print("Calculating deriv2")
    return multiplier * objfct.deriv2(objfct.simulation.model, v)


def _store_model(objfct, model):
    objfct.simulation.model = model


def _get_jtj_diag(objfct, model):
    jtj = objfct.simulation.getJtJdiag(objfct.simulation.model, objfct.W)
    return jtj.flatten()


def _validate_type_or_future_of_type(
    property_name,
    objects,
    obj_type,
    client,
    workers=None,
    return_workers=False,
):
    # try:
    #     # validate as a list of things that need to be sent.
    workers = [(worker.worker_address,) for worker in client.cluster.workers.values()]
    objects = validate_list_of_types(
        property_name, objects, obj_type, ensure_unique=True
    )
    workload = [[]]
    count = 0
    for obj in objects:
        if count == len(workers):
            count = 0
            workload.append([])
        obj.simulation.simulations[0].worker = workers[count]
        future = client.scatter([obj], workers=workers[count])[0]

        if hasattr(obj, "name"):
            future.name = obj.name

        workload[-1].append(future)
        count += 1

        # objects[0].simulation.simulations[0].worker = workers[0]
        # if workers is None:
        #     objects = client.scatter(objects)
        # else:
        #     tmp = []
        #     for obj, worker in zip(objects, workers):
        #         tmp.append(client.scatter([obj], workers=worker)[0])
        #     objects = tmp
    # except TypeError:
    #     pass
    # ensure list of futures
    # objects = validate_list_of_types(
    #     property_name,
    #     objects,
    #     Future,
    # )
    # Figure out where everything lives

    # who = client.who_has(workload)
    # # if workers is None:
    # #     workers = []
    # for ii, worker in enumerate(who.values()):
    #     if worker != workers[ii % len(workers)]:
    #         warnings.warn(
    #             f"{property_name} {i} is not on the expected worker.", stacklevel=2
    #         )
    #         # obj = client.submit(_set_worker, obj, worker)

    # Ensure this runs on the expected worker
    futures = []
    for work in workload:

        for obj, worker in zip(work, workers):
            futures.append(
                client.submit(
                    lambda v: not isinstance(v, obj_type), obj, workers=worker
                )
            )
    is_not_obj = np.array(client.gather(futures))
    if np.any(is_not_obj):
        raise TypeError(f"{property_name} futures must be an instance of {obj_type}")

    if return_workers:
        return workload, workers
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

        values = []
        count = 0
        for futures in self._futures:
            for objfct, worker in zip(futures, self._workers):

                if self.multipliers[count] == 0.0:
                    continue

                values.append(
                    client.submit(
                        _calc_objective,
                        objfct,
                        self.multipliers[count],
                        m_future,
                        workers=worker,
                    )
                )

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

        # if f is None:
        #     f = self.fields(m)

        derivs = 0.0
        count = 0
        write_message("Calculating deriv")
        for futures in self._futures:
            future_deriv = []
            for objfct, worker in zip(futures, self._workers):
                if self.multipliers[count] == 0.0:  # don't evaluate the fct
                    continue

                future_deriv.append(
                    client.submit(
                        _deriv,
                        objfct,
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
        write_message("Calculating deriv2")
        for futures in self._futures:

            future_derivs = []
            for objfct, worker in zip(futures, self._workers):
                if self.multipliers[count] == 0.0:  # don't evaluate the fct
                    continue

                future_derivs.append(
                    client.submit(
                        _deriv2,
                        objfct,
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

    def get_dpred(self, m, f=None):
        self.model = m

        client = self.client
        m_future = self._m_as_future
        dpred = []
        write_message("Calculating dpred")
        for futures in self._futures:
            future_preds = []
            for objfct, worker in zip(futures, self._workers):
                future_preds.append(
                    client.submit(
                        _calc_dpred,
                        objfct,
                        m_future,
                        workers=worker,
                    )
                )
            dpred += client.gather(future_preds)

        return dpred

    def getJtJdiag(self, m, f=None):
        self.model = m
        m_future = self._m_as_future
        if getattr(self, "_jtjdiag", None) is None:

            jtj_diag = 0.0
            client = self.client

            write_message("Calculating JtJdiag")
            # if f is None:
            #     f = self.fields(m)
            for ii, futures in enumerate(self._futures):
                work = []
                ct = time()
                for objfct, worker in zip(futures, self._workers):
                    work.append(
                        client.submit(
                            _get_jtj_diag,
                            objfct,
                            m_future,
                            # field,
                            workers=worker,
                        )
                    )

                work = client.gather(work)
                write_message(
                    f"Future {ii} of {len(self._futures)} in {time() - ct:.3f} sec"
                )
                jtj_diag += np.sum(work, axis=0)

            self._jtjdiag = jtj_diag

        return self._jtjdiag

    def fields(self, m):
        self.model = m
        client = self.client
        m_future = self._m_as_future
        if getattr(self, "_stashed_fields", None) is not None:
            return self._stashed_fields
        # The above should pass the model to all the internal simulations.
        f = []
        write_message("Calculating fields")
        for futures in self._futures:
            f.append([])
            for objfct, worker in zip(futures, self._workers):
                f[-1].append(
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
        for futures in self._futures:
            for objfct, worker in zip(futures, self._workers):
                stores.append(
                    client.submit(
                        _store_model,
                        objfct,
                        self._m_as_future,
                        workers=worker,
                    )
                )
        self.client.gather(stores)  # blocking call to ensure all models were stored
        self._model = value

    @property
    def objfcts(self):
        return self._objfcts

    @objfcts.setter
    def objfcts(self, objfcts):
        client = self.client

        futures, workers = _validate_type_or_future_of_type(
            "objfcts",
            objfcts,
            L2DataMisfit,
            client,
            workers=self.workers,
            return_workers=True,
        )

        self._objfcts = objfcts
        self._futures = futures
        self._workers = workers

    def residuals(self, m, f=None):
        """
        Compute the residual for the data misfit.
        """
        self.model = m

        client = self.client
        m_future = self._m_as_future
        residuals = []
        write_message("Calculating residuals")
        for futures in self._futures:
            future_residuals = []
            for objfct, worker in zip(futures, self._workers):
                future_residuals.append(
                    client.submit(
                        _calc_residual,
                        objfct,
                        m_future,
                        workers=worker,
                    )
                )
            residuals += client.gather(future_residuals)

        return residuals
