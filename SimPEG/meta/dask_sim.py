import numpy as np

from SimPEG.simulation import BaseSimulation
from SimPEG.survey import BaseSurvey
from SimPEG.maps import IdentityMap
from SimPEG.utils import validate_list_of_types, validate_type
from SimPEG.props import HasModel
import itertools
from dask.distributed import Client
from dask.distributed import Future
from .simulation import MetaSimulation, SumMetaSimulation
import scipy.sparse as sp
from operator import add
import warnings


def _store_model(mapping, sim, model):
    sim.model = mapping * model


def _calc_fields(mapping, sim, model, apply_map=False):
    if apply_map:
        sim.model = mapping @ model
    return sim.fields(m=sim.model)


def _calc_dpred(mapping, sim, model, field, apply_map=False):
    if apply_map:
        sim.model = mapping @ model
    return sim.dpred(m=sim.model, f=field)


def _j_vec_op(mapping, sim, model, field, v, apply_map=False):
    if apply_map is not None:
        sim.model = mapping @ model
    sim_v = mapping.deriv(model) @ v
    return sim.Jvec(sim.model, sim_v, f=field)


def _jt_vec_op(mapping, sim, model, field, v, apply_map=False):
    if apply_map is not None:
        sim.model = mapping @ model
    return mapping.deriv(model).T @ sim.Jtvec(sim.model, v, f=field)


def _get_jtj_diag(mapping, sim, model, field, w, apply_map=False):
    if apply_map is not None:
        sim.model = mapping @ model
    w = sp.diags(w)
    sim_jtj = sp.diags(np.sqrt(sim.getJtJdiag(sim.model, w, f=field)))
    m_deriv = mapping.deriv(model)
    return np.asarray((sim_jtj @ m_deriv).power(2).sum(axis=0)).flatten()


def _reduce(client, operation, items):
    while len(items) > 1:
        new_reduce = client.map(operation, items[::2], items[1::2])
        if len(items) % 2 == 1:
            new_reduce[-1] = client.submit(operation, new_reduce[-1], items[-1])
        items = new_reduce
    return client.gather(items[0])


def _validate_type_or_future_of_type(
    property_name,
    objects,
    obj_type,
    client,
    workers=None,
    return_workers=False,
):
    try:
        # validate as a list of things that need to be sent.
        objects = validate_list_of_types(
            property_name, objects, obj_type, ensure_unique=True
        )
        if workers is None:
            objects = client.scatter(objects)
        else:
            tmp = []
            for obj, worker in zip(objects, workers):
                tmp.append(client.scatter([obj], workers=worker)[0])
            objects = tmp
    except TypeError:
        pass
    # ensure list of futures
    objects = validate_list_of_types(
        property_name,
        objects,
        Future,
    )
    # Figure out where everything lives
    who = client.who_has(objects)
    if workers is None:
        workers = []
        for obj in objects:
            workers.append(who[obj.key])
    else:
        # Issue a warning if the future is not on the expected worker
        for i, (obj, worker) in enumerate(zip(objects, workers)):
            obj_owner = client.who_has(obj)[obj.key]
            if obj_owner != worker:
                warnings.warn(
                    f"{property_name} {i} is not on the expected worker.", stacklevel=2
                )

    # Ensure this runs on the expected worker
    futures = []
    for obj, worker in zip(objects, workers):
        futures.append(
            client.submit(lambda v: not isinstance(v, obj_type), obj, workers=worker)
        )
    is_not_obj = np.array(client.gather(futures))
    if np.any(is_not_obj):
        raise TypeError(f"{property_name} futures must be an instance of {obj_type}")

    if return_workers:
        return objects, workers
    else:
        return objects


class DaskMetaSimulation(MetaSimulation):
    def __init__(self, simulations, mappings, client):
        self._client = validate_type("client", client, Client, cast=False)
        super().__init__(simulations, mappings)

    def _make_survey(self):
        survey = BaseSurvey([])
        vnD = []
        client = self.client
        for sim, worker in zip(self.simulations, self._workers):
            vnD.append(client.submit(lambda s: s.survey.nD, sim, workers=worker))
        vnD = client.gather(vnD)
        survey._vnD = vnD
        return survey

    @property
    def simulations(self):
        """The future list of simulations.

        Returns
        -------
        (n_sim) list of distributed.client.Future SimPEG.simulation.BaseSimulation
        """
        return self._simulations

    @simulations.setter
    def simulations(self, value):
        client = self.client
        simulations, workers = _validate_type_or_future_of_type(
            "simulations", value, BaseSimulation, client, return_workers=True
        )
        self._simulations = simulations
        self._workers = workers

    @property
    def mappings(self):
        """The future mappings paired to each simulation.

        Every mapping should accept the same length model, and output
        a model that is consistent with the simulation.

        Returns
        -------
        (n_sim) list of distributed.client.Future SimPEG.maps.IdentityMap
        """
        return self._mappings

    @mappings.setter
    def mappings(self, value):
        client = self.client
        if self._repeat_sim:
            mappings, workers = _validate_type_or_future_of_type(
                "mappings", value, IdentityMap, client, return_workers=True
            )
        else:
            workers = self._workers
            if len(value) != len(self.simulations):
                raise ValueError(
                    "Must provide the same number of mappings and simulations."
                )
            mappings = _validate_type_or_future_of_type(
                "mappings", value, IdentityMap, client, workers=workers
            )

        # validate mapping shapes and simulation shapes
        model_len = client.submit(lambda v: v.shape[1], mappings[0]).result()

        def check_mapping(mapping, sim, model_len):
            if mapping.shape[1] != model_len:
                # Bad mapping model length
                return 1
            map_out_shape = mapping.shape[0]
            for name in sim._act_map_names:
                sim_mapping = getattr(sim, name)
                sim_in_shape = sim_mapping.shape[1]
                if (
                    map_out_shape != "*"
                    and sim_in_shape != "*"
                    and sim_in_shape != map_out_shape
                ):
                    # Inconsistent simulation input and mapping output
                    return 2
            # All good
            return 0

        error_checks = []
        for mapping, sim, worker in zip(mappings, self.simulations, workers):
            # if it was a repeat sim, this should cause the simulation to be transfered
            # to each worker.
            error_checks.append(
                client.submit(check_mapping, mapping, sim, model_len, workers=worker)
            )
        error_checks = np.asarray(client.gather(error_checks))

        if np.any(error_checks == 1):
            raise ValueError("All mappings must have the same input length")
        if np.any(error_checks == 2):
            raise ValueError(
                f"Simulations and mappings at indices {np.where(error_checks==2)}"
                f" are inconsistent."
            )

        self._mappings = mappings
        if self._repeat_sim:
            self._workers = workers

    @property
    def _model_map(self):
        # create a bland mapping that has the correct input shape
        # to test against model inputs, avoids pulling the first
        # mapping back to the main task.
        if not hasattr(self, "__model_map"):
            client = self.client
            n_m = client.submit(
                lambda v: v.shape[1],
                self.mappings[0],
                workers=self._workers[0],
            )
            n_m = client.gather(n_m)
            self.__model_map = IdentityMap(nP=n_m)
        return self.__model_map

    @property
    def client(self):
        """The distributed client that handles the internal tasks.

        Returns
        -------
        distributed.Client
        """
        return self._client

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        updated = HasModel.model.fset(self, value)
        # Only send the model to the internal simulations if it was updated.
        if updated:
            client = self.client
            [self._m_as_future] = client.scatter([self._model], broadcast=True)
            if not self._repeat_sim:
                futures = []
                for mapping, sim, worker in zip(
                    self.mappings, self.simulations, self._workers
                ):
                    futures.append(
                        client.submit(
                            _store_model,
                            mapping,
                            sim,
                            self._m_as_future,
                            workers=worker,
                        )
                    )
                self.client.gather(
                    futures
                )  # blocking call to ensure all models were stored

    def fields(self, m):
        self.model = m
        client = self.client
        m_future = self._m_as_future
        # The above should pass the model to all the internal simulations.
        f = []
        for mapping, sim, worker in zip(self.mappings, self.simulations, self._workers):
            f.append(
                client.submit(
                    _calc_fields,
                    mapping,
                    sim,
                    m_future,
                    self._repeat_sim,
                    workers=worker,
                )
            )
        return f

    def dpred(self, m=None, f=None):
        if f is None:
            if m is None:
                m = self.model
            f = self.fields(m)
        client = self.client
        m_future = self._m_as_future
        dpred = []
        for mapping, sim, worker, field in zip(
            self.mappings, self.simulations, self._workers, f
        ):
            dpred.append(
                client.submit(
                    _calc_dpred,
                    mapping,
                    sim,
                    m_future,
                    field,
                    self._repeat_sim,
                    workers=worker,
                )
            )
        return np.concatenate(client.gather(dpred))

    def Jvec(self, m, v, f=None):
        self.model = m
        m_future = self._m_as_future
        if f is None:
            f = self.fields(m)
        client = self.client
        [v_future] = client.scatter([v], broadcast=True)
        j_vec = []
        for mapping, sim, worker, field in zip(
            self.mappings, self.simulations, self._workers, f
        ):
            j_vec.append(
                client.submit(
                    _j_vec_op,
                    mapping,
                    sim,
                    m_future,
                    field,
                    v_future,
                    self._repeat_sim,
                    workers=worker,
                )
            )
        return np.concatenate(self.client.gather(j_vec))

    def Jtvec(self, m, v, f=None):
        self.model = m
        m_future = self._m_as_future
        if f is None:
            f = self.fields(m)
        jt_vec = []
        client = self.client
        for i, (mapping, sim, worker, field) in enumerate(
            zip(self.mappings, self.simulations, self._workers, f)
        ):
            jt_vec.append(
                client.submit(
                    _jt_vec_op,
                    mapping,
                    sim,
                    m_future,
                    field,
                    v[self._data_offsets[i] : self._data_offsets[i + 1]],
                    self._repeat_sim,
                    workers=worker,
                )
            )
        # Do the sum by a reduction operation to avoid gathering a vector
        # of size n_simulations by n_model parameters on the head.
        return _reduce(client, add, jt_vec)

    def getJtJdiag(self, m, W=None, f=None):
        self.model = m
        m_future = self._m_as_future
        if getattr(self, "_jtjdiag", None) is None:
            if W is None:
                W = np.ones(self.survey.nD)
            else:
                W = W.diagonal()
            jtj_diag = []
            client = self.client
            if f is None:
                f = self.fields(m)
            for i, (mapping, sim, worker, field) in enumerate(
                zip(self.mappings, self.simulations, self._workers, f)
            ):
                sim_w = W[self._data_offsets[i] : self._data_offsets[i + 1]]
                jtj_diag.append(
                    client.submit(
                        _get_jtj_diag,
                        mapping,
                        sim,
                        m_future,
                        field,
                        sim_w,
                        self._repeat_sim,
                        workers=worker,
                    )
                )
            self._jtjdiag = _reduce(client, add, jtj_diag)

        return self._jtjdiag


class DaskSumMetaSimulation(DaskMetaSimulation, SumMetaSimulation):
    """An extension of the MetaSimulation that sums the data outputs.

    This class requires the model mappings have the same input length
    and output data for each simulation to have the same number of data.
    """

    def __init__(self, simulations, mappings, client):
        super().__init__(simulations, mappings, client)

    def _make_survey(self):
        survey = BaseSurvey([])
        client = self.client
        n_d = client.submit(lambda s: s.survey.nD, self.simulations[0]).result()
        survey._vnD = [
            n_d,
        ]
        return survey

    @DaskMetaSimulation.simulations.setter
    def simulations(self, value):
        client = self.client
        simulations, workers = _validate_type_or_future_of_type(
            "simulations", value, BaseSimulation, client, return_workers=True
        )
        n_d = client.submit(lambda s: s.survey.nD, simulations[0], workers=workers[0])
        sim_check = []
        for sim, worker in zip(simulations, workers):
            sim_check.append(
                client.submit(lambda s, n: s.survey.nD != n, sim, n_d, workers=worker)
            )
        if np.any(client.gather(sim_check)):
            raise ValueError("All simulations must have the same number of data.")
        self._simulations = simulations
        self._workers = workers

    def dpred(self, m=None, f=None):
        if f is None:
            if m is None:
                m = self.model
            f = self.fields(m)
        client = self.client
        dpred = []
        for sim, worker, field in zip(self.simulations, self._workers, f):
            dpred.append(
                client.submit(_calc_dpred, None, sim, None, field, workers=worker)
            )
        return _reduce(client, add, dpred)

    def Jvec(self, m, v, f=None):
        self.model = m
        if f is None:
            f = self.fields(m)
        client = self.client
        [v_future] = client.scatter([v], broadcast=True)
        j_vec = []
        for mapping, sim, worker, field in zip(
            self.mappings, self._simulations, self._workers, f
        ):
            j_vec.append(
                client.submit(
                    _j_vec_op,
                    mapping,
                    sim,
                    self._m_as_future,
                    field,
                    v_future,
                    workers=worker,
                )
            )
        return _reduce(client, add, j_vec)

    def Jtvec(self, m, v, f=None):
        self.model = m
        if f is None:
            f = self.fields(m)
        jt_vec = []
        client = self.client
        for mapping, sim, worker, field in zip(
            self.mappings, self._simulations, self._workers, f
        ):
            jt_vec.append(
                client.submit(
                    _jt_vec_op,
                    mapping,
                    sim,
                    self._m_as_future,
                    field,
                    v,
                    workers=worker,
                )
            )
        # Do the sum by a reduction operation to avoid gathering a vector
        # of size n_simulations by n_model parameters on the head.
        return _reduce(client, add, jt_vec)

    def getJtJdiag(self, m, W=None, f=None):
        self.model = m
        if getattr(self, "_jtjdiag", None) is None:
            jtj_diag = []
            if W is None:
                W = np.ones(self.survey.nD)
            else:
                W = W.diagonal()
            client = self.client
            if f is None:
                f = self.fields(m)
            for mapping, sim, worker, field in zip(
                self.mappings, self._simulations, self._workers, f
            ):
                jtj_diag.append(
                    client.submit(
                        _get_jtj_diag,
                        mapping,
                        sim,
                        self._m_as_future,
                        field,
                        W,
                        workers=worker,
                    )
                )
            self._jtjdiag = _reduce(client, add, jtj_diag)

        return self._jtjdiag


class DaskRepeatedSimulation(DaskMetaSimulation):
    """A MetaSimulation where a single simulation is used repeatedly.

    This is most useful for linear simulations where a sensitivity matrix can be
    reused with different models. For Non-linear simulations it will often be quicker
    to use the MultiSimulation class with multiple copies of the same simulation.
    """

    _repeat_sim = True

    def __init__(self, simulation, mappings, client):
        self._client = validate_type("client", client, Client, cast=False)

        self.simulation = simulation
        self.mappings = mappings

        self.survey = self._make_survey()
        self._data_offsets = np.cumsum(np.r_[0, self.survey.vnD])

    def _make_survey(self):
        survey = BaseSurvey([])
        nD = self.client.submit(lambda s: s.survey.nD, self.simulation).result()
        survey._vnD = len(self.mappings) * [nD]
        return survey

    @property
    def simulations(self):
        return itertools.repeat(self.simulation)

    @property
    def simulation(self):
        return self._simulation

    @simulation.setter
    def simulation(self, value):
        client = self.client
        if isinstance(value, BaseSimulation):
            # Scatter sim to every client
            [
                value,
            ] = client.scatter([value], broadcast=True)
        if not (
            isinstance(value, Future)
            and client.submit(lambda s: isinstance(s, BaseSimulation), value).result()
        ):
            raise TypeError(
                "simulation must be an instance of BaseSimulation or a Future that returns"
                " a BaseSimulation"
            )
        self._simulation = value
