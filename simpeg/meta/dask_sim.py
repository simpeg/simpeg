import numpy as np

from simpeg.simulation import BaseSimulation
from simpeg.survey import BaseSurvey
from simpeg.maps import IdentityMap
from simpeg.utils import validate_list_of_types, validate_type
from simpeg.props import HasModel
import itertools
from dask.distributed import Client
from dask.distributed import Future
from .simulation import MetaSimulation, SumMetaSimulation
import scipy.sparse as sp
from operator import add
import warnings

from dask.base import normalize_token


@normalize_token.register(IdentityMap)
def _normalize_map(mapping):
    return mapping._uuid.hex


@normalize_token.register(BaseSimulation)
def _normalize_simulation(sim):
    return sim._uuid.hex


def _store_model(mapping, sim, model):
    sim.model = mapping * model


def _calc_fields(mapping, sim, model, apply_map=False):
    if apply_map and model is not None:
        return sim.fields(m=mapping @ model)
    else:
        return sim.fields(m=sim.model)


def _calc_dpred(mapping, sim, model, field, apply_map=False):
    if apply_map and model is not None:
        return sim.dpred(m=mapping @ model)
    else:
        return sim.dpred(m=sim.model, f=field)


def _j_vec_op(mapping, sim, model, field, v, apply_map=False):
    sim_v = mapping.deriv(model) @ v
    if apply_map:
        return sim.Jvec(mapping @ model, sim_v, f=field)
    else:
        return sim.Jvec(sim.model, sim_v, f=field)


def _jt_vec_op(mapping, sim, model, field, v, apply_map=False):
    if apply_map:
        jtv = sim.Jtvec(mapping @ model, v, f=field)
    else:
        jtv = sim.Jtvec(sim.model, v, f=field)
    return mapping.deriv(model).T @ jtv


def _get_jtj_diag(mapping, sim, model, field, w, apply_map=False):
    w = sp.diags(w)
    if apply_map:
        jtj = sim.getJtJdiag(mapping @ model, w, f=field)
    else:
        jtj = sim.getJtJdiag(sim.model, w, f=field)
    sim_jtj = sp.diags(np.sqrt(jtj))
    m_deriv = mapping.deriv(model)
    return np.asarray((sim_jtj @ m_deriv).power(2).sum(axis=0)).flatten()


def _reduce(client, operation, items, workers):
    # first sort by workers so items on the same workers are mapped together.
    items = [val for (_, val) in sorted(zip(workers, items), key=lambda x: x[0])]
    while len(items) > 1:
        new_reduce = client.map(operation, items[::2], items[1::2], pure=False)
        if len(items) % 2 == 1:
            new_reduce[-1] = client.submit(
                operation, new_reduce[-1], items[-1], pure=False
            )
        items = new_reduce
    return items[0].result()


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
            # If workers are already set, move the object to the respective worker.
            tmp = []
            for obj, worker in zip(objects, workers):
                future = client.scatter(obj, workers=worker)
                tmp.append(future)
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
        # Because we only ever want to allow execution on a single consistent
        # worker for each simulation-mapping pair, we need to do a bit of sanity
        # checking to choose which worker if the object exists on multiple
        # workers.

        # find out how objects have been assigned to each worker.
        workers_assign_count = {}
        for obj in objects:
            workers = who[obj.key]
            for worker in workers:
                workers_assign_count[worker] = workers_assign_count.get(worker, 0) + 1

        # then loop through and if they exist on multiple workers,
        # choose the worker with the fewest assignments.
        # then decrement any other workers
        worker_assignments = []
        for obj in objects:
            workers = who[obj.key]
            n_assigned = len(objects)
            assigned = None
            for worker in workers:
                n_test = workers_assign_count[worker]
                # choose the worker with the least assigned tasks:
                if n_test < n_assigned:
                    assigned = worker
                    n_assigned = n_test
            # discount workers who had this object but were not chosen:
            for worker in workers:
                if worker != assigned:
                    workers_assign_count[worker] -= 1
            worker_assignments.append(assigned)
        workers = worker_assignments
    else:
        # Issue a warning if the future is not on the expected worker
        for i, (obj, worker) in enumerate(zip(objects, workers)):
            obj_owners = client.who_has(obj)[obj.key]
            if worker not in obj_owners:
                warnings.warn(
                    f"{property_name} {i} is not on the expected worker.", stacklevel=2
                )

    # Ensure this runs on the expected worker
    futures = []
    for obj, worker in zip(objects, workers):
        futures.append(
            client.submit(
                lambda v: not isinstance(v, obj_type), obj, workers=worker, pure=False
            )
        )
    is_not_obj = np.array(client.gather(futures))
    if np.any(is_not_obj):
        raise TypeError(f"{property_name} futures must be an instance of {obj_type}")

    if return_workers:
        return objects, workers
    else:
        return objects


class DaskMetaSimulation(MetaSimulation):
    """Dask Distributed version of simulation of simulations.

    This class makes use of `dask.distributed` module to provide
    concurrency, executing the internal simulations in parallel. This class
    is meant to be a (mostly) drop in replacement for :class:`.MetaSimulation`.
    If you want to test your implementation, we recommend starting with a
    small problem using `MetaSimulation`, then switching it to this class.
    the serial version of this class is good for testing correctness.

    Parameters
    ----------
    simulations : (n_sim) list of simpeg.simulation.BaseSimulation or list of dask.distributed.Future
        The list of unique simulations (or futures that would return a simulation)
        that each handle a piece of the problem.
    mappings : (n_sim) list of simpeg.maps.IdentityMap or list of dask.distributed.Future
        The map for every simulation (or futures that would return a map). Every
        map should accept the  same length model, and output a model appropriate
        for its paired simulation.
    client : dask.distributed.Client, optional
        The dask client to use for communication.
    """

    def __init__(self, simulations, mappings, client):
        self._client = validate_type("client", client, Client, cast=False)
        super().__init__(simulations, mappings)

    def _make_survey(self):
        survey = BaseSurvey([])
        vnD = []
        client = self.client
        for sim, worker in zip(self.simulations, self._workers):
            vnD.append(
                client.submit(lambda s: s.survey.nD, sim, workers=worker, pure=False)
            )
        vnD = client.gather(vnD)
        survey._vnD = vnD
        return survey

    @property
    def simulations(self):
        """The future list of simulations.

        Returns
        -------
        (n_sim) list of distributed.Future simpeg.simulation.BaseSimulation
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
        (n_sim) list of distributed.Future simpeg.maps.IdentityMap
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
        model_len = client.submit(
            lambda v: v.shape[1], mappings[0], pure=False
        ).result()

        def check_mapping(mapping, sim, model_len):
            if mapping.shape[1] != model_len:
                # Bad mapping model length
                return 1
            map_out_shape = mapping.shape[0]
            for name in sim._mapped_properties:
                sim_mapping = sim._prop_map(name)
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
            # if it was a repeat sim, this should cause the simulation to be transferred
            # to each worker if it was originally passed as a future.
            error_checks.append(
                client.submit(
                    check_mapping, mapping, sim, model_len, workers=worker, pure=False
                )
            )
        error_checks = np.asarray(client.gather(error_checks))

        if np.any(error_checks == 1):
            raise ValueError("All mappings must have the same input length")
        if np.any(error_checks == 2):
            raise ValueError(
                f"Simulations and mappings at indices {np.where(error_checks == 2)}"
                f" are inconsistent."
            )

        self._meta_prop = IdentityMap(nP=model_len)
        self._mappings = mappings
        if self._repeat_sim:
            self._workers = workers

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
            self._m_as_future = client.scatter(self._model, broadcast=True, hash=False)
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
                            pure=False,
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
                    pure=False,
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
                    pure=False,
                )
            )
        return np.concatenate(client.gather(dpred))

    def Jvec(self, m, v, f=None):
        self.model = m
        m_future = self._m_as_future
        if f is None:
            f = self.fields(m)
        client = self.client
        v_future = client.scatter(v, broadcast=True, hash=False)
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
                    pure=False,
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
                    pure=False,
                )
            )
        # Do the sum by a reduction operation to avoid gathering a vector
        # of size n_simulations by n_model parameters on the head.
        return _reduce(client, add, jt_vec, workers=self._workers)

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
                        pure=False,
                    )
                )
            self._jtjdiag = _reduce(client, add, jtj_diag, workers=self._workers)

        return self._jtjdiag


class DaskSumMetaSimulation(DaskMetaSimulation, SumMetaSimulation):
    """A dask distributed version of :class:`.SumMetaSimulation`.

    A meta simulation that sums the results of the many individual
    simulations.

    Parameters
    ----------
    simulations : (n_sim) list of simpeg.simulation.BaseSimulation or list of dask.distributed.Future
        The list of unique simulations that each handle a piece
        of the problem.
    mappings : (n_sim) list of simpeg.maps.IdentityMap or list of dask.distributed.Future        The map for every simulation. Every map should accept the
        same length model, and output a model appropriate for its
        paired simulation.
    client : dask.distributed.Client, optional
        The dask client to use for communication.
    """

    def __init__(self, simulations, mappings, client):
        super().__init__(simulations, mappings, client)

    def _make_survey(self):
        survey = BaseSurvey([])
        client = self.client
        n_d = client.submit(
            lambda s: s.survey.nD, self.simulations[0], pure=False
        ).result()
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
        n_d = client.submit(
            lambda s: s.survey.nD, simulations[0], workers=workers[0], pure=False
        )
        sim_check = []
        for sim, worker in zip(simulations, workers):
            sim_check.append(
                client.submit(
                    lambda s, n: s.survey.nD != n, sim, n_d, workers=worker, pure=False
                )
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
                client.submit(
                    _calc_dpred, None, sim, None, field, workers=worker, pure=False
                )
            )
        return _reduce(client, add, dpred, workers=self._workers)

    def Jvec(self, m, v, f=None):
        self.model = m
        if f is None:
            f = self.fields(m)
        client = self.client
        v_future = client.scatter(v, broadcast=True, hash=False)
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
                    pure=False,
                )
            )
        return _reduce(client, add, j_vec, workers=self._workers)

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
                    pure=False,
                )
            )
        # Do the sum by a reduction operation to avoid gathering a vector
        # of size n_simulations by n_model parameters on the head.
        return _reduce(client, add, jt_vec, workers=self._workers)

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
                        pure=False,
                    )
                )
            self._jtjdiag = _reduce(client, add, jtj_diag, workers=self._workers)

        return self._jtjdiag


class DaskRepeatedSimulation(DaskMetaSimulation):
    """A multiprocessing version of the :class:`.RepeatedSimulation`.

    This class makes use of a single simulation that is copied to each internal
    process, but only once per process.

    This simulation shares internals with the :class:`.MultiprocessingMetaSimulation`.
    class, as such please see that documentation for details regarding how to properly
    use multiprocessing on your operating system.

    Parameters
    ----------
    simulation : simpeg.simulation.BaseSimulation or dask.distributed.Future
        The simulation to use repeatedly with different mappings.
    mappings : (n_sim) list of simpeg.maps.IdentityMap or list of dask.distributed.Future
        The list of different mappings to use (or futures that each return a mapping).
    client : dask.distributed.Client, optional
        The dask client to use for communication.
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
        nD = self.client.submit(
            lambda s: s.survey.nD, self.simulation, pure=False
        ).result()
        survey._vnD = len(self.mappings) * [nD]
        return survey

    @property
    def simulations(self):
        return itertools.repeat(self.simulation)

    @property
    def simulation(self):
        """The internal simulation.

        Returns
        -------
        distributed.Future of simpeg.simulation.BaseSimulation
        """
        return self._simulation

    @simulation.setter
    def simulation(self, value):
        client = self.client
        if isinstance(value, BaseSimulation):
            # Scatter sim to every client
            value = client.scatter(value, broadcast=True)
        if not (
            isinstance(value, Future)
            and client.submit(
                lambda s: isinstance(s, BaseSimulation), value, pure=False
            ).result()
        ):
            raise TypeError(
                "simulation must be an instance of BaseSimulation or a Future that returns"
                " a BaseSimulation"
            )
        self._simulation = value
