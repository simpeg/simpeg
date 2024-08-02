from multiprocessing import Process, Queue, cpu_count
from simpeg.meta import MetaSimulation, SumMetaSimulation, RepeatedSimulation
from simpeg.props import HasModel
import uuid
import numpy as np


class SimpleFuture:
    """Represents an object stored on a seperate simulation process."""

    def __init__(self, item_id, t_queue, r_queue):
        self.item_id = item_id
        self.t_queue = t_queue
        self.r_queue = r_queue

    # This doesn't quite work well yet,
    # Due to the fact that some fields objects from the PDE
    # classes stash the simulation, so this requires serializing
    # the simulation (something we explicitly want to avoid in all cases).
    # def result(self):
    #     self.t_queue.put(("get_item", (self.item_id,)))
    #     item = self.r_queue.get()
    #     if isinstance(item, Exception):
    #         raise item
    #     return item

    def __del__(self):
        # Tell the child process that this object is no longer needed in its cache.
        try:
            self.t_queue.put(("del_item", (self.item_id,)))
        except ValueError:
            # if the queue was already closed it will throw a value error
            # so catch it here gracefully and continue on.
            pass


class _SimulationProcess(Process):
    """A very simple Simulation Actor process.

    It essentially encloses a single simulation in a process that will
    then respond to requests to perform operations with its simulation.
    it will also cache field objects created on this process instead of
    returning them to the main processes, unless explicitly asked for...
    """

    def __init__(self):
        super().__init__()
        self.task_queue = Queue()
        self.result_queue = Queue()

    def run(self):
        # everything here is local to the process
        # a place to cache items locally
        _cached_items = {}

        # The queues are shared between the head process and the worker processes
        # We use them to communicate between the two.
        t_queue = self.task_queue
        r_queue = self.result_queue
        while True:
            # Get a task from the queue
            task = t_queue.get()
            if task is None:
                # None is a poison pill message to kill this loop.
                break
            op, args = task
            try:
                if op == "set_sim":
                    (sim,) = args
                    sim_key = uuid.uuid4().hex
                    _cached_items[sim_key] = sim
                    r_queue.put(sim_key)
                elif op == "get_item":
                    (key,) = args
                    r_queue.put(_cached_items[key])
                elif op == "del_item":
                    (key,) = args
                    _cached_items.pop(key, None)
                elif op == 0:
                    # store_model
                    sim_key, m = args
                    sim = _cached_items[sim_key]
                    sim.model = m
                elif op == 1:
                    # create fields
                    (sim_key,) = args
                    sim = _cached_items[sim_key]
                    f_key = uuid.uuid4().hex
                    r_queue.put(f_key)
                    fields = sim.fields(sim.model)
                    _cached_items[f_key] = fields
                elif op == 2:
                    # do dpred
                    sim_key, f_key = args
                    sim = _cached_items[sim_key]
                    fields = _cached_items[f_key]
                    d_pred = sim.dpred(sim.model, fields)
                    r_queue.put(d_pred)
                elif op == 3:
                    # do jvec
                    sim_key, v, f_key = args
                    sim = _cached_items[sim_key]
                    fields = _cached_items[f_key]
                    jvec = sim.Jvec(sim.model, v, fields)
                    r_queue.put(jvec)
                elif op == 4:
                    # do jtvec
                    sim_key, v, f_key = args
                    sim = _cached_items[sim_key]
                    fields = _cached_items[f_key]
                    jtvec = sim.Jtvec(sim.model, v, fields)
                    r_queue.put(jtvec)
                elif op == 5:
                    # do jtj_diag
                    sim_key, w, f_key = args
                    sim = _cached_items[sim_key]
                    fields = _cached_items[f_key]
                    jtj = sim.getJtJdiag(sim.model, w, fields)
                    r_queue.put(jtj)
            except Exception as err:
                r_queue.put(err)

    def set_sim(self, sim):
        self._check_closed()
        self.task_queue.put(("set_sim", (sim,)))
        key = self.result_queue.get()
        future = SimpleFuture(key, self.task_queue, self.result_queue)
        self._my_sim = future
        return future

    def store_model(self, m):
        self._check_closed()
        sim = self._my_sim
        self.task_queue.put((0, (sim.item_id, m)))

    def get_fields(self):
        self._check_closed()
        sim = self._my_sim
        self.task_queue.put((1, (sim.item_id,)))
        key = self.result_queue.get()
        future = SimpleFuture(key, self.task_queue, self.result_queue)
        return future

    def start_dpred(self, f_future):
        self._check_closed()
        sim = self._my_sim
        self.task_queue.put((2, (sim.item_id, f_future.item_id)))

    def start_j_vec(self, v, f_future):
        self._check_closed()
        sim = self._my_sim
        self.task_queue.put((3, (sim.item_id, v, f_future.item_id)))

    def start_jt_vec(self, v, f_future):
        self._check_closed()
        sim = self._my_sim
        self.task_queue.put((4, (sim.item_id, v, f_future.item_id)))

    def start_jtj_diag(self, w, f_future):
        self._check_closed()
        sim = self._my_sim
        self.task_queue.put(
            (
                5,
                (
                    sim.item_id,
                    w,
                    f_future.item_id,
                ),
            )
        )

    def result(self):
        self._check_closed()
        return self.result_queue.get()

    def join(self, timeout=None):
        self._check_closed()
        self.task_queue.put(None)
        self.task_queue.close()
        self.result_queue.close()
        self.task_queue.join_thread()
        self.result_queue.join_thread()
        super().join(timeout=timeout)


class MultiprocessingMetaSimulation(MetaSimulation):
    """Multiprocessing version of simulation of simulations.

    This class makes use of the `multiprocessing` module to provide
    concurrency, executing the internal simulations in parallel. This class
    is meant to be a (mostly) drop in replacement for :class:`.MetaSimulation`.
    If you want to test your implementation, we recommend starting with a
    small problem using `MetaSimulation`, then switching it to this class.
    the serial version of this class is good for testing correctness.

    If using this class, please be conscious of your operating system's
    default method of spawning new processes. On Windows systems this
    means that the user must be sure that this code is only executed on
    the main process. Usually this is solved in your main script by
    protecting your function calls by checking if you are in `__main__`
    with:

    >>> from simpeg.meta import MultiprocessingMetaSimulation
    >>> if __name__ == '__main__':
    ...     # Do processing here
    ...     sim = MultiprocessingMetaSimulation(...)
    ...     sim.dpred(model)

    You must also be sure to call `sim.join()` before discarding
    this worker to kill the subprocesses that are created, as you would with
    any other multiprocessing process.

    >>> sim.join()

    Parameters
    ----------
    simulations : (n_sim) list of simpeg.simulation.BaseSimulation
        The list of unique simulations that each handle a piece
        of the problem.
    mappings : (n_sim) list of simpeg.maps.IdentityMap
        The map for every simulation. Every map should accept the
        same length model, and output a model appropriate for its
        paired simulation.
    n_processes : optional
        The number of processes to spawn internally. This will default
        to `multiprocessing.cpu_count()`. The number of processes spawned
        will be the minimum of this number and the number of simulations.

    Notes
    -----
    On Unix systems with python version 3.8 the default `fork` method of starting the
    processes has lead to program stalls in certain cases. If you encounter this
    try setting the start method to `spawn'.

    >>> import multiprocessing as mp
    >>> mp.set_start_method("spawn")
    """

    def __init__(self, simulations, mappings, n_processes=None):
        super().__init__(simulations, mappings)

        if n_processes is None:
            n_processes = cpu_count()

        # split simulation,mappings up into chunks
        # (Which are currently defined using MetaSimulations)
        n_sim = len(simulations)
        chunk_sizes = min(n_processes, n_sim) * [n_sim // n_processes]
        for i in range(n_sim % n_processes):
            chunk_sizes[i] += 1

        i_start = 0
        chunk_nd = []
        processes = []
        for chunk in chunk_sizes:
            if chunk == 0:
                continue
            i_end = i_start + chunk
            sim_chunk = MetaSimulation(
                self.simulations[i_start:i_end], self.mappings[i_start:i_end]
            )
            chunk_nd.append(sim_chunk.survey.nD)
            p = _SimulationProcess()
            processes.append(p)
            p.start()
            p.set_sim(sim_chunk)
            i_start = i_end

        self._sim_processes = processes
        self._data_offsets = np.cumsum(np.r_[0, chunk_nd])

    @MetaSimulation.model.setter
    def model(self, value):
        updated = HasModel.model.fset(self, value)
        # Only send the model to the internal simulations if it was updated.
        if updated:
            for p in self._sim_processes:
                p.store_model(self._model)

    def fields(self, m):
        """Create fields for every simulation.

        The returned list contains the field object from each simulation.

        Parameters
        ----------
        m : array_like
            The full model vector.

        Returns
        -------
        (n_sim) list of SimpleFuture
            The list of references to the fields stored on the separate processes.
        """
        self.model = m
        # The above should pass the model to all the internal simulations.
        f = []
        for p in self._sim_processes:
            f.append(p.get_fields())
        return f

    def dpred(self, m=None, f=None):
        if f is None:
            if m is None:
                m = self.model
            f = self.fields(m)
        for p, field in zip(self._sim_processes, f):
            p.start_dpred(field)

        d_pred = []
        for p in self._sim_processes:
            d_pred.append(p.result())
        return np.concatenate(d_pred)

    def Jvec(self, m, v, f=None):
        self.model = m
        if f is None:
            f = self.fields(m)
        for p, field in zip(self._sim_processes, f):
            p.start_j_vec(v, field)
        j_vec = []
        for p in self._sim_processes:
            j_vec.append(p.result())
        return np.concatenate(j_vec)

    def Jtvec(self, m, v, f=None):
        self.model = m
        if f is None:
            f = self.fields(m)
        for i, (p, field) in enumerate(zip(self._sim_processes, f)):
            chunk_v = v[self._data_offsets[i] : self._data_offsets[i + 1]]
            p.start_jt_vec(chunk_v, field)

        jt_vec = []
        for p in self._sim_processes:
            jt_vec.append(p.result())
        return np.sum(jt_vec, axis=0)

    def getJtJdiag(self, m, W=None, f=None):
        self.model = m
        if getattr(self, "_jtjdiag", None) is None:
            if W is None:
                W = np.ones(self.survey.nD)
            else:
                W = W.diagonal()
            if f is None:
                f = self.fields(m)
            for i, (p, field) in enumerate(zip(self._sim_processes, f)):
                chunk_w = W[self._data_offsets[i] : self._data_offsets[i + 1]]
                p.start_jtj_diag(chunk_w, field)
            jtj_diag = []
            for p in self._sim_processes:
                jtj_diag.append(p.result())
            self._jtjdiag = np.sum(jtj_diag, axis=0)
        return self._jtjdiag

    def join(self, timeout=None):
        for p in self._sim_processes:
            if p.is_alive():
                p.join(timeout=timeout)


class MultiprocessingSumMetaSimulation(
    MultiprocessingMetaSimulation, SumMetaSimulation
):
    """A multiprocessing version of :class:`.SumMetaSimulation`.

    See the documentation of :class:`.MultiprocessingMetaSimulation` for
    details on how to use multiprocessing for you operating system.

    Parameters
    ----------
    simulations : (n_sim) list of simpeg.simulation.BaseSimulation
        The list of unique simulations that each handle a piece
        of the problem.
    mappings : (n_sim) list of simpeg.maps.IdentityMap
        The map for every simulation. Every map should accept the
        same length model, and output a model appropriate for its
        paired simulation.
    n_processes : optional
        The number of processes to spawn internally. This will default
        to `multiprocessing.cpu_count()`. The number of processes spawned
        will be the minimum of this number and the number of simulations.
    """

    def dpred(self, m=None, f=None):
        if f is None:
            if m is None:
                m = self.model
            f = self.fields(m)
        for p, field in zip(self._sim_processes, f):
            p.start_dpred(field)

        d_pred = 0
        for p in self._sim_processes:
            d_pred += p.result()
        return d_pred

    def Jvec(self, m, v, f=None):
        self.model = m
        if f is None:
            f = self.fields(m)
        for p, field in zip(self._sim_processes, f):
            p.start_j_vec(v, field)
        j_vec = []
        for p in self._sim_processes:
            j_vec.append(p.result())
        return np.sum(j_vec, axis=0)

    def Jtvec(self, m, v, f=None):
        self.model = m
        if f is None:
            f = self.fields(m)
        for p, field in zip(self._sim_processes, f):
            p.start_jt_vec(v, field)

        jt_vec = []
        for p in self._sim_processes:
            jt_vec.append(p.result())
        return np.sum(jt_vec, axis=0)

    def getJtJdiag(self, m, W=None, f=None):
        self.model = m
        if getattr(self, "_jtjdiag", None) is None:
            if f is None:
                f = self.fields(m)
            for p, field in zip(self._sim_processes, f):
                p.start_jtj_diag(W, field)
            jtj_diag = []
            for p in self._sim_processes:
                jtj_diag.append(p.result())
            self._jtjdiag = np.sum(jtj_diag, axis=0)
        return self._jtjdiag


class MultiprocessingRepeatedSimulation(
    MultiprocessingMetaSimulation, RepeatedSimulation
):
    """A multiprocessing version of the :class:`.RepeatedSimulation`.

    This class makes use of a single simulation that is copied to each internal
    process, but only once per process.

    This simulation shares internals with the :class:`.MultiprocessingMetaSimulation`.
    class, as such please see that documentation for details regarding how to properly
    use multiprocessing on your operating system.

    Parameters
    ----------
    simulation : simpeg.simulation.BaseSimulation
        The simulation to use repeatedly with different mappings.
    mappings : (n_sim) list of simpeg.maps.IdentityMap
        The list of different mappings to use.
    n_processes : optional
        The number of processes to spawn internally. This will default
        to `multiprocessing.cpu_count()`. The number of processes spawned
        will be the minimum of this number and the number of simulations.
    """

    def __init__(self, simulation, mappings, n_processes=None):
        # do this to call the initializer of the Repeated Sim
        super(MultiprocessingMetaSimulation, self).__init__(simulation, mappings)

        if n_processes is None:
            n_processes = cpu_count()

        # split mappings up into chunks
        n_sim = len(mappings)
        chunk_sizes = min(n_processes, n_sim) * [n_sim // n_processes]
        for i in range(n_sim % n_processes):
            chunk_sizes[i] += 1

        processes = []
        i_start = 0
        chunk_nd = []
        for chunk in chunk_sizes:
            if chunk == 0:
                continue
            i_end = i_start + chunk
            sim_chunk = RepeatedSimulation(
                self.simulation, self.mappings[i_start:i_end]
            )
            chunk_nd.append(sim_chunk.survey.nD)
            p = _SimulationProcess()
            processes.append(p)
            p.start()
            p.set_sim(sim_chunk)
            i_start = i_end

        self._data_offsets = np.cumsum(np.r_[0, chunk_nd])
        self._sim_processes = processes
