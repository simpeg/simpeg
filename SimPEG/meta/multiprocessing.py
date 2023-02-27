from multiprocessing import Process, Queue, cpu_count
from SimPEG.meta import MetaSimulation
from SimPEG.props import HasModel
import uuid
import numpy as np


class SimpleFuture:
    def __init__(self, item_id, t_queue, r_queue):
        self.item_id = item_id
        self.t_queue = t_queue
        self.r_queue = r_queue

    def result(self):
        self.t_queue.put(("get_item", (self.item_id,)))
        item = self.r_queue.get()
        if isinstance(item, Exception):
            raise item
        return item

    def __del__(self):
        self.t_queue.put(("del_item", (self.item_id,)))


class _SimulationProcess(Process):
    """A very simple Simulation Actor process.

    It essentially encloses a single simulation in a process that will
    then respond to requests to perform operations with its simulation.
    it will also cache field objects created on this process instead of
    returning them to the main processes, unless explicitly asked for...
    """

    def __init__(self, sim_chunk):
        self.sim_chunk = sim_chunk
        self.task_queue = Queue()
        self.result_queue = Queue()
        super().__init__()

    def run(self):
        # everything here is local to the process
        # this sim is actually local to the running process and will
        # persist between calls to field, dprec, jvec,...
        sim = self.sim_chunk
        # a place to cache the field items locally
        _cached_items = {}

        # The queues are shared between the head process and the worker processes
        t_queue = self.task_queue
        r_queue = self.result_queue
        while True:
            if not t_queue.empty():
                # Get a task from the queue
                task = t_queue.get()
                if task is None:
                    break
                op, args = task
                if op == "get_item":
                    (key,) = args
                    try:
                        r_queue.put(_cached_items[key])
                    except Exception as err:
                        r_queue.put(err)
                elif op == "del_item":
                    (key,) = args
                    _cached_items.pop(key, None)
                else:
                    if op == 0:
                        # store_model
                        (m,) = args
                        sim.model = m
                    elif op == 1:
                        # create fields
                        f_key = uuid.uuid4().hex
                        r_queue.put(f_key)
                        fields = sim.fields(sim.model)
                        _cached_items[f_key] = fields
                    elif op == 2:
                        # do dpred
                        (f_key,) = args
                        fields = _cached_items[f_key]
                        r_queue.put(sim.dpred(sim.model, fields))
                    elif op == 3:
                        # do jvec
                        v, f_key = args
                        fields = _cached_items[f_key]
                        r_queue.put(sim.Jvec(sim.model, v, fields))
                    elif op == 4:
                        # do jtvec
                        v, f_key = args
                        fields = _cached_items[f_key]
                        r_queue.put(sim.Jtvec(sim.model, v, fields))
                    elif op == 5:
                        # do jtj_diag
                        f_key = args
                        fields = _cached_items[f_key]
                        r_queue.put(sim.getJtJdiag(sim.model, fields))

    def store_model(self, m):
        self.task_queue.put((0, (m,)))

    def get_fields(self):
        self.task_queue.put((1, None))
        key = self.result_queue.get()
        future = SimpleFuture(key, self.task_queue, self.result_queue)
        return future

    def start_dpred(self, f_future):
        self.task_queue.put((2, (f_future.item_id,)))

    def start_j_vec(self, v, f_future):
        self.task_queue.put((3, (v, f_future.item_id)))

    def start_jt_vec(self, v, f_future):
        self.task_queue.put((4, (v, f_future.item_id)))

    def start_jtj_diag(self, f_future):
        self.task_queue.put((4, (f_future.item_id,)))


class MultiprocessingMetaSimulation(MetaSimulation):
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

        processes = []
        i_start = 0
        chunk_nd = []
        for chunk in chunk_sizes:
            if chunk == 0:
                continue
            i_end = i_start + chunk
            sim_chunk = MetaSimulation(
                self.simulations[i_start:i_end], self.mappings[i_start:i_end]
            )
            chunk_nd.append(sim_chunk.survey.nD)
            p = _SimulationProcess(sim_chunk)
            processes.append(p)
            p.start()
            i_start = i_end

        self._data_offsets = np.cumsum(np.r_[0, chunk_nd])
        self._sim_processes = processes
        self._is_alive = True

    def _check_alive(self):
        if not self._is_alive:
            raise RuntimeError(
                "Child processes have already been killed. This simulation is now invalid."
            )

    @MetaSimulation.model.setter
    def model(self, value):
        self._check_alive()
        updated = HasModel.model.fset(self, value)
        # Only send the model to the internal simulations if it was updated.
        if updated:
            for p in self._sim_processes:
                p.store_model(self._model)

    def fields(self, m):
        self._check_alive()
        self.model = m
        # The above should pass the model to all the internal simulations.
        f = []
        for p in self._sim_processes:
            f.append(p.get_fields())
        return f

    def dpred(self, m=None, f=None):
        self._check_alive()
        if f is None:
            if m is None:
                m = self.model
            f = self.fields(m)
        for p, field in zip(self._sim_processes, f):
            p.start_dpred(field)

        d_pred = []
        for p in self._sim_processes:
            d_pred.append(p.result_queue.get())
        return np.concatenate(d_pred)

    def Jvec(self, m, v, f=None):
        self._check_alive()
        self.model = m
        if f is None:
            f = self.fields(m)
        for p, field in zip(self._sim_processes, f):
            p.start_j_vec(v, field)
        j_vec = []
        for p in self._sim_processes:
            j_vec.append(p.result_queue.get())
        return np.concatenate(j_vec)

    def Jtvec(self, m, v, f=None):
        self._check_alive()
        self.model = m
        if f is None:
            f = self.fields(m)
        for i, (p, field) in enumerate(zip(self._sim_processes, f)):
            chunk_v = v[self._data_offsets[i] : self._data_offsets[i + 1]]
            p.start_jt_vec(chunk_v, field)

        jt_vec = 0
        for p in self._sim_processes:
            jt_vec += p.result_queue.get()
        return jt_vec

    def join(self):
        self._is_alive = False
        # kill the subprocesses
        for p in self._sim_processes:
            if p.is_alive():
                p.task_queue.put(None)
                p.join()

    def __del__(self):
        self.join()
