import numpy as np

from ..simulation import BaseSimulation
from ..survey import BaseSurvey
from ..maps import IdentityMap
from ..utils import validate_list_of_types, validate_type
from ..props import HasModel
import itertools
from dask.distributed import Client
from .simulation import MultiSimulation
import scipy.sparse as sp


def _store_model(sim, mapping, model):
    sim.model = mapping * model


def _apply_mapping(mapping, model):
    return mapping * model


def _calc_fields(sim, sim_model=None):
    if sim_model is None:
        sim_model = sim.model
    return sim.fields(m=sim_model)


def _calc_dpred(sim, field, sim_model=None):
    if sim_model is None:
        sim_model = sim.model
    return sim.dpred(m=sim.model, f=field)


def _j_vec_op(sim, mapping, model, field, v, sim_model=None):
    if sim_model is None:
        sim_model = sim.model
    sim_v = mapping.deriv(model) @ v
    return sim.Jvec(sim_model, sim_v, f=field)


def _jt_vec_op(sim, mapping, model, field, v, sim_model=None):
    if sim_model is None:
        sim_model = sim.model
    return mapping.deriv(model).T @ sim.Jtvec(sim_model, v, f=field)


def _get_jtj_diag(sim, mapping, model, field, w, sim_model=None):
    if sim_model is None:
        sim_model = sim.model
    w = sp.diags(w)
    sim_jtj = sp.diags(np.sqrt(sim.getJtJdiag(sim_model, w)))
    m_deriv = mapping.deriv(model)
    return np.asarray((sim_jtj @ m_deriv).power(2).sum(axis=0)).flatten()


def _sum_op(val1, val2):
    return val1 + val2


def _reduce(client, operation, items):
    while len(items) > 1:
        new_reduce = client.map(operation, items[::2], items[1::2])
        if len(items) % 2 == 1:
            new_reduce[-1] = client.submit(operation, new_reduce[-1], items[-1])
        items = new_reduce
    return client.gather(items[0])


class DaskMultiSimulation(MultiSimulation):
    def __init__(self, simulations, mappings, client=None):
        if client is None:
            client = Client()
        self.client = client
        super().__init__(simulations, mappings)
        self._scattered_mappings = self.client.scatter(self.mappings)
        self._scattered_simulations = self.client.scatter(self.simulations)

    @property
    def client(self):
        return self._client

    @client.setter
    def client(self, value):
        self._client = validate_type("client", value, Client, cast=False)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        updated = HasModel.model.fset(self, value)
        # Only send the model to the internal simulations if it was updated.
        if updated:
            [m_future] = self.client.scatter([self._model], broadcast=True)
            self._m_as_future = m_future
            client = self.client
            futures = []
            for mapping, sim in zip(
                self._scattered_mappings, self._scattered_simulations
            ):
                futures.append(client.submit(_store_model, sim, mapping, m_future))
            client.gather(futures)  # blocking call

    def fields(self, m):
        self.model = m
        # The above should pass the model to all the internal simulations.
        f = self.client.map(_calc_fields, self._scattered_simulations)
        return f

    def dpred(self, m=None, f=None):
        if f is None:
            if m is None:
                m = self.model
            f = self.fields(m)
        d_pred = self.client.map(_calc_dpred, self._scattered_simulations, f)
        return np.concatenate(self.client.gather(d_pred))

    def Jvec(self, m, v, f=None):
        self.model = m
        if f is None:
            f = self.fields(m)
        client = self.client
        [v_future] = client.scatter([v], broadcast=True)
        j_vec = []
        for mapping, sim, field in zip(
            self._scattered_mappings, self._scattered_simulations, f
        ):
            j_vec.append(
                client.submit(
                    _j_vec_op,
                    sim,
                    mapping,
                    self._m_as_future,
                    field,
                    v_future,
                )
            )
        return np.concatenate(self.client.gather(j_vec))

    def Jtvec(self, m, v, f=None):
        self.model = m
        if f is None:
            f = self.fields(m)
        jt_vec = []
        client = self.client
        for i, (mapping, sim, field) in enumerate(
            zip(self._scattered_mappings, self._scattered_simulations, f)
        ):
            jt_vec.append(
                client.submit(
                    _jt_vec_op,
                    sim,
                    mapping,
                    self._m_as_future,
                    field,
                    v[self._data_offsets[i] : self._data_offsets[i + 1]],
                )
            )
        # Do the sum by a reduction operation to avoid gathering a vector
        # of size n_simulations by n_model parameters on the head.
        return _reduce(client, _sum_op, jt_vec)

    def getJtJdiag(self, m, W=None, f=None):
        self.model = m
        if getattr(self, "_jtjdiag", None) is None:
            if W is None:
                W = np.ones(self.survey.nD)
            else:
                W = W.diagonal()
            jtj_diag = []
            client = self.client
            for i, (mapping, sim) in enumerate(
                zip(self._scattered_mappings, self._scattered_simulations)
            ):
                sim_w = W[self._data_offsets[i] : self._data_offsets[i + 1]]
                jtj_diag.append(
                    client.submit(
                        _get_jtj_diag, sim, mapping, self._m_as_future, f, sim_w
                    )
                )
            self._jtjdiag = _reduce(client, _sum_op, jtj_diag)

        return self._jtjdiag


class DaskSumMultiSimulation(DaskMultiSimulation):
    """An extension of the MultiSimulation that sums the data outputs.

    This class requires the model mappings have the same input length
    and output data for each simulation to have the same number of data.
    """

    def __init__(self, simulations, mappings, client=None):
        super().__init__(simulations=simulations, mappings=mappings, client=client)
        # give myself a BaseSurvey
        survey = BaseSurvey([])
        survey._vnD = [
            self.simulations[0].survey.nD,
        ]
        self.survey = survey

    @DaskMultiSimulation.simulations.setter
    def simulations(self, value):
        value = validate_list_of_types(
            "simulations", value, BaseSimulation, ensure_unique=True
        )
        nD = value[0].survey.nD
        for sim in value:
            if sim.survey.nD != nD:
                raise ValueError("All simulations must have the same number of data.")
        self._simulations = value

    def dpred(self, m=None, f=None):
        if f is None:
            if m is None:
                m = self.model
            f = self.fields(m)
        d_pred = self.client.map(_calc_dpred, self._scattered_simulations, f)
        return _reduce(self.client, _sum_op, d_pred)

    def Jvec(self, m, v, f=None):
        self.model = m
        if f is None:
            f = self.fields(m)
        client = self.client
        [v_future] = client.scatter([v], broadcast=True)
        j_vec = []
        for i, (mapping, sim, field) in enumerate(
            zip(self._scattered_mappings, self._scattered_simulations, f)
        ):
            j_vec.append(
                client.submit(
                    _j_vec_op,
                    sim,
                    mapping,
                    self._m_as_future,
                    field,
                    v_future,
                )
            )
        return _reduce(client, _sum_op, j_vec)

    def Jtvec(self, m, v, f=None):
        self.model = m
        if f is None:
            f = self.fields(m)
        jt_vec = []
        client = self.client
        [v_future] = client.scatter([v], broadcast=True)
        for i, (mapping, sim, field) in enumerate(
            zip(self._scattered_mappings, self._scattered_simulations, f)
        ):
            jt_vec.append(
                client.submit(
                    _jt_vec_op,
                    sim,
                    mapping,
                    self._m_as_future,
                    field,
                    v_future,
                )
            )
        # Do the sum by a reduction operation to avoid gathering a vector
        # of size n_simulations by n_model parameters on the head.
        return _reduce(client, _sum_op, jt_vec)

    def getJtJdiag(self, m, W=None, f=None):
        self.model = m
        if getattr(self, "_jtjdiag", None) is None:
            jtj_diag = []
            if W is None:
                W = np.ones(self.survey.nD)
            else:
                W = W.diagonal()
            client = self.client
            [w_future] = client.scatter([W], broadcast=True)
            for i, (mapping, sim) in enumerate(
                zip(self._scattered_mappings, self._scattered_simulations)
            ):
                jtj_diag.append(
                    client.submit(
                        _get_jtj_diag, sim, mapping, self._m_as_future, f, w_future
                    )
                )
            self._jtjdiag = _reduce(client, _sum_op, jtj_diag)

        return self._jtjdiag


class DaskRepeatedSimulation(DaskMultiSimulation):
    """A MultiSimulation where a single simulation is used repeatedly.

    This is most useful for linear simulations where a sensitivity matrix can be
    reused with different models. For Non-linear simulations it will often be quicker
    to use the MultiSimulation class with multiple copies of the same simulation.
    """

    def __init__(self, simulation, mappings, client=None):
        if client is None:
            client = Client()
        self.client = client

        self.simulation = simulation
        self.mappings = mappings
        survey = BaseSurvey([])
        vnD = [sim.survey.nD for sim in self.simulations]
        survey._vnD = vnD
        self.survey = survey
        self._data_offsets = np.cumsum(np.r_[0, vnD])

        self._scattered_mappings = self.client.scatter(self.mappings)
        [self._sim_future] = self.client.scatter([self._simulation], broadcast=True)

    @property
    def simulations(self):
        return itertools.repeat(self.simulation, len(self.mappings))

    @property
    def simulation(self):
        return self._simulation

    @simulation.setter
    def simulation(self, value):
        self._simulation = validate_type(
            "simulation", value, BaseSimulation, cast=False
        )

    @MultiSimulation.mappings.setter
    def mappings(self, value):
        value = validate_list_of_types("mappings", value, IdentityMap)
        model_len = value[0].shape[1]
        sim = self.simulation
        for i, mapping in enumerate(value):
            if mapping.shape[1] != model_len:
                raise ValueError("All mappings must have the same input length")
            map_out_shape = mapping.shape[0]
            for name in sim._act_map_names:
                sim_mapping = getattr(sim, name)
                sim_in_shape = sim_mapping.shape[1]
                if (
                    map_out_shape != "*"
                    and sim_in_shape != "*"
                    and sim_in_shape != map_out_shape
                ):
                    raise ValueError(
                        f"Simulation and mapping at index {i} inconsistent. "
                        f"Simulation mapping shape {sim_in_shape} incompatible with "
                        f"input mapping shape {map_out_shape}."
                    )
        self._mappings = value

    @MultiSimulation.model.setter
    def model(self, value):
        updated = HasModel.model.fset(self, value)
        if updated:
            [self._m_as_future] = self.client.scatter([self._model], broadcast=True)

    def fields(self, m):
        self.model = m
        # The above should pass the model to all the internal simulations.
        f = []
        client = self.client
        for mapping in self._scattered_mappings:
            model = client.submit(_apply_mapping, mapping, self._m_as_future)
            f.append(client.submit(_calc_fields, self._sim_future, model))
        return f

    def dpred(self, m=None, f=None):
        if m is not None:
            self.model = m
        if f is None:
            f = self.fields(self.model)
        d_pred = []
        client = self.client
        for mapping, field in zip(self._scattered_mappings, f):
            model = client.submit(_apply_mapping, mapping, self._m_as_future)
            d_pred.append(_calc_dpred, self._sim_future, field, sim_model=model)
        return np.concatenate(self.client.gather(d_pred))

    def Jvec(self, m, v, f=None):
        self.model = m
        if f is None:
            f = self.fields(m)
        client = self.client
        [v_future] = client.scatter([v], broadcast=True)
        j_vec = []
        sim = self._sim_future
        for mapping, field in zip(self._scattered_mappings, f):
            model = client.submit(_apply_mapping, mapping, self._m_as_future)
            j_vec.append(
                client.submit(
                    _j_vec_op,
                    sim,
                    mapping,
                    self._m_as_future,
                    field,
                    v_future,
                    sim_model=model,
                )
            )
        return np.concatenate(self.client.gather(j_vec))

    def Jtvec(self, m, v, f=None):
        self.model = m
        if f is None:
            f = self.fields(m)
        jt_vec = []
        client = self.client
        sim = self._sim_future
        for i, (mapping, field) in enumerate(zip(self._scattered_mappings, f)):
            model = client.submit(_apply_mapping, mapping, self._m_as_future)
            jt_vec.append(
                client.submit(
                    _jt_vec_op,
                    sim,
                    mapping,
                    self._m_as_future,
                    field,
                    v[self._data_offsets[i] : self._data_offsets[i + 1]],
                    sim_model=model,
                )
            )
        # Do the sum by a reduction operation to avoid gathering a vector
        # of size n_simulations by n_model parameters on the head.
        return _reduce(client, _sum_op, jt_vec)

    def getJtJdiag(self, m, W=None, f=None):
        self.model = m
        if getattr(self, "_jtjdiag", None) is None:
            if W is None:
                W = np.ones(self.survey.nD)
            else:
                W = W.diagonal()
            jtj_diag = []
            client = self.client
            for i, (mapping, sim) in enumerate(
                zip(self._scattered_mappings, self._scattered_simulations)
            ):
                model = client.submit(_apply_mapping, mapping, self._m_as_future)
                sim_w = W[self._data_offsets[i] : self._data_offsets[i + 1]]
                jtj_diag.append(
                    client.submit(
                        _get_jtj_diag,
                        sim,
                        mapping,
                        self._m_as_future,
                        f,
                        sim_w,
                        sim_model=model,
                    )
                )
            self._jtjdiag = _reduce(client, _sum_op, jtj_diag)

        return self._jtjdiag
