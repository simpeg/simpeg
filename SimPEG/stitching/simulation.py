import numpy as np

from ..simulation import BaseSimulation
from ..survey import BaseSurvey
from ..maps import IdentityMap
from ..utils import validate_list_of_types
from ..props import HasModel


class ComboSimulation(BaseSimulation):
    def __init__(self, simulations, model_mappings):
        # Ensure each Simulation's input mapping matches the output mappings
        #
        self.simulations = simulations
        self.model_mappings = model_mappings
        # give myself a BaseSurvey that has the number of data equal to the sum of the
        # sims data
        survey = BaseSurvey([])
        vnD = [sim.survey.nD for sim in self.simulations]
        survey._vnD = vnD
        self.survey = survey

    @property
    def simulations(self):
        """The list of simulations."""
        return self._simulations

    @simulations.setter
    def simulations(self, value):
        self._simulations = validate_list_of_types("simulations", value, BaseSimulation)

    @property
    def model_mappings(self):
        return self._model_mappings

    @model_mappings.setter
    def model_mappings(self, value):
        value = validate_list_of_types("model_mappings", value, IdentityMap)
        if len(value) != len(self.simulations):
            raise ValueError(
                "Must provide the same number of model_mappings and simulations."
            )
        model_len = value[0].shape[1]
        for mapping, sim in zip(value, self.simulations):
            if mapping.shape[1] != model_len:
                raise ValueError("All mappings must have the same input length")
        self._model_mappings = value

    @property
    def _act_map_names(self):
        # Implement this here to trick the model setter to know about
        # the list of models and their shape.
        # essentially it points to the first mapping.
        return ["_model_map"]

    @property
    def _model_map(self):
        # all of the model_mappings have the same input shape, so just return
        # the first one.
        return self.model_mappings[0]

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        updated = HasModel.model.fset(self, value)
        # Only send the model to the internal simulations if it was updated.
        if updated:
            for mapping, sim in zip(self.model_mappings, self.simulations):
                sim.model = mapping * self._model

    def fields(self, m):
        self.model = m
        # The above should pass the model to all the internal simulations.
        f = []
        multi_sim = len(self.simulations)
        for mapping, sim in zip(self.model_mappings, self.simulations):
            f.append(sim.fields(m=sim.model))
        return f

    def dpred(self, m=None, f=None):
        if f is None:
            if m is None:
                m = self.model
            f = self.fields(m)
        d_pred = []
        for sim, field in zip(self.simulations, f):
            d_pred.append(sim.dpred(m=sim.model, f=field))
        return np.concatenate(d_pred)

    def Jvec(self, m, v, f=None):
        if f is None:
            if m is None:
                m = self.model
            f = self.fields(m)
        j_vec = []
        for mapping, sim, field in zip(self.model_mappings, self.simulations, f):
            # Every d_pred needs to be setup to grab the current model
            # if given m=None as an argument.
            sim_v = mapping.deriv(self.model) @ v
            j_vec.append(sim.Jvec(sim.model, sim_v, f=field))
        return np.concatenate(j_vec)

    def Jtvec(self, m, v, f=None):
        if f is None:
            if m is None:
                m = self.model
            f = self.fields(m)
        jt_vec = 0
        ind_v_start = 0
        for mapping, sim, field in zip(self.model_mappings, self.simulations, f):
            ind_v_end = ind_v_start + sim.survey.nD
            sim_v = v[ind_v_start:ind_v_end]
            ind_v_start = ind_v_end
            # every simulation needs to have a survey that knows its
            # number of data.
            jt_vec += mapping.deriv(self.model).T @ sim.Jtvec(sim.model, sim_v, f=field)
        return jt_vec


class AdditiveComboSimulation(ComboSimulation):
    """An extension of the ComboSimulation that sums the data outputs.

    This class requires the model mappings have the same input length
    and output data for each simulation to have the same number of data.
    """

    def dpred(self, m=None, f=None):
        if f is None:
            if m is None:
                m = self.model
            f = self.fields(m)
        d_pred = 0
        for sim, field in zip(self.simulations, f):
            d_pred += sim.dpred(m=sim.model, f=field)
        return np.concatenate(d_pred)

    def Jvec(self, m, v, f=None):
        if f is None:
            if m is None:
                m = self.model
            f = self.fields(m)
        j_vec = 0
        for mapping, sim, field in zip(self.model_mappings, self.simulations, f):
            # Every d_pred needs to be setup to grab the current model
            # if given m=None as an argument.
            sim_v = mapping.deriv(self.model) @ v
            j_vec += sim.Jvec(sim.model, sim_v, f=field)
        return j_vec

    def Jtvec(self, m, v, f=None):
        if f is None:
            if m is None:
                m = self.model
            f = self.fields(m)
        jt_vec = 0
        for mapping, sim, field in zip(self.model_mappings, self.simulations, f):
            jt_vec += mapping.deriv(self.model).T @ sim.Jtvec(sim.model, v, f=field)
        return jt_vec
