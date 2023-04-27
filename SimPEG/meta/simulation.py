import numpy as np
import scipy.sparse as sp

from ..simulation import BaseSimulation
from ..survey import BaseSurvey
from ..maps import IdentityMap
from ..utils import validate_list_of_types, validate_type
from ..props import HasModel
import itertools
import warnings


class MetaSimulation(BaseSimulation):
    """Combine multiple simulations into a single one.

    This class is used to encapsulate multiple simulations into a
    single simulation. Each simulation and mapping pair will
    perform its own work, then concatenate the results together.

    For each mapping and simulation pair, given a model, this first
    applies the mapping, then passes the resulting model to the simulation.

    With the proper mappings this can be useful for setting up time-lapse,
    tiled, stitched, or any other simulation that can be broken into many
    individual simulations.

    Parameters
    ----------
    simulations : (n_sim) list of SimPEG.simulation.BaseSimulation
        The list of unique simulations that each handle a piece
        of the problem.
    mappings : (n_sim) list of SimPEG.maps.IdentityMap
        The map for every simulation. Every map should accept the
        same length model, and output a model appropriate for its
        paired simulation.

    Examples
    --------
    Create a list of 1D simulations that perform a piece of a
    stitched problem.

    >>> from SimPEG.simulation import ExponentialSinusoidSimulation
    >>> from SimPEG import maps
    >>> from SimPEG.meta import MetaSimulation
    >>> from discretize import TensorMesh
    >>> import matplotlib.pyplot as plt

    Create a mesh for space and time, then one that represents
    the full dimensionality of the model.
    >>> mesh_space = TensorMesh([100])
    >>> mesh_time = TensorMesh([5])
    >>> full_mesh = TensorMesh([5, 100])

    Lets say we have observations at 5 locations in time. For simplicity
    we will just use the same times from the time mesh, but this is not
    required. Then create a simulation for each of these times. We also
    create an operator that maps the model in full space to the model for
    each simulation.
    >>> obs_times = mesh_time.cell_centers_x
    >>> sims, mappings = [], []
    >>> for time in obs_times:
    ...     sims.append(ExponentialSinusoidSimulation(
    ...         mesh=mesh_space,
    ...         model_map=maps.IdentityMap(),
    ...     ))
    ...     ccs = mesh_space.cell_centers
    ...     p_ave = full_mesh.get_interpolation_matrix(
    ...         np.c_[np.full_like(ccs, time), ccs]
    ...     )
    ...     mappings.append(maps.LinearMap(p_ave))
    >>> sim = MetaSimulation(sims, mappings)

    This simulation acts like a single simulation, which can be used for modeling
    and inversion. This model is a moving box car.
    >>> true_model = np.zeros(full_mesh.shape_cells)
    >>> speed, start, width = 0.8, 0.1, 0.2
    >>> for i, time in enumerate(mesh_time.cell_centers):
    ...     center = speed * time  + start
    ...     in_box = np.abs(mesh_space.cell_centers - center) <= width/2
    ...     true_model[i, in_box] = 1.0
    >>> true_model = true_model.reshape(-1, order='F')

    Then use the simulation to create data.
    >>> d_pred = sim.dpred(true_model)
    >>> plt.plot(d_pred.reshape(5, -1).T)
    >>> plt.show()
    """

    _repeat_sim = False

    def __init__(self, simulations, mappings):
        warnings.warn(
            "The MetaSimulation class is a work in progress and might change in the future",
            stacklevel=2,
        )
        self.simulations = simulations
        self.mappings = mappings
        self.model = None
        # give myself a BaseSurvey that has the number of data equal
        # to the sum of the sims' data.
        survey = BaseSurvey([])
        vnD = [sim.survey.nD for sim in self.simulations]
        survey._vnD = vnD
        self.survey = survey
        self._data_offsets = np.cumsum(np.r_[0, vnD])

    @property
    def simulations(self):
        """The list of simulations.

        Returns
        -------
        (n_sim) list of SimPEG.simulation.BaseSimulation
        """
        return self._simulations

    @simulations.setter
    def simulations(self, value):
        self._simulations = validate_list_of_types(
            "simulations", value, BaseSimulation, ensure_unique=True
        )

    @property
    def mappings(self):
        """The mappings paired to each simulation.

        Every mapping should accept the same length model, and output
        a model that is consistent with the simulation.

        Returns
        -------
        (n_sim) list of SimPEG.maps.IdentityMap
        """
        return self._mappings

    @mappings.setter
    def mappings(self, value):
        value = validate_list_of_types("mappings", value, IdentityMap)
        if not self._repeat_sim and len(value) != len(self.simulations):
            raise ValueError(
                "Must provide the same number of mappings and simulations."
            )
        model_len = value[0].shape[1]
        for i, (mapping, sim) in enumerate(zip(value, self.simulations)):
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

    @property
    def _act_map_names(self):
        # Implement this here to trick the model setter to know about
        # how long an input model should be.
        # essentially it points to the first mapping.
        return ["_model_map"]

    @property
    def _model_map(self):
        # all of the mappings have the same input shape, so just return
        # the first one.
        return self.mappings[0]

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        updated = HasModel.model.fset(self, value)
        # Only send the model to the internal simulations if it was updated.
        if not self._repeat_sim and updated:
            for mapping, sim in zip(self.mappings, self.simulations):
                if value is not None:
                    sim.model = mapping * self._model
                else:
                    sim.model = value

    def fields(self, m):
        """Create fields for every simulation.

        The returned list contains the field object from each simulation.

        Returns
        -------
        (n_sim) list
            The type of each item of the list is determined by the internal
            simulation that created it.
        """
        self.model = m
        # The above should pass the model to all the internal simulations.
        f = []
        for mapping, sim in zip(self.mappings, self.simulations):
            if self._repeat_sim and self.model is not None:
                sim.model = mapping * self.model
            f.append(sim.fields(sim.model))
        return f

    def dpred(self, m=None, f=None):
        if f is None:
            if m is None:
                m = self.model
            f = self.fields(m)
        d_pred = []
        for mapping, sim, field in zip(self.mappings, self.simulations, f):
            if self._repeat_sim:
                sim.model = mapping * self.model
            d_pred.append(sim.dpred(m=sim.model, f=field))
        return np.concatenate(d_pred)

    def Jvec(self, m, v, f=None):
        self.model = m
        if f is None:
            f = self.fields(m)
        j_vec = []
        for mapping, sim, field in zip(self.mappings, self.simulations, f):
            if self._repeat_sim:
                sim.model = mapping * self.model
            sim_v = mapping.deriv(self.model) @ v
            j_vec.append(sim.Jvec(sim.model, sim_v, f=field))
        return np.concatenate(j_vec)

    def Jtvec(self, m, v, f=None):
        self.model = m
        if f is None:
            f = self.fields(m)
        jt_vec = 0
        for i, (mapping, sim, field) in enumerate(
            zip(self.mappings, self.simulations, f)
        ):
            if self._repeat_sim:
                sim.model = mapping * self.model
            sim_v = v[self._data_offsets[i] : self._data_offsets[i + 1]]
            jt_vec += mapping.deriv(self.model).T @ sim.Jtvec(sim.model, sim_v, f=field)
        return jt_vec

    def getJtJdiag(self, m, W=None, f=None):
        """Return the squared sum of columns of the Jacobian.

        Evaluates the weighted squared norm of each column
        of the Jacobian matrix. This is usually used to construct
        sensitivity weighting matrices or for diagonal preconditioners
        to iterative solvers.

        Parameters
        ----------
        m : (n_m) numpy.ndarray
            The model to evalute the Jacobian at.
        W : (n_d, n_d) scipy.sparse.csr_matrix, optional
            A diagonal data weighting matrix.
        f : fields, optional
            The fields object created from this class.

        Returns
        -------
        (n_m) numpy.ndarray
            Squared sum of columns of the Jacobian matrix

        Notes
        -----
        Internally, this function evaluates the ``getJtJdiag`` method of each
        simulation, then applies the model mapping to the output as:

        >>> sq_sum = 0
        >>> for i in range(n_sim):
        ...    row = sim[i].getJtJdiag(model)
        ...    sq_sum += W[i] * (sp.diag(sqrt(row)) @ mapping[i].deriv()).power(2).sum(axis=0)

        This approach is correct for mapping that match input parameters to a
        single output parameter, (i.e. the `mapping.deriv` has only 1 element in each column).
        For other mappings, it is usually close within a scaling factor, whose accuracy is
        then controlled by how diagonally dominant ``J.T @ J`` is.
        """
        self.model = m
        if getattr(self, "_jtjdiag", None) is None:
            if W is None:
                W = np.ones(self.survey.nD)
            else:
                W = W.diagonal()
            jtj_diag = 0.0
            # approximate the JtJ diag on the full model space as:
            # sum((diag(sqrt(jtj_diag)) @ M_deriv))**2)
            # Which is correct for mappings that match input parameters to only 1 output parameter.
            # (i.e. projections, multipliers, etc.).
            # It is usually close within a scaling factor for others, whose accuracy is controlled
            # by how diagonally dominant JtJ is.
            if f is None:
                f = self.fields(m)
            for i, (mapping, sim, field) in enumerate(
                zip(self.mappings, self.simulations, f)
            ):
                if self._repeat_sim:
                    sim.model = mapping * self.model
                sim_w = sp.diags(W[self._data_offsets[i] : self._data_offsets[i + 1]])
                sim_jtj = sp.diags(np.sqrt(sim.getJtJdiag(sim.model, sim_w, f=field)))
                m_deriv = mapping.deriv(self.model)
                jtj_diag += np.asarray(
                    (sim_jtj @ m_deriv).power(2).sum(axis=0)
                ).flatten()
            self._jtjdiag = jtj_diag

        return self._jtjdiag

    @property
    def deleteTheseOnModelUpdate(self):
        return super().deleteTheseOnModelUpdate + ["_jtjdiag"]


class SumMetaSimulation(MetaSimulation):
    """An extension of the MetaSimulation that sums the data outputs.

    This class requires the mappings have the same input length
    and each simulation to have the same number of data.

    This could be useful for a linear problem where each simulation
    tackles a different subset of the model.

    Parameters
    ----------
    simulations : (n_sim) list of SimPEG.simulation.BaseSimulation
    mappings : (n_sim) list of SimPEG.maps.IdentityMap
    """

    _repeat_sim = False

    def __init__(self, simulations, mappings):
        warnings.warn(
            "The SumMetaSimulation class is a work in progress and might change in the future",
            stacklevel=2,
        )
        self.simulations = simulations
        self.mappings = mappings
        self.model = None
        # give myself a BaseSurvey
        survey = BaseSurvey([])
        survey._vnD = [
            self.simulations[0].survey.nD,
        ]
        self.survey = survey

    @MetaSimulation.simulations.setter
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
        d_pred = 0
        for sim, field in zip(self.simulations, f):
            d_pred += sim.dpred(m=sim.model, f=field)
        return d_pred

    def Jvec(self, m, v, f=None):
        self.model = m
        if f is None:
            f = self.fields(m)
        j_vec = 0
        for mapping, sim, field in zip(self.mappings, self.simulations, f):
            # Every d_pred needs to be setup to grab the current model
            # if given m=None as an argument.
            sim_v = mapping.deriv(self.model) @ v
            j_vec += sim.Jvec(sim.model, sim_v, f=field)
        return j_vec

    def Jtvec(self, m, v, f=None):
        self.model = m
        if f is None:
            f = self.fields(m)
        jt_vec = 0
        for mapping, sim, field in zip(self.mappings, self.simulations, f):
            jt_vec += mapping.deriv(self.model).T @ sim.Jtvec(sim.model, v, f=field)
        return jt_vec

    def getJtJdiag(self, m, W=None, f=None):
        self.model = m
        if getattr(self, "_jtjdiag", None) is None:
            jtj_diag = 0.0
            if f is None:
                f = self.fields(m)
            for mapping, sim, field in zip(self.mappings, self.simulations, f):
                sim_jtj = sp.diags(np.sqrt(sim.getJtJdiag(sim.model, W, f=field)))
                m_deriv = mapping.deriv(self.model)
                jtj_diag += np.asarray(
                    (sim_jtj @ m_deriv).power(2).sum(axis=0)
                ).flatten()
            self._jtjdiag = jtj_diag

        return self._jtjdiag


class RepeatedSimulation(MetaSimulation):
    """A MetaSimulation where a single simulation is used repeatedly.

    This is most useful for linear simulations where a sensitivity matrix can be
    reused with different models. For non-linear simulations it will often be quicker
    to use the MetaSimulation class with multiple copies of the same simulation.

    Parameters
    ----------
    simulation : SimPEG.simulation.BaseSimulation
    mappings : (n_sim) list of SimPEG.maps.IdentityMap
    """

    _repeat_sim = True

    def __init__(self, simulation, mappings):
        warnings.warn(
            "The RepeatedSimulation class is a work in progress and might change in the future",
            stacklevel=2,
        )
        self.simulation = simulation
        self.mappings = mappings
        self.model = None
        survey = BaseSurvey([])
        vnD = len(self.mappings) * [self.simulation.survey.nD]
        survey._vnD = vnD
        self.survey = survey
        self._data_offsets = np.cumsum(np.r_[0, vnD])

    @property
    def simulations(self):
        return itertools.repeat(self.simulation)

    @property
    def simulation(self):
        """The internal simulation.

        Returns
        -------
        SimPEG.simulation.BaseSimulation
        """
        return self._simulation

    @simulation.setter
    def simulation(self, value):
        self._simulation = validate_type(
            "simulation", value, BaseSimulation, cast=False
        )
