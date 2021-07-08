import numpy as np

from discretize.utils import requires

from ...utils import mkvc
from .simulation import BaseFDEMSimulation
from .sources import ElectricWire
from memory_profiler import profile

# emg3d is a soft dependency
try:
    import emg3d
except ImportError:
    emg3d = False


@requires({"emg3d": emg3d})
class Simulation3DEMG3D(BaseFDEMSimulation):
    """3D simulation of electromagnetic fields using emg3d as a solver.

    .. note::

        - Only isotropic model implemented so far. Needs extension to tri-axial
          anisotropy, mu_r, and epsilon_r.

        - Currently, only electric "point"-dipole sources and electric/magnetic
          point receivers are implemented. emg3d could do more, e.g., finite
          length electric dipoles or arbitrary electric wires (and therefore
          loops). These are *not yet* implemented here.


    Parameters
    ----------
    simulation_opts : dict
        Input parameters forward to ``emg3d.Simulation``. See the emg3d
        documentation for all the possibilities.

    """

    _solutionType = "eSolution"
    _formulation = "EB"
    storeJ = False
    _Jmatrix = None

    def __init__(self, mesh, **kwargs):
        """Initialize Simulation using emg3d as solver."""

        # Store simulation options.
        self.simulation_opts = kwargs.pop('simulation_opts', {})

        super().__init__(mesh, **kwargs)

        # TODO : Count to store data per iteration.
        # - Should be replaced by a proper count form SimPEG.inversion.
        # - Should probably be made optional to store data at each step.
        self._it_count = 0

    @property
    def emg3d_sim(self):
        """emg3d simulation; obtained from SimPEG simulation."""

        if getattr(self, "_emg3d_sim", None) is None:

            # Create twin Simulation in emg3d.
            self._emg3d_sim = emg3d.Simulation(
                survey=self.emg3d_survey,
                model=emg3d.Model(self.mesh),  # Dummy values of 1 for init.
                **{'name': 'Simulation created by SimPEG',
                   'gridding': 'same',               # Change this eventually!
                   'tqdm_opts': {'disable': True},   # Switch-off tqdm
                   'receiver_interpolation': 'linear',  # Should be linear
                   **self.simulation_opts}              # User input
            )

        return self._emg3d_sim

    @property
    def emg3d_survey(self):
        """emg3d survey; obtained from SimPEG survey."""

        if getattr(self, "_emg3d_survey", None) is None:

            # Get and store emg3d-survey and data map.
            survey, dmap = survey_to_emg3d(self.survey)
            self._emg3d_survey = survey
            self._dmap_simpeg_emg3d = dmap

            # Create emg3d data dummy; can be re-used.
            self._emg3d_array = np.full(survey.shape, np.nan+1j*np.nan)

        return self._emg3d_survey

    @emg3d_survey.setter
    def emg3d_survey(self, emg3d_survey):
        """emg3d survey; obtained from SimPEG survey."""

        # Store survey.
        self._emg3d_survey = emg3d_survey

        # Store emg3d-to-SimPEG mapping.
        try:

            # Get dmap from the stored indices.
            indices = np.zeros((self.survey.nD, 3), dtype=int)
            for i in range(self.survey.nD):
                indices[i, :] = np.r_[
                        np.where(emg3d_survey.data.indices.data == i)]

            # Store dmap.
            self._dmap_simpeg_emg3d = tuple(indices.T)

        except:
            raise AttributeError(
                "Provided emg3d-survey misses the indices data array."
            )

        # Create emg3d data dummy; can be re-used.
        self._emg3d_array = np.full(emg3d_survey.shape, np.nan+1j*np.nan)

    @property
    def emg3d_model(self):
        """emg3d conductivity model; obtained from SimPEG conductivities."""
        self._emg3d_model = emg3d.Model(
            self.mesh,
            property_x=self.sigma.reshape(self.mesh.shape_cells, order='F'),
            # property_y=None,  Not yet implemented
            # property_z=None,   "
            # mu_r=None,         "
            # epsilon_r=None,    "
            mapping='Conductivity',
        )
        return self._emg3d_model

    @property
    def _emg3d_simulation_update(self):
        """Updates emg3d simulation with new model, resetting the fields."""

        # We have to replace the model
        # Warning:
        # This will not work for automatic model extension possible in emg3d.
        self.emg3d_sim.model = self.emg3d_model

        # Re-initiate all dicts _except_ grids (automatic gridding)
        self.emg3d_sim._dict_model = self.emg3d_sim._dict_initiate
        self.emg3d_sim._dict_efield = self.emg3d_sim._dict_initiate
        self.emg3d_sim._dict_hfield = self.emg3d_sim._dict_initiate
        self.emg3d_sim._dict_efield_info = self.emg3d_sim._dict_initiate
        self.emg3d_sim._gradient = None
        self.emg3d_sim._misfit = None
        self.emg3d_sim._vec = None  # TODO check back when emg3d-side finished!

    @profile
    def Jvec(self, m, v, f=None):
        """
        Sensitivity times a vector.

        :param numpy.ndarray m: inversion model (nP,)
        :param numpy.ndarray v: vector which we take sensitivity product with
            (nP,)
        :param SimPEG.electromagnetics.frequency_domain.fields.FieldsFDEM f: fields object
        :rtype: numpy.ndarray
        :return: Jvec (ndata,)
        """
        if self.verbose:
            print("Compute Jvec")

        if self.storeJ:
            J = self.getJ(m, f=f)
            Jv = mkvc(np.dot(J, v))
            return Jv

        self.model = m

        if f is None:
            f = self.fields(m=m)

        dsig_dm_v = self.sigmaDeriv @ v
        j_vec = emg3d.optimize.jvec(f, vec=dsig_dm_v)

        # Map emg3d-data-array to SimPEG-data-vector
        return j_vec[self._dmap_simpeg_emg3d]

    @profile
    def Jtvec(self, m, v, f=None):
        """
        Sensitivity transpose times a vector

        :param numpy.ndarray m: inversion model (nP,)
        :param numpy.ndarray v: vector which we take adjoint product with (ndata,)
        :param SimPEG.electromagnetics.frequency_domain.fields.FieldsFDEM f: fields object
        :rtype: numpy.ndarray
        :return: Jtvec (nP,)
        """
        if self.verbose:
            print("Compute Jtvec")

        if self.storeJ:

            # Put v onto emg3d data-array.
            self._emg3d_array[self._dmap_simpeg_emg3d] = v

            J = self.getJ(m, f=f)
            Jtv = mkvc(np.dot(J.T, self._emg3d_array))

            return Jtv

        self.model = m

        if f is None:
            f = self.fields(m=m)

        return self._Jtvec(m, v=v, f=f)

    @profile
    def _Jtvec(self, m, v=None, f=None):
        """Compute adjoint sensitivity matrix (J^T) and vector (v) product.

        Full J matrix can be computed by setting v=None (not implemented yet).
        """

        if v is not None:
            # Put v onto emg3d data-array.
            self._emg3d_array[self._dmap_simpeg_emg3d] = v

            # Replace residual by vector if provided
            f.survey.data['residual'][...] = self._emg3d_array
            jt_sigma_vec = np.empty(self.model.size)
            # Get gradient with `v` as residual.
            jt_sigma_vec = emg3d.optimize.gradient(f).flatten('F')

            jt_vec = self.sigmaDeriv.T @ jt_sigma_vec
            return jt_vec

        else:
            # This is for forming full sensitivity matrix
            # Currently, it is not correct.
            # Requires a fix in optimize.gradient
            # Jt is supposed to be a complex value ...
            # Jt = np.zeros((self.model.size, self.survey.nD), order="F")
            # for i_datum in range(self.survey.nD):
            #     vec = np.zeros(self.survey.nD)
            #     vec[i_datum] = 1.
            #     vec = vec.reshape(self.emg3d_survey.shape)
            #     jt_sigma_vec = emg3d.optimize.gradient(f, vector=vec)
            #     Jt[:, i_datum] = self.sigmaDeriv.T @ jt_sigma_vec
            # return Jt

            raise NotImplementedError

    def getJ(self, m, f=None):
        """Generate full sensitivity matrix."""

        if self._Jmatrix is not None:
            return self._Jmatrix

        else:
            if self.verbose:
                print("Calculating J and storing")
            self.model = m
            if f is None:
                f = self.fields(m)
            self._Jmatrix = (self._Jtvec(m, v=None, f=f)).T

        return self._Jmatrix

    def dpred(self, m=None, f=None):
        r"""Return the predicted (modelled) data for a given model.

        The fields, f, (if provided) will be used for the predicted data
        instead of recalculating the fields (which may be expensive!).

        .. math::

            d_\text{pred} = P(f(m))

        Where P is a projection of the fields onto the data space.

        :param numpy.ndarray m: model
        :param SimPEG.electromagnetics.frequency_domain.fields.FieldsFDEM f: fields object
        :rtype: numpy.ndarray
        :return: data, the data
        """

        if self.verbose:
            print("Compute predicted")

        if f is None:
            f = self.fields(m=m)

        # Map emg3d-data-array to SimPEG-data-vector
        return f.data.synthetic.data[self._dmap_simpeg_emg3d]

    @profile
    def fields(self, m=None):
        """Return the electric fields for a given model.

        :param numpy.ndarray m: model
        :rtype: numpy.ndarray
        :return: f, the fields
        """

        if self.verbose:
            print("Compute fields")

        if m is not None:
            # Store model.
            self.model = m

            # Update simulation.
            self._emg3d_simulation_update

        # Compute forward model and sets initial residuals.
        _ = self.emg3d_sim.misfit

        # Store the data at each step in the survey-xarray
        if m is not None:
            current_data = self.emg3d_survey.data.synthetic.copy()
            self.emg3d_survey.data[f"it_{self._it_count}"] = current_data
            self._it_count += 1  # Update counter

        return self.emg3d_sim


def survey_to_emg3d(survey):
    """Return emg3d survey from provided SimPEG survey.


    Parameters
    ----------
    survey : Survey
        SimPEG survey instance.


    Returns
    -------
    emg3d_survey : Survey
        emg3d survey instance, containing the data set `indices`.

    data_map : tuple
        Indices to map SimPEG-data to emg3d data and vice-versa.

        To put SimPEG data array on, e.g., the emg3d synthetic xarray:

           emg3d_survey.data.synthetic.data[dmap] = simpeg_array

        To obtain SimPEG data array from, e.g., the emg3d synthetic xarray:

           simpeg_array = emg3d_survey.data.synthetic.data[dmap]

    """

    # Allocate lists to create data to/from dicts.
    src_list = []
    freq_list = []
    rec_list = []
    data_dict = {}
    rec_uid = {}
    indices = np.zeros((survey.nD, 3), dtype=int)

    # Counter for SimPEG data object (lists the data continuously).
    ind = 0

    # Loop over sources.
    for src in survey.source_list:

        # Create emg3d source.
        if isinstance(src, ElectricWire):
            source = emg3d.TxElectricWire(
                src.locations,
                strength=src.strength
            )
        else:
            source = emg3d.TxElectricDipole(
                (*src.location, src.azimuth, src.elevation),
                strength=src.strength, length=src.length
            )

        # New frequency: add.
        if src.frequency not in freq_list:
            f_ind = len(freq_list)
            freq_list.append(src.frequency)

        # Existing source: get index.
        else:
            f_ind = freq_list.index(src.frequency)

        # New source: add.
        if source not in src_list:
            s_ind = len(src_list)
            data_dict[s_ind] = {f_ind: {}}
            src_list.append(source)

        # Existing source: get index.
        else:
            s_ind = src_list.index(source)

            # If new frequency for existing source, add:
            if f_ind not in data_dict[s_ind].keys():
                data_dict[s_ind][f_ind] = {}

        # Loop over receiver lists.
        rec_types = [emg3d.RxElectricPoint, emg3d.RxMagneticPoint]
        for rec in src.receiver_list:

            # If this SimPEG receiver was already processed, store it.
            if rec._uid in rec_uid.keys():
                li = len(rec_uid[rec._uid])
                indices[ind:ind+li, 0] = s_ind
                indices[ind:ind+li, 1] = rec_uid[rec._uid]
                indices[ind:ind+li, 2] = f_ind
                ind += li
                continue
            else:
                rec_uid[rec._uid] = []

            if rec.projField not in ['e', 'h']:
                raise NotImplementedError(
                    "Only projField = {'e'; 'h'} implemented."
                )

            if rec.orientation not in ['x', 'y', 'z']:
                raise NotImplementedError(
                    "Only orientation = {'x'; 'y'; 'z'} implemented."
                )

            # Get type, azimuth, elevation.
            rec_type = rec_types[rec.projField == 'h']
            azimuth = [0, 90][rec.orientation == 'y']
            elevation = [0, 90][rec.orientation == 'z']

            # Loop over receivers.
            for i in range(rec.locations[:, 0].size):

                # Create emg3d receiver.
                receiver = rec_type(
                        (*rec.locations[i, :], azimuth, elevation))

                # New receiver: add.
                if receiver not in rec_list:
                    r_ind = len(rec_list)
                    data_dict[s_ind][f_ind][r_ind] = ind
                    rec_list.append(receiver)

                # Existing receiver: get index.
                else:
                    r_ind = rec_list.index(receiver)

                    # If new receiver for existing src-freq, add:
                    existing = data_dict[s_ind][f_ind].keys()
                    if r_ind not in existing:
                        data_dict[s_ind][f_ind][r_ind] = ind

                    # Else, throw an error.
                    else:
                        raise ValueError(
                            "Duplicate source-receiver-frequency."
                        )

                # Store receiver index, in case the entire receiver
                # is used several times.
                rec_uid[rec._uid].append(r_ind)

                # Store the SimPEG<->emg3d mapping for this receiver
                indices[ind, :] = [s_ind, r_ind, f_ind]
                ind += 1

    # Create and store survey.
    emg3d_survey = emg3d.Survey(
        name='Survey created by SimPEG',
        sources=emg3d.surveys.txrx_lists_to_dict(src_list),
        receivers=emg3d.surveys.txrx_lists_to_dict(rec_list),
        frequencies=freq_list,
        noise_floor=1.,       # We deal with std in SimPEG.
        relative_error=None,  #  "   "   "
    )

    # Store data-mapping SimPEG <-> emg3d
    data_map = tuple(indices.T)

    # Add reverse map to emg3d-data (is saved with survey).
    ind = np.full(emg3d_survey.shape, -1)
    ind[data_map] = np.arange(survey.nD)
    emg3d_survey.data['indices'] = emg3d_survey.data.observed.copy(data=ind)

    return emg3d_survey, data_map
