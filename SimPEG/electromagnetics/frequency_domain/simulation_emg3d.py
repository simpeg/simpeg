import emg3d
import numpy as np

from ...utils import mkvc
from .simulation import BaseFDEMSimulation
from .fields import Fields3DElectricField


# TODO Have to make emg3d optional and check if it is installed or not, fail
#      graciously otherwise (e.g., like matplotlib in discretize).

class Simulation3DEMG3D(BaseFDEMSimulation):
    """Same as Simulation3DElectricField, but using emg3d as solver.

    General thought
    ---------------
    I still think we should simplify this and generalize this in such a way
    that the regular class `Simulation3DElectricField` can be used, with
    `solver=emg3d` and the `solver_opts` used as `simulation_opts`.

    Maybe as a mixin.


    Notes regarding input parameters
    --------------------------------

    - ``solver`` is ignored.
    - ``simulation_opts`` can contain any parameter accepted in
      ``emg3d.Simulation``.


    Notes
    -----

    - Only isotropic model implemented so far. Needs extension to tri-axial
      anisotropy, mu_r, and epsilon_r.

    - Currently, only electric "point"-dipole sources and electric/magnetic
      point receivers are implemented. emg3d could do more, e.g., finite length
      electric dipoles or arbitrary electric wires (and therefore loops). These
      are *not yet* implemented here.

    - emg3d currently supports two receiver placement types (which can be mixed
      in an emg3d-survey):

      - Positioned absolutely;
      - Positioned relative to source position.

      Here are currently only absolutely positioned receivers implemented.

      Source, receiver, and frequencies are stored in an nsrc x nrec x nfreq
      xarray in emg3d. Data  for non-existing source-receiver-frequency
      combination contain NaN's.

    - ...

    """

    _solutionType = "eSolution"
    _formulation = "EB"
    fieldsPair = Fields3DElectricField  # <= TODO not used yet (see fields-fct)

    storeJ = False
    _Jmatrix = None

    def __init__(self, mesh, **kwargs):
        """Initialize Simulation using emg3d as solver."""
        # Store simulation options
        self.simulation_opts = kwargs.pop('simulation_opts', {})

        unknown = ['solver', 'solver_opts', 'Solver', 'solverOpts']
        for keyword in unknown:
            if keyword in kwargs:
                raise AttributeError(
                    f"Keyword input '{keyword}' is not a known input for "
                    "Simulation3DEMG3D"
                )
        # Should we raise a Warning if `solver` or `solver_opts` is provided?
        super().__init__(mesh, **kwargs)

    @property
    def emg3d_survey(self):
        """emg3d survey; obtained from SimPEG survey."""

        if getattr(self, "_emg3d_survey", None) is None:

            # TODO: Should link to src_id, rec_id, i; store.
            #       This could then be used to model emg3d-data to SimPEG data.
            src_dict = {}

            src_list = []
            freq_list = []
            rec_list = []

            # Loop over sources.
            for src in self.survey.source_list:

                # Create emg3d source.
                source = emg3d.TxElectricDipole(
                    (*src.location, src.azimuth, src.elevation),
                    strength=src.strength, length=src.length
                )

                # New source: add.
                if source not in src_list:
                    s_ind = len(src_list)
                    src_dict[s_ind] = {src.frequency: {}}
                    src_list.append(source)

                # Existing source: get index.
                else:
                    s_ind = src_list.index(source)

                    # If new frequency for existing source, add:
                    if src.frequency not in src_dict[s_ind].keys():
                        src_dict[s_ind][src.frequency] = {}

                # New frequency: add.
                if src.frequency not in freq_list:
                    freq_list.append(src.frequency)

                # Loop over receiver lists.
                rec_types = [emg3d.RxElectricPoint, emg3d.RxMagneticPoint]
                for rec in src.receiver_list:

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
                            src_dict[s_ind][src.frequency][r_ind] = True
                            rec_list.append(receiver)

                        # Existing receiver: get index.
                        else:
                            r_ind = rec_list.index(receiver)

                            # If new receiver for existing src-freq, add:
                            existing = src_dict[s_ind][src.frequency].keys()
                            if r_ind not in existing:
                                src_dict[s_ind][src.frequency][r_ind] = True

                            # Else, throw an error.
                            else:
                                raise ValueError(
                                    "Duplicate source-receiver-frequency."
                                )

            # Create and store survey.
            self._emg3d_survey = emg3d.Survey(
                name='Survey created by SimPEG',
                sources=emg3d.surveys.txrx_lists_to_dict(src_list),
                receivers=emg3d.surveys.txrx_lists_to_dict(rec_list),
                frequencies=freq_list,
                noise_floor=1.,       # We deal with std in SimPEG.
                relative_error=None,  #  "   "   "
            )

        return self._emg3d_survey

    @property
    def emg3d_model(self):
        """emg3d conductivity model; obtained from SimPEG conductivities."""
        self._emg3d_model = emg3d.Model(
            self.mesh,
            property_x=self.sigma.reshape(self.mesh.vnC, order='F'),
            # property_y=None,  Not yet implemented
            # property_z=None,   "
            # mu_r=None,         "
            # epsilon_r=None,    "
            mapping='Conductivity',
        )
        return self._emg3d_model

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
        return j_vec

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
            J = self.getJ(m, f=f)
            Jtv = mkvc(np.dot(J.T, v))
            return Jtv

        self.model = m

        if f is None:
            f = self.fields(m=m)

        return self._Jtvec(m, v=v, f=f)

    def _Jtvec(self, m, v=None, f=None):
        """Compute adjoint sensitivity matrix (J^T) and vector (v) product.

        Full J matrix can be computed by setting v=None (not implemented yet).
        """

        if v is not None:

            vec = v.reshape(self.emg3d_survey.shape)
            jt_sigma_vec = emg3d.optimize.gradient(f, vector=vec)
            jt_vec = self.sigmaDeriv.T @ jt_sigma_vec.ravel('F')
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

        # TODO this does not work for various frequency, or for
        # source-receiver-frequency pairs without data, etc. Needs more
        # generalisations and using the actual Data object.

        if self.verbose:
            print("Compute predicted")

        # currently this does not change once self.fields is computed.
        if f is None:
            f = self.fields(m=m)

        # this may not be robust, need to be changed...
        data = f.data.synthetic.values.flatten()

        return data

    def fields(self, m=None):
        """Return the electric fields for a given model.

        :param numpy.ndarray m: model
        :rtype: numpy.ndarray
        :return: f, the fields
        """

        # TODO 1: The `fields` should return the fields, not the simulation,
        #         using the fancy field storage fieldsPair.
        #
        # TODO 2: MOVE SIMULATION out of fields
        #         should only be generated ONCE, then we just replace parts of
        #         it. Also, combine it with survey?

        if self.verbose:
            print("Compute fields")

        self.model = m

        # Default values for inputs which can be overwritten by simulation_opts
        sim_input = {
            'name': 'Simulation created by SimPEG',
            'gridding': 'same',                  # Change this eventually!
            'tqdm_opts': {'disable': True},      # Switch-off tqdm
            'receiver_interpolation': 'linear',  # Should be linear
            **self.simulation_opts,
        }
        print(sim_input)

        sim = emg3d.Simulation(
            survey=self.emg3d_survey,
            model=self.emg3d_model,
            **sim_input,
        )

        # Store weights  # WHY???
        std = sim.survey.standard_deviation
        sim.data['weights'] = std**-2

        sim.compute()
        return sim
