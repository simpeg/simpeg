import numpy as np
import scipy.sparse as sp
from scipy.constants import mu_0
import properties
from ...utils.code_utils import deprecate_class

from ... import props
from ...data import Data
from ...utils import mkvc
from .simulation import BaseFDEMSimulation
from ..utils import omega
from .survey import Survey
from .fields import (
    Fields3DElectricField,
)
import emg3d
from emg3d import models, surveys, simulations, optimize


class Simulation3DEMG3D(BaseFDEMSimulation):
    """
    By eliminating the magnetic flux density using

        .. math ::

            \mathbf{b} = \\frac{1}{i \omega}\\left(-\mathbf{C} \mathbf{e} +
            \mathbf{s_m}\\right)


    we can write Maxwell's equations as a second order system in
    \\\(\\\mathbf{e}\\\) only:

    .. math ::

        \\left(\mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f} \mathbf{C} +
        i \omega \mathbf{M^e_{\sigma}} \\right)\mathbf{e} =
        \mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f}\mathbf{s_m}
        - i\omega\mathbf{M^e}\mathbf{s_e}

    which we solve for :math:`\mathbf{e}`.

    :param discretize.base.BaseMesh mesh: mesh
    """

    _solutionType = "eSolution"
    _formulation = "EB"
    fieldsPair = Fields3DElectricField
    max_workers = 1
    storeJ = False
    _Jmatrix = None

    def __init__(self, mesh, **kwargs):
        super(Simulation3DEMG3D, self).__init__(mesh, **kwargs)

    @property
    def emg3d_survey(self):
        if getattr(self, "_emg3d_survey", None) is None:
            self._emg3d_survey = self.generate_emg3d_survey_from_simpeg_survey()
        return self._emg3d_survey

    @property
    def emg3d_sigma(self):
        self._emg3d_sigma = models.Model(
            self.mesh,
            self.sigma.reshape(self.mesh.vnC, order='F'),
            mapping='Conductivity'
        )
        return self._emg3d_sigma

    @ property
    def emg3d_simulation_inputs(self):
        if getattr(self, "_emg3d_simulation_inputs", None) is None:
            solver_opts = {
                'sslsolver': True,
                'semicoarsening': True,
                'linerelaxation': True,
            #     'tol': 5e-5,  # Reduce tolerance to speed-up
            }
            self._emg3d_simulation_inputs = {
                'name': 'Testing',
                'survey': self.emg3d_survey,
                'solver_opts': solver_opts,
                'max_workers': self.max_workers,
                'gridding': 'same',
                'verb': -1,
                'receiver_interpolation': 'linear',
                'tqdm_opts': {'disable': True},  # Avoid verb. for inv.
            }
        return self._emg3d_simulation_inputs


    def generate_emg3d_survey_from_simpeg_survey(self):

        source_list_emg3d = []
        src_rx_uids = []
        source_frequency = []
        for src in self.survey.srcList:
            source_frequency.append(src.frequency)
            src_emg3d = emg3d.TxElectricDipole(
                (src.location[0], src.location[1], src.location[2], src.azimuth, src.dip)
            )
            source_list_emg3d.append(src_emg3d)
            for rx in src.rxList:
                src_rx_uids.append([src._uid, rx._uid])
        src_rx_uids = np.vstack(src_rx_uids)

        # assume all sources have the same receiver locations now
        if np.unique(src_rx_uids[:,1]).size == 1:
            # also assume rx only contains a single component
            if rx.projField == 'e':
                emg3d_rx_object = emg3d.RxElectricPoint
            elif rx.projField == 'h':
                emg3d_rx_object = emg3d.RxMagneticPoint

            if rx.orientation == 'x':
                azimuth, dip = 0, 0
            elif rx.orientation == 'y':
                azimuth, dip = 90, 0
            elif rx.orientation == 'z':
                azimuth, dip = 0, 90
            else:
                raise Exception()
            receivers = emg3d.surveys.txrx_coordinates_to_dict(
                emg3d_rx_object,
                (rx.locations[:,0], rx.locations[:,1], rx.locations[:,2], azimuth, dip),
            )
        else:
            raise Exception()
        sources = emg3d.surveys.txrx_lists_to_dict(source_list_emg3d)
        source_frequency = np.unique(np.array(source_frequency))
        emg3d_survey = surveys.Survey(
            name='emg3d survey',
            sources=sources,
            receivers=receivers,
            frequencies=source_frequency,
            noise_floor=1.,
            relative_error=None,
        )
        self.n_source = len(source_list_emg3d)
        self.n_frequency = len(source_frequency)
        self.n_receiver = rx.locations.shape[0]
        return emg3d_survey

    def Jvec(self, m, v, f=None):
        if self.verbose:
            print ("Compute Jvec")

        if self.storeJ:
            J = self.getJ(m, f=f)
            Jv = mkvc(np.dot(J, v))
            return Jv

        self.model = m

        if f is None:
            f = self.fields(m=m)

        dsig_dm_v = self.sigmaDeriv @ v
        j_vec = optimize.jvec(f, vec=dsig_dm_v)
        return j_vec

    def Jtvec(self, m, v, f=None):
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
        """
            Compute adjoint sensitivity matrix (J^T) and vector (v) product.
            Full J matrix can be computed by inputing v=None
        """

        if v is not None:

            jt_sigma_vec = optimize.gradient(f, vec=v)
            jt_vec = self.sigmaDeriv.T @ jt_sigma_vec
            return jt_vec

        else:
            # This is for forming full sensitivity matrix
            # Currently, it is not correct.
            # Requires a fix in optimize.gradient
            # Jt is supposed to be a complex value ...
            Jt = np.zeros((self.model.size, self.survey.nD), order="F")
            for i_datum in range(self.survey.nD):
                v = np.zeros(self.survey.nD)
                v[i_datum] = 1.
                jt_sigma_vec = optimize.gradient(f, vec=v)
                Jt[:, i_datum] = self.sigmaDeriv.T @ jt_sigma_vec

            return Jt

    def getJ(self, m, f=None):
        """
            Generate Full sensitivity matrix
        """
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
        if self.verbose:
            print("Compute predicted")
        # currently this does not change once self.fields is computed.
        if f is None:
            f = self.fields(m=m)
        # this may not be robust, need to be changed...
        data = f.data.synthetic.values.flatten()
        return data

    def fields(self, m=None):
        if self.verbose:
            print("Compute fields")
        self.model = m

        f = simulations.Simulation(
            model=self.emg3d_sigma,
            **self.emg3d_simulation_inputs
        )
        std = f.survey.standard_deviation
        # Store weights
        f.data['weights'] = std**-2
        f.compute()
        return f
