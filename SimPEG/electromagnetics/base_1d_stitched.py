from scipy.constants import mu_0
import numpy as np
from ..simulation import BaseSimulation
from .. import props
from .. import utils
from ..utils.code_utils import (
    validate_integer,
    validate_ndarray_with_shape,
    validate_type,
)

###############################################################################
#                                                                             #
#                             BaseStitchedEM1DSimulation                      #
#                                                                             #
###############################################################################

__all__ = ["BaseStitchedEM1DSimulation"]


class BaseStitchedEM1DSimulation(BaseSimulation):
    """
    Base class for the stitched 1D simulation. This simulation models the EM
    response for a set of 1D EM soundings.
    """

    _formulation = "1D"

    # Properties for electrical conductivity/resistivity
    sigma, sigmaMap, sigmaDeriv = props.Invertible(
        "Electrical conductivity at infinite frequency (S/m)"
    )

    eta = props.PhysicalProperty("Intrinsic chargeability (V/V), 0 <= eta < 1")
    tau = props.PhysicalProperty("Time constant for Cole-Cole model (s)")
    c = props.PhysicalProperty("Frequency Dependency for Cole-Cole model, 0 < c < 1")

    # Properties for magnetic susceptibility
    mu, muMap, muDeriv = props.Invertible(
        "Magnetic permeability at infinite frequency (SI)"
    )
    chi = props.PhysicalProperty(
        "DC magnetic susceptibility for viscous remanent magnetization contribution (SI)"
    )
    tau1 = props.PhysicalProperty(
        "Lower bound for log-uniform distribution of time-relaxation constants for viscous remanent magnetization (s)"
    )
    tau2 = props.PhysicalProperty(
        "Upper bound for log-uniform distribution of time-relaxation constants for viscous remanent magnetization (s)"
    )

    # Additional properties
    h, hMap, hDeriv = props.Invertible("Receiver Height (m), h > 0")

    thicknesses, thicknessesMap, thicknessesDeriv = props.Invertible(
        "layer thicknesses (m)"
    )

    def __init__(
        self,
        sigma=None,
        sigmaMap=None,
        thicknesses=None,
        thicknessesMap=None,
        mu=mu_0,
        muMap=None,
        h=None,
        hMap=None,
        eta=None,
        tau=None,
        c=None,
        dchi=None,
        tau1=None,
        tau2=None,
        fix_Jmatrix=False,
        topo=None,
        parallel=False,
        n_cpu=None,
        **kwargs,
    ):
        super().__init__(mesh=None, **kwargs)
        self.sigma = sigma
        self.sigmaMap = sigmaMap
        self.mu = mu
        self.muMap = muMap
        self.h = h
        self.hMap = hMap
        if thicknesses is None:
            thicknesses = np.array([])
        self.thicknesses = thicknesses
        self.thicknessesMap = thicknessesMap
        self.eta = eta
        self.tau = tau
        self.c = c
        self.dchi = dchi
        self.tau1 = tau1
        self.tau2 = tau2
        self.fix_Jmatrix = fix_Jmatrix
        self.topo = topo
        if self.topo is None:
            self.set_null_topography()

        self.parallel = parallel
        self.n_cpu = n_cpu

        if self.parallel:
            if self.verbose:
                print(">> Use multiprocessing for parallelization")
                if self.n_cpu is None:
                    self.n_cpu = multiprocessing.cpu_count()
                print((">> n_cpu: %i") % (self.n_cpu))
        else:
            if self.verbose:
                print(">> Serial version is used")

    @property
    def fix_Jmatrix(self):
        """Whether to fix the sensitivity matrix.

        Returns
        -------
        bool
        """
        return self._fix_Jmatrix

    @fix_Jmatrix.setter
    def fix_Jmatrix(self, value):
        self._fix_Jmatrix = validate_type("fix_Jmatrix", value, bool)

    @property
    def topo(self):
        """Topography.

        Returns
        -------
        numpy.ndarray of float
        """
        return self._topo

    @topo.setter
    def topo(self, value):
        self._topo = validate_ndarray_with_shape("topo", value, shape=("*", 3))

    @property
    def parallel(self):
        """Parallel

        Returns
        -------
        bool
        """
        return self._parallel

    @parallel.setter
    def parallel(self, value):
        self._parallel = validate_type("parallel", value, bool)

    @property
    def n_cpu(self):
        """Number of cpus

        Returns
        -------
        int
        """
        return self._n_cpu

    @n_cpu.setter
    def n_cpu(self, value):
        self._n_cpu = validate_integer("n_cpu", value, min_val=1)

    @property
    def invert_height(self):
        if self.hMap is None:
            return False
        else:
            return True

    @property
    def halfspace_switch(self):
        """True = halfspace, False = layered Earth"""
        if (self.thicknesses is None) | (len(self.thicknesses) == 0):
            return True
        else:
            return False

    @property
    def n_layer(self):
        if self.thicknesses is None:
            return 1
        else:
            return len(self.thicknesses) + 1

    @property
    def n_sounding(self):
        return len(self.survey.source_location_by_sounding_dict)

    @property
    def data_index(self):
        return self.survey.data_index

    # ------------- For physical properties ------------- #
    @property
    def Sigma(self):
        if getattr(self, "_Sigma", None) is None:
            # Ordering: first z then x
            self._Sigma = self.sigma.reshape((self.n_sounding, self.n_layer))
        return self._Sigma

    @property
    def Thicknesses(self):
        if getattr(self, "_Thicknesses", None) is None:
            # Ordering: first z then x
            if len(self.thicknesses) == int(self.n_sounding * (self.n_layer - 1)):
                self._Thicknesses = self.thicknesses.reshape(
                    (self.n_sounding, self.n_layer - 1)
                )
            else:
                self._Thicknesses = np.tile(self.thicknesses, (self.n_sounding, 1))
        return self._Thicknesses

    @property
    def Eta(self):
        if getattr(self, "_Eta", None) is None:
            # Ordering: first z then x
            if self.eta is None:
                self._Eta = np.zeros(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._Eta = self.eta.reshape((self.n_sounding, self.n_layer))
        return self._Eta

    @property
    def Tau(self):
        if getattr(self, "_Tau", None) is None:
            # Ordering: first z then x
            if self.tau is None:
                self._Tau = 1e-3 * np.ones(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._Tau = self.tau.reshape((self.n_sounding, self.n_layer))
        return self._Tau

    @property
    def C(self):
        if getattr(self, "_C", None) is None:
            # Ordering: first z then x
            if self.c is None:
                self._C = np.ones(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._C = self.c.reshape((self.n_sounding, self.n_layer))
        return self._C

    @property
    def Chi(self):
        if getattr(self, "_Chi", None) is None:
            # Ordering: first z then x
            if self.chi is None:
                self._Chi = np.zeros(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._Chi = self.chi.reshape((self.n_sounding, self.n_layer))
        return self._Chi

    @property
    def dChi(self):
        if getattr(self, "_dChi", None) is None:
            # Ordering: first z then x
            if self.dchi is None:
                self._dChi = np.zeros(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._dChi = self.dchi.reshape((self.n_sounding, self.n_layer))
        return self._dChi

    @property
    def Tau1(self):
        if getattr(self, "_Tau1", None) is None:
            # Ordering: first z then x
            if self.tau1 is None:
                self._Tau1 = 1e-10 * np.ones(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._Tau1 = self.tau1.reshape((self.n_sounding, self.n_layer))
        return self._Tau1

    @property
    def Tau2(self):
        if getattr(self, "_Tau2", None) is None:
            # Ordering: first z then x
            if self.tau2 is None:
                self._Tau2 = 100.0 * np.ones(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._Tau2 = self.tau2.reshape((self.n_sounding, self.n_layer))
        return self._Tau2

    @property
    def JtJ_sigma(self):
        return self._JtJ_sigma

    def JtJ_height(self):
        return self._JtJ_height

    @property
    def H(self):
        if self.hMap is None:
            h = self.source_locations_for_sounding[:, 2] - self.topo[:, 2]
            return h
        else:
            return self.h

    # ------------- Etcetra .... ------------- #
    @property
    def IJLayers(self):
        if getattr(self, "_IJLayers", None) is None:
            # Ordering: first z then x
            self._IJLayers = self.set_ij_n_layer()
        return self._IJLayers

    @property
    def IJHeight(self):
        if getattr(self, "_IJHeight", None) is None:
            # Ordering: first z then x
            self._IJHeight = self.set_ij_n_layer(n_layer=1)
        return self._IJHeight

    # ------------- For physics ------------- #

    def get_uniq_soundings(self):
        self._sounding_types_uniq, self._ind_sounding_uniq = np.unique(
            self.survey._sounding_types, return_index=True
        )

    def input_args(self, i_sounding, output_type="forward"):
        output = (
            self.survey.get_sources_by_sounding_number(i_sounding),
            self.topo[i_sounding, :],
            self.Thicknesses[i_sounding, :],
            self.Sigma[i_sounding, :],
            self.Eta[i_sounding, :],
            self.Tau[i_sounding, :],
            self.C[i_sounding, :],
            self.Chi[i_sounding, :],
            self.dChi[i_sounding, :],
            self.Tau1[i_sounding, :],
            self.Tau2[i_sounding, :],
            self.H[i_sounding],
            output_type,
            # False,
            # self._coefficients[i_sounding],
        )
        return output

    # This is the most expensive process, but required once
    # May need to find unique
    # def input_args_for_coeff(self, i_sounding):
    #     output = (
    #         self.survey.get_sources_by_sounding_number(i_sounding),
    #         self.topo[i_sounding, :],
    #         self.Thicknesses[i_sounding,:],
    #         self.Sigma[i_sounding, :],
    #         self.Eta[i_sounding, :],
    #         self.Tau[i_sounding, :],
    #         self.C[i_sounding, :],
    #         self.Chi[i_sounding, :],
    #         self.dChi[i_sounding, :],
    #         self.Tau1[i_sounding, :],
    #         self.Tau2[i_sounding, :],
    #         self.H[i_sounding],
    #         'forward',
    #         True,
    #         [],
    #     )
    #     return output

    def fields(self, m):
        if self.verbose:
            print("Compute fields")

        return self.forward(m)

    def dpred(self, m, f=None):
        """
        Return predicted data.
        Predicted data, (`_pred`) are computed when
        self.fields is called.
        """
        if f is None:
            f = self.fields(m)

        return f

    @property
    def sounding_number(self):
        self._sounding_number = [
            key for key in self.survey.source_location_by_sounding_dict.keys()
        ]
        return self._sounding_number

    @property
    def n_chunk(self):
        self._n_chunk = len(self.sounding_number_chunks)
        return self._n_chunk

    @property
    def source_locations_for_sounding(self):
        if getattr(self, "_source_locations_for_sounding", None) is None:
            self._source_locations_for_sounding = np.vstack(
                [
                    self.survey._source_location_by_sounding_dict[ii][0]
                    for ii in range(self.n_sounding)
                ]
            )
        return self._source_locations_for_sounding

    # def chunks(self, lst, n):
    #     """Yield successive n-sized chunks from lst."""
    #     for i in range(0, len(lst), n):
    #         yield lst[i:i + n]

    # @property
    # def sounding_number_chunks(self):
    #     self._sounding_number_chunks = list(self.chunks(self.sounding_number, self.n_sounding_for_chunk))
    #     return self._sounding_number_chunks

    # def input_args_by_chunk(self, i_chunk, output_type):
    #     args_by_chunks = []
    #     for i_sounding in self.sounding_number_chunks[i_chunk]:
    #         args_by_chunks.append(self.input_args(i_sounding, output_type))
    #     return args_by_chunks

    def set_null_topography(self):
        self.topo = self.source_locations_for_sounding.copy()
        self.topo[:, 2] = 0.0

    def set_ij_n_layer(self, n_layer=None):
        """
        Compute (I, J) indicies to form sparse sensitivity matrix
        This will be used in GlobalEM1DSimulation when after sensitivity matrix
        for each sounding is computed
        """
        I = []
        J = []
        shift_for_J = 0
        shift_for_I = 0
        if n_layer is None:
            m = self.n_layer
        else:
            m = n_layer

        for i_sounding in range(self.n_sounding):
            n = self.survey.vnD_by_sounding_dict[i_sounding]
            J_temp = np.tile(np.arange(m), (n, 1)) + shift_for_J
            I_temp = (
                np.tile(np.arange(n), (1, m)).reshape((n, m), order="F") + shift_for_I
            )
            J.append(utils.mkvc(J_temp))
            I.append(utils.mkvc(I_temp))
            shift_for_J += m
            shift_for_I = I_temp[-1, -1] + 1
        J = np.hstack(J).astype(int)
        I = np.hstack(I).astype(int)
        return (I, J)

    def set_ij_height(self):
        """
        Compute (I, J) indicies to form sparse sensitivity matrix
        This will be used in GlobalEM1DSimulation when after sensitivity matrix
        for each sounding is computed
        """
        J = []
        I = np.arange(self.survey.nD)
        for i_sounding in range(self.n_sounding):
            n = self.survey.vnD_by_sounding_dict[i_sounding]
            J.append(np.ones(n) * i_sounding)
        J = np.hstack(J).astype(int)
        return (I, J)

    def Jvec(self, m, v, f=None):
        J_sigma = self.getJ_sigma(m)
        Jv = J_sigma @ (self.sigmaDeriv @ v)
        if self.hMap is not None:
            J_height = self.getJ_height(m)
            Jv += J_height @ (self.hDeriv @ v)
        return Jv

    def Jtvec(self, m, v, f=None):
        J_sigma = self.getJ_sigma(m)
        Jtv = self.sigmaDeriv.T @ (J_sigma.T @ v)
        if self.hMap is not None:
            J_height = self.getJ_height(m)
            Jtv += self.hDeriv.T @ (J_height.T @ v)
        return Jtv

    # Revisit this
    def getJtJdiag(self, m, W=None, threshold=1e-8):
        """
        Compute diagonal component of JtJ or
        trace of sensitivity matrix (J)
        """
        if getattr(self, "_gtgdiag", None) is None:
            J_sigma = self.getJ_sigma(m)
            J_matrix = J_sigma @ (self.sigmaDeriv)

            if self.hMap is not None:
                J_height = self.getJ_height(m)
                J_matrix += J_height * self.hDeriv

            if W is None:
                W = utils.speye(J_matrix.shape[0])
            J_matrix = W * J_matrix
            gtgdiag = (J_matrix.T * J_matrix).diagonal()
            gtgdiag /= gtgdiag.max()
            gtgdiag += threshold
            self._gtgdiag = gtgdiag
        return self._gtgdiag

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = super().deleteTheseOnModelUpdate
        if self.fix_Jmatrix is False:
            toDelete += [
                "_Sigma",
                "_J",
                "_Jmatrix_sigma",
                "_Jmatrix_height",
                "_gtg_diag",
            ]
        return toDelete
