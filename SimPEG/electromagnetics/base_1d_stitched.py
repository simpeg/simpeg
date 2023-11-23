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
        """
        Run the computation as a parallel process.

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
        return len(self.survey.source_location_by_sounding)

    @property
    def data_index(self):
        return self.survey.data_index

    # ------------- For physical properties ------------- #
    @property
    def sigma_matrix(self):
        if getattr(self, "_sigma_matrix", None) is None:
            # Ordering: first z then x
            self._sigma_matrix = self.sigma.reshape((self.n_sounding, self.n_layer))
        return self._sigma_matrix

    @property
    def thickness_matrix(self):
        if getattr(self, "_thickness_matrix", None) is None:
            # Ordering: first z then x
            if len(self.thicknesses) == int(self.n_sounding * (self.n_layer - 1)):
                self._thickness_matrix = self.thicknesses.reshape(
                    (self.n_sounding, self.n_layer - 1)
                )
            else:
                self._thickness_matrix = np.tile(self.thicknesses, (self.n_sounding, 1))
        return self._thickness_matrix

    @property
    def eta_matrix(self):
        if getattr(self, "_eta_matrix", None) is None:
            # Ordering: first z then x
            if self.eta is None:
                self._eta_matrix = np.zeros(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._eta_matrix = self.eta.reshape((self.n_sounding, self.n_layer))
        return self._eta_matrix

    @property
    def tau_matrix(self):
        if getattr(self, "_tau_matrix", None) is None:
            # Ordering: first z then x
            if self.tau is None:
                self._tau_matrix = 1e-3 * np.ones(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._tau_matrix = self.tau.reshape((self.n_sounding, self.n_layer))
        return self._tau_matrix

    @property
    def c_matrix(self):
        if getattr(self, "_c_matrix", None) is None:
            # Ordering: first z then x
            if self.c is None:
                self._c_matrix = np.ones(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._c_matrix = self.c.reshape((self.n_sounding, self.n_layer))
        return self._c_matrix

    @property
    def chi_matrix(self):
        if getattr(self, "_chi_matrix", None) is None:
            # Ordering: first z then x
            if self.chi is None:
                self._chi_matrix = np.zeros(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._chi_matrix = self.chi.reshape((self.n_sounding, self.n_layer))
        return self._chi_matrix

    @property
    def dchi_matrix(self):
        if getattr(self, "_dchi_matrix", None) is None:
            # Ordering: first z then x
            if self.dchi is None:
                self._dchi_matrix = np.zeros(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._dchi_matrix = self.dchi.reshape((self.n_sounding, self.n_layer))
        return self._dchi_matrix

    @property
    def tau1_matrix(self):
        if getattr(self, "_tau1_matrix", None) is None:
            # Ordering: first z then x
            if self.tau1 is None:
                self._tau1_matrix = 1e-10 * np.ones(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._tau1_matrix = self.tau1.reshape((self.n_sounding, self.n_layer))
        return self._tau1_matrix

    @property
    def tau2_matrix(self):
        if getattr(self, "_tau2_matrix", None) is None:
            # Ordering: first z then x
            if self.tau2 is None:
                self._tau2_matrix = 100.0 * np.ones(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._tau2_matrix = self.tau2.reshape((self.n_sounding, self.n_layer))
        return self._tau2_matrix

    @property
    def JtJ_sigma(self):
        return self._JtJ_sigma

    def JtJ_height(self):
        return self._JtJ_height

    @property
    def h_vector(self):
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
            self.thickness_matrix[i_sounding, :],
            self.sigma_matrix[i_sounding, :],
            self.eta_matrix[i_sounding, :],
            self.tau_matrix[i_sounding, :],
            self.c_matrix[i_sounding, :],
            self.chi_matrix[i_sounding, :],
            self.dchi_matrix[i_sounding, :],
            self.tau1_matrix[i_sounding, :],
            self.tau2_matrix[i_sounding, :],
            self.h_vector[i_sounding],
            output_type,
        )
        return output

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
            key for key in self.survey.source_location_by_sounding.keys()
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
                    self.survey._source_location_by_sounding[ii][0]
                    for ii in range(self.n_sounding)
                ]
            )
        return self._source_locations_for_sounding

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
            n = self.survey.vnD_by_sounding[i_sounding]
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
            n = self.survey.vnD_by_sounding[i_sounding]
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
                "_sigma_matrix",
                "_J",
                "_Jmatrix_sigma",
                "_Jmatrix_height",
                "_gtg_diag",
            ]
        return toDelete
