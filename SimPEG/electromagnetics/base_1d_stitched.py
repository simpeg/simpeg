from __future__ import annotations

from scipy.constants import mu_0
import scipy.sparse as sp
import numpy as np
from .. import props
from .. import utils
from ..utils.code_utils import (
    validate_integer,
    validate_ndarray_with_shape,
    validate_type,
)

from ..simulation import BaseSimulation
from ..survey import BaseSurvey

from multiprocessing import cpu_count, Pool
from .base_1d import run_em1d_simulation, OutputType, ColeColeParameters


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
    _simulation_type: type(BaseSimulation)
    _survey_type: type(BaseSurvey)

    # Properties for electrical conductivity/resistivity
    sigma, sigmaMap, sigmaDeriv = props.Invertible(
        "Electrical conductivity at infinite frequency (S/m)"
    )
    # Properties for magnetic susceptibility
    mu, muMap, muDeriv = props.Invertible(
        "Magnetic permeability at infinite frequency (SI)"
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
        cole_cole_parameters: ColeColeParameters | None = None,
        fix_Jmatrix=False,
        topo=None,
        parallel=False,
        n_cpu=None,
        **kwargs,
    ):
        super().__init__(mesh=None, **kwargs)
        self._cole_cole_parameters: ColeColeParameters | None = None
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
        self.cole_cole_parameters = cole_cole_parameters
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
                    self.n_cpu = cpu_count()
                print((">> n_cpu: %i") % (self.n_cpu))
        else:
            if self.verbose:
                print(">> Serial version is used")

    @property
    def cole_cole_parameters(self) -> ColeColeParameters | None:
        """
        Physical properties defining the Cole-Cole model.
        """
        return self._cole_cole_parameters

    @cole_cole_parameters.setter
    def cole_cole_parameters(self, value):
        if value is None:
            value = ColeColeParameters()

        if not isinstance(value, ColeColeParameters):
            raise TypeError(
                f"cole_cole_parameters must be of type ColeColeParameters, {type(value)} provided."
            )

        self._cole_cole_parameters = value

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
            if isinstance(self.cole_cole_parameters.eta, float):
                self._eta_matrix = self.cole_cole_parameters.eta * np.ones(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._eta_matrix = self.cole_cole_parameters.eta.reshape(
                    (self.n_sounding, self.n_layer)
                )

        return self._eta_matrix

    @property
    def tau_matrix(self):
        if getattr(self, "_tau_matrix", None) is None:
            # Ordering: first z then x
            if isinstance(self.cole_cole_parameters.tau, float):
                self._tau_matrix = self.cole_cole_parameters.tau * np.ones(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._tau_matrix = self.cole_cole_parameters.tau.reshape(
                    (self.n_sounding, self.n_layer)
                )

        return self._tau_matrix

    @property
    def c_matrix(self):
        if getattr(self, "_c_matrix", None) is None:
            # Ordering: first z then x
            if isinstance(self.cole_cole_parameters.c, float):
                self._c_matrix = self.cole_cole_parameters.c * np.ones(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._c_matrix = self.cole_cole_parameters.c.reshape(
                    (self.n_sounding, self.n_layer)
                )

        return self._c_matrix

    @property
    def chi_matrix(self):
        if getattr(self, "_chi_matrix", None) is None:
            # Ordering: first z then x
            if isinstance(self.cole_cole_parameters.chi, float):
                self._chi_matrix = self.cole_cole_parameters.chi * np.ones(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._chi_matrix = self.cole_cole_parameters.chi.reshape(
                    (self.n_sounding, self.n_layer)
                )

        return self._chi_matrix

    @property
    def tau1_matrix(self):
        if getattr(self, "_tau1_matrix", None) is None:
            # Ordering: first z then x
            if isinstance(self.cole_cole_parameters.tau1, float):
                self._tau1_matrix = self.cole_cole_parameters.tau1 * np.ones(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._tau1_matrix = self.cole_cole_parameters.tau1.reshape(
                    (self.n_sounding, self.n_layer)
                )

        return self._tau1_matrix

    @property
    def tau2_matrix(self):
        if getattr(self, "_tau2_matrix", None) is None:
            # Ordering: first z then x
            if isinstance(self.cole_cole_parameters.tau2, float):
                self._tau2_matrix = self.cole_cole_parameters.tau2 * np.ones(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._tau2_matrix = self.cole_cole_parameters.tau2.reshape(
                    (self.n_sounding, self.n_layer)
                )

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

    def input_args(self, i_sounding, output_type=OutputType.FORWARD):
        output = (
            self._simulation_type,
            self._survey_type,
            self.survey.get_sources_by_sounding_number(i_sounding),
            self.topo[i_sounding, :],
            self.thickness_matrix[i_sounding, :],
            self.sigma_matrix[i_sounding, :],
            self.eta_matrix[i_sounding, :],
            self.tau_matrix[i_sounding, :],
            self.c_matrix[i_sounding, :],
            self.h_vector[i_sounding],
            output_type,
        )
        return output

    def fields(self, m=None):
        if self.verbose:
            print("Compute fields")

        return self.forward(m)

    def dpred(self, m=None, f=None):
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

    def forward(self, m):
        self.model = m

        if self.verbose:
            print(">> Compute response")

        if self.parallel:
            if self.verbose:
                print("parallel")
            # This assumes the same # of layers for each of sounding
            # if self.n_sounding_for_chunk is None:
            pool = Pool(self.n_cpu)
            result = pool.map(
                run_em1d_simulation,
                [
                    self.input_args(i, output_type=OutputType.FORWARD)
                    for i in range(self.n_sounding)
                ],
            )

            pool.close()
            pool.join()
        else:
            result = [
                run_em1d_simulation(*self.input_args(i, output_type=OutputType.FORWARD))
                for i in range(self.n_sounding)
            ]

        return np.hstack(result)

    def getJ(self, m):
        """
        Compute d F / d sigma
        """
        self.model = m
        if getattr(self, "_J", None) is None:
            if self.verbose:
                print(">> Compute J")

            if self.parallel:
                if self.verbose:
                    print(">> Start pooling")

                pool = Pool(self.n_cpu)

                # Deprecate this for now, but revisit later
                # It is an idea of chunking for parallelization
                # if self.n_sounding_for_chunk is None:
                self._J = pool.map(
                    run_em1d_simulation,
                    [
                        self.input_args(i, output_type=OutputType.SENSITIVITY)
                        for i in range(self.n_sounding)
                    ],
                )

                if self.verbose:
                    print(">> End pooling and form J matrix")

            else:
                self._J = [
                    run_em1d_simulation(
                        *self.input_args(i, output_type=OutputType.SENSITIVITY)
                    )
                    for i in range(self.n_sounding)
                ]
        return self._J

    def getJ_sigma(self, m):
        """
        Compute d F / d sigma
        """
        if getattr(self, "_Jmatrix_sigma", None) is None:
            J = self.getJ(m)
            self._Jmatrix_sigma = np.hstack(
                [utils.mkvc(J[i]["ds"]) for i in range(self.n_sounding)]
            )
            self._Jmatrix_sigma = sp.coo_matrix(
                (self._Jmatrix_sigma, self.IJLayers), dtype=float
            ).tocsr()
        return self._Jmatrix_sigma

    def getJ_height(self, m):
        if getattr(self, "_Jmatrix_height", None) is None:
            J = self.getJ(m)
            self._Jmatrix_height = np.hstack(
                [utils.mkvc(J[i]["dh"]) for i in range(self.n_sounding)]
            )
            self._Jmatrix_height = sp.coo_matrix(
                (self._Jmatrix_height, self.IJHeight), dtype=float
            ).tocsr()
        return self._Jmatrix_height
