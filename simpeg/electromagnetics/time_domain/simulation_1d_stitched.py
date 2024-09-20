import numpy as np
from scipy import sparse as sp
from ... import utils
from ..base_1d_stitched import BaseStitchedEM1DSimulation
from .simulation_1d import Simulation1DLayered
from .survey import Survey
from ... import maps
from multiprocessing import Pool


def run_simulation_time_domain(args):
    import os

    os.environ["MKL_NUM_THREADS"] = "1"
    """
    This method simulates the EM response or computes the sensitivities for
    a single sounding. The method allows for parallelization of
    the stitched 1D problem.
    :param src: a EM1DTM source object
    :param topo: Topographic location (x, y, z)
    :param np.array thicknesses: np.array(N-1,) layer thicknesses for a single sounding
    :param np.array sigma: np.array(N,) layer conductivities for a single sounding
    :param np.array eta: np.array(N,) intrinsic chargeabilities for a single sounding
    :param np.array tau: np.array(N,) Cole-Cole time constant for a single sounding
    :param np.array c: np.array(N,) Cole-Cole frequency distribution constant for a single sounding
    :param np.array chi: np.array(N,) magnetic susceptibility for a single sounding
    :param np.array dchi: np.array(N,) DC susceptibility for magnetic viscosity for a single sounding
    :param np.array tau1: np.array(N,) lower time-relaxation constant for magnetic viscosity for a single sounding
    :param np.array tau2: np.array(N,) upper time-relaxation constant for magnetic viscosity for a single sounding
    :param float h: source height for a single sounding
    :param list freq_to_time_matricies for a single sounding
    :param list hankel_coefficients for a single sounding
    :param bool is_invert_h: boolean switch for inverting for source height
    :return: response or sensitivities or hankel coefficients or freq_to_time matricies
    """

    (
        source_list,
        topo,
        thicknesses,
        sigma,
        eta,
        tau,
        c,
        chi,
        dchi,
        tau1,
        tau2,
        h,
        output_type,
        return_projection,
        freq_to_time_matricies,
        hankel_coefficients,
        is_invert_h,
    ) = args

    n_layer = len(thicknesses) + 1
    n_src = len(source_list)

    local_survey = Survey(source_list)
    wires = maps.Wires(("sigma", n_layer), ("h", n_src))
    sigma_map = wires.sigma
    h_map = wires.h

    sim = Simulation1DLayered(
        survey=local_survey,
        thicknesses=thicknesses,
        sigmaMap=sigma_map,
        hMap=h_map,
        eta=eta,
        tau=tau,
        c=c,
        topo=topo,
        hankel_filter="key_101_2009",
    )

    model = np.r_[sigma, h * np.ones(n_src)]

    if return_projection:
        sim.model = model
        if output_type == "hankel":
            return sim.get_hankel_coefficients()
        elif output_type == "freq_to_time":
            return sim.get_freq_to_time_matricies()

    sim._set_freq_to_time_matricies(freq_to_time_matricies)
    if is_invert_h is False:
        sim._set_hankel_coefficients(hankel_coefficients)

    if output_type == "sensitivity":
        J = sim.getJ(model)
        # we assumed the tx heights in a sounding is fixed
        J["dh"] = J["dh"].sum(axis=1)
        return J
    else:
        em_response = sim.dpred(model)
        return em_response


#######################################################################
#       STITCHED 1D SIMULATION CLASS AND GLOBAL FUNCTIONS
#######################################################################


class Simulation1DLayeredStitched(BaseStitchedEM1DSimulation):
    _simulation_type = "time"
    _freq_to_time_matricies = []
    _freq_to_time_matricies_set = False

    def input_args(self, i_sounding, output_type="forward"):
        if self._is_invert_h:
            hankel_coefficients = []
        else:
            hankel_coefficients = self._hankel_coefficients[i_sounding]
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
            False,
            self._freq_to_time_matricies[self._inv_index[i_sounding]],
            hankel_coefficients,
            self._is_invert_h,
        )
        return output

    def input_args_for_coefficients(self, i_sounding, output_type="freq_to_time"):
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
            True,
            [],
            [],
            self._is_invert_h,
        )
        return output

    def get_hankel_coefficients(self):
        run_simulation = run_simulation_time_domain
        if getattr(self, "_hankel_coefficients", None) is None:
            if self.verbose:
                print(">> Calculate hankel coefficients")
            if self.parallel:
                # This assumes the same # of layers for each of sounding
                # if self.n_sounding_for_chunk is None:
                pool = Pool(self.n_cpu)
                self._hankel_coefficients = pool.map(
                    run_simulation,
                    [
                        self.input_args_for_coefficients(i, output_type="hankel")
                        for i in range(self.n_sounding)
                    ],
                )

                pool.close()
                pool.join()
            else:
                self._hankel_coefficients = [
                    run_simulation(
                        self.input_args_for_coefficients(i, output_type="hankel")
                    )
                    for i in range(self.n_sounding)
                ]
        return self._hankel_coefficients

    def get_freq_to_time_matricies(self):
        run_simulation = run_simulation_time_domain
        if self.verbose:
            print(">> Calculate freq_to_time matricies")

        self._freq_to_time_matricies = [
            run_simulation(
                self.input_args_for_coefficients(i, output_type="freq_to_time")
            )
            for i in self._uniq_index
        ]
        self._freq_to_time_matricies_set = True

    def forward(self, m):
        self.model = m

        # Set flat topo at zero
        # if self.topo is None:

        run_simulation = run_simulation_time_domain

        # TODOs:
        # Check when height is not inverted, then store hankel coefficients
        if self._freq_to_time_matricies_set is False:
            self.get_freq_to_time_matricies()
        self.get_hankel_coefficients()
        if self.parallel:

            if self.verbose:
                print(">> Compute response")
            # This assumes the same # of layers for each of sounding
            # if self.n_sounding_for_chunk is None:
            pool = Pool(self.n_cpu)
            result = pool.map(
                run_simulation,
                [
                    self.input_args(i, output_type="forward")
                    for i in range(self.n_sounding)
                ],
            )

            pool.close()
            pool.join()
        else:
            result = [
                run_simulation(self.input_args(i, output_type="forward"))
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

            if self._freq_to_time_matricies_set is False:
                self.get_freq_to_time_matricies()
            self.get_hankel_coefficients()
            run_simulation = run_simulation_time_domain

            if self.parallel:
                if self.verbose:
                    print(">> Start pooling")

                pool = Pool(self.n_cpu)

                # Deprecate this for now, but revisit later
                # It is an idea of chunking for parallelization
                # if self.n_sounding_for_chunk is None:
                self._J = pool.map(
                    run_simulation,
                    [
                        self.input_args(i, output_type="sensitivity")
                        for i in range(self.n_sounding)
                    ],
                )

                if self.verbose:
                    print(">> End pooling and form J matrix")

            else:
                self._J = [
                    run_simulation(self.input_args(i, output_type="sensitivity"))
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

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = super().deleteTheseOnModelUpdate
        if self.fix_Jmatrix is False:
            toDelete += [
                "_hankel_coefficients",
            ]
        return toDelete
