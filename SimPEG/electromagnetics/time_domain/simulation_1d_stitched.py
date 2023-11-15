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
    :param string output_type: "response", "sensitivity_sigma", "sensitivity_height"
    :param bool invert_height: boolean switch for inverting for source height
    :return: response or sensitivities
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
        # return_projection,
        # coefficients
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

    def get_coefficients(self):
        run_simulation = run_simulation_time_domain

        if self.verbose:
            print(">> Calculate coefficients")
        if self.parallel:
            pool = Pool(self.n_cpu)
            self._coefficients = pool.map(
                run_simulation,
                [self.input_args_for_coeff(i) for i in range(self.n_sounding)],
            )
            self._coefficients_set = True
            pool.close()
            pool.join()
        else:
            self._coefficients = [
                run_simulation(self.input_args_for_coeff(i))
                for i in range(self.n_sounding)
            ]

    def forward(self, m):
        self.model = m

        if self.verbose:
            print(">> Compute response")

        # Set flat topo at zero
        # if self.topo is None:

        run_simulation = run_simulation_time_domain

        if self.parallel:
            if self.verbose:
                print("parallel")
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
