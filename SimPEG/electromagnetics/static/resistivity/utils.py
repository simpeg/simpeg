import numpy as np

from . import receivers
from . import sources
from .survey import Survey

from ..utils import *


def WennerSrcList(nElecs, aSpacing, in2D=False, plotIt=False):
    """
    Source list for a Wenner Array
    """

    elocs = np.arange(0, aSpacing * nElecs, aSpacing)
    elocs -= (nElecs * aSpacing - aSpacing) / 2
    space = 1
    WENNER = np.zeros((0,), dtype=int)
    for ii in range(nElecs):
        for jj in range(nElecs):
            test = np.r_[jj, jj + space, jj + space * 2, jj + space * 3]
            if np.any(test >= nElecs):
                break
            WENNER = np.r_[WENNER, test]
        space += 1
    WENNER = WENNER.reshape((-1, 4))

    if plotIt:
        for i, s in enumerate("rbkg"):
            plt.plot(elocs[WENNER[:, i]], s + ".")
        plt.show()

    # Create sources and receivers
    i = 0
    if in2D:

        def getLoc(ii, abmn):
            return np.r_[elocs[WENNER[ii, abmn]], 0]

    else:

        def getLoc(ii, abmn):
            return np.r_[elocs[WENNER[ii, abmn]], 0, 0]

    srcList = []
    for i in range(WENNER.shape[0]):
        rx = receivers.Dipole(
            getLoc(i, 1).reshape([1, -1]), getLoc(i, 2).reshape([1, -1])
        )
        src = sources.Dipole([rx], getLoc(i, 0), getLoc(i, 3))
        srcList += [src]

    return srcList


def _mini_pole_pole(survey, verbose=False):
    """ Function to miniaturize a survey for use in DCSimulation.

    Miniaturizes the survey into the minimum number of unique pole-pole electrode
    combinations for AM, AN, BM, BN pairs (also taking into account reciprocity),
    for use in a DCSimulation only.
    """
    A = survey.locations_a
    B = survey.locations_b
    M = survey.locations_m
    N = survey.locations_n

    elecs, inverse = np.unique(np.r_[A, B, M, N], axis=0, return_inverse=True)

    inv_A, inv_B, inv_M, inv_N = inverse.reshape(4, -1)
    dipole_tx = inv_A != inv_B
    dipole_rx = inv_M != inv_N

    AM = np.sort(np.c_[inv_A, inv_M])
    AN = np.sort(np.c_[inv_A[dipole_rx], inv_N[dipole_rx]])
    BM = np.sort(np.c_[inv_B[dipole_tx], inv_M[dipole_tx]])
    BN = np.sort(np.c_[inv_B[dipole_tx & dipole_rx], inv_N[dipole_tx & dipole_rx]])
    unique_pole_poles, pole_pole_inv = np.unique(
        np.r_[AM, AN, BM, BN], axis=0, return_inverse=True
    )

    inv_AM, pole_pole_inv = pole_pole_inv[: len(AM)], pole_pole_inv[len(AM) :]
    inv_AN, pole_pole_inv = pole_pole_inv[: len(AN)], pole_pole_inv[len(AN) :]
    inv_BM, inv_BN = pole_pole_inv[: len(BM)], pole_pole_inv[len(BM) :]

    if verbose:
        print(f"There are {unique_pole_poles.shape[0]} unique pole-pole combinations.")

    unique_sources = []
    last_src = None
    i_d = 0
    while i_d < len(unique_pole_poles):
        if last_src != unique_pole_poles[i_d, 0]:
            last_src = unique_pole_poles[i_d, 0]
            rxs = []
        else:
            while (
                i_d < len(unique_pole_poles) and last_src == unique_pole_poles[i_d, 0]
            ):
                rxs.append(unique_pole_poles[i_d, 1])
                i_d += 1
            rxs = np.array(rxs, dtype=int)
            rxs = receivers.Pole(elecs[rxs])
            unique_sources.append(sources.Pole([rxs], elecs[last_src]))

    dipoles = [dipole_rx, dipole_tx]
    invs = [inv_AM, inv_AN, inv_BM, inv_BN]
    mini_survey = Survey(unique_sources)
    return dipoles, invs, mini_survey
