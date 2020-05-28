from __future__ import print_function
import unittest
import numpy as np
import sys
from scipy.constants import mu_0

from discretize import TensorMesh
from ... import maps, utils

from SimPEG import SolverLU
from SimPEG.electromagnetics import frequency_domain as fdem

FLR = 1e-20  # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0
freq = 5e-1


def getFDEMProblem(fdemType, comp, SrcList, freq, useMu=False, verbose=False):
    cs = 10.0
    ncx, ncy, ncz = 0, 0, 0
    npad = 8
    hx = [(cs, npad, -1.3), (cs, ncx), (cs, npad, 1.3)]
    hy = [(cs, npad, -1.3), (cs, ncy), (cs, npad, 1.3)]
    hz = [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
    mesh = TensorMesh([hx, hy, hz], ["C", "C", "C"])

    if useMu is True:
        mapping = [("sigma", maps.ExpMap(mesh)), ("mu", maps.IdentityMap(mesh))]
    else:
        mapping = maps.ExpMap(mesh)

    x = (
        np.array(
            [np.linspace(-5.0 * cs, -2.0 * cs, 3), np.linspace(5.0 * cs, 2.0 * cs, 3)]
        )
        + cs / 4.0
    )  # don't sample right by the source, slightly off alignment from either staggered grid
    XYZ = utils.ndgrid(x, x, np.linspace(-2.0 * cs, 2.0 * cs, 5))
    Rx0 = getattr(fdem.Rx, "Point" + comp[0])
    if comp[-1] == "r":
        real_or_imag = "real"
    elif comp[-1] == "i":
        real_or_imag = "imag"
    rx0 = Rx0(XYZ, comp[1], real_or_imag)

    Src = []

    for SrcType in SrcList:
        if SrcType == "MagDipole":
            Src.append(fdem.Src.MagDipole([rx0], freq=freq, loc=np.r_[0.0, 0.0, 0.0]))
        elif SrcType == "MagDipole_Bfield":
            Src.append(
                fdem.Src.MagDipole_Bfield([rx0], freq=freq, loc=np.r_[0.0, 0.0, 0.0])
            )
        elif SrcType == "CircularLoop":
            Src.append(
                fdem.Src.CircularLoop([rx0], freq=freq, loc=np.r_[0.0, 0.0, 0.0])
            )
        elif SrcType == "RawVec":
            if fdemType == "e" or fdemType == "b":
                S_m = np.zeros(mesh.nF)
                S_e = np.zeros(mesh.nE)
                S_m[
                    utils.closestPoints(mesh, [0.0, 0.0, 0.0], "Fz")
                    + np.sum(mesh.vnF[:1])
                ] = 1e-3
                S_e[
                    utils.closestPoints(mesh, [0.0, 0.0, 0.0], "Ez")
                    + np.sum(mesh.vnE[:1])
                ] = 1e-3
                Src.append(
                    fdem.Src.RawVec([rx0], freq, S_m, mesh.getEdgeInnerProduct() * S_e)
                )

            elif fdemType == "h" or fdemType == "j":
                S_m = np.zeros(mesh.nE)
                S_e = np.zeros(mesh.nF)
                S_m[
                    utils.closestPoints(mesh, [0.0, 0.0, 0.0], "Ez")
                    + np.sum(mesh.vnE[:1])
                ] = 1e-3
                S_e[
                    utils.closestPoints(mesh, [0.0, 0.0, 0.0], "Fz")
                    + np.sum(mesh.vnF[:1])
                ] = 1e-3
                Src.append(
                    fdem.Src.RawVec([rx0], freq, mesh.getEdgeInnerProduct() * S_m, S_e)
                )

    if verbose:
        print("  Fetching {0!s} problem".format((fdemType)))

    if fdemType == "e":
        survey = fdem.Survey(Src)
        prb = fdem.Simulation3DElectricField(mesh, sigmaMap=mapping)

    elif fdemType == "b":
        survey = fdem.Survey(Src)
        prb = fdem.Simulation3DMagneticFluxDensity(mesh, sigmaMap=mapping)

    elif fdemType == "j":
        survey = fdem.Survey(Src)
        prb = fdem.Simulation3DCurrentDensity(mesh, sigmaMap=mapping)

    elif fdemType == "h":
        survey = fdem.Survey(Src)
        prb = fdem.Simulation3DMagneticField(mesh, sigmaMap=mapping)

    else:
        raise NotImplementedError()
    prb.pair(survey)

    try:
        from pymatsolver import Pardiso

        prb.Solver = Pardiso
    except ImportError:
        prb.Solver = SolverLU
    # prb.solverOpts = dict(check_accuracy=True)

    return prb


def crossCheckTest(
    SrcList,
    fdemType1,
    fdemType2,
    comp,
    addrandoms=False,
    useMu=False,
    TOL=1e-5,
    verbose=False,
):

    l2norm = lambda r: np.sqrt(r.dot(r))

    prb1 = getFDEMProblem(fdemType1, comp, SrcList, freq, useMu, verbose)
    mesh = prb1.mesh
    print(
        "Cross Checking Forward: {0!s}, {1!s} formulations - {2!s}".format(
            fdemType1, fdemType2, comp
        )
    )

    logsig = np.log(np.ones(mesh.nC) * CONDUCTIVITY)
    mu = np.ones(mesh.nC) * MU

    if addrandoms is True:
        logsig += np.random.randn(mesh.nC) * np.log(CONDUCTIVITY) * 1e-1
        mu += np.random.randn(mesh.nC) * MU * 1e-1

    if useMu is True:
        m = np.r_[logsig, mu]
    else:
        m = logsig

    survey1 = prb1.survey
    d1 = prb1.dpred(m)

    if verbose:
        print("  Problem 1 solved")

    prb2 = getFDEMProblem(fdemType2, comp, SrcList, freq, useMu, verbose)

    survey2 = prb2.survey
    d2 = prb2.dpred(m)

    if verbose:
        print("  Problem 2 solved")

    r = d2 - d1
    l2r = l2norm(r)

    tol = np.max([TOL * (10 ** int(np.log10(0.5 * (l2norm(d1) + l2norm(d2))))), FLR])
    print(l2norm(d1), l2norm(d2), l2r, tol, l2r < tol)
    return l2r < tol
