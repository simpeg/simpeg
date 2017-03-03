from __future__ import print_function
import unittest
import numpy as np
from SimPEG import Mesh, Maps, Utils, SolverLU
from SimPEG import EM
import sys
from scipy.constants import mu_0

FLR = 1e-20 # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0
freq = 5e-1


def getFDEMProblem(fdemType, comp, SrcList, freq, useMu=False, verbose=False):
    cs = 10.
    ncx, ncy, ncz = 0, 0, 0
    npad = 8
    hx = [(cs,npad,-1.3), (cs,ncx), (cs,npad,1.3)]
    hy = [(cs,npad,-1.3), (cs,ncy), (cs,npad,1.3)]
    hz = [(cs,npad,-1.3), (cs,ncz), (cs,npad,1.3)]
    mesh = Mesh.TensorMesh([hx,hy,hz],['C','C','C'])

    if useMu is True:
        mapping = [('sigma', Maps.ExpMap(mesh)), ('mu', Maps.IdentityMap(mesh))]
    else:
        mapping = Maps.ExpMap(mesh)

    x = np.array([np.linspace(-5.*cs,-2.*cs,3),np.linspace(5.*cs,2.*cs,3)]) + cs/4. #don't sample right by the source, slightly off alignment from either staggered grid
    XYZ = Utils.ndgrid(x,x,np.linspace(-2.*cs,2.*cs,5))
    Rx0 = getattr(EM.FDEM.Rx, 'Point_' + comp[0])
    if comp[2] == 'r':
        real_or_imag = 'real'
    elif comp[2] == 'i':
        real_or_imag = 'imag'
    rx0 = Rx0(XYZ, comp[1], 'imag')

    Src = []

    for SrcType in SrcList:
        if SrcType is 'MagDipole':
            Src.append(EM.FDEM.Src.MagDipole([rx0], freq=freq, loc=np.r_[0.,0.,0.]))
        elif SrcType is 'MagDipole_Bfield':
            Src.append(EM.FDEM.Src.MagDipole_Bfield([rx0], freq=freq, loc=np.r_[0.,0.,0.]))
        elif SrcType is 'CircularLoop':
            Src.append(EM.FDEM.Src.CircularLoop([rx0], freq=freq, loc=np.r_[0.,0.,0.]))
        elif SrcType is 'RawVec':
            if fdemType is 'e' or fdemType is 'b':
                S_m = np.zeros(mesh.nF)
                S_e = np.zeros(mesh.nE)
                S_m[Utils.closestPoints(mesh,[0.,0.,0.],'Fz') + np.sum(mesh.vnF[:1])] = 1e-3
                S_e[Utils.closestPoints(mesh,[0.,0.,0.],'Ez') + np.sum(mesh.vnE[:1])] = 1e-3
                Src.append(EM.FDEM.Src.RawVec([rx0], freq, S_m, mesh.getEdgeInnerProduct()*S_e))

            elif fdemType is 'h' or fdemType is 'j':
                S_m = np.zeros(mesh.nE)
                S_e = np.zeros(mesh.nF)
                S_m[Utils.closestPoints(mesh,[0.,0.,0.],'Ez') + np.sum(mesh.vnE[:1])] = 1e-3
                S_e[Utils.closestPoints(mesh,[0.,0.,0.],'Fz') + np.sum(mesh.vnF[:1])] = 1e-3
                Src.append(EM.FDEM.Src.RawVec([rx0], freq, mesh.getEdgeInnerProduct()*S_m, S_e))

    if verbose:
        print('  Fetching {0!s} problem'.format((fdemType)))

    if fdemType == 'e':
        survey = EM.FDEM.Survey(Src)
        prb = EM.FDEM.Problem3D_e(mesh, sigmaMap=mapping)

    elif fdemType == 'b':
        survey = EM.FDEM.Survey(Src)
        prb = EM.FDEM.Problem3D_b(mesh, sigmaMap=mapping)

    elif fdemType == 'j':
        survey = EM.FDEM.Survey(Src)
        prb = EM.FDEM.Problem3D_j(mesh, sigmaMap=mapping)

    elif fdemType == 'h':
        survey = EM.FDEM.Survey(Src)
        prb = EM.FDEM.Problem3D_h(mesh, sigmaMap=mapping)

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

def crossCheckTest(SrcList, fdemType1, fdemType2, comp, addrandoms = False, useMu=False, TOL=1e-5, verbose=False):

    l2norm = lambda r: np.sqrt(r.dot(r))

    prb1 = getFDEMProblem(fdemType1, comp, SrcList, freq, useMu, verbose)
    mesh = prb1.mesh
    print('Cross Checking Forward: {0!s}, {1!s} formulations - {2!s}'.format(fdemType1, fdemType2, comp))

    logsig = np.log(np.ones(mesh.nC)*CONDUCTIVITY)
    mu = np.ones(mesh.nC)*MU

    if addrandoms is True:
        logsig  += np.random.randn(mesh.nC)*np.log(CONDUCTIVITY)*1e-1
        mu += np.random.randn(mesh.nC)*MU*1e-1

    if useMu is True:
        m = np.r_[logsig, mu]
    else:
        m = logsig

    survey1 = prb1.survey
    d1 = survey1.dpred(m)

    if verbose:
        print('  Problem 1 solved')


    prb2 = getFDEMProblem(fdemType2, comp, SrcList, freq, useMu, verbose)

    survey2 = prb2.survey
    d2 = survey2.dpred(m)

    if verbose:
        print('  Problem 2 solved')

    r = d2-d1
    l2r = l2norm(r)

    tol = np.max([TOL*(10**int(np.log10(0.5* (l2norm(d1) + l2norm(d2)) ))),FLR])
    print(l2norm(d1), l2norm(d2),  l2r , tol, l2r < tol)
    return l2r < tol
