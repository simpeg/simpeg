import unittest
from SimPEG import *
from SimPEG import EM
import sys
from scipy.constants import mu_0

def getFDEMProblem(fdemType, comp, SrcList, freq, verbose=False):
    cs = 10.
    ncx, ncy, ncz = 0, 0, 0
    npad = 8
    hx = [(cs,npad,-1.3), (cs,ncx), (cs,npad,1.3)]
    hy = [(cs,npad,-1.3), (cs,ncy), (cs,npad,1.3)]
    hz = [(cs,npad,-1.3), (cs,ncz), (cs,npad,1.3)]
    mesh = Mesh.TensorMesh([hx,hy,hz],['C','C','C'])

    mapping = Maps.ExpMap(mesh)

    x = np.array([np.linspace(-5.*cs,-2.*cs,3),np.linspace(5.*cs,2.*cs,3)]) #don't sample right by the source
    XYZ = Utils.ndgrid(x,x,np.linspace(-2.*cs,2.*cs,5))
    Rx0 = EM.FDEM.Rx(XYZ, comp)

    Src = []

    for SrcType in SrcList:
        if SrcType is 'MagDipole':
            Src.append(EM.FDEM.Src.MagDipole([Rx0], freq=freq, loc=np.r_[0.,0.,0.]))
        elif SrcType is 'MagDipole_Bfield':
            Src.append(EM.FDEM.Src.MagDipole_Bfield([Rx0], freq=freq, loc=np.r_[0.,0.,0.]))
        elif SrcType is 'CircularLoop':
            Src.append(EM.FDEM.Src.CircularLoop([Rx0], freq=freq, loc=np.r_[0.,0.,0.]))
        elif SrcType is 'RawVec':
            if fdemType is 'e' or fdemType is 'b':
                S_m = np.zeros(mesh.nF)
                S_e = np.zeros(mesh.nE)
                S_m[Utils.closestPoints(mesh,[0.,0.,0.],'Fz') + np.sum(mesh.vnF[:1])] = 1.
                S_e[Utils.closestPoints(mesh,[0.,0.,0.],'Ez') + np.sum(mesh.vnE[:1])] = 1.
                Src.append(EM.FDEM.Src.RawVec([Rx0], freq, S_m, S_e))

            elif fdemType is 'h' or fdemType is 'j':
                S_m = np.zeros(mesh.nE)
                S_e = np.zeros(mesh.nF)
                S_m[Utils.closestPoints(mesh,[0.,0.,0.],'Ez') + np.sum(mesh.vnE[:1])] = 1.
                S_e[Utils.closestPoints(mesh,[0.,0.,0.],'Fz') + np.sum(mesh.vnF[:1])] = 1.
                Src.append(EM.FDEM.Src.RawVec([Rx0], freq, S_m, S_e))

    if verbose:
        print '  Fetching %s problem' % (fdemType)

    if fdemType == 'e':
        survey = EM.FDEM.Survey(Src)
        prb = EM.FDEM.Problem_e(mesh, mapping=mapping)

    elif fdemType == 'b':
        survey = EM.FDEM.Survey(Src)
        prb = EM.FDEM.Problem_b(mesh, mapping=mapping)

    elif fdemType == 'j':
        survey = EM.FDEM.Survey(Src)
        prb = EM.FDEM.Problem_j(mesh, mapping=mapping)

    elif fdemType == 'h':
        survey = EM.FDEM.Survey(Src)
        prb = EM.FDEM.Problem_h(mesh, mapping=mapping)

    else:
        raise NotImplementedError()
    prb.pair(survey)

    try:
        from pymatsolver import MumpsSolver
        prb.Solver = MumpsSolver
    except ImportError, e:
        prb.Solver = SolverLU

    return prb