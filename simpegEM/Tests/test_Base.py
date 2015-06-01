import unittest
from SimPEG import *
import simpegEM as EM
import sys
from scipy.constants import mu_0
import copy

testDerivs = False
testCrossCheck = True
testAdjoint = False
testEB = True
testHJ = True

verbose = False

TOL = 1e-4
FLR = 1e-20 # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0
freq = [1e-1, 2e-1]
addrandoms = True

def getProblem(fdemType):
    cs = 5.
    ncx, ncy, ncz = 6, 6, 6
    npad = 3
    hx = [(cs,npad,-1.3), (cs,ncx), (cs,npad,1.3)]
    hy = [(cs,npad,-1.3), (cs,ncy), (cs,npad,1.3)]
    hz = [(cs,npad,-1.3), (cs,ncz), (cs,npad,1.3)]
    mesh = Mesh.TensorMesh([hx,hy,hz],['C','C','C'])

    mapping = Maps.ExpMap(mesh)

    x = np.array([np.linspace(-30,-15,3),np.linspace(15,30,3)]) #don't sample right by the source
    XYZ = Utils.ndgrid(x,x,np.r_[0.])
    Src0 = EM.FDEM.SrcFDEM_MagDipole([], loc=np.r_[0.,0.,0.], freq=freq[0])
    Src1 = EM.FDEM.SrcFDEM_MagDipole([], loc=np.r_[0.,0.,0.], freq=freq[1])

    survey = EM.FDEM.SurveyFDEM([Src0])


    if verbose:
        print '  Fetching %s problem' % (fdemType)

    if fdemType == 'e':
        prb = EM.FDEM.ProblemFDEM_e(mesh, mapping=mapping)
    elif fdemType == 'b':
        prb = EM.FDEM.ProblemFDEM_b(mesh, mapping=mapping)
    elif fdemType == 'j':
        prb = EM.FDEM.ProblemFDEM_j(mesh, mapping=mapping)
    elif fdemType == 'h':
        prb = EM.FDEM.ProblemFDEM_h(mesh, mapping=mapping)
    else:
        raise NotImplementedError()
    prb.pair(survey)

    try:
        from pymatsolver import MumpsSolver
        prb.Solver = MumpsSolver
    except ImportError, e:
        pass

    return prb

def test_MassMatDeriv()
