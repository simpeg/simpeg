import unittest
from SimPEG import *
from SimPEG import EM
import sys
from scipy.constants import mu_0

testCrossCheck = True
testEB = True
testHJ = True

verbose = False

TOL = 1e-5
FLR = 1e-20 # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0
freq = 1e-1
addrandoms = True

SrcType = 'RawVec' #or 'MAgDipole_Bfield', 'CircularLoop', 'RawVec'


def getProblem(fdemType, comp):
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
    Rx0 = EM.FDEM.RxFDEM(XYZ, comp)

    if SrcType is 'MagDipole':
        Src = EM.FDEM.SrcFDEM_MagDipole([Rx0], freq=freq, loc=np.r_[0.,0.,0.])
    elif SrcType is 'MagDipole_Bfield':
        Src = EM.FDEM.SrcFDEM_MagDipole_Bfield([Rx0], freq=freq, loc=np.r_[0.,0.,0.])
    elif SrcType is 'CircularLoop':
        Src2 = EM.FDEM.SrcFDEM_CircularLoop([Rx0], freq=freq, loc=np.r_[0.,0.,0.])

    if verbose:
        print '  Fetching %s problem' % (fdemType)

    if fdemType == 'e':
        if SrcType is 'RawVec':
            S_m = np.zeros(mesh.nF)
            S_e = np.zeros(mesh.nE)
            S_m[Utils.closestPoints(mesh,[0.,0.,0.],'Fz') + np.sum(mesh.vnF[:1])] = 1.
            S_e[Utils.closestPoints(mesh,[0.,0.,0.],'Ez') + np.sum(mesh.vnE[:1])] = 1.
            Src = EM.FDEM.SrcFDEM_RawVec([Rx0], freq, S_m, S_e)

        survey = EM.FDEM.SurveyFDEM([Src])
        prb = EM.FDEM.ProblemFDEM_e(mesh, mapping=mapping)

    elif fdemType == 'b':
        if SrcType is 'RawVec':
            S_m = np.zeros(mesh.nF)
            S_e = np.zeros(mesh.nE)
            S_m[Utils.closestPoints(mesh,[0.,0.,0.],'Fz') + np.sum(mesh.vnF[:1])] = 1.
            S_e[Utils.closestPoints(mesh,[0.,0.,0.],'Ez') + np.sum(mesh.vnE[:1])] = 1.
            Src = EM.FDEM.SrcFDEM_RawVec([Rx0], freq, S_m, S_e)

        survey = EM.FDEM.SurveyFDEM([Src])
        prb = EM.FDEM.ProblemFDEM_b(mesh, mapping=mapping)

    elif fdemType == 'j':
        if SrcType is 'RawVec':
            S_m = np.zeros(mesh.nE)
            S_e = np.zeros(mesh.nF)
            S_m[Utils.closestPoints(mesh,[0.,0.,0.],'Ez') + np.sum(mesh.vnE[:1])] = 1.
            S_e[Utils.closestPoints(mesh,[0.,0.,0.],'Fz') + np.sum(mesh.vnF[:1])] = 1.
            Src = EM.FDEM.SrcFDEM_RawVec([Rx0], freq, S_m, S_e)

        survey = EM.FDEM.SurveyFDEM([Src])
        prb = EM.FDEM.ProblemFDEM_j(mesh, mapping=mapping)

    elif fdemType == 'h':
        if SrcType is 'RawVec':
            S_m = np.zeros(mesh.nE)
            S_e = np.zeros(mesh.nF)
            S_m[Utils.closestPoints(mesh,[0.,0.,0.],'Ez') + np.sum(mesh.vnE[:1])] = 1.
            S_e[Utils.closestPoints(mesh,[0.,0.,0.],'Fz') + np.sum(mesh.vnF[:1])] = 1.
            Src = EM.FDEM.SrcFDEM_RawVec([Rx0], freq, S_m, S_e)

        survey = EM.FDEM.SurveyFDEM([Src])
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


def crossCheckTest(fdemType, comp):

    l2norm = lambda r: np.sqrt(r.dot(r))

    prb1 = getProblem(fdemType, comp)
    mesh = prb1.mesh
    print 'Cross Checking Forward: %s formulation - %s' % (fdemType, comp)
    m = np.log(np.ones(mesh.nC)*CONDUCTIVITY)
    mu = np.log(np.ones(mesh.nC)*MU)

    if addrandoms is True:
        m  = m + np.random.randn(mesh.nC)*np.log(CONDUCTIVITY)*1e-1
        mu = mu + np.random.randn(mesh.nC)*MU*1e-1

    # prb1.PropMap.PropModel.mu = mu
    # prb1.PropMap.PropModel.mui = 1./mu
    survey1 = prb1.survey
    d1 = survey1.dpred(m)

    if verbose:
        print '  Problem 1 solved'

    if fdemType == 'e':
        prb2 = getProblem('b', comp)
    elif fdemType == 'b':
        prb2 = getProblem('e', comp)
    elif fdemType == 'j':
        prb2 = getProblem('h', comp)
    elif fdemType == 'h':
        prb2 = getProblem('j', comp)
    else:
        raise NotImplementedError()

    # prb2.mu = mu
    survey2 = prb2.survey
    d2 = survey2.dpred(m)

    if verbose:
        print '  Problem 2 solved'

    r = d2-d1
    l2r = l2norm(r)

    tol = np.max([TOL*(10**int(np.log10(l2norm(d1)))),FLR])
    print l2norm(d1), l2norm(d2),  l2r , tol, l2r < tol
    return l2r < tol


if testCrossCheck:
    if testEB:
        def test_EB_CrossCheck_exr_Eform(self):
            self.assertTrue(crossCheckTest('e', 'exr'))
        def test_EB_CrossCheck_eyr_Eform(self):
            self.assertTrue(crossCheckTest('e', 'eyr'))
        def test_EB_CrossCheck_ezr_Eform(self):
            self.assertTrue(crossCheckTest('e', 'ezr'))
        def test_EB_CrossCheck_exi_Eform(self):
            self.assertTrue(crossCheckTest('e', 'exi'))
        def test_EB_CrossCheck_eyi_Eform(self):
            self.assertTrue(crossCheckTest('e', 'eyi'))
        def test_EB_CrossCheck_ezi_Eform(self):
            self.assertTrue(crossCheckTest('e', 'ezi'))

        def test_EB_CrossCheck_bxr_Eform(self):
            self.assertTrue(crossCheckTest('e', 'bxr'))
        def test_EB_CrossCheck_byr_Eform(self):
            self.assertTrue(crossCheckTest('e', 'byr'))
        def test_EB_CrossCheck_bzr_Eform(self):
            self.assertTrue(crossCheckTest('e', 'bzr'))
        def test_EB_CrossCheck_bxi_Eform(self):
            self.assertTrue(crossCheckTest('e', 'bxi'))
        def test_EB_CrossCheck_byi_Eform(self):
            self.assertTrue(crossCheckTest('e', 'byi'))
        def test_EB_CrossCheck_bzi_Eform(self):
            self.assertTrue(crossCheckTest('e', 'bzi'))

    if testHJ:
        def test_HJ_CrossCheck_jxr_Jform(self):
            self.assertTrue(crossCheckTest('j', 'jxr'))
        def test_HJ_CrossCheck_jyr_Jform(self):
            self.assertTrue(crossCheckTest('j', 'jyr'))
        def test_HJ_CrossCheck_jzr_Jform(self):
            self.assertTrue(crossCheckTest('j', 'jzr'))
        def test_HJ_CrossCheck_jxi_Jform(self):
            self.assertTrue(crossCheckTest('j', 'jxi'))
        def test_HJ_CrossCheck_jyi_Jform(self):
            self.assertTrue(crossCheckTest('j', 'jyi'))
        def test_HJ_CrossCheck_jzi_Jform(self):
            self.assertTrue(crossCheckTest('j', 'jzi'))

        def test_HJ_CrossCheck_hxr_Jform(self):
            self.assertTrue(crossCheckTest('j', 'hxr'))
        def test_HJ_CrossCheck_hyr_Jform(self):
            self.assertTrue(crossCheckTest('j', 'hyr'))
        def test_HJ_CrossCheck_hzr_Jform(self):
            self.assertTrue(crossCheckTest('j', 'hzr'))
        def test_HJ_CrossCheck_hxi_Jform(self):
            self.assertTrue(crossCheckTest('j', 'hxi'))
        def test_HJ_CrossCheck_hyi_Jform(self):
            self.assertTrue(crossCheckTest('j', 'hyi'))
        def test_HJ_CrossCheck_hzi_Jform(self):
            self.assertTrue(crossCheckTest('j', 'hzi'))

if __name__ == '__main__':
    unittest.main()