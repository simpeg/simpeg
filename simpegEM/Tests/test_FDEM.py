import unittest
from SimPEG import *
import simpegEM as EM
import sys  
from scipy.constants import mu_0
import copy

testDerivs = True
testCrossCheck = True
testAdjoint = True
testEB = True
testHJ = True

verbose = False

TOL = 1e-4
FLR = 1e-20 # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0
freq = 1e-1
addrandoms = True

SrcType = 'MagDipole' #or 'MAgDipole_Bfield', 'CircularLoop', 'RawVec'


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

def adjointTest(fdemType, comp):
    prb = getProblem(fdemType, comp)
    print 'Adjoint %s formulation - %s' % (fdemType, comp)

    m  = np.log(np.ones(prb.mesh.nC)*CONDUCTIVITY)
    mu = np.ones(prb.mesh.nC)*MU

    if addrandoms is True:
        m  = m + np.random.randn(prb.mesh.nC)*np.log(CONDUCTIVITY)*1e-1 
        mu = mu + np.random.randn(prb.mesh.nC)*MU*1e-1

    survey = prb.survey
    prb.PropMap.PropModel.mu = mu
    prb.PropMap.PropModel.mui = 1./mu
    u = prb.fields(m)

    v = np.random.rand(survey.nD)
    # print prb.PropMap.PropModel.nP
    w = np.random.rand(prb.mesh.nC)

    vJw = v.dot(prb.Jvec(m, w, u))
    wJtv = w.dot(prb.Jtvec(m, v, u))
    tol = np.max([TOL*(10**int(np.log10(np.abs(vJw)))),FLR]) 
    print vJw, wJtv, vJw - wJtv, tol, np.abs(vJw - wJtv) < tol
    return np.abs(vJw - wJtv) < tol


def derivTest(fdemType, comp):

    prb = getProblem(fdemType, comp)
    print '%s formulation - %s' % (fdemType, comp)
    x0 = np.log(np.ones(prb.mesh.nC)*CONDUCTIVITY)
    mu = np.log(np.ones(prb.mesh.nC)*MU)

    if addrandoms is True:
        x0 = x0 + np.random.randn(prb.mesh.nC)*np.log(CONDUCTIVITY)*1e-1 
        mu = mu + np.random.randn(prb.mesh.nC)*MU*1e-1

    prb.PropMap.PropModel.mu = mu
    prb.PropMap.PropModel.mui = 1./mu

    survey = prb.survey
    def fun(x):
        return survey.dpred(x), lambda x: prb.Jvec(x0, x)
    return Tests.checkDerivative(fun, x0, num=3, plotIt=False, eps=FLR)


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

    prb1.PropMap.PropModel.mu = mu
    prb1.PropMap.PropModel.mui = 1./mu
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
    
    prb2.mu = mu
    survey2 = prb2.survey
    d2 = survey2.dpred(m)

    if verbose:
        print '  Problem 2 solved'

    r = d2-d1
    l2r = l2norm(r) 

    tol = np.max([TOL*(10**int(np.log10(l2norm(d1)))),FLR]) 
    print l2norm(d1), l2norm(d2),  l2r , tol, l2r < tol
    return l2r < tol    





class FDEM_DerivTests(unittest.TestCase):

    if testDerivs:
        if testEB:
            def test_Jvec_exr_Eform(self):
                self.assertTrue(derivTest('e', 'exr'))
            def test_Jvec_eyr_Eform(self):
                self.assertTrue(derivTest('e', 'eyr'))
            def test_Jvec_ezr_Eform(self):
                self.assertTrue(derivTest('e', 'ezr'))
            def test_Jvec_exi_Eform(self):
                self.assertTrue(derivTest('e', 'exi'))
            def test_Jvec_eyi_Eform(self):
                self.assertTrue(derivTest('e', 'eyi'))
            def test_Jvec_ezi_Eform(self):
                self.assertTrue(derivTest('e', 'ezi'))

            def test_Jvec_bxr_Eform(self):
                self.assertTrue(derivTest('e', 'bxr'))
            def test_Jvec_byr_Eform(self):
                self.assertTrue(derivTest('e', 'byr'))
            def test_Jvec_bzr_Eform(self):
                self.assertTrue(derivTest('e', 'bzr'))
            def test_Jvec_bxi_Eform(self):
                self.assertTrue(derivTest('e', 'bxi'))
            def test_Jvec_byi_Eform(self):
                self.assertTrue(derivTest('e', 'byi'))
            def test_Jvec_bzi_Eform(self):
                self.assertTrue(derivTest('e', 'bzi'))

            def test_Jvec_exr_Bform(self):
                self.assertTrue(derivTest('b', 'exr'))
            def test_Jvec_eyr_Bform(self):
                self.assertTrue(derivTest('b', 'eyr'))
            def test_Jvec_ezr_Bform(self):
                self.assertTrue(derivTest('b', 'ezr'))
            def test_Jvec_exi_Bform(self):
                self.assertTrue(derivTest('b', 'exi'))
            def test_Jvec_eyi_Bform(self):
                self.assertTrue(derivTest('b', 'eyi'))
            def test_Jvec_ezi_Bform(self):
                self.assertTrue(derivTest('b', 'ezi'))

            def test_Jvec_bxr_Bform(self):
                self.assertTrue(derivTest('b', 'bxr'))
            def test_Jvec_byr_Bform(self):
                self.assertTrue(derivTest('b', 'byr'))
            def test_Jvec_bzr_Bform(self):
                self.assertTrue(derivTest('b', 'bzr'))
            def test_Jvec_bxi_Bform(self):
                self.assertTrue(derivTest('b', 'bxi'))
            def test_Jvec_byi_Bform(self):
                self.assertTrue(derivTest('b', 'byi'))
            def test_Jvec_bzi_Bform(self):
                self.assertTrue(derivTest('b', 'bzi'))

        if testHJ: 
            def test_Jvec_jxr_Jform(self):
                self.assertTrue(derivTest('j', 'jxr'))
            def test_Jvec_jyr_Jform(self):
                self.assertTrue(derivTest('j', 'jyr'))
            def test_Jvec_jzr_Jform(self):
                self.assertTrue(derivTest('j', 'jzr'))
            def test_Jvec_jxi_Jform(self):
                self.assertTrue(derivTest('j', 'jxi'))
            def test_Jvec_jyi_Jform(self):
                self.assertTrue(derivTest('j', 'jyi'))
            def test_Jvec_jzi_Jform(self):
                self.assertTrue(derivTest('j', 'jzi'))

            def test_Jvec_hxr_Jform(self):
                self.assertTrue(derivTest('j', 'hxr'))
            def test_Jvec_hyr_Jform(self):
                self.assertTrue(derivTest('j', 'hyr'))
            def test_Jvec_hzr_Jform(self):
                self.assertTrue(derivTest('j', 'hzr'))
            def test_Jvec_hxi_Jform(self):
                self.assertTrue(derivTest('j', 'hxi'))
            def test_Jvec_hyi_Jform(self):
                self.assertTrue(derivTest('j', 'hyi'))
            def test_Jvec_hzi_Jform(self):
                self.assertTrue(derivTest('j', 'hzi'))

            def test_Jvec_hxr_Hform(self):
                self.assertTrue(derivTest('h', 'hxr'))
            def test_Jvec_hyr_Hform(self):
                self.assertTrue(derivTest('h', 'hyr'))
            def test_Jvec_hzr_Hform(self):
                self.assertTrue(derivTest('h', 'hzr'))
            def test_Jvec_hxi_Hform(self):
                self.assertTrue(derivTest('h', 'hxi'))
            def test_Jvec_hyi_Hform(self):
                self.assertTrue(derivTest('h', 'hyi'))
            def test_Jvec_hzi_Hform(self):
                self.assertTrue(derivTest('h', 'hzi'))

            def test_Jvec_hxr_Hform(self):
                self.assertTrue(derivTest('h', 'jxr'))
            def test_Jvec_hyr_Hform(self):
                self.assertTrue(derivTest('h', 'jyr'))
            def test_Jvec_hzr_Hform(self):
                self.assertTrue(derivTest('h', 'jzr'))
            def test_Jvec_hxi_Hform(self):
                self.assertTrue(derivTest('h', 'jxi'))
            def test_Jvec_hyi_Hform(self):
                self.assertTrue(derivTest('h', 'jyi'))
            def test_Jvec_hzi_Hform(self):
                self.assertTrue(derivTest('h', 'jzi'))


    if testAdjoint:
        if testEB:
            def test_Jtvec_adjointTest_exr_Eform(self):
                self.assertTrue(adjointTest('e', 'exr'))
            def test_Jtvec_adjointTest_eyr_Eform(self):
                self.assertTrue(adjointTest('e', 'eyr'))
            def test_Jtvec_adjointTest_ezr_Eform(self):
                self.assertTrue(adjointTest('e', 'ezr'))
            def test_Jtvec_adjointTest_exi_Eform(self):
                self.assertTrue(adjointTest('e', 'exi'))
            def test_Jtvec_adjointTest_eyi_Eform(self):
                self.assertTrue(adjointTest('e', 'eyi'))
            def test_Jtvec_adjointTest_ezi_Eform(self):
                self.assertTrue(adjointTest('e', 'ezi'))

            def test_Jtvec_adjointTest_bxr_Eform(self):
                self.assertTrue(adjointTest('e', 'bxr'))
            def test_Jtvec_adjointTest_byr_Eform(self):
                self.assertTrue(adjointTest('e', 'byr'))
            def test_Jtvec_adjointTest_bzr_Eform(self):
                self.assertTrue(adjointTest('e', 'bzr'))
            def test_Jtvec_adjointTest_bxi_Eform(self):
                self.assertTrue(adjointTest('e', 'bxi'))
            def test_Jtvec_adjointTest_byi_Eform(self):
                self.assertTrue(adjointTest('e', 'byi'))
            def test_Jtvec_adjointTest_bzi_Eform(self):
                self.assertTrue(adjointTest('e', 'bzi'))

            def test_Jtvec_adjointTest_exr_Bform(self):
                self.assertTrue(adjointTest('b', 'exr'))
            def test_Jtvec_adjointTest_eyr_Bform(self):
                self.assertTrue(adjointTest('b', 'eyr'))
            def test_Jtvec_adjointTest_ezr_Bform(self):
                self.assertTrue(adjointTest('b', 'ezr'))
            def test_Jtvec_adjointTest_exi_Bform(self):
                self.assertTrue(adjointTest('b', 'exi'))
            def test_Jtvec_adjointTest_eyi_Bform(self):
                self.assertTrue(adjointTest('b', 'eyi'))
            def test_Jtvec_adjointTest_ezi_Bform(self):
                self.assertTrue(adjointTest('b', 'ezi'))
            def test_Jtvec_adjointTest_bxr_Bform(self):
                self.assertTrue(adjointTest('b', 'bxr'))
            def test_Jtvec_adjointTest_byr_Bform(self):
                self.assertTrue(adjointTest('b', 'byr'))
            def test_Jtvec_adjointTest_bzr_Bform(self):
                self.assertTrue(adjointTest('b', 'bzr'))
            def test_Jtvec_adjointTest_bxi_Bform(self):
                self.assertTrue(adjointTest('b', 'bxi'))
            def test_Jtvec_adjointTest_byi_Bform(self):
                self.assertTrue(adjointTest('b', 'byi'))
            def test_Jtvec_adjointTest_bzi_Bform(self):
                self.assertTrue(adjointTest('b', 'bzi'))


        if testHJ: 
            def test_Jtvec_adjointTest_jxr_Jform(self):
                self.assertTrue(adjointTest('j', 'jxr'))
            def test_Jtvec_adjointTest_jyr_Jform(self):
                self.assertTrue(adjointTest('j', 'jyr'))
            def test_Jtvec_adjointTest_jzr_Jform(self):
                self.assertTrue(adjointTest('j', 'jzr'))
            def test_Jtvec_adjointTest_jxi_Jform(self):
                self.assertTrue(adjointTest('j', 'jxi'))
            def test_Jtvec_adjointTest_jyi_Jform(self):
                self.assertTrue(adjointTest('j', 'jyi'))
            def test_Jtvec_adjointTest_jzi_Jform(self):
                self.assertTrue(adjointTest('j', 'jzi'))

            def test_Jtvec_adjointTest_hxr_Jform(self):
                self.assertTrue(adjointTest('j', 'hxr'))
            def test_Jtvec_adjointTest_hyr_Jform(self):
                self.assertTrue(adjointTest('j', 'hyr'))
            def test_Jtvec_adjointTest_hzr_Jform(self):
                self.assertTrue(adjointTest('j', 'hzr'))
            def test_Jtvec_adjointTest_hxi_Jform(self):
                self.assertTrue(adjointTest('j', 'hxi'))
            def test_Jtvec_adjointTest_hyi_Jform(self):
                self.assertTrue(adjointTest('j', 'hyi'))
            def test_Jtvec_adjointTest_hzi_Jform(self):
                 self.assertTrue(adjointTest('j', 'hzi'))

            def test_Jtvec_adjointTest_hxr_Hform(self):
                self.assertTrue(adjointTest('h', 'hxr'))
            def test_Jtvec_adjointTest_hyr_Hform(self):
                self.assertTrue(adjointTest('h', 'hyr'))
            def test_Jtvec_adjointTest_hzr_Hform(self):
                self.assertTrue(adjointTest('h', 'hzr'))
            def test_Jtvec_adjointTest_hxi_Hform(self):
                self.assertTrue(adjointTest('h', 'hxi'))
            def test_Jtvec_adjointTest_hyi_Hform(self):
                self.assertTrue(adjointTest('h', 'hyi'))
            def test_Jtvec_adjointTest_hzi_Hform(self):
                self.assertTrue(adjointTest('h', 'hzi'))

            def test_Jtvec_adjointTest_hxr_Hform(self):
                self.assertTrue(adjointTest('h', 'jxr'))
            def test_Jtvec_adjointTest_hyr_Hform(self):
                self.assertTrue(adjointTest('h', 'jyr'))
            def test_Jtvec_adjointTest_hzr_Hform(self):
                self.assertTrue(adjointTest('h', 'jzr'))
            def test_Jtvec_adjointTest_hxi_Hform(self):
                self.assertTrue(adjointTest('h', 'jxi'))
            def test_Jtvec_adjointTest_hyi_Hform(self):
                self.assertTrue(adjointTest('h', 'jyi'))
            def test_Jtvec_adjointTest_hzi_Hform(self):
                self.assertTrue(adjointTest('h', 'jzi'))


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
