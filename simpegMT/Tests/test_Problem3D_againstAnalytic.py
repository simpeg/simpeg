# Test functions
from glob import glob
import numpy as np, sys, os, time, scipy, subprocess
import simpegMT as simpegmt, SimPEG as simpeg
import unittest
import SimPEG as simpeg
import simpegMT as simpegmt
from SimPEG.Utils import meshTensor
from scipy.constants import mu_0

TOLr = 5e-2
TOL = 1e-4
FLR = 1e-20 # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0
freq = [1e-1, 2e-1]
addrandoms = True


def getInputs():
    """
    Function that returns Mesh, freqs, rx_loc, elev.
    """
    # Make a mesh
    # M = simpeg.Mesh.TensorMesh([[(100,5,-1.5),(100.,10),(100,5,1.5)],[(100,5,-1.5),(100.,10),(100,5,1.5)],[(100,5,1.6),(100.,10),(100,3,2)]], x0=['C','C',-3529.5360])
    M = simpeg.Mesh.TensorMesh([[(1000,6,-1.5),(1000.,6),(1000,6,1.5)],[(1000,6,-1.5),(1000.,2),(1000,6,1.5)],[(1000,10,-1.3),(1000.,2),(1000,10,1.3)]], x0=['C','C','C'])# Setup the model
    # Set the frequencies
    freqs = np.logspace(1,-3,5)
    elev = 0

    ## Setup the the survey object
    # Receiver locations
    rx_x, rx_y = np.meshgrid(np.arange(-3000,3001,500),np.arange(-1000,1001,500))
    rx_loc = np.array([[0, 0, 0]]) #np.hstack((simpeg.Utils.mkvc(rx_x,2),simpeg.Utils.mkvc(rx_y,2),elev+np.zeros((np.prod(rx_x.shape),1))))

    return M, freqs, rx_loc, elev


def halfSpace(conds):
    ''' Returns a halfspace model based on the inputs'''
    M, freqs, rx_loc, elev = getInputs()

    # Model
    ccM = M.gridCC
    # conds = [1e-2]
    groundInd = ccM[:,2] < elev
    sig = np.zeros(M.nC) + 1e-8
    sig[groundInd] = conds
    # Set the background, not the same as the model
    sigBG = np.zeros(M.nC) + 1e-8
    sigBG[groundInd] = conds

    return (M, freqs, sig, sigBG, rx_loc)

def twoLayer(conds):
    ''' Returns a 2 layer model based on the conductivity values given'''
    M, freqs, rx_loc, elev = getInputs()

    # Model
    ccM = M.gridCC
    groundInd = ccM[:,2] < elev
    botInd = ccM[:,2] < -3000
    sig = np.zeros(M.nC) + 1e-8
    sig[groundInd] = conds[1]
    sig[botInd] = conds[0]
    # Set the background, not the same as the model
    sigBG = np.zeros(M.nC) + 1e-8
    sigBG[groundInd] = conds[1]


    return (M, freqs, sig, sigBG, rx_loc)

def setupSimpegMTfwd_eForm_ps(inputSetup,comp='All',singleFreq=False):
    M,freqs,sig,sigBG,rx_loc = inputSetup
    # Make a receiver list
    rxList = []
    if comp == 'All':
        for rxType in ['zxyr','zxyi','zyxr','zyxi']:
                rxList.append(simpegmt.SurveyMT.RxMT(rx_loc,rxType))
    else:
        rxList.append(simpegmt.SurveyMT.RxMT(rx_loc,comp))
    # Source list
    srcList =[]
    sigma1d = M.r(sigBG,'CC','CC','M')[0,0,:]
    if singleFreq:
        srcList.append(simpegmt.SurveyMT.srcMT_polxy_1Dprimary(rxList,freqs[2]))
    else:
        for freq in freqs:
            srcList.append(simpegmt.SurveyMT.srcMT_polxy_1Dprimary(rxList,freq))
    # Survey MT
    survey = simpegmt.SurveyMT.SurveyMT(srcList)

    ## Setup the problem object
    problem = simpegmt.ProblemMT3D.eForm_ps(M,sigmaPrimary=sigma1d)
    problem.verbose = False
    try:
        from pymatsolver import MumpsSolver
        problem.Solver = MumpsSolver
    except:
        pass
    problem.pair(survey)
    problem.curMod = sig
    problem.mapping = simpeg.Maps.ExpMap(problem.mesh)

    return (survey, problem)

def getAppResPhs(MTdata):
    # Make impedance
    def appResPhs(freq,z):
        app_res = ((1./(8e-7*np.pi**2))/freq)*np.abs(z)**2
        app_phs = np.arctan2(z.imag,z.real)*(180/np.pi)
        return app_res, app_phs
    recData = MTdata.toRecArray('Complex')
    return appResPhs(recData['freq'],recData['zxy']), appResPhs(recData['freq'],recData['zyx'])

def adjointTest(inputSetup):

    survey, problem = setupSimpegMTfwd_eForm_ps(inputSetup)
    print 'Adjoint test of eForm primary/secondary\n'

    m  = problem.curMod

    # if addrandoms is True:
    #     m  = m + np.random.randn(problem.mesh.nC)*CONDUCTIVITY*1e-1

    u = problem.fields(m)

    v = np.random.rand(survey.nD)
    # print problem.PropMap.PropModel.nP
    w = np.random.rand(problem.mesh.nC)

    vJw = v.dot(problem.Jvec(m, w, u))
    wJtv = w.dot(problem.Jtvec(m, v, u))
    tol = np.max([TOL*(10**int(np.log10(np.abs(vJw)))),FLR])
    print ' vJw   wJtv  vJw - wJtv     tol    abs(vJw - wJtv) < tol'
    print vJw, wJtv, vJw - wJtv, tol, np.abs(vJw - wJtv) < tol
    return np.abs(vJw - wJtv) < tol


def derivProjfields(inputSetup):

    survey, problem = setupSimpegMTfwd_eForm_ps(inputSetup)
    print 'Derivative test of data projection for eFormulation primary/secondary\n\n'

    # Define a src and rx
    src = survey.srcList[-1]
    rx = src.rxList[1]
    u0 = np.random.randn(survey.mesh.nE)+np.random.randn(survey.mesh.nE)*1j
    f0 = problem.fieldsPair(survey.mesh,survey)
    f0[src,'e_pxSolution'] = u0
    f0[src,'e_pySolution'] = u0
    def fun(u):
        f = problem.fieldsPair(survey.mesh,survey)
        f[src,'e_pxSolution'] = u.ravel()
        f[src,'e_pySolution'] = u.ravel()
        return rx.projectFields(src,survey.mesh,f), lambda t: rx.projectFieldsDeriv(src,survey.mesh,f0,simpeg.mkvc(t,2))

    return simpeg.Tests.checkDerivative(fun, u0, num=3, plotIt=False, eps=FLR)


def appResPhsHalfspace_eFrom_ps_Norm(sigmaHalf,appR=True):
    if appR:
        label = 'resistivity'
    else:
        label = 'phase'
    # Make the survey and the problem
    survey, problem = setupSimpegMTfwd_eForm_ps(halfSpace(sigmaHalf))
    print 'Apperent {:s} test of eFormulation primary/secondary at {:g}\n\n'.format(label,sigmaHalf)

    data = problem.dataPair(survey,survey.dpred(problem.curMod))
    # Calculate the app  phs
    app_rpxy, app_rpyx = np.array(getAppResPhs(data))
    if appR:
        return np.all(np.abs(app_rpxy[0,:] - np.ones(survey.nFreq)/sigmaHalf) * sigmaHalf < .35)
    else:
        return np.all(np.abs(app_rpxy[1,:] + np.ones(survey.nFreq)*135) / 135 < .35)

class TestAnalytics(unittest.TestCase):

    def setUp(self):
        pass
    # def test_appRes2en1(self):self.assertTrue(appResPhsHalfspace_eFrom_ps_Norm(2e-1))
    def test_appRes1en2(self):self.assertTrue(appResPhsHalfspace_eFrom_ps_Norm(1e-2))
    def test_appPhs1en2(self):self.assertTrue(appResPhsHalfspace_eFrom_ps_Norm(1e-2,False))

    def test_appRes1en3(self):self.assertTrue(appResPhsHalfspace_eFrom_ps_Norm(1e-3))
    def test_appPhs1en3(self):self.assertTrue(appResPhsHalfspace_eFrom_ps_Norm(1e-3,False))

    # Do a derivative test
    def test_deriv1(self):self.assertTrue(derivProjfields(halfSpace(1e-3)))

    # Test the adjoint of Jvec and Jtvec
    def test_adjointDeriv1(self):self.assertTrue(adjointTest(halfSpace(1e-3)))

if __name__ == '__main__':
    unittest.main()