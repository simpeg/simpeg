from __future__ import print_function
# Test functions
from glob import glob
import numpy as np, sys, os, time, scipy, subprocess
import SimPEG as simpeg
import unittest
from SimPEG import MT
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
    # M = simpeg.Mesh.TensorMesh([[(1000,6,-1.5),(1000.,6),(1000,6,1.5)],[(1000,6,-1.5),(1000.,2),(1000,6,1.5)],[(1000,6,-1.3),(1000.,6),(1000,6,1.3)]], x0=['C','C','C'])# Setup the model
    M = simpeg.Mesh.TensorMesh([[(1000,6,-1.5),(1000.,4),(1000,6,1.5)],[(1000,6,-1.5),(1000.,4),(1000,6,1.5)],[(500,8,-1.3),(500.,8),(500,8,1.3)]], x0=['C','C','C'])# Setup the model
    # Set the frequencies
    freqs = np.logspace(1,-3,5)
    elev = 0

    ## Setup the the survey object
    # Receiver locations
    rx_x, rx_y = np.meshgrid(np.arange(-1000,1001,500),np.arange(-1000,1001,500))
    rx_loc = np.hstack((simpeg.Utils.mkvc(rx_x,2),simpeg.Utils.mkvc(rx_y,2),elev+np.zeros((np.prod(rx_x.shape),1))))

    return M, freqs, rx_loc, elev

def random(conds):
    ''' Returns a halfspace model based on the inputs'''
    M, freqs, rx_loc, elev = getInputs()

    # Backround
    sigBG = np.ones(M.nC)*conds
    # Add randomness to the model (10% of the value).
    sig = np.exp( np.log(sigBG) + np.random.randn(M.nC)*(conds)*1e-1 )

    return (M, freqs, sig, sigBG, rx_loc)

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

def blockInhalfSpace(conds):
    ''' Returns a halfspace model based on the inputs'''
    M, freqs, rx_loc, elev = getInputs()

    # Model
    ccM = M.gridCC
    # conds = [1e-2]
    groundInd = ccM[:,2] < elev
    sig = simpeg.Utils.ModelBuilder.defineBlock(M.gridCC,np.array([-1000,-1000,-1500]),np.array([1000,1000,-1000]),conds)
    sig[~groundInd] = 1e-8
    # Set the background, not the same as the model
    sigBG = np.zeros(M.nC) + 1e-8
    sigBG[groundInd] = conds[1]

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



def setupSimpegMTfwd_eForm_ps(inputSetup,comp='Imp',singleFreq=False,expMap=True):
    M,freqs,sig,sigBG,rx_loc = inputSetup
    # Make a receiver list
    rxList = []
    if comp == 'All':
        for rxType in ['zxxr','zxxi','zxyr','zxyi','zyxr','zyxi','zyyr','zyyi','tzxr','tzxi','tzyr','tzyi']:
            rxList.append(MT.Rx(rx_loc,rxType))
    elif comp == 'Imp':
        for rxType in ['zxxr','zxxi','zxyr','zxyi','zyxr','zyxi','zyyr','zyyi']:
            rxList.append(MT.Rx(rx_loc,rxType))
    elif comp == 'Tip':
        for rxType in ['tzxr','tzxi','tzyr','tzyi']:
            rxList.append(MT.Rx(rx_loc,rxType))
    else:
        rxList.append(MT.Rx(rx_loc,comp))
    # Source list
    srcList =[]

    if singleFreq:
        srcList.append(MT.SrcMT.polxy_1Dprimary(rxList,singleFreq))
    else:
        for freq in freqs:
            srcList.append(MT.SrcMT.polxy_1Dprimary(rxList,freq))
    # Survey MT
    survey = MT.Survey(srcList)

    ## Setup the problem object
    sigma1d = M.r(sigBG,'CC','CC','M')[0,0,:]
    if expMap:
        problem = MT.Problem3D.eForm_ps(M,sigmaPrimary= np.log(sigma1d) )
        problem.mapping = simpeg.Maps.ExpMap(problem.mesh)
        problem.curModel = np.log(sig)
    else:
        problem = MT.Problem3D.eForm_ps(M,sigmaPrimary= sigma1d)
        problem.curModel = sig
    problem.pair(survey)
    problem.verbose = False
    try:
        from pymatsolver import PardisoSolver
        problem.Solver = PardisoSolver
    except:
        pass

    return (survey, problem)

def getAppResPhs(MTdata):
    # Make impedance
    def appResPhs(freq,z):
        app_res = ((1./(8e-7*np.pi**2))/freq)*np.abs(z)**2
        app_phs = np.arctan2(z.imag,z.real)*(180/np.pi)
        return app_res, app_phs
    recData = MTdata.toRecArray('Complex')
    return appResPhs(recData['freq'],recData['zxy']), appResPhs(recData['freq'],recData['zyx'])

def JvecAdjointTest(inputSetup,comp='All',freq=False):
    (M, freqs, sig, sigBG, rx_loc) = inputSetup
    survey, problem = setupSimpegMTfwd_eForm_ps(inputSetup,comp='All',singleFreq=freq)
    print('Adjoint test of eForm primary/secondary for {!s:s} comp at {!s:s}\n'.format(comp,str(survey.freqs)))

    m  = sig
    u = problem.fields(m)

    v = np.random.rand(survey.nD,)
    # print(problem.PropMap.PropModel.nP)
    w = np.random.rand(problem.mesh.nC,)

    vJw = v.ravel().dot(problem.Jvec(m, w, u))
    wJtv = w.ravel().dot(problem.Jtvec(m, v, u))
    tol = np.max([TOL*(10**int(np.log10(np.abs(vJw)))),FLR])
    print(' vJw   wJtv  vJw - wJtv     tol    abs(vJw - wJtv) < tol')
    print(vJw, wJtv, vJw - wJtv, tol, np.abs(vJw - wJtv) < tol)
    return np.abs(vJw - wJtv) < tol

# Test the Jvec derivative
def DerivJvecTest(inputSetup,comp='All',freq=False,expMap=True):
    (M, freqs, sig, sigBG, rx_loc) = inputSetup
    survey, problem = setupSimpegMTfwd_eForm_ps(inputSetup,comp=comp,singleFreq=freq,expMap=expMap)
    print('Derivative test of Jvec for eForm primary/secondary for {!s:s} comp at {!s:s}\n'.format(comp,survey.freqs))
    # problem.mapping = simpeg.Maps.ExpMap(problem.mesh)
    # problem.sigmaPrimary = np.log(sigBG)
    x0 = np.log(sigBG)
    # cond = sig[0]
    # x0 = np.log(np.ones(problem.mesh.nC)*cond)
    # problem.sigmaPrimary = x0
    # if True:
    #     x0  = x0 + np.random.randn(problem.mesh.nC)*cond*1e-1
    survey = problem.survey
    def fun(x):
        return survey.dpred(x), lambda x: problem.Jvec(x0, x)
    return simpeg.Tests.checkDerivative(fun, x0, num=3, plotIt=False, eps=FLR)

def DerivProjfieldsTest(inputSetup,comp='All',freq=False):

    survey, problem = setupSimpegMTfwd_eForm_ps(inputSetup,comp,freq)
    print('Derivative test of data projection for eFormulation primary/secondary\n\n')
    # problem.mapping = simpeg.Maps.ExpMap(problem.mesh)
    # Initate things for the derivs Test
    src = survey.srcList[0]
    rx = src.rxList[0]

    u0x = np.random.randn(survey.mesh.nE)+np.random.randn(survey.mesh.nE)*1j
    u0y = np.random.randn(survey.mesh.nE)+np.random.randn(survey.mesh.nE)*1j
    u0 = np.vstack((simpeg.mkvc(u0x,2),simpeg.mkvc(u0y,2)))
    f0 = problem.fieldsPair(survey.mesh,survey)
    # u0 = np.hstack((simpeg.mkvc(u0_px,2),simpeg.mkvc(u0_py,2)))
    f0[src,'e_pxSolution'] =  u0[:len(u0)/2]#u0x
    f0[src,'e_pySolution'] = u0[len(u0)/2::]#u0y

    def fun(u):
        f = problem.fieldsPair(survey.mesh,survey)
        f[src,'e_pxSolution'] = u[:len(u)/2]
        f[src,'e_pySolution'] = u[len(u)/2::]
        return rx.eval(src,survey.mesh,f), lambda t: rx.evalDeriv(src,survey.mesh,f0,simpeg.mkvc(t,2))

    return simpeg.Tests.checkDerivative(fun, u0, num=3, plotIt=False, eps=FLR)

def appResPhsHalfspace_eFrom_ps_Norm(sigmaHalf,appR=True,expMap=False):
    if appR:
        label = 'resistivity'
    else:
        label = 'phase'
    # Make the survey and the problem
    survey, problem = setupSimpegMTfwd_eForm_ps(halfSpace(sigmaHalf),expMap=expMap)
    print('Apperent {!s:s} test of eFormulation primary/secondary at {:g}\n\n'.format(label,sigmaHalf))

    data = problem.dataPair(survey,survey.dpred(problem.curModel))
    # Calculate the app  phs
    app_rpxy, app_rpyx = np.array(getAppResPhs(data))
    if appR:
        return np.all(np.abs(app_rpxy[0,:] - 1./sigmaHalf) * sigmaHalf < .4)
    else:
        return np.all(np.abs(app_rpxy[1,:] + 135) / 135 < .4)

class TestAnalytics(unittest.TestCase):

    def setUp(self):
        pass
    # # Test apparent resistivity and phase
    def test_appRes1en2(self):self.assertTrue(appResPhsHalfspace_eFrom_ps_Norm(1e-2))
    def test_appPhs1en2(self):self.assertTrue(appResPhsHalfspace_eFrom_ps_Norm(1e-2,False))

    def test_appRes1en3(self):self.assertTrue(appResPhsHalfspace_eFrom_ps_Norm(1e-3))
    def test_appPhs1en3(self):self.assertTrue(appResPhsHalfspace_eFrom_ps_Norm(1e-3,False))

    # Do a derivative test of Jvec
    # def test_derivJvec_zxxr(self):self.assertTrue(DerivJvecTest(random(1e-2),'zxxr',.1))
    # def test_derivJvec_zxxi(self):self.assertTrue(DerivJvecTest(random(1e-2),'zxxi',.1))
    # def test_derivJvec_zxyr(self):self.assertTrue(DerivJvecTest(random(1e-2),'zxyr',.1))
    # def test_derivJvec_zxyi(self):self.assertTrue(DerivJvecTest(random(1e-2),'zxyi',.1))
    # def test_derivJvec_zyxr(self):self.assertTrue(DerivJvecTest(random(1e-2),'zyxr',.1))
    # def test_derivJvec_zyxi(self):self.assertTrue(DerivJvecTest(random(1e-2),'zyxi',.1))
    # def test_derivJvec_zyyr(self):self.assertTrue(DerivJvecTest(random(1e-2),'zyyr',.1))
    # def test_derivJvec_zyyi(self):self.assertTrue(DerivJvecTest(random(1e-2),'zyyi',.1))
    def test_derivJvec_All(self):self.assertTrue(DerivJvecTest(random(1e-2),'All',.1))

    # Test the adjoint of Jvec and Jtvec
    # def test_JvecAdjoint_zxxr(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zxxr',.1))
    # def test_JvecAdjoint_zxxi(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zxxi',.1))
    # def test_JvecAdjoint_zxyr(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zxyr',.1))
    # def test_JvecAdjoint_zxyi(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zxyi',.1))
    # def test_JvecAdjoint_zyxr(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zyxr',.1))
    # def test_JvecAdjoint_zyxi(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zyxi',.1))
    # def test_JvecAdjoint_zyyr(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zyyr',.1))
    # def test_JvecAdjoint_zyyi(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zyyi',.1))
    def test_JvecAdjoint_All(self):self.assertTrue(JvecAdjointTest(random(1e-2),'All',.1))

if __name__ == '__main__':
    unittest.main()
