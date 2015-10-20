# Test functions
from glob import glob
import numpy as np, sys, os, time, scipy, subprocess
import simpegMT as simpegmt, SimPEG as simpeg
import unittest
import SimPEG as simpeg
import simpegMT as simpegmt
from SimPEG.Utils import meshTensor


TOLr = 5e-2

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

def twoLayer():
    M, freqs, rx_loc, elev = getInputs()

    # Model
    ccM = M.gridCC
    conds = [1e-2,1]
    groundInd = ccM[:,2] < elev
    botInd = ccM[:,2] < -3000
    sig = np.zeros(M.nC) + 1e-8
    sig[groundInd] = conds[1]
    sig[botInd] = conds[0]
    # Set the background, not the same as the model
    sigBG = np.zeros(M.nC) + 1e-8
    sigBG[groundInd] = conds[1]


    return (M, freqs, sig, sigBG, rx_loc)

def runSimpegMTfwd_eForm_ps(inputsProblem,singleFreq=False):
    M,freqs,sig,sigBG,rx_loc = inputsProblem
    # Make a receiver list
    rxList = []
    for rxType in ['zxyr','zxyi','zyxr','zyxi']:
            rxList.append(simpegmt.SurveyMT.RxMT(rx_loc,rxType))
    # Source list
    srcList =[]
    sigma1d = M.r(sigBG,'CC','CC','M')[0,0,:]
    if singleFreq:
        srcList.append(simpegmt.SurveyMT.srcMT_polxy_1Dprimary(rxList,freqs[-1]))
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

    fields = problem.fields(sig)
    mtData = survey.projectFields(fields)

    return (survey, problem, fields, mtData)


def getAppResPhs(MTdata):
    # Make impedance
    def appResPhs(freq,z):
        app_res = ((1./(8e-7*np.pi**2))/freq)*np.abs(z)**2
        app_phs = np.arctan2(z.imag,z.real)*(180/np.pi)
        return app_res, app_phs
    recData = MTdata.toRecArray('Complex')
    return appResPhs(recData['freq'],recData['zxy']), appResPhs(recData['freq'],recData['zyx'])

def appResPhsHalfspace_eFrom_ps_Norm(sigmaHalf,appR=True):

    # Make the survey
    survey, problem, fields, data = runSimpegMTfwd_eForm_ps(halfSpace(sigmaHalf))
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
    def test_appRes1en3(self):self.assertTrue(appResPhsHalfspace_eFrom_ps_Norm(1e-3))
    # def test_appRes2en1(self):self.assertTrue(appResPhsHalfspace_eFrom_ps_Norm(2e-1,False))
    def test_appPhs1en2(self):self.assertTrue(appResPhsHalfspace_eFrom_ps_Norm(1e-2,False))
    def test_appPhs1en3(self):self.assertTrue(appResPhsHalfspace_eFrom_ps_Norm(1e-3,False))
if __name__ == '__main__':
    unittest.main()