from __future__ import print_function
import unittest
import SimPEG as simpeg
from SimPEG import MT
from SimPEG.Utils import meshTensor
import numpy as np
# Define the tolerances
TOLr = 5e-2
TOLp = 5e-2


def setupSurvey(sigmaHalf,tD=True):

    # Frequency
    nFreq = 33
    freqs = np.logspace(3,-3,nFreq)
    # Make the mesh
    ct = 5
    air = meshTensor([(ct,25,1.3)])
    # coreT0 = meshTensor([(ct,15,1.2)])
    # coreT1 = np.kron(meshTensor([(coreT0[-1],15,1.3)]),np.ones((7,)))
    core = np.concatenate( (  np.kron(meshTensor([(ct,15,-1.2)]),np.ones((10,))) , meshTensor([(ct,20)]) ) )
    bot = meshTensor([(core[0],10,-1.3)])
    x0 = -np.array([np.sum(np.concatenate((core,bot)))])
    m1d = simpeg.Mesh.TensorMesh([np.concatenate((bot,core,air))], x0=x0)
    # Make the model
    sigma = np.zeros(m1d.nC) + sigmaHalf
    sigma[m1d.gridCC > 0 ] = 1e-8

    rxList = []
    for rxType in ['z1dr','z1di']:
        rxList.append(MT.Rx(simpeg.mkvc(np.array([0.0]),2).T,rxType))
    # Source list
    srcList =[]
    if tD:
        for freq in freqs:
            srcList.append(MT.SrcMT.polxy_1DhomotD(rxList,freq))
    else:
        for freq in freqs:
            srcList.append(MT.SrcMT.polxy_1Dprimary(rxList,freq))

    survey = MT.Survey(srcList)
    return survey, sigma, m1d

def getAppResPhs(MTdata):
    # Make impedance
    def appResPhs(freq,z):
        app_res = ((1./(8e-7*np.pi**2))/freq)*np.abs(z)**2
        app_phs = np.arctan2(z.imag,z.real)*(180/np.pi)
        return app_res, app_phs
    zList = []
    for src in MTdata.survey.srcList:
        zc = [src.freq]
        for rx in src.rxList:
            if 'i' in rx.rxType:
                m=1j
            else:
                m = 1
            zc.append(m*MTdata[src,rx])
        zList.append(zc)
    return [appResPhs(zList[i][0],np.sum(zList[i][1:3])) for i in np.arange(len(zList))]

def appRes_TotalFieldNorm(sigmaHalf):

    # Make the survey
    survey, sigma, mesh = setupSurvey(sigmaHalf)
    problem = MT.Problem1D.eForm_TotalField(mesh)
    problem.pair(survey)

    # Get the fields
    fields = problem.fields(sigma)

    # Project the data
    data = survey.eval(fields)

    # Calculate the app res and phs
    app_r = np.array(getAppResPhs(data))[:,0]

    return np.linalg.norm(np.abs(app_r - np.ones(survey.nFreq)/sigmaHalf)*sigmaHalf)

def appPhs_TotalFieldNorm(sigmaHalf):

    # Make the survey
    survey, sigma, mesh = setupSurvey(sigmaHalf)
    problem = MT.Problem1D.eForm_TotalField(mesh)
    problem.pair(survey)

    # Get the fields
    fields = problem.fields(sigma)

    # Project the data
    data = survey.eval(fields)

    # Calculate the app  phs
    app_p = np.array(getAppResPhs(data))[:,1]

    return np.linalg.norm(np.abs(app_p - np.ones(survey.nFreq)*45)/ 45)

def appRes_psFieldNorm(sigmaHalf):

    # Make the survey
    survey, sigma, mesh = setupSurvey(sigmaHalf,False)
    problem = MT.Problem1D.eForm_psField(mesh, sigmaPrimary = sigma)
    problem.pair(survey)

    # Get the fields
    fields = problem.fields(sigma)

    # Project the data
    data = survey.eval(fields)

    # Calculate the app res and phs
    app_r = np.array(getAppResPhs(data))[:,0]

    return np.linalg.norm(np.abs(app_r - np.ones(survey.nFreq)/sigmaHalf)*sigmaHalf)

def appPhs_psFieldNorm(sigmaHalf):

    # Make the survey
    survey, sigma, mesh = setupSurvey(sigmaHalf,False)
    problem = MT.Problem1D.eForm_psField(mesh, sigmaPrimary = sigma)
    problem.pair(survey)

    # Get the fields
    fields = problem.fields(sigma)

    # Project the data
    data = survey.eval(fields)

    # Calculate the app  phs
    app_p = np.array(getAppResPhs(data))[:,1]

    return np.linalg.norm(np.abs(app_p - np.ones(survey.nFreq)*45)/ 45)

class TestAnalytics(unittest.TestCase):

    def setUp(self):
        pass
    # Total Fields
    # def test_appRes2en1(self):self.assertLess(appRes_TotalFieldNorm(2e-1), TOLr)
    # def test_appPhs2en1(self):self.assertLess(appPhs_TotalFieldNorm(2e-1), TOLp)

    # def test_appRes2en2(self):self.assertLess(appRes_TotalFieldNorm(2e-2), TOLr)
    # def test_appPhs2en2(self):self.assertLess(appPhs_TotalFieldNorm(2e-2), TOLp)

    # def test_appRes2en3(self):self.assertLess(appRes_TotalFieldNorm(2e-3), TOLr)
    # def test_appPhs2en3(self):self.assertLess(appPhs_TotalFieldNorm(2e-3), TOLp)

    # def test_appRes2en4(self):self.assertLess(appRes_TotalFieldNorm(2e-4), TOLr)
    # def test_appPhs2en4(self):self.assertLess(appPhs_TotalFieldNorm(2e-4), TOLp)

    # def test_appRes2en5(self):self.assertLess(appRes_TotalFieldNorm(2e-5), TOLr)
    # def test_appPhs2en5(self):self.assertLess(appPhs_TotalFieldNorm(2e-5), TOLp)

    # def test_appRes2en6(self):self.assertLess(appRes_TotalFieldNorm(2e-6), TOLr)
    # def test_appPhs2en6(self):self.assertLess(appPhs_TotalFieldNorm(2e-6), TOLp)

    # Primary/secondary
    def test_appRes2en2_ps(self):self.assertLess(appRes_psFieldNorm(2e-2), TOLr)
    def test_appPhs2en2_ps(self):self.assertLess(appPhs_psFieldNorm(2e-2), TOLp)

if __name__ == '__main__':
    unittest.main()
