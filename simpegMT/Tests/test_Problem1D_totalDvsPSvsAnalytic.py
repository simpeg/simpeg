import unittest
import SimPEG as simpeg
import simpegMT as simpegmt
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
    bot = meshTensor([(core[0],15,-1.3)])
    x0 = -np.array([np.sum(np.concatenate((core,bot)))])
    m1d = simpeg.Mesh.TensorMesh([np.concatenate((bot,core,air))], x0=x0)
    # Make the model
    sigma = np.zeros(m1d.nC) + sigmaHalf
    sigma[m1d.gridCC > 0 ] = 1e-8
    sigmaBack = sigma.copy()
    # Add structure
    shallow = (m1d.gridCC < -200) * (m1d.gridCC > -600)
    deep = (m1d.gridCC < -3000) * (m1d.gridCC > -5000)
    sigma[shallow] = 1
    sigma[deep] = 0.1

    rxList = []
    for rxType in ['z1dr','z1di']:
        rxList.append(simpegmt.SurveyMT.RxMT(simpeg.mkvc(np.array([0.0]),2).T,rxType))
    # Source list
    srcList =[]
    if tD:
        for freq in freqs:
            srcList.append(simpegmt.SurveyMT.srcMT_polxy_1DhomotD(rxList,freq))
    else:
        for freq in freqs:
            srcList.append(simpegmt.SurveyMT.srcMT_polxy_1Dprimary(rxList,freq,sigmaBack))

    survey = simpegmt.SurveyMT.SurveyMT(srcList)
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

def calculateAnalyticSolution(srcList,mesh,model):
    surveyAna = simpegmt.SurveyMT.SurveyMT(srcList)
    data1D = simpegmt.DataMT.DataMT(surveyAna)
    for src in surveyAna.srcList:
        elev = src.rxList[0].locs[0]
        anaEd, anaEu, anaHd, anaHu = simpegmt.Utils.MT1Danalytic.getEHfields(mesh,model,src.freq,elev)
        anaE = anaEd+anaEu
        anaH = anaHd+anaHu
        # Scale the solution
        # anaE = (anaEtemp/anaEtemp[-1])#.conj()
        # anaH = (anaHtemp/anaEtemp[-1])#.conj()
        anaZ = anaE/anaH
        for rx in src.rxList:
            data1D[src,rx] = getattr(anaZ, rx.projComp)
    return data1D

def dataMis_AnalyticTotalDomain(sigmaHalf):

    # Make the survey

    # Total domain solution
    surveyTD, sigma, mesh = setupSurvey(sigmaHalf)
    problemTD = simpegmt.ProblemMT1D.eForm_TotalField(mesh)
    problemTD.pair(surveyTD)
    # Analytic data
    dataAnaObj = calculateAnalyticSolution(surveyTD.srcList,mesh,sigma)
    # dataTDObj = simpegmt.DataMT.DataMT(surveyTD, surveyTD.dpred(sigma))
    dataTD = surveyTD.dpred(sigma)
    dataAna = simpeg.mkvc(dataAnaObj)
    return np.all((dataTD - dataAna)/dataAna < 2.)
    # surveyTD.dtrue = -simpeg.mkvc(dataAna,2)
    # surveyTD.dobs = -simpeg.mkvc(dataAna,2)
    # surveyTD.Wd = np.ones(surveyTD.dtrue.shape) #/(np.abs(surveyTD.dtrue)*0.01)
    # # Setup the data misfit
    # dmis = simpeg.DataMisfit.l2_DataMisfit(surveyTD)
    # dmis.Wd = surveyTD.Wd
    # return dmis.eval(sigma)


def dataMis_AnalyticPrimarySecondary(sigmaHalf):

    # Make the survey
    # Primary secondary
    surveyPS, sigmaPS, mesh = setupSurvey(sigmaHalf,False)
    problemPS = simpegmt.ProblemMT1D.eForm_psField(mesh,sigmaPS)
    problemPS.pair(surveyPS)
    # Analytic data
    dataAna = calculateAnalyticSolution(surveyPS.srcList,mesh,sigma)

    surveyPS.dtrue = dataAna
    # Project the data
    data = surveyPS.dpred(sigmaPS)

    # Setup the data misfit
    dmis = simpeg.DataMisfit.l2_DataMisfit(survey)
    return dmis.eval(sigma)



class TestNumericVsAnalytics(unittest.TestCase):

    def setUp(self):
        pass
    # Total Fields
    # def test_appRes2en1(self):self.assertLess(appRes_TotalFieldNorm(2e-1), TOLr)
    # def test_appPhs2en1(self):self.assertLess(appPhs_TotalFieldNorm(2e-1), TOLp)

    def test_appRes2en2(self):self.assertTrue(dataMis_AnalyticTotalDomain(2e-2))
    # def test_appPhs2en2(self):self.assert(appPhs_TotalFieldNorm(2e-2), TOLp)

    # def test_appRes2en3(self):self.assertLess(appRes_TotalFieldNorm(2e-3), TOLr)
    # def test_appPhs2en3(self):self.assertLess(appPhs_TotalFieldNorm(2e-3), TOLp)

    # def test_appRes2en4(self):self.assertLess(appRes_TotalFieldNorm(2e-4), TOLr)
    # def test_appPhs2en4(self):self.assertLess(appPhs_TotalFieldNorm(2e-4), TOLp)

    # def test_appRes2en5(self):self.assertLess(appRes_TotalFieldNorm(2e-5), TOLr)
    # def test_appPhs2en5(self):self.assertLess(appPhs_TotalFieldNorm(2e-5), TOLp)

    # def test_appRes2en6(self):self.assertLess(appRes_TotalFieldNorm(2e-6), TOLr)
    # def test_appPhs2en6(self):self.assertLess(appPhs_TotalFieldNorm(2e-6), TOLp)

    # Primary/secondary
    # def test_appRes2en2_ps(self):self.assertLess(appRes_psFieldNorm(2e-2), TOLr)
    # def test_appPhs2en2_ps(self):self.assertLess(appPhs_psFieldNorm(2e-2), TOLp)

if __name__ == '__main__':
    unittest.main()