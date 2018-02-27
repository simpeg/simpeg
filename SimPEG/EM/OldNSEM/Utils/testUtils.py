from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np

import SimPEG as simpeg
from SimPEG.Utils import meshTensor
from SimPEG.EM.NSEM.RxNSEM import Point_impedance1D, Point_impedance3D, Point_tipper3D
from SimPEG.EM.NSEM.SurveyNSEM import Survey
from SimPEG.EM.NSEM.SrcNSEM import Planewave_xy_1Dprimary, Planewave_xy_1DhomotD
from SimPEG.EM.NSEM.ProblemNSEM import Problem3D_ePrimSec
from .dataUtils import appResPhs

np.random.seed(1100)
# Define the tolerances
TOLr = 5e-2
TOLp = 5e-2


def getAppResPhs(NSEMdata):
    # Make impedance

    zList = []
    for src in NSEMdata.survey.srcList:
        zc = [src.freq]
        for rx in src.rxList:
            if 'imag' in rx.component:
                m = 1j
            else:
                m = 1
            zc.append(m*NSEMdata[src, rx])
        zList.append(zc)
    return [appResPhs(zList[i][0],np.sum(zList[i][1:3])) for i in np.arange(len(zList))]


def setup1DSurvey(sigmaHalf, tD=False, structure=False):

    # Frequency
    nFreq = 33
    freqs = np.logspace(3,-3,nFreq)
    # Make the mesh
    ct = 5
    air = meshTensor([(ct,25,1.3)])
    # coreT0 = meshTensor([(ct,15,1.2)])
    # coreT1 = np.kron(meshTensor([(coreT0[-1],15,1.3)]),np.ones((7,)))
    core = np.concatenate((np.kron(meshTensor([(ct, 15, -1.2)]), np.ones((10, ))), meshTensor([(ct, 20)])))
    bot = meshTensor([(core[0], 20, -1.3)])
    x0 = -np.array([np.sum(np.concatenate((core, bot)))])
    m1d = simpeg.Mesh.TensorMesh([np.concatenate((bot, core, air))], x0=x0)
    # Make the model
    sigma = np.zeros(m1d.nC) + sigmaHalf
    sigma[m1d.gridCC > 0 ] = 1e-8
    sigmaBack = sigma.copy()
    # Add structure
    if structure:
        shallow = (m1d.gridCC < -200) * (m1d.gridCC > -600)
        deep = (m1d.gridCC < -3000) * (m1d.gridCC > -5000)
        sigma[shallow] = 1
        sigma[deep] = 0.1

    rxList = []
    for rxType in ['z1d', 'z1d']:
        rxList.append(Point_impedance1D(simpeg.mkvc(np.array([0.0]), 2).T, 'real'))
        rxList.append(Point_impedance1D(simpeg.mkvc(np.array([0.0]), 2).T, 'imag'))
    # Source list
    srcList = []
    if tD:
        for freq in freqs:
            srcList.append(Planewave_xy_1DhomotD(rxList,freq))
    else:
        for freq in freqs:
            srcList.append(Planewave_xy_1Dprimary(rxList,freq))

    survey = Survey(srcList)
    return (survey, sigma, sigmaBack, m1d)


def setupSimpegNSEM_ePrimSec(inputSetup, comp='Imp', singleFreq=False, expMap=True):

    M, freqs, sig, sigBG, rx_loc = inputSetup
    # Make a receiver list
    rxList = []
    if comp == 'All':
        rx_type_list = ['xx', 'xy', 'yx', 'yy', 'zx', 'zy']
    elif comp == 'Imp':
        rx_type_list = ['xx', 'xy', 'yx', 'yy']
    elif comp == 'Tip':
        rx_type_list = ['zx', 'zy']
    else:
        rx_type_list = [comp]

    for rx_type in rx_type_list:
        if rx_type in ['xx', 'xy', 'yx', 'yy']:
            rxList.append(Point_impedance3D(rx_loc, rx_type, 'real'))
            rxList.append(Point_impedance3D(rx_loc, rx_type, 'imag'))
        if rx_type in ['zx', 'zy']:
            rxList.append(Point_tipper3D(rx_loc, rx_type, 'real'))
            rxList.append(Point_tipper3D(rx_loc, rx_type, 'imag'))

    # Source list
    srcList = []

    if singleFreq:
        srcList.append(Planewave_xy_1Dprimary(rxList, singleFreq))
    else:
        for freq in freqs:
            srcList.append(Planewave_xy_1Dprimary(rxList, freq))
    # Survey NSEM
    survey = Survey(srcList)

    # Setup the problem object
    sigma1d = M.r(sigBG, 'CC', 'CC', 'M')[0, 0, :]

    if expMap:
        problem = Problem3D_ePrimSec(M, sigmaPrimary=np.log(sigma1d))
        problem.sigmaMap = simpeg.Maps.ExpMap(problem.mesh)
        problem.model = np.log(sig)
    else:
        problem = Problem3D_ePrimSec(M, sigmaPrimary=sigma1d)
        problem.sigmaMap = simpeg.Maps.IdentityMap(problem.mesh)
        problem.model = sig
    problem.pair(survey)
    problem.verbose = False
    try:
        from pymatsolver import Pardiso
        problem.Solver = Pardiso
    except:
        pass

    return (survey, problem)


def getInputs():
    """
    Function that returns Mesh, freqs, rx_loc, elev.
    """
    # Make a mesh
    M = simpeg.Mesh.TensorMesh([[(200, 6, -1.5), (200., 4), (200, 6, 1.5)], [(200, 6, -1.5), (200., 4), (200, 6, 1.5)], [(200, 8, -1.5), (200., 8), (200, 8, 1.5)]],  x0=['C', 'C', 'C'])# Setup the model
    # Set the frequencies
    freqs = np.logspace(1, -3, 5)
    elev = 0

    # Setup the the survey object
    # Receiver locations
    rx_x, rx_y = np.meshgrid(np.arange(-350, 350, 200), np.arange(-350, 350, 200))
    rx_loc = np.hstack((simpeg.Utils.mkvc(rx_x, 2), simpeg.Utils.mkvc(rx_y, 2), elev+np.zeros((np.prod(rx_x.shape), 1))))

    return M, freqs, rx_loc, elev


def random(conds):
    """ Returns a random model based on the inputs"""
    M, freqs, rx_loc, elev = getInputs()

    # Backround
    sigBG = np.ones(M.nC)*conds
    # Add randomness to the model (10% of the value).
    sig = np.exp( np.log(sigBG) + np.random.randn(M.nC)*(conds)*1e-1 )

    return (M, freqs, sig, sigBG, rx_loc)


def halfSpace(conds):
    """ Returns a halfspace model based on the inputs"""
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
    """ Returns a block in a halfspace model based on the inputs"""
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
    """ Returns a 2 layer model based on the conductivity values given"""
    M, freqs, rx_loc, elev = getInputs()

    # Model
    ccM = M.gridCC
    groundInd = ccM[:, 2] < elev
    botInd = ccM[:, 2] < -3000
    sig = np.zeros(M.nC) + 1e-8
    sig[groundInd] = conds[1]
    sig[botInd] = conds[0]
    # Set the background, not the same as the model
    sigBG = np.zeros(M.nC) + 1e-8
    sigBG[groundInd] = conds[1]

    return (M, freqs, sig, sigBG, rx_loc)
