from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.recfunctions as recFunc
from scipy.constants import mu_0
from scipy import interpolate as sciint

import SimPEG as simpeg
from SimPEG.EM.NSEM.SurveyNSEM import Survey, Data
from SimPEG.EM.NSEM.RxNSEM import (Point_impedance1D,
    Point_impedance3D, Point_tipper3D)
from SimPEG.EM.NSEM.SrcNSEM import Planewave_xy_1Dprimary
from SimPEG.EM.NSEM.Utils import MT1Danalytic, plotDataTypes as pDt



def rotate_data(NSEMdata, rot_angle):
    '''
    Function that rotates clockwise by rotation angle
        (- negative for a counter-clockwise rotation)

    **Required**
    :param SimPEG.EM.NSEM.Data NSEMdata: NSEM data object to process
    :param float rot_angle: Rotation angel in degrees,
        positive for clockwise rotation
    '''
    recData = NSEMdata.toRecArray('Complex')
    impData = rec_to_ndarr(
        recData[['zxx', 'zxy', 'zyx', 'zyy']].copy(), complex)
    # Make the rotation matrix
    # c,s,zxx,zxy,zyx,zyy = sympy.symbols('c,s,zxx,zxy,zyx,zyy')
    # rotM = sympy.Matrix([[c,-s],[s, c]])
    # zM = sympy.Matrix([[zxx,zxy],[zyx,zyy]])
    # rotM*zM*rotM.T
    # [c*(c*zxx - s*zyx) - s*(c*zxy - s*zyy), c*(c*zxy - s*zyy) + s*(c*zxx - s*zyx)],
    # [c*(c*zyx + s*zxx) - s*(c*zyy + s*zxy), c*(c*zyy + s*zxy) + s*(c*zyx + s*zxx)]])
    s = np.sin(-np.deg2rad(rot_angle))
    c = np.cos(-np.deg2rad(rot_angle))
    rotMat = np.array([[c, -s], [s, c]])
    rotData = (
        rotMat.dot(impData.reshape(-1, 2, 2).dot(rotMat.T))
    ).transpose(1, 0, 2).reshape(-1, 4)
    outRec = recData.copy()
    for nr, comp in enumerate(['zxx', 'zxy', 'zyx', 'zyy']):
        outRec[comp] = rotData[:, nr]

    return Data.fromRecArray(outRec)

# Function to get data and data info
def extract_data_info(NSEMdata):
    """
    Simple function that extracts data, frequency and receiver type lists.

    Useful when assigning uncertainties to data based on frequencies and
    receiver types.

    **Required**
    :param SimPEG.EM.NSEM.Data NSEMdata: NSEM data object to process

    """
    dL, freqL, rxTL = [], [], []

    for src in NSEMdata.survey.srcList:
        for rx in src.rxList:
            dL.append(NSEMdata[src, rx])
            freqL.append(np.ones(rx.nD) * src.freq)
            if isinstance(rx, Point_impedance3D):
                rxTL.extend((('z' + rx.orientation + ' ') * rx.nD).split())
            if isinstance(rx, Point_tipper3D):
                rxTL.extend((('t' + rx.orientation + ' ') * rx.nD).split())
    return np.concatenate(dL), np.concatenate(freqL), np.array(rxTL)


def resample_data(NSEMdata, locs='All', freqs='All', rxs='All', verbose=False):
    """
    Function that selects locations from all the receivers in the survey
    (uses the numerator location as a reference). Also gives the option
    of selecting frequencies and receiver.

    **Required**
    :param SimPEG.EM.NSEM.Data NSEMdata: NSEM data object to process

    **Optional**
    :param numpy.ndarray locs: receiver locations to use (default is 'All' locations)
    :param numpy.ndarray freqs: frequencies to use (default is 'All' frequencies))
    :param string rxs: list of receiver sting types to use (default is 'All' types)
        Can be any componation of ['zxx','zxy','zyx','zyy','tzx','tzy']
    """

    # Initiate new objects
    new_srcList = []
    data_list = []
    std_list = []
    floor_list = []

    # Sort out input frequencies
    if locs is 'All':
        locations = NSEMdata._unique_locations()
    elif isinstance(locs, np.ndarray):
        locations = locs
    else:
        raise IOError('Incorrect input type for locs. \n' +
                      'Can be \'All\' or ndarray ')
    # Sort out input frequencies
    if freqs is 'All':
        frequencies = NSEMdata.survey.freqs
    elif isinstance(freqs, np.ndarray):
        frequencies = freqs
    elif isinstance(freqs, list):
        frequencies = np.array(freqs)
    else:
        raise IOError('Incorrect input type for freqs. \n' +
                      'Can be \'All\'; ndarray or a list')
    # Sort out input rxs
    if rxs is 'All':
        rx_comp = True
    elif isinstance(rxs, list):
        rx_comp = []
        for rxT in rxs:
            if'z' in rxT[0]:
                rxtype = Point_impedance3D
            elif 't' in rxT[0]:
                rxtype = Point_tipper3D
            else:
                raise IOError('Unknown rx type string')
            orient = rxT[1:3]
            rx_comp.append((rxtype, orient))

    else:
        raise IOError('Incorrect input type for rxs. \n' +
                      'Can be \'All\' or a list')

    # Filter the data
    for src in NSEMdata.survey.srcList:
        if src.freq in frequencies:
            new_rxList = []
            for rx in src.rxList:
                if rx_comp is True or np.any(
                        [(isinstance(rx, ct) and rx.orientation in co)
                            for (ct, co) in rx_comp]):
                    if len(rx.locs.shape) == 3:
                        ind_loc = np.sum(
                            np.concatenate(
                                [(np.sqrt(np.sum((rx.locs[:, :, 0] - location) ** 2, axis=1)) < 0.1).reshape(-1, 1)
                                 for location in locations],
                                axis=1), axis=1, dtype=bool)
                        new_locs = rx.locs[ind_loc,:,:]
                    else:
                        ind_loc = np.sum(
                            np.concatenate(
                                [(np.sqrt(np.sum((rx.locs[:, :] - location) ** 2, axis=1)) < 0.1).reshape(-1, 1)
                                 for location in locations],
                                axis=1), axis=1, dtype=bool)
                        new_locs = rx.locs[ind_loc, :]
                    new_rx = type(rx)
                    new_rxList.append(
                        new_rx(new_locs, rx.orientation, rx.component))
                    data_list.append(NSEMdata[src, rx][ind_loc])
                    try:
                        std_list.append(
                            NSEMdata.standard_deviation[src, rx][ind_loc])
                        floor_list.append(NSEMdata.floor[src, rx][ind_loc])
                    except Exception as e:
                        if verbose:
                            print('No standard deviation or floor assigned')

            new_src = type(src)
            new_srcList.append(new_src(new_rxList, src.freq))

    survey = Survey(new_srcList)
    if std_list or floor_list:
        return Data(
            survey, np.concatenate(data_list),
            np.concatenate(std_list), np.concatenate(floor_list))
    else:
        return Data(survey, np.concatenate(data_list))

def convert3Dto1Dobject(NSEMdata, rxType3D='yx'):
    """
    Function that converts a 3D NSEMdata of a list of
    1D NSEMdata objects for running 1D inversions for.

    **Required**
    :param SimPEG.EM.NSEM.Data NSEMdata: NSEM data object to process

    **Optional**
    :param string rxType3D: component of the NSEMdata to use
        Can be 'xy', 'yx' or 'det'
    """

    # Find the unique locations
    # Need to find the locations
    recDataTemp = NSEMdata.toRecArray().data.flatten()
    # Check if survey.std has been assigned.
    ## NEED TO: write this...
    # Calculte and add the DET of the tensor to the recArray
    if 'det' in rxType3D:
        Zon = ((recDataTemp['zxxr'] + 1j * recDataTemp['zxxi']) *
               (recDataTemp['zyyr'] + 1j * recDataTemp['zyyi']))
        Zoff = ((recDataTemp['zxyr'] + 1j * recDataTemp['zxyi']) *
                (recDataTemp['zyxr'] + 1j * recDataTemp['zyxi']))
        det = np.sqrt(Zon - Zoff)
        recData = recFunc.append_fields(
            recDataTemp, ['zdetr', 'zdeti'], [det.real, det.imag])
    else:
        recData = recDataTemp

    uniLocs = rec_to_ndarr(np.unique(recData[['x', 'y', 'z']].copy()))
    mtData1DList = []
    if 'xy' in rxType3D:
        corr = -1
        # Shift the data to comply with the quadtrature of the 1d problem
    else:
        corr = 1
    for loc in uniLocs:
        # Make the receiver list
        rx1DList = []
        rx1DList.append(Point_impedance1D(simpeg.mkvc(loc, 2).T, 'real'))
        rx1DList.append(Point_impedance1D(simpeg.mkvc(loc, 2).T, 'imag'))
        # Source list
        locrecData = recData[np.sqrt(
            np.sum((rec_to_ndarr(
                recData[['x', 'y', 'z']].copy()) - loc)**2, axis=1)
        ) < 1e-5]
        dat1DList = []
        src1DList = []
        for freq in locrecData['freq']:
            src1DList.append(Planewave_xy_1Dprimary(rx1DList, freq))
            for comp in ['r', 'i']:
                dat1DList.append(
                    corr *
                    locrecData[rxType3D + comp][locrecData['freq'] == freq])

        # Make the survey
        sur1D = Survey(src1DList)

        # Make the data
        dataVec = np.hstack(dat1DList)
        dat1D = Data(sur1D, dataVec)
        sur1D.dobs = dataVec
        # Need to take NSEMdata.survey.std and split it as well.
        std = 0.05
        sur1D.std = np.abs(sur1D.dobs * std)
        mtData1DList.append(dat1D)

    # Return the the list of data.
    return mtData1DList

### Other utils, that don't take NSEM as an input
def appResPhs(freq, z):
    app_res = ((1. / (8e-7 * np.pi**2)) / freq) * np.abs(z)**2
    app_phs = np.arctan2(z.imag, z.real) * (180 / np.pi)
    return app_res, app_phs

def skindepth(rho, freq):
    ''' Function to calculate the skindepth of EM waves'''
    return np.sqrt((rho * ((1 / (freq * mu_0 * np.pi)))))

def rec_to_ndarr(rec_arr, data_type=float):
    """
    Function to transform a numpy record array to a nd array.
    """
    return rec_arr.copy().view((data_type, len(rec_arr.dtype.names)))

def makeAnalyticSolution(mesh, model, elev, freqs):


    data1D = []
    for freq in freqs:
        anaEd, anaEu, anaHd, anaHu = MT1Danalytic.getEHfields(
            mesh, model, freq, elev)
        anaE = anaEd + anaEu
        anaH = anaHd + anaHu

        anaZ = anaE / anaH
        # Add to the list
        data1D.append((freq, 0, 0, elev, anaZ[0]))
    dataRec = np.array(
        data1D,
        dtype=[
            ('freq', float), ('x', float), ('y', float),
            ('z', float), ('zyx', complex)])
    return dataRec


def plotMT1DModelData(problem, models, symList=None):

    # Setup the figure
    fontSize = 15

    fig = plt.figure(figsize=[9, 7])
    axM = fig.add_axes([0.075, .1, .25, .875])
    axM.set_xlabel('Resistivity [Ohm*m]', fontsize=fontSize)
    axM.set_xlim(1e-1, 1e5)
    axM.set_ylim(-10000, 5000)
    axM.set_ylabel('Depth [km]', fontsize=fontSize)
    axR = fig.add_axes([0.42, .575, .5, .4])
    axR.set_xscale('log')
    axR.set_yscale('log')
    axR.invert_xaxis()
    # axR.set_xlabel('Frequency [Hz]')
    axR.set_ylabel('Apparent resistivity [Ohm m]', fontsize=fontSize)

    axP = fig.add_axes([0.42, .1, .5, .4])
    axP.set_xscale('log')
    axP.invert_xaxis()
    axP.set_ylim(0, 90)
    axP.set_xlabel('Frequency [Hz]', fontsize=fontSize)
    axP.set_ylabel('Apparent phase [deg]', fontsize=fontSize)

    # if not symList:
    #   symList = ['x']*len(models)
    # Loop through the models.
    modelList = [problem.survey.mtrue]
    modelList.extend(models)
    if False:
        modelList = [problem.sigmaMap * mod for mod in modelList]
    for nr, model in enumerate(modelList):
        # Calculate the data
        if nr == 0:
            data1D = problem.dataPair(
                problem.survey, problem.survey.dobs).toRecArray('Complex')
        else:
            data1D = problem.dataPair(
                problem.survey,
                problem.survey.dpred(model)).toRecArray('Complex')
        # Plot the data and the model
        colRat = nr / ((len(modelList) - 1.999) * 1.)
        if colRat > 1.:
            col = 'k'
        else:
            col = plt.cm.seismic(1 - colRat)
        # The model - make the pts to plot
        meshPts = np.concatenate((
            problem.mesh.gridN[0:1],
            np.kron(problem.mesh.gridN[1::], np.ones(2))[:-1]))
        modelPts = np.kron(1. / (problem.sigmaMap * model), np.ones(2,))
        axM.semilogx(modelPts, meshPts, color=col)

        ## Data
        loc = rec_to_ndarr(np.unique(data1D[['x', 'y']]).copy())
        # Appres
        pDt.plotIsoStaImpedance(axR, loc, data1D, 'zyx', 'res', pColor=col)
        # Appphs
        pDt.plotIsoStaImpedance(axP, loc, data1D, 'zyx', 'phs', pColor=col)
        try:
            allData = np.concatenate((
                allData, simpeg.mkvc(data1D['zyx'], 2)), 1)
        except:
            allData = simpeg.mkvc(data1D['zyx'], 2)
    freq = simpeg.mkvc(data1D['freq'], 2)
    res, phs = appResPhs(freq, allData)

    if False:
        stdCol = 'gray'
        axRtw = axR.twinx()
        axRtw.set_ylabel('Std of log10', color=stdCol)
        [(t.set_color(stdCol), t.set_rotation(-45))
            for t in axRtw.get_yticklabels()]
        axPtw = axP.twinx()
        axPtw.set_ylabel('Std ', color=stdCol)
        [t.set_color(stdCol) for t in axPtw.get_yticklabels()]
        axRtw.plot(freq, np.std(np.log10(res), 1), '--', color=stdCol)
        axPtw.plot(freq, np.std(phs, 1), '--', color=stdCol)

    # Fix labels and ticks

    # yMtick = [l/1000 for l in axM.get_yticks().tolist()]
    # axM.set_yticklabels(yMtick)
    [l.set_rotation(90) for l in axM.get_yticklabels()]
    [l.set_rotation(90) for l in axR.get_yticklabels()]
    # [(t.set_color(stdCol), t.set_rotation(-45)) for t in axRtw.get_yticklabels()]
    # [t.set_color(stdCol) for t in axPtw.get_yticklabels()]
    for ax in [axM, axR, axP]:
        ax.xaxis.set_tick_params(labelsize=fontSize)
        ax.yaxis.set_tick_params(labelsize=fontSize)
    return fig

def plotImpAppRes(dataArrays, plotLoc, textStr=[]):
    '''
    Plots amplitude impedance and phase
    '''
    # Make the figure and axes
    fig,axT=plt.subplots(2, 2, sharex=True)
    axes = axT.ravel()
    fig.set_size_inches((13.5, 7.0))
    fig.suptitle(
        '{:s}\nStation at: {:.1f}x ; {:.1f}y'.format(
            textStr, plotLoc[0], plotLoc[1]))
    # Have to deal with axes
    # Set log
    for ax in axes.ravel():
        ax.set_xscale('log')

    axes[0].invert_xaxis()
    axes[0].set_yscale('log')
    axes[2].set_yscale('log')
    # Set labels
    axes[2].set_xlabel('Frequency [Hz]')
    axes[3].set_xlabel('Frequency [Hz]')
    axes[0].set_ylabel('Apperent resistivity [Ohm m]')
    axes[1].set_ylabel('Apperent phase [degrees]')
    axes[1].set_ylim(-180, 180)
    axes[2].set_ylabel('Impedance amplitude [V/A]')
    axes[3].set_ylim(-180, 180)
    axes[3].set_ylabel('Impedance angle [degrees]')


    # Plot the data
    for nr,dataArray in enumerate(dataArrays):
        if nr==1:
            parSym = '*'
        else:
            parSym = 's'
        # app res
        pDt.plotIsoStaImpedance(
            axes[0], plotLoc, dataArray, 'zxy', par='res', pSym=parSym)
        pDt.plotIsoStaImpedance(
            axes[0], plotLoc, dataArray, 'zyx', par='res', pSym=parSym)
        # app phs
        pDt.plotIsoStaImpedance(
            axes[1], plotLoc, dataArray, 'zxy', par='phs', pSym=parSym)
        pDt.plotIsoStaImpedance(
            axes[1], plotLoc, dataArray, 'zyx', par='phs', pSym=parSym)
        # imp abs
        pDt.plotIsoStaImpedance(
            axes[2], plotLoc, dataArray, 'zxx', par='abs', pSym=parSym)
        pDt.plotIsoStaImpedance(
            axes[2], plotLoc, dataArray, 'zxy', par='abs', pSym=parSym)
        pDt.plotIsoStaImpedance(
            axes[2], plotLoc, dataArray, 'zyx', par='abs', pSym=parSym)
        pDt.plotIsoStaImpedance(
            axes[2], plotLoc, dataArray, 'zyy', par='abs', pSym=parSym)
        # imp abs
        pDt.plotIsoStaImpedance(
            axes[3], plotLoc, dataArray, 'zxx', par='phs', pSym=parSym)
        pDt.plotIsoStaImpedance(
            axes[3], plotLoc, dataArray, 'zxy', par='phs', pSym=parSym)
        pDt.plotIsoStaImpedance(
            axes[3], plotLoc, dataArray, 'zyx', par='phs', pSym=parSym)
        pDt.plotIsoStaImpedance(
            axes[3], plotLoc, dataArray, 'zyy', par='phs', pSym=parSym)

    return (fig, axes)


def printTime():
    """
    Small function to print the current time
    """
    import time
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime()))
