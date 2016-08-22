from __future__ import print_function
# Utils used for the data,
import numpy as np, matplotlib.pyplot as plt, sys
import SimPEG as simpeg
import numpy.lib.recfunctions as recFunc
from scipy.constants import mu_0
from scipy import interpolate as sciint

def getAppRes(MTdata):
    # Make impedance
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

def rotateData(MTdata, rotAngle):
    '''
    Function that rotates clockwist by rotAngle (- negative for a counter-clockwise rotation)
    '''
    recData = MTdata.toRecArray('Complex')
    impData = rec2ndarr(recData[['zxx','zxy','zyx','zyy']],complex)
    # Make the rotation matrix
    # c,s,zxx,zxy,zyx,zyy = sympy.symbols('c,s,zxx,zxy,zyx,zyy')
    # rotM = sympy.Matrix([[c,-s],[s, c]])
    # zM = sympy.Matrix([[zxx,zxy],[zyx,zyy]])
    # rotM*zM*rotM.T
    # [c*(c*zxx - s*zyx) - s*(c*zxy - s*zyy), c*(c*zxy - s*zyy) + s*(c*zxx - s*zyx)],
    # [c*(c*zyx + s*zxx) - s*(c*zyy + s*zxy), c*(c*zyy + s*zxy) + s*(c*zyx + s*zxx)]])
    s = np.sin(-np.deg2rad(rotAngle))
    c = np.cos(-np.deg2rad(rotAngle))
    rotMat = np.array([[c,-s],[s,c]])
    rotData = (rotMat.dot(impData.reshape(-1,2,2).dot(rotMat.T))).transpose(1,0,2).reshape(-1,4)
    outRec = recData.copy()
    for nr,comp in enumerate(['zxx','zxy','zyx','zyy']):
        outRec[comp] = rotData[:,nr]

    from SimPEG import MT
    return MT.Data.fromRecArray(outRec)


def appResPhs(freq, z):
    app_res = ((1./(8e-7*np.pi**2))/freq)*np.abs(z)**2
    app_phs = np.arctan2(z.imag,z.real)*(180/np.pi)
    return app_res, app_phs

def skindepth(rho, freq):
    ''' Function to calculate the skindepth of EM waves'''
    return np.sqrt( (rho*((1/(freq * mu_0 * np.pi )))))

def rec2ndarr(x, dt=float):
    return x.view((dt, len(x.dtype.names)))

def makeAnalyticSolution(mesh, model, elev, freqs):
    from SimPEG import MT
    data1D = []
    for freq in freqs:
        anaEd, anaEu, anaHd, anaHu = MT.Utils.MT1Danalytic.getEHfields(mesh,model,freq,elev)
        anaE = anaEd+anaEu
        anaH = anaHd+anaHu

        anaZ = anaE/anaH
        # Add to the list
        data1D.append((freq,0,0,elev,anaZ[0]))
    dataRec = np.array(data1D,dtype=[('freq',float),('x',float),('y',float),('z',float),('zyx',complex)])
    return dataRec

def plotMT1DModelData(problem, models, symList=None):
    from SimPEG import MT
    # Setup the figure
    fontSize = 15

    fig = plt.figure(figsize=[9,7])
    axM = fig.add_axes([0.075,.1,.25,.875])
    axM.set_xlabel('Resistivity [Ohm*m]',fontsize=fontSize)
    axM.set_xlim(1e-1,1e5)
    axM.set_ylim(-10000,5000)
    axM.set_ylabel('Depth [km]',fontsize=fontSize)
    axR = fig.add_axes([0.42,.575,.5,.4])
    axR.set_xscale('log')
    axR.set_yscale('log')
    axR.invert_xaxis()
    # axR.set_xlabel('Frequency [Hz]')
    axR.set_ylabel('Apparent resistivity [Ohm m]',fontsize=fontSize)

    axP = fig.add_axes([0.42,.1,.5,.4])
    axP.set_xscale('log')
    axP.invert_xaxis()
    axP.set_ylim(0,90)
    axP.set_xlabel('Frequency [Hz]',fontsize=fontSize)
    axP.set_ylabel('Apparent phase [deg]',fontsize=fontSize)

    # if not symList:
    #   symList = ['x']*len(models)
    from SimPEG.MT.Utils import plotDataTypes as pDt
    # Loop through the models.
    modelList = [problem.survey.mtrue]
    modelList.extend(models)
    if False:
        modelList = [problem.mapping.sigmaMap*mod for mod in modelList]
    for nr, model in enumerate(modelList):
        # Calculate the data
        if nr==0:
            data1D = problem.dataPair(problem.survey,problem.survey.dobs).toRecArray('Complex')
        else:
            data1D = problem.dataPair(problem.survey,problem.survey.dpred(model)).toRecArray('Complex')
        # Plot the data and the model
        colRat = nr/((len(modelList)-1.999)*1.)
        if colRat > 1.:
            col = 'k'
        else:
            col = plt.cm.seismic(1-colRat)
        # The model - make the pts to plot
        meshPts = np.concatenate((problem.mesh.gridN[0:1],np.kron(problem.mesh.gridN[1::],np.ones(2))[:-1]))
        modelPts = np.kron(1./(problem.mapping.sigmaMap*model),np.ones(2,))
        axM.semilogx(modelPts,meshPts,color=col)

        ## Data
        # Appres
        pDt.plotIsoStaImpedance(axR,np.array([0,0]),data1D,'zyx','res',pColor=col)
        # Appphs
        pDt.plotIsoStaImpedance(axP,np.array([0,0]),data1D,'zyx','phs',pColor=col)
        try:
            allData = np.concatenate((allData,simpeg.mkvc(data1D['zyx'],2)),1)
        except:
            allData = simpeg.mkvc(data1D['zyx'],2)
    freq = simpeg.mkvc(data1D['freq'],2)
    res, phs = appResPhs(freq,allData)

    stdCol = 'gray'
    axRtw = axR.twinx()
    axRtw.set_ylabel('Std of log10',color=stdCol)
    [(t.set_color(stdCol), t.set_rotation(-45)) for t in axRtw.get_yticklabels()]
    axPtw = axP.twinx()
    axPtw.set_ylabel('Std ',color=stdCol)
    [t.set_color(stdCol) for t in axPtw.get_yticklabels()]
    axRtw.plot(freq, np.std(np.log10(res),1),'--',color=stdCol)
    axPtw.plot(freq, np.std(phs,1),'--',color=stdCol)

    # Fix labels and ticks

    yMtick = [l/1000 for l in axM.get_yticks().tolist()]
    axM.set_yticklabels(yMtick)
    [ l.set_rotation(90) for l in axM.get_yticklabels()]
    [ l.set_rotation(90) for l in axR.get_yticklabels()]
    [(t.set_color(stdCol), t.set_rotation(-45)) for t in axRtw.get_yticklabels()]
    [t.set_color(stdCol) for t in axPtw.get_yticklabels()]
    for ax in [axM,axR,axP]:
        ax.xaxis.set_tick_params(labelsize=fontSize)
        ax.yaxis.set_tick_params(labelsize=fontSize)
    return fig

def printTime():
    import time
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime()))

def convert3Dto1Dobject(MTdata,rxType3D='zyx'):
    from SimPEG import MT
    # Find the unique locations
    # Need to find the locations
    recDataTemp = MTdata.toRecArray()
    # Check if survey.std has been assigned.
    ## NEED TO: write this...
    # Calculte and add the DET of the tensor to the recArray
    if 'det' in rxType3D:
        Zon = (recDataTemp['zxxr']+1j*recDataTemp['zxxi'])*(recDataTemp['zyyr']+1j*recDataTemp['zyyi'])
        Zoff = (recDataTemp['zxyr']+1j*recDataTemp['zxyi'])*(recDataTemp['zyxr']+1j*recDataTemp['zyxi'])
        det = np.sqrt(Zon.data - Zoff.data)
        recData = recFunc.append_fields(recDataTemp,['zdetr','zdeti'],[det.real,det.imag] )
    else:
        recData = recDataTemp

    uniLocs = rec2ndarr(np.unique(recData[['x','y','z']])).data
    mtData1DList = []
    if 'zxy' in rxType3D:
        corr = -1 # Shift the data to comply with the quadtrature of the 1d problem
    else:
        corr = 1
    for loc in uniLocs:
        # Make the receiver list
        rx1DList = []
        for rxType in ['z1dr','z1di']:
            rx1DList.append(MT.Rx(simpeg.mkvc(loc,2).T,rxType))
        # Source list
        locrecData = recData[np.sqrt(np.sum( (rec2ndarr(recData[['x','y','z']]).data - loc )**2,axis=1)) < 1e-5]
        dat1DList = []
        src1DList = []
        for freq in locrecData['freq']:
            src1DList.append(MT.SrcMT.src_polxy_1Dprimary(rx1DList,freq))
            for comp  in ['r','i']:
                dat1DList.append( corr * locrecData[rxType3D+comp][locrecData['freq']== freq].data )

        # Make the survey
        sur1D = MT.Survey(src1DList)

        # Make the data
        dataVec = np.hstack(dat1DList)
        dat1D = MT.Data(sur1D,dataVec)
        sur1D.dobs = dataVec
        # Need to take MTdata.survey.std and split it as well.
        std=0.05
        sur1D.std =  np.abs(sur1D.dobs*std) #+ 0.01*np.linalg.norm(sur1D.dobs)
        mtData1DList.append(dat1D)

    # Return the the list of data.
    return mtData1DList

def resampleMTdataAtFreq(MTdata,freqs):
    """
    Function to resample MTdata at set of frequencies

    """
    from SimPEG import MT
    # Make a rec array
    MTrec = MTdata.toRecArray().data

    # Find unique locations
    uniLoc = np.unique(MTrec[['x','y','z']])
    uniFreq = MTdata.survey.freqs
    # Get the comps
    dNames = MTrec.dtype

    # Loop over all the locations and interpolate
    for loc in uniLoc:
        # Find the index of the station
        ind = np.sqrt(np.sum((rec2ndarr(MTrec[['x','y','z']]) - rec2ndarr(loc))**2,axis=1)) < 1. # Find dist of 1 m accuracy
        # Make a temporary recArray and interpolate all the components
        tArrRec = np.concatenate((simpeg.mkvc(freqs,2),np.ones((len(freqs),1))*rec2ndarr(loc),np.nan*np.ones((len(freqs),12))),axis=1).view(dNames)
        for comp in ['zxxr','zxxi','zxyr','zxyi','zyxr','zyxi','zyyr','zyyi','tzxr','tzxi','tzyr','tzyi']:
            int1d = sciint.interp1d(MTrec[ind]['freq'],MTrec[ind][comp],bounds_error=False)
            tArrRec[comp] = simpeg.mkvc(int1d(freqs),2)

        # Join together
        try:
            outRecArr = recFunc.stack_arrays((outRecArr,tArrRec))
        except NameError:
            outRecArr = tArrRec

    # Make the MTdata and return
    return MT.Data.fromRecArray(outRecArr)
