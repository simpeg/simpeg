# Utils used for the data,
import numpy as np, matplotlib.pyplot as plt, sys
import SimPEG as simpeg

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


def appResPhs(freq,z):
    app_res = ((1./(8e-7*np.pi**2))/freq)*np.abs(z)**2
    app_phs = np.arctan2(z.imag,z.real)*(180/np.pi)
    return app_res, app_phs

def rec2ndarr(x,dt=float):
    return x.view((dt, len(x.dtype.names)))

def makeAnalyticSolution(mesh,model,elev,freqs):
  data1D = []
  for freq in freqs:
      anaEd, anaEu, anaHd, anaHu = simpegmt.Utils.MT1Danalytic.getEHfields(mesh,model,freq,elev)
      anaE = anaEd+anaEu
      anaH = anaHd+anaHu

      anaZ = anaE/anaH
      # Add to the list
      data1D.append((freq,0,0,elev,anaZ[0]))
  dataRec = np.array(data1D,dtype=[('freq',float),('x',float),('y',float),('z',float),('zyx',complex)])
  return dataRec

def plotMT1DModelData(problem,models,symList=None):

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
    sys.path.append('/home/gudni/Dropbox/code/python/MTview')
    import plotDataTypes as pDt
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
        colRat = nr/((len(modelList)-2)*1.)
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