# Test script to use simpegMT platform to forward model synthetic data.

# Import 
import simpegMT as simpegmt, SimPEG as simpeg 
import numpy as np

# Make a mesh
M = simpeg.Mesh.TensorMesh([[(100,5,-1.5),(100.,10),(100,5,1.5)],[(100,5,-1.5),(100.,10),(100,5,1.5)],[(100,5,1.6),(100.,10),(100,3,2)]], x0=['C','C',-3529.5360])
# Setup the model
conds = [1e-2,1]
sig = simpeg.Utils.ModelBuilder.defineBlock(M.gridCC,[-1000,-1000,-400],[1000,1000,-200],conds)
sig[M.gridCC[:,2]>0] = 1e-8
sig[M.gridCC[:,2]<-600] = 1e-1
sigBG = np.zeros(M.nC) + conds[0]
sigBG[M.gridCC[:,2]>0] = 1e-8

## Setup the the survey object
# Receiver locations
rx_x, rx_y = np.meshgrid(np.arange(-500,501,50),np.arange(-500,501,50))
rx_loc = np.hstack((simpeg.Utils.mkvc(rx_x,2),simpeg.Utils.mkvc(rx_y,2),np.zeros((np.prod(rx_x.shape),1))))
# Make a receiver list
rxList = []
for loc in rx_loc:
    # NOTE: loc has to be a (1,3) np.ndarray otherwise errors accure
    for rxType in ['zxxr','zxxi','zxyr','zxyi','zyxr','zyxi','zyyr','zyyi']:
        rxList.append(simpegmt.SurveyMT.RxMT(simpeg.mkvc(loc,2).T,rxType))
# Source list
srcList =[]
for freq in np.logspace(3,-1,5):
    srcList.append(simpegmt.SurveyMT.srcMT(freq,rxList))
# Survey MT 
survey = simpegmt.SurveyMT.SurveyMT(srcList)

## Setup the problem object
problem = simpegmt.ProblemMT.MTProblem(M)
problem.pair(survey)

problem.fields(sig,sigBG)
mtData = survey.projectFields(fields)

def torecarray(MTdata,returnType='RealImag'):
    '''
    Function that returns a numpy.recarray for a SimpegMT data object.

    '''

    def rec2ndarr(x,dt=float):
        return x.view((dt, len(x.dtype.names)))
    # Define the record fields
    dtRI = [('freq',float),('x',float),('y',float),('z',float),('zxxr',float),('zxxi',float),('zxyr',float),('zxyi',float),('zyxr',float),('zyxi',float),('zyyr',float),('zyyi',float)]
    dtCP = [('freq',float),('x',float),('y',float),('z',float),('zxx',complex),('zxy',complex),('zyx',complex),('zyy',complex)]
    impList = ['zxxr','zxxi','zxyr','zxyi','zyxr','zyxi','zyyr','zyyi']
    for src in MTdata.survey.srcList:
        # Temp array for all the receivers of the source.
        tArrRec = np.array([(src.freq,rx.locs[0,0],rx.locs[0,1],rx.locs[0,2],np.nan ,np.nan ,np.nan ,np.nan ,np.nan ,np.nan ,np.nan ,np.nan ) for rx in src.rxList],dtype=dtRI)
        # Get the type and the value for the mtdata object as a list
        typeList = [[rx.rxType,MTdata[src,rx][0]] for rx in src.rxList]
        # Insert the values to the temp array
        for nr,(key,val) in enumerate(typeList):
            tArrRec[key][nr] = val
        # Masked array 
        mArrRec = np.ma.MaskedArray(rec2ndarr(tArrRec),mask=np.isnan(rec2ndarr(tArrRec))).view(dtype=tArrRec.dtype)
        # Unique freq and loc of the masked array
        uniFLmarr = np.unique(mArrRec[['freq','x','y','z']])
        if 'RealImag' in returnType:
            dt = dtRI
            for uniFL in uniFLmarr:
                mTemp = rec2ndarr(mArrRec[np.ma.where(mArrRec[['freq','x','y','z']].data == np.array(uniFL))][impList]).sum(axis=0)
                try:
                    outArr = np.concatenate((outArr,simpeg.mkvc(np.concatenate((rec2ndarr(uniFL),mTemp.data)),2).T),axis=0)
                except NameError as e:
                    outArr = simpeg.mkvc(np.concatenate((rec2ndarr(uniFL),mTemp.data)),2).T
        elif 'Complex' in returnType:
            # Add the real and imaginary to a complex number
            dt = dtCP 
            for uniFL in uniFLmarr:
                mTemp = simpeg.mkvc(rec2ndarr(mArrRec[np.ma.where(mArrRec[['freq','x','y','z']].data == np.array(uniFL))][impList]).sum(axis=0),2).T
                dataBlock = np.sum(mTemp.data.reshape((mTemp.shape[0],4,2))*np.array([[[1,1j],[1,1j],[1,1j],[1,1j]]]),axis=2)
                try:
                    outArr = np.concatenate((outArr,simpeg.mkvc(np.concatenate((rec2ndarr(uniFL),dataBlock)),2).T),axis=0)
                except NameError as e:
                    outArr = simpeg.mkvc(np.concatenate((rec2ndarr(uniFL),dataBlock)),2).T

    # Return 
    return outArr.view(dt)
