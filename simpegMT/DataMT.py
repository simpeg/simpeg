from SimPEG import Survey, Utils, Problem, np, sp, mkvc
from simpegMT.Utils import rec2ndarr
import simpegMT
from scipy.constants import mu_0
import sys
from numpy.lib import recfunctions as recFunc


############
### Data ###
############
class DataMT(Survey.Data):
    '''
    Data class for MTdata

    :param SimPEG survey object survey:
    :param v vector with data

    '''
    def __init__(self, survey, v=None):
        # Pass the variables to the "parent" method
        Survey.Data.__init__(self, survey, v)

    # # Import data
    # @classmethod
    # def fromEDIFiles():
    #     pass

    def toRecArray(self,returnType='RealImag'):
        '''
        Function that returns a numpy.recarray for a SimpegMT impedance data object.

        :param str returnType: Switches between returning a rec array where the impedance is split to real and imaginary ('RealImag') or is a complex ('Complex')

        '''

        # Define the record fields
        dtRI = [('freq',float),('x',float),('y',float),('z',float),('zxxr',float),('zxxi',float),('zxyr',float),('zxyi',float),('zyxr',float),('zyxi',float),('zyyr',float),('zyyi',float)]
        dtCP = [('freq',float),('x',float),('y',float),('z',float),('zxx',complex),('zxy',complex),('zyx',complex),('zyy',complex)]
        impList = ['zxxr','zxxi','zxyr','zxyi','zyxr','zyxi','zyyr','zyyi']
        for src in self.survey.srcList:
            # Temp array for all the receivers of the source.
            # Note: needs to be written more generally, using diffterent rxTypes and not all the data at the locaitons
            # Assume the same locs for all RX
            locs = src.rxList[0].locs
            if locs.shape[1] == 1:
                locs = np.hstack((np.array([[0.0,0.0]]),locs))
            elif locs.shape[1] == 2:
                locs = np.hstack((np.array([[0.0]]),locs))
            tArrRec = np.concatenate((src.freq*np.ones((locs.shape[0],1)),locs,np.nan*np.ones((locs.shape[0],8))),axis=1).view(dtRI)
            # np.array([(src.freq,rx.locs[0,0],rx.locs[0,1],rx.locs[0,2],np.nan ,np.nan ,np.nan ,np.nan ,np.nan ,np.nan ,np.nan ,np.nan ) for rx in src.rxList],dtype=dtRI)
            # Get the type and the value for the DataMT object as a list
            typeList = [[rx.rxType.replace('z1d','zyx'),self[src,rx]] for rx in src.rxList]
            # Insert the values to the temp array
            for nr,(key,val) in enumerate(typeList):
                tArrRec[key] = mkvc(val,2)
            # Masked array
            mArrRec = np.ma.MaskedArray(rec2ndarr(tArrRec),mask=np.isnan(rec2ndarr(tArrRec))).view(dtype=tArrRec.dtype)
            # Unique freq and loc of the masked array
            uniFLmarr = np.unique(mArrRec[['freq','x','y','z']]).copy()

            try:
                outTemp = recFunc.stack_arrays((outTemp,mArrRec))
                #outTemp = np.concatenate((outTemp,dataBlock),axis=0)
            except NameError as e:
                outTemp = mArrRec

            if 'RealImag' in returnType:
                outArr = outTemp
            elif 'Complex' in returnType:
                # Add the real and imaginary to a complex number
                outArr = np.empty(outTemp.shape,dtype=dtCP)
                for comp in ['freq','x','y','z']:
                    outArr[comp] = outTemp[comp].copy()
                for comp in ['zxx','zxy','zyx','zyy']:
                    outArr[comp] = outTemp[comp+'r'].copy() + 1j*outTemp[comp+'i'].copy()
            else:
                raise NotImplementedError('{:s} is not implemented, as to be RealImag or Complex.')

        # Return
        return outArr

    @classmethod
    def fromRecArray(cls, recArray, srcType='primary'):
        """
        Class method that reads in a numpy record array to MTdata object.

        Only imports the impedance data.

        """
        if srcType=='primary':
            src = simpegMT.SurveyMT.srcMT_polxy_1Dprimary
        elif srcType=='total':
            src = sdsimpegMT.SurveyMT.srcMT_polxy_1DhomotD
        else:
            raise NotImplementedError('{:s} is not a valid source type for MTdata')

        # Find all the frequencies in recArray
        uniFreq = np.unique(recArray['freq'])
        srcList = []
        dataList = []
        for freq in uniFreq:
            # Initiate rxList
            rxList = []
            # Find that data for freq
            dFreq = recArray[recArray['freq'] == freq].copy()
            # Find the impedance rxTypes in the recArray.
            rxTypes = [ comp for comp in recArray.dtype.names if len(comp)==4 and 'z' in comp and ('r' in comp or 'i' in comp)]
            for rxType in rxTypes:
                # Find index of not nan values in rxType
                notNaNind = ~np.isnan(dFreq[rxType])
                if np.any(notNaNind): # Make sure that there is any data to add.
                    locs = rec2ndarr(dFreq[['x','y','z']][notNaNind].copy())
                    rxList.append(simpegMT.SurveyMT.RxMT(locs,rxType))
                    dataList.append(dFreq[rxType][notNaNind].copy())
            srcList.append(src(rxList,freq))

        # Make a survey
        survey = simpegMT.SurveyMT.SurveyMT(srcList)
        dataVec = np.hstack(dataList)
        return cls(survey,dataVec)