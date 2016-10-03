from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sys
import numpy as np
from numpy.lib import recfunctions as recFunc

from SimPEG import Survey as SimPEGsurvey, mkvc
from .RxNSEM import Point_impedance3D, Point_tipper3D
from .SrcNSEM import BaseNSEMSrc, Planewave_xy_1Dprimary, Planewave_xy_1DhomotD

#################
###  Survey   ###
#################


class Survey(SimPEGsurvey.BaseSurvey):
    """
    Survey class for NSEM. Contains all the sources associated with the survey.

    :param list srcList: List of sources associated with the survey

    """
    srcPair = BaseNSEMSrc

    def __init__(self, srcList, **kwargs):
        # Sort these by frequency
        self.srcList = srcList
        SimPEGsurvey.BaseSurvey.__init__(self, **kwargs)

        _freqDict = {}
        for src in srcList:
            if src.freq not in _freqDict:
                _freqDict[src.freq] = []
            _freqDict[src.freq] += [src]

        self._freqDict = _freqDict
        self._freqs = sorted([f for f in self._freqDict])

    @property
    def freqs(self):
        """Frequencies"""
        return self._freqs

    @property
    def nFreq(self):
        """Number of frequencies"""
        return len(self._freqDict)

    def getSrcByFreq(self, freq):
        """Returns the sources associated with a specific frequency."""
        assert freq in self._freqDict, "The requested frequency is not in this survey."
        return self._freqDict[freq]

    def eval(self, f):
        data = Data(self)
        for src in self.srcList:
            sys.stdout.flush()
            for rx in src.rxList:
                data[src, rx] = rx.eval(src, self.mesh, f)
        return data

    def evalDeriv(self, f):
        raise Exception('Use Sources to project fields deriv.')

#################
# Data
#################


class Data(SimPEGsurvey.Data):
    '''
    Data class for NSEMdata. Stores the data vector indexed by the survey.

    :param SimPEG.EM.NSEM.SurveyNSEM survey: NSEM survey object
    :param numpy.ndarray v: Vector of the data in order matching of the survey

    '''
    def __init__(self, survey, v=None):
        # Pass the variables to the "parent" method
        SimPEGsurvey.Data.__init__(self, survey, v)


    def toRecArray(self,returnType='RealImag'):
        '''
        Function that returns a numpy.recarray for a SimpegNSEM impedance data object.

        :param str returnType: Switches between returning a rec array where the impedance
            is split to real and imaginary ('RealImag') or is a complex ('Complex')
        :rtype: numpy.recarray
        :return: Record array with data, with indexed columns

        '''

        # Define the record fields
        dtRI = [('freq',float),('x',float),('y',float),('z',float),('zxxr',float),('zxxi',float),('zxyr',float),('zxyi',float),
        ('zyxr',float),('zyxi',float),('zyyr',float),('zyyi',float),('tzxr',float),('tzxi',float),('tzyr',float),('tzyi',float)]
        dtCP = [('freq',float),('x',float),('y',float),('z',float),('zxx',complex),('zxy',complex),('zyx',complex),('zyy',complex),('tzx',complex),('tzy',complex)]
        for src in self.survey.srcList:
            # Temp array for all the receivers of the source.
            # Note: needs to be written more generally, using diffterent rxTypes and not all the data at the locaitons
            # Assume the same locs for all RX
            locs = src.rxList[0].locs
            if locs.shape[1] == 1:
                locs = np.hstack((np.array([[0.0,0.0]]),locs))
            elif locs.shape[1] == 2:
                locs = np.hstack((np.array([[0.0]]),locs))
            tArrRec = np.concatenate((src.freq*np.ones((locs.shape[0],1)),locs,np.nan*np.ones((locs.shape[0],12))),axis=1).view(dtRI)
            # np.array([(src.freq,rx.locs[0,0],rx.locs[0,1],rx.locs[0,2],np.nan ,np.nan ,np.nan ,np.nan ,np.nan ,np.nan ,np.nan ,np.nan ) for rx in src.rxList],dtype=dtRI)
            # Get the type and the value for the DataNSEM object as a list
            typeList = [[rx.orientation,rx.component,self[src,rx]] for rx in src.rxList]
            # Insert the values to the temp array
            for k,c,val in typeList:
                key = 'z' + k + c[0]
                tArrRec[key] = mkvc(val,2)
            # Masked array
            # mArrRec = np.ma.MaskedArray(_rec_to_ndarr(tArrRec),mask=np.isnan(_rec_to_ndarr(tArrRec))).view(dtype=tArrRec.dtype)

            try:
                outTemp = recFunc.stack_arrays((outTemp,tArrRec))
            except NameError:
                outTemp = tArrRec.copy()

            if 'RealImag' in returnType:
                outArr = outTemp.copy()
            elif 'Complex' in returnType:
                # Add the real and imaginary to a complex number
                outArr = np.empty(outTemp.shape,dtype=dtCP)
                for comp in ['freq','x','y','z']:
                    outArr[comp] = outTemp[comp].copy()
                for comp in ['zxx','zxy','zyx','zyy','tzx','tzy']:
                    outArr[comp] = outTemp[comp+'r'].copy() + 1j*outTemp[comp+'i'].copy()
            else:
                raise NotImplementedError('{:s} is not implemented, as to be RealImag or Complex.')

        # Return
        return outArr

    @classmethod
    def fromRecArray(cls, recArray, srcType='primary'):
        """
        Class method that reads in a numpy record array to NSEMdata object.

        :param numpy.recarray recArray: Record array with the data. Has to have
            ('freq','x','y','z') columns and some ('zxx','zxy','zyx','zyy','tzx','tzy')
        :param string srcType: The type of SimPEG.EM.NSEM.SrcNSEM to be used

        """
        if srcType=='primary':
            src = Planewave_xy_1Dprimary
        elif srcType=='total':
            src = Planewave_xy_1DhomotD
        else:
            raise NotImplementedError('{:s} is not a valid source type for NSEMdata')

        # Find all the frequencies in recArray
        uniFreq = np.unique(recArray['freq'].copy())
        srcList = []
        dataList = []
        for freq in uniFreq:
            # Initiate rxList
            rxList = []
            # Find that data for freq
            dFreq = recArray[recArray['freq'] == freq].copy()
            # Find the impedance rxTypes in the recArray.
            rxTypes = [ comp for comp in recArray.dtype.names if (len(comp)==4 or len(comp)==3) and 'z' in comp]
            for rxType in rxTypes:
                # Find index of not nan values in rxType
                notNaNind = ~np.isnan(dFreq[rxType].copy())
                if np.any(notNaNind): # Make sure that there is any data to add.
                    locs = _rec_to_ndarr(dFreq[['x','y','z']][notNaNind].copy())
                    if dFreq[rxType].dtype.name in 'complex128':
                        if 'z' in rxType:
                            rxList.append(Point_impedance3D(locs,rxType[1:3],'real'))
                            dataList.append(dFreq[rxType][notNaNind].real.copy())
                            rxList.append(Point_impedance3D(locs,rxType[1:3],'imag'))
                            dataList.append(dFreq[rxType][notNaNind].imag.copy())
                        elif 't' in rxType:
                            rxList.append(Point_tipper3D(locs,rxType[1:3],'real'))
                            dataList.append(dFreq[rxType][notNaNind].real.copy())
                            rxList.append(Point_tipper3D(locs,rxType[1:3],'imag'))
                            dataList.append(dFreq[rxType][notNaNind].imag.copy())
                    else:
                        component = 'real' if 'r' in rxType else 'imag'
                        if 'z' in rxType:
                            rxList.append(
                                Point_impedance3D(locs, rxType[1:3], component))
                            dataList.append(dFreq[rxType][notNaNind].copy())
                        if 't' in rxType:
                            rxList.append(Point_tipper3D(locs, rxType[1:3], component))
                            dataList.append(dFreq[rxType][notNaNind].copy())

            srcList.append(src(rxList, freq))

        # Make a survey
        survey=Survey(srcList)
        dataVec=np.hstack(dataList)
        return cls(survey, dataVec)


def _rec_to_ndarr(rec_arr, data_type=float):
    """
    Function to transform a numpy record array to a nd array.
    """
    return rec_arr.view((data_type, len(rec_arr.dtype.names)))
