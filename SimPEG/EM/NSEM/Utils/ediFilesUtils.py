from __future__ import print_function
# Functions to import and export MT EDI files.
from SimPEG import mkvc
from scipy.constants import mu_0
from numpy.lib import recfunctions as recFunc
from .dataUtils import rec_to_ndarr

# Import modules
import numpy as np
import os, sys, re


class EDIimporter:
    """
    A class to import EDIfiles.

    """


    # Define data converters
    _impUnitEDI2SI = 4*np.pi*1e-4 # Convert Z[mV/km/nT] (as in EDI)to Z[V/A] SI unit
    _impUnitSI2EDI = 1./_impUnitEDI2SI # ConvertZ[V/A] SI unit to Z[mV/km/nT] (as in EDI)

    # Properties
    filesList = None
    comps = None

    # Hidden properties
    _outEPSG = None # Project info
    _2out = None    # The projection operator


    def __init__(self, EDIfilesList, compList=None, outEPSG=None):

        # Set the fileList
        self.filesList = EDIfilesList
        # Set the components to import
        if compList is None:
            self.comps = ['ZXXR','ZXYR','ZYXR','ZYYR','ZXXI','ZXYI','ZYXI','ZYYI','ZXX.VAR','ZXY.VAR','ZYX.VAR','ZYY.VAR']
        else:
            self.comps = compList
        if outEPSG is not None:
            self._outEPSG = outEPSG

    def __call__(self,comps=None):

        if comps is None:
            return self._data

        return self._data[comps]

    def importFiles(self):
        """
        Function to import EDI files into a object.


        """

        # Constants that are needed for convertion of units

        # Temp lists
        tmpStaList = []

        tmpCompList = ['freq','x','y','z']
        tmpCompList.extend(self.comps)
        # Make the outarray
        dtRI = [(compS.lower().replace('.',''),float) for compS in tmpCompList]
        # Loop through all the files
        for nrEDI, EDIfile in enumerate(self.filesList):
            # Read the file into a list of the lines
            with open(EDIfile,'r') as fid:
                EDIlines = fid.readlines()
            # Find the location
            latD, longD, elevM = _findLatLong(EDIlines)
            # Transfrom coordinates
            transCoord = self._transfromPoints(longD,latD)
            # Extract the name of the file (station)
            EDIname = EDIfile.split(os.sep)[-1].split('.')[0]
            # Arrange the data
            staList = [EDIname, EDIfile, transCoord[0], transCoord[1], elevM[0]]
            # Add to the station list
            tmpStaList.extend(staList)

            # Read the frequency data
            freq = _findEDIcomp('>FREQ',EDIlines)
            # Make the temporary rec array.
            tArrRec = ( np.nan*np.ones( (len(freq),len(dtRI)) ) ).view(dtRI)     #np.concatenate((freq*np.ones((locs.shape[0],1)),locs,np.nan*np.ones((locs.shape[0],8))),axis=1).view(dtRI)
            # Add data to the array
            tArrRec['freq'] = mkvc(freq,2)
            tArrRec['x'] = mkvc(np.ones((len(freq),1))*transCoord[0],2)
            tArrRec['y'] = mkvc(np.ones((len(freq),1))*transCoord[1],2)
            tArrRec['z'] = mkvc(np.ones((len(freq),1))*elevM[0],2)
            for comp in self.comps:
                # Deal with converting units of the impedance tensor
                if 'Z' in comp:
                    unitConvert = self._impUnitEDI2SI
                else:
                    unitConvert = 1
                # Rotate the data since EDI x is *north, y *east but Simpeg uses x *east, y *north (* means internal reference frame)
                key = [comp.lower().replace('.','').replace(s,t) for s,t in [['xx','yy'],['xy','yx'],['yx','xy'],['yy','xx']] if s in comp.lower()][0]
                tArrRec[key] = mkvc(unitConvert*_findEDIcomp('>'+comp,EDIlines),2)
            # Make a masked array
            mArrRec = np.ma.MaskedArray(rec_to_ndarr(tArrRec),mask=np.isnan(rec_to_ndarr(tArrRec))).view(dtype=tArrRec.dtype)
            try:
                outTemp = recFunc.stack_arrays((outTemp,mArrRec))
            except NameError:
                outTemp = mArrRec

        # Assign the data
        self._data = outTemp

    # % Assign the data to the obj
    # nOutData=length(obj.data);
    # obj.data(nOutData+1:nOutData+length(TEMP.data),:) = TEMP.data;
    def _transfromPoints(self,longD,latD):
        # Import the coordinate projections
        try:
            import osr
        except ImportError as e:
            print('Could not import osr, missing the gdal package\nCan not project coordinates')
            raise e
        # Coordinates convertor
        if self._2out is None:
            src = osr.SpatialReference()
            src.ImportFromEPSG(4326)
            out = osr.SpatialReference()
            if self._outEPSG is None:
                # Find the UTM EPSG number
                Nnr =  700 if latD < 0.0 else 600
                utmZ = int(1+(longD+180.0)/6.0)
                self._outEPSG = 32000 + Nnr + utmZ
            out.ImportFromEPSG(self._outEPSG)
            self._2out = osr.CoordinateTransformation(src,out)
        # Return the transfrom
        return self._2out.TransformPoint(longD,latD)

# Hidden functions
def _findLatLong(fileLines):
    latDMS = np.array(fileLines[_findLine('LAT=',fileLines)[0]].split('=')[1].split()[0].split(':'),float)
    longDMS = np.array(fileLines[_findLine('LONG=',fileLines)[0]].split('=')[1].split()[0].split(':'),float)
    elevM = np.array([fileLines[_findLine('ELEV=',fileLines)[0]].split('=')[1].split()[0]],float)
    # Convert to D.ddddd values
    latS = np.sign(latDMS[0])
    longS = np.sign(longDMS[0])
    latD = latDMS[0] + latS*latDMS[1]/60 + latS*latDMS[2]/3600
    longD = longDMS[0] + longS*longDMS[1]/60 + longS*longDMS[2]/3600
    return latD, longD, elevM

def _findLine(comp,fileLines):
    """ Find a line number in the file"""
    # Line counter
    c = 0
    # List of indices for found lines
    found = []
    # Loop through all the lines
    for line in fileLines:
        if comp in line:
            # Append if found
            found.append(c)
        # Increse the counter
        c += 1
    # Return the found indices
    return found

def _findEDIcomp(comp,fileLines,dt=float):
    """
    Extract the data vector.

    Returns a list of the data.
    """
    # Find the data
    headLine, indHead = [(st,nr) for nr,st in enumerate(fileLines) if re.search(comp,st)][0]
    # Extract the data
    nrVec = int(headLine.split('//')[-1])
    c = 0
    dataList = []
    while c < nrVec:
        indHead += 1
        dataList.extend(fileLines[indHead].split())
        c = len(dataList)
    return np.array(dataList,dt)
