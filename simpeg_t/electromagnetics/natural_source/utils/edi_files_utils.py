# Functions to import and export MT EDI files.
from simpeg import mkvc
from numpy.lib import recfunctions as recFunc
from .data_utils import rec_to_ndarr
from discretize.utils import requires

# Import modules
import numpy as np
import os
import re

try:
    import utm
except ImportError:
    utm = False


@requires({"utm": utm})
class EDIimporter:
    """
    A class to import EDIfiles.

    """

    # Define data converters
    # Convert Z[mV/km/nT] (as in EDI)to Z[V/A] SI unit
    _impUnitEDI2SI = 4 * np.pi * 1e-4
    # ConvertZ[V/A] SI unit to Z[mV/km/nT] (as in EDI)_
    _impUnitSI2EDI = 1.0 / _impUnitEDI2SI

    # Properties
    filesList = None
    comps = None

    # Hidden properties
    _outEPSG = None  # Project info
    _2out = None  # The projection operator

    def __init__(self, EDIfilesList, compList=None, outEPSG=None):
        # Set the fileList
        self.filesList = EDIfilesList
        # Set the components to import
        if compList is None:
            self.comps = [
                "ZXXR",
                "ZXYR",
                "ZYXR",
                "ZYYR",
                "ZXXI",
                "ZXYI",
                "ZYXI",
                "ZYYI",
                "ZXX.VAR",
                "ZXY.VAR",
                "ZYX.VAR",
                "ZYY.VAR",
            ]
        else:
            self.comps = compList
        if outEPSG is not None:
            self._outEPSG = outEPSG

    def __call__(self, comps=None):
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

        tmpCompList = ["freq", "x", "y", "z"]
        tmpCompList.extend(self.comps)
        # List of how to "rotate/shift" the data to comply with
        shift_list = [["xx", "yy"], ["xy", "yx"], ["yx", "xy"], ["yy", "xx"]]
        # Make the outarray
        dtRI = [(compS.lower().replace(".", ""), float) for compS in tmpCompList]
        # Loop through all the files
        for EDIfile in self.filesList:
            # Read the file into a list of the lines
            with open(EDIfile, "r") as fid:
                EDIlines = fid.readlines()
            # Find the location
            latD, longD, elevM = _findLatLong(EDIlines)
            # Transfrom coordinates
            transCoord = utm.from_latlon(latD, longD)
            # Extract the name of the file (station)
            EDIname = EDIfile.split(os.sep)[-1].split(".")[0]
            # Arrange the data
            staList = [EDIname, EDIfile, transCoord[0], transCoord[1], elevM[0]]
            # Add to the station list
            tmpStaList.extend(staList)

            # Read the frequency data
            freq = _findEDIcomp(">FREQ", EDIlines)
            # Make the temporary rec array.
            tArrRec = (np.nan * np.ones((len(freq), len(dtRI)))).view(dtRI)
            tArrRec["freq"] = mkvc(freq, 2)
            tArrRec["x"] = mkvc(np.ones((len(freq), 1)) * transCoord[0], 2)
            tArrRec["y"] = mkvc(np.ones((len(freq), 1)) * transCoord[1], 2)
            tArrRec["z"] = mkvc(np.ones((len(freq), 1)) * elevM[0], 2)
            for comp in self.comps:
                # Deal with converting units of the impedance tensor
                if "Z" in comp:
                    unitConvert = self._impUnitEDI2SI
                else:
                    unitConvert = 1
                # Rotate the data since EDI x is *north, y *east but Simpeg
                # uses x *east, y *north (* means internal reference frame)
                key = [
                    comp.lower().replace(".", "").replace(s, t)
                    for s, t in shift_list
                    if s in comp.lower()
                ][0]
                tArrRec[key] = mkvc(unitConvert * _findEDIcomp(">" + comp, EDIlines), 2)
            # Make a masked array
            mArrRec = np.ma.MaskedArray(
                rec_to_ndarr(tArrRec), mask=np.isnan(rec_to_ndarr(tArrRec))
            ).view(dtype=tArrRec.dtype)
            try:
                outTemp = recFunc.stack_arrays((outTemp, mArrRec))
            except NameError:
                outTemp = mArrRec

        # Assign the data
        self._data = outTemp


# Hidden functions
def _findLatLong(fileLines):
    latDMS = np.array(
        fileLines[_findLine("LAT=", fileLines)[0]].split("=")[1].split()[0].split(":"),
        float,
    )
    longDMS = np.array(
        fileLines[_findLine("LONG=", fileLines)[0]].split("=")[1].split()[0].split(":"),
        float,
    )
    elevM = np.array(
        [fileLines[_findLine("ELEV=", fileLines)[0]].split("=")[1].split()[0]], float
    )
    # Convert to D.ddddd values
    latS = np.sign(latDMS[0])
    longS = np.sign(longDMS[0])
    latD = latDMS[0] + latS * latDMS[1] / 60 + latS * latDMS[2] / 3600
    longD = longDMS[0] + longS * longDMS[1] / 60 + longS * longDMS[2] / 3600
    return latD, longD, elevM


def _findLine(comp, fileLines):
    """Find a line number in the file"""
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


def _findEDIcomp(comp, fileLines, dt=float):
    """
    Extract the data vector.

    Returns a list of the data.
    """
    # Find the data
    headLine, indHead = [
        (st, nr) for nr, st in enumerate(fileLines) if re.search(comp, st)
    ][0]
    # Extract the data
    if "NFREQ" in headLine:
        breakup = headLine.split("=")
        breakup2 = breakup[1].split()[0]
        # print(breakup, breakup2)
        nrVec = int(breakup2)
    else:
        nrVec = int(headLine.split("//")[-1])
    c = 0
    dataList = []
    while c < nrVec:
        indHead += 1
        dataList.extend(fileLines[indHead].split())
        c = len(dataList)
    return np.array(dataList, dt)
