import numpy as np
from numpy.lib import recfunctions as recFunc

from ..frequency_domain.survey import Survey
from ...data import Data as BaseData
from ...utils import mkvc
from .sources import PlanewaveXYPrimary
from .receivers import PointNaturalSource, Point3DTipper
from .utils.plot_utils import DataNSEMPlotMethods

#########
# Data
#########


class Data(BaseData, DataNSEMPlotMethods):
    """Data class for NSEMdata.

    Stores the data vector indexed by the survey.

    Parameters
    ----------
    survey : SimPEG.survey.Survey
        Natural source EM survey
    dobs : numpy.ndarray
        Observed data
    relative_error : numpy.ndarray, optional
        Relative error
    noise_floor : numpy.ndarray, optional
        Noise floor
    """
    def __init__(self, survey, dobs=None, relative_error=None, noise_floor=None):
        BaseData.__init__(self, survey, dobs, relative_error, noise_floor)

    def toRecArray(self, returnType="RealImag"):
        """
        Returns a numpy.recarray for a SimpegNSEM impedance data object.
        
        Parameters
        ----------
        returnType : str, default: "RealImag:
            Switches between returning a receiver array where the impedance is split to real and imaginary ('RealImag') or is a complex ('Complex')
        
        Returns
        -------
        numpy.recarray
            Record array with data, with indexed columns
        """

        # Define the record fields
        dtRI = [
            ("freq", float),
            ("x", float),
            ("y", float),
            ("z", float),
            ("zxxr", float),
            ("zxxi", float),
            ("zxyr", float),
            ("zxyi", float),
            ("zyxr", float),
            ("zyxi", float),
            ("zyyr", float),
            ("zyyi", float),
            ("tzxr", float),
            ("tzxi", float),
            ("tzyr", float),
            ("tzyi", float),
        ]
        dtCP = [
            ("freq", float),
            ("x", float),
            ("y", float),
            ("z", float),
            ("zxx", complex),
            ("zxy", complex),
            ("zyx", complex),
            ("zyy", complex),
            ("tzx", complex),
            ("tzy", complex),
        ]

        for src in self.survey.source_list:
            # Temp array for all the receivers of the source.
            # Note: needs to be written more generally,
            # using diffterent rxTypes and not all the data at the locations
            # Assume the same locs for all RX
            locs = src.receiver_list[0].locations
            if locs.shape[1] == 1:
                locs = np.hstack((np.array([[0.0, 0.0]]), locs))
            elif locs.shape[1] == 2:
                locs = np.hstack((np.array([[0.0]]), locs))
            tArrRec = np.concatenate(
                (
                    src.frequency * np.ones((locs.shape[0], 1)),
                    locs,
                    np.nan * np.ones((locs.shape[0], 12)),
                ),
                axis=1,
            ).view(dtRI)
            # Get the type and the value for the DataNSEM object as a list
            typeList = [
                [rx.orientation, rx.component, self[src, rx]]
                for rx in src.receiver_list
            ]
            # Insert the values to the temp array
            for nr, (k, c, val) in enumerate(typeList):
                zt_type = "t" if "z" in k else "z"
                key = zt_type + k + c[0]
                tArrRec[key] = mkvc(val, 2)
            # Masked array

            try:
                outTemp = recFunc.stack_arrays((outTemp, tArrRec))
            except NameError:
                outTemp = tArrRec.copy()

            if "RealImag" in returnType:
                outArr = outTemp.copy()
            elif "Complex" in returnType:
                # Add the real and imaginary to a complex number
                outArr = np.empty(outTemp.shape, dtype=dtCP)
                for comp in ["freq", "x", "y", "z"]:
                    outArr[comp] = outTemp[comp].copy()
                for comp in ["zxx", "zxy", "zyx", "zyy", "tzx", "tzy"]:
                    outArr[comp] = (
                        outTemp[comp + "r"].copy() + 1j * outTemp[comp + "i"].copy()
                    )
            else:
                raise NotImplementedError(
                    "{:s} is not implemented, as to be RealImag or Complex."
                )

        # Return
        return outArr

    @classmethod
    def fromRecArray(cls, recArray, srcType="primary"):
        """Class method that reads in a numpy record array to NSEMdata object.

        Parameters
        ----------
        recArray : numpy.ndarray
            Record array with the data. Has to have ('freq','x','y','z') columns and some ('zxx','zxy','zyx','zyy','tzx','tzy')
        srcType : str, default: "primary"
            The type of SimPEG.EM.NSEM.SrcNSEM to be used. Either "primary" or "total"

        Returns
        -------
        SimPEG.electromagnetics.natural_source.sources.SrcNSEM
            Natural source
        """
        if srcType == "primary":
            src = PlanewaveXYPrimary
        elif srcType == "total":
            src = Planewave_xy_1DhomotD
        else:
            raise NotImplementedError("{:s} is not a valid source type for NSEMdata")

        # Find all the frequencies in recArray
        uniFreq = np.unique(recArray["freq"].copy())
        source_list = []
        dataList = []
        for freq in uniFreq:
            # Initiate receiver_list
            receiver_list = []
            # Find that data for freq
            dFreq = recArray[recArray["freq"] == freq].copy()
            # Find the impedance rxTypes in the recArray.
            rxTypes = [
                comp
                for comp in recArray.dtype.names
                if (len(comp) == 4 or len(comp) == 3) and "z" in comp
            ]
            for rxType in rxTypes:
                # Find index of not nan values in rxType
                notNaNind = ~np.isnan(dFreq[rxType].copy())
                if np.any(notNaNind):  # Make sure that there is any data to add.
                    locs = _rec_to_ndarr(dFreq[["x", "y", "z"]][notNaNind].copy())
                    if dFreq[rxType].dtype.name in "complex128":
                        if "t" in rxType:
                            receiver_list.append(
                                Point3DTipper(locs, rxType[1:3], "real")
                            )
                            dataList.append(dFreq[rxType][notNaNind].real.copy())
                            receiver_list.append(
                                Point3DTipper(locs, rxType[1:3], "imag")
                            )
                            dataList.append(dFreq[rxType][notNaNind].imag.copy())
                        elif "z" in rxType:
                            receiver_list.append(
                                PointNaturalSource(locs, rxType[1:3], "real")
                            )
                            dataList.append(dFreq[rxType][notNaNind].real.copy())
                            receiver_list.append(
                                PointNaturalSource(locs, rxType[1:3], "imag")
                            )
                            dataList.append(dFreq[rxType][notNaNind].imag.copy())
                    else:
                        component = "real" if "r" in rxType else "imag"
                        if "z" in rxType:
                            receiver_list.append(
                                PointNaturalSource(locs, rxType[1:3], component)
                            )
                            dataList.append(dFreq[rxType][notNaNind].copy())
                        if "t" in rxType:
                            receiver_list.append(
                                Point3DTipper(locs, rxType[1:3], component)
                            )
                            dataList.append(dFreq[rxType][notNaNind].copy())

            source_list.append(src(receiver_list, freq))

        # Make a survey
        survey = Survey(source_list)
        dataVec = np.hstack(dataList)
        return cls(survey, dataVec)


def _rec_to_ndarr(rec_arr, data_type=float):
    """
    Function to transform a numpy record array to a nd array.
    dupe of SimPEG.electromagnetics.natural_source.utils.rec_to_ndarr to avoid circular import
    """
    # fix for numpy >= 1.16.0
    # https://numpy.org/devdocs/release/1.16.0-notes.html#multi-field-views-return-a-view-instead-of-a-copy
    return np.array(
        recFunc.structured_to_unstructured(
            recFunc.repack_fields(rec_arr[list(rec_arr.dtype.names)])
        ),
        dtype=data_type,
    )
