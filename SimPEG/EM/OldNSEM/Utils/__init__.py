""" module SimPEG.EM.NSEM.Utils

Collection of utilities that are usefull for the NSEM problem

NOTE: These utilities are not well test, use with care

"""
from __future__ import absolute_import

from .MT1Dsolutions import get1DEfields  # Add the names of the functions
from .MT1Danalytic import getEHfields, getImpedance
from .dataUtils import (appResPhs, rec_to_ndarr, rotate_data,
                        skindepth, makeAnalyticSolution, plotMT1DModelData,
                        plotImpAppRes, printTime, convert3Dto1Dobject,
                        resample_data, extract_data_info)
from .ediFilesUtils import (EDIimporter,
                            _findLatLong, _findLine, _findEDIcomp)
from .testUtils import (getAppResPhs, setup1DSurvey, setupSimpegNSEM_ePrimSec,
                        random, halfSpace, blockInhalfSpace, twoLayer)
