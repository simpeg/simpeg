""" module SimPEG.EM.NSEM.Utils

Collection of utilities that are usefull for the NSEM problem

NOTE: These utilities are not well test, use with care

"""
from __future__ import absolute_import

from .solutions_1d import get1DEfields  # Add the names of the functions
from .analytic_1d import getEHfields, getImpedance
from .data_utils import (
    appResPhs,
    rec_to_ndarr,
    rotate_data,
    skindepth,
    makeAnalyticSolution,
    plotMT1DModelData,
    plotImpAppRes,
    printTime,
    convert3Dto1Dobject,
    resample_data,
    extract_data_info,
)
from .edi_files_utils import EDIimporter, _findLatLong, _findLine, _findEDIcomp
from .test_utils import (
    getAppResPhs,
    setup1DSurvey,
    setupSimpegNSEM_ePrimSec,
    random,
    halfSpace,
    blockInhalfSpace,
    twoLayer,
)
