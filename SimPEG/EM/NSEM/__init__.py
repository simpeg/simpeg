""" module SimPEG.EM.NSEM

SimPEG implementation of the natural source problem
(including magenetotelluric, tipper and ZTEM)



"""
from __future__ import absolute_import

import SimPEG.EM.NSEM.Utils
import SimPEG.EM.NSEM.SrcNSEM
from .SurveyNSEM import Survey, Data
from .RxNSEM import (rx_Point_impedance1D,
					 rx_Point_impedance3D, rx_Point_tipper3D)
from .FieldsNSEM import Fields1D_ePrimSec, Fields3D_ePrimSec
from .ProblemNSEM import Problem1D_ePrimSec, Problem3D_ePrimSec
