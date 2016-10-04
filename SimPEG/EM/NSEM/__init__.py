""" module SimPEG.EM.NSEM

SimPEG implementation of the natural source problem
(including magenetotelluric, tipper and ZTEM)



"""
from __future__ import absolute_import

from . import Utils
from . import SrcNSEM as Src
from . import RxNSEM as Rx
from .SurveyNSEM import Survey, Data
from .FieldsNSEM import Fields1D_ePrimSec, Fields3D_ePrimSec
from .ProblemNSEM import Problem1D_ePrimSec, Problem3D_ePrimSec
