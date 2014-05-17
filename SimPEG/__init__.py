import numpy as np
import scipy.sparse as sp
import Utils
from Utils.SolverUtils import *
import Mesh
import Maps
import Problem
import Survey
import Regularization
import ObjFunction
import Optimization
import Directives
import Inversion
import Tests


import scipy.version as _v
if _v.version < '0.13.0':
    print 'Warning: upgrade your scipy to 0.13.0'
