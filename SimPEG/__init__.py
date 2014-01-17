import numpy as np
import scipy.sparse as sp
import Utils
Solver = Utils.Solver
import Mesh
import Model
import Problem
import Data
import Inversion
import Optimization
import Regularization
import Examples
import Tests


import scipy.version as _v
if _v.version < '0.13.0':
    print 'Warning: upgrade your scipy to 0.13.0'
