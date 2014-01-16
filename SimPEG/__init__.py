import numpy as np
import scipy.sparse as sp
import utils
Solver = utils.Solver
import mesh
import forward
import inverse
import examples
import tests

Data = forward.Data

import scipy.version as _v
if _v.version < '0.13.0':
    print 'Warning: upgrade your scipy to 0.13.0'
