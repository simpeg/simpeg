import numpy as np
import scipy.sparse as sp
import utils
from utils import Solver
import mesh
import forward
import inverse
import visualize
import examples
import tests

import scipy.version as _v
if _v.version < '0.13.0':
    print 'Warning: upgrade your scipy to 0.13.0'
