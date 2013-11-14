import utils
from utils import Solver
import mesh
import inverse
import forward
import regularization


import scipy.version as _v
if _v.version < '0.13.0':
    print 'Warning: upgrade your scipy to 0.13.0'
