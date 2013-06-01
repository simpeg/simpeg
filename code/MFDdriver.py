# from scipy.sparse import linalg
from numpy import *
#from numpy.linalg import *
from numpy.random import randn
from utils import *
from getDiffOps import getCurlMatrix, getNodalGradient
from sputils import *
from meshUtils import *
from getFaceInnerProduct import getFaceInnerProduct
from getEdgeInnerProduct import getEdgeInnerProduct
#from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import *
from pylab import *

n1 = 14
n2 = 14
n3 = 15

X, Y, Z = ndgrid(linspace(0, 1, n1), linspace(0, 1, n2), linspace(0, 1, n3))
sigma = 1e-2*ones([n1-1, n2-1, n3-1])
sigma[:, :, (n3-1)/2:] = 1e-6
mu = 4*pi*1e-7*ones([n1-1, n2-1, n3-1])
w = 10

CURL = getCurlMatrix(X, Y, Z)
GRAD = getNodalGradient(X, Y, Z)
Mf = getFaceInnerProduct(X, Y, Z, 1/mu)
Me = getEdgeInnerProduct(X, Y, Z, sigma)

A = CURL.T * Mf * CURL + 1j * w * Me

ne = shape(A)
b = matrix(randn(ne[0])).T
# clean b
DIVb = GRAD.T*b
p = dsolve.spsolve(GRAD.T*GRAD, DIVb, use_umfpack=True).T
b = b - GRAD*p

#x = spsolve(A, b)
x = dsolve.spsolve(A, b, use_umfpack=True).T

t = norm(A*x-b)/norm(b)
print t
