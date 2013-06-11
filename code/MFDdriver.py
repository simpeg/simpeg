import numpy as np
from numpy.random import randn
from utils import ndgrid
from getDiffOps import getCurlMatrix, getNodalGradient
from getFaceInnerProduct import getFaceInnerProduct
from getEdgeInnerProduct import getEdgeInnerProduct
from scipy.sparse.linalg import dsolve
from pylab import norm

n = np.array([14, 14, 15])

X, Y, Z = ndgrid(*[np.linspace(0, 1, x) for x in n])
sigma = 1e-2*np.ones(n-1)
sigma[:, :, (n[2]-1)/2:] = 1e-6
mu = 4*np.pi*1e-7*np.ones(n-1)
w = 10

CURL = getCurlMatrix(X, Y, Z)
GRAD = getNodalGradient(X, Y, Z)
Mf = getFaceInnerProduct(X, Y, Z, 1/mu)
Me = getEdgeInnerProduct(X, Y, Z, sigma)

A = CURL.T * Mf * CURL + 1j * w * Me

ne = np.shape(A)
b = np.matrix(randn(ne[0])).T
# clean b
DIVb = GRAD.T*b
p = dsolve.spsolve(GRAD.T*GRAD, DIVb, use_umfpack=True).T
b = b - GRAD*p

#x = spsolve(A, b)
x = dsolve.spsolve(A, b, use_umfpack=True).T

t = norm(A*x-b)/norm(b)
print t
