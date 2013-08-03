import numpy as np
from utils import mkvc
import scipy.sparse.linalg as spla
import scipy.sparse as sp


def matmul(A,B):
    
    # first check shape
    if np.shape(A)[1] != np.shape(B)[0]:
    	print 'error in sizes'
    	return

    # Check types
    sA = sp.issparse(A)
    sB = sp.issparse(B)

    if ((sA == False) & (sB == True)):  # doesno't work unless we trick it
    	return (B.T.dot(A.T)).T
    else:
    	return A.dot(B)


def dot(A,B):
	A = mkvc(A,1)
	B = mkvc(B,1)
	return np.dot(A,B)

def inner(A,B):
	A = mkvc(A,1)
	B = mkvc(B,1)
	return np.dot(A,B)


if __name__ == '__main__':
	import numpy as np
	from utils import mkvc
	import scipy.sparse as sp

	# generate sparse and dense matrices
	A = sp.rand(100, 200, density=0.05, format='csr', dtype=None)
	B = sp.rand(200, 150, density=0.05, format='csr', dtype=None)
	C = np.random.rand(200,150)
	D = np.random.rand(150,100)
	b = mkvc(np.arange(200),1)
	c = np.reshape(b,(1,200))
	matmul(A,B)
	matmul(A,C)
	matmul(C,D)
	matmul(D,A)
	matmul(A,b)
	dot(c,b)
	dot(C,C)
	print np.shape(c), np.shape(b)[0]
	print matmul(c,b),dot(c,b)







