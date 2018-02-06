import multiprocessing
import numpy as np

def f(x):

	a, b = x
	print(a, b)
	return np.dot(np.ones(int(1e6)), np.ones(int(1e6)))

class ClassName(object):
	"""docstring for ClassName"""
	def __init__(self):
		super(ClassName, self).__init__()

	def getJmat(self):
		pool = multiprocessing.Pool(8)
		results = pool.map(f, [(1, 2), (1, 3)])
		pool.close()
		pool.join()
		args = (1, 2)
		results = f(args)


		return results





if __name__ == '__main__':
    obj = ClassName()
    obj.getJmat()
