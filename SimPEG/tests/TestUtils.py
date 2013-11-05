import numpy as np
import matplotlib.pyplot as plt
from pylab import norm
from SimPEG.utils import mkvc, sdiag
from SimPEG import utils
from SimPEG.mesh import TensorMesh, LogicallyOrthogonalMesh
import numpy as np
import unittest
import inspect

happiness = ['The test be workin!', 'You get a gold star!', 'Yay passed!', 'Happy little convergence test!', 'That was easy!', 'Testing is important.', 'You are awesome.', 'Go Test Go!', 'Once upon a time, a happy little test passed.', 'And then everyone was happy.']
sadness = ['No gold star for you.','Try again soon.','Thankfully, persistence is a great substitute for talent.','It might be easier to call this a feature...','Coffee break?', 'Boooooooo  :(', 'Testing is important. Do it again.']

class OrderTest(unittest.TestCase):
    """

    OrderTest is a base class for testing convergence orders with respect to mesh
    sizes of integral/differential operators.

    Mathematical Problem:

        Given are an operator A and its discretization A[h]. For a given test function f
        and h --> 0  we compare:

        .. math::
            error(h) = \| A[h](f) - A(f) \|_{\infty}

        Note that you can provide any norm.

        Test is passed when estimated rate order of convergence is  at least within the specified tolerance of the
        estimated rate supplied by the user.

    Minimal example for a curl operator::

        class TestCURL(OrderTest):
            name = "Curl"

            def getError(self):
                # For given Mesh, generate A[h], f and A(f) and return norm of error.


                fun  = lambda x: np.cos(x)  # i (cos(y)) + j (cos(z)) + k (cos(x))
                sol = lambda x: np.sin(x)  # i (sin(z)) + j (sin(x)) + k (sin(y))


                Ex = fun(self.M.gridEx[:, 1])
                Ey = fun(self.M.gridEy[:, 2])
                Ez = fun(self.M.gridEz[:, 0])
                f = np.concatenate((Ex, Ey, Ez))

                Fx = sol(self.M.gridFx[:, 2])
                Fy = sol(self.M.gridFy[:, 0])
                Fz = sol(self.M.gridFz[:, 1])
                Af = np.concatenate((Fx, Fy, Fz))

                # Generate DIV matrix
                Ah = self.M.edgeCurl

                curlE = Ah*E
                err = np.linalg.norm((Ah*f -Af), np.inf)
                return err

            def test_order(self):
                # runs the test
                self.orderTest()

    See also: test_operatorOrder.py

    """

    name = "Order Test"
    expectedOrders = 2.  # This can be a list of orders, must be the same length as meshTypes
    tolerance = 0.85     # This can also be a list, must be the same length as meshTypes
    meshSizes = [4, 8, 16, 32]
    meshTypes = ['uniformTensorMesh']
    _meshType = meshTypes[0]
    meshDimension = 3

    def setupMesh(self, nc):
        """
        For a given number of cells nc, generate a TensorMesh with uniform cells with edge length h=1/nc.
        """
        if 'TensorMesh' in self._meshType:
            if 'uniform' in self._meshType:
                h1 = np.ones(nc)/nc
                h2 = np.ones(nc)/nc
                h3 = np.ones(nc)/nc
                h = [h1, h2, h3]
            elif 'random' in self._meshType:
                h1 = np.random.rand(nc)
                h2 = np.random.rand(nc)
                h3 = np.random.rand(nc)
                h = [hi/np.sum(hi) for hi in [h1, h2, h3]]  # normalize
            else:
                raise Exception('Unexpected meshType')

            self.M = TensorMesh(h[:self.meshDimension])
            max_h = max([np.max(hi) for hi in self.M.h])
            return max_h

        elif 'LOM' in self._meshType:
            if 'uniform' in self._meshType:
                kwrd = 'rect'
            elif 'rotate' in self._meshType:
                kwrd = 'rotate'
            else:
                raise Exception('Unexpected meshType')
            if self.meshDimension == 2:
                X, Y = utils.exampleLomGird([nc, nc], kwrd)
                self.M = LogicallyOrthogonalMesh([X, Y])
            if self.meshDimension == 3:
                X, Y, Z = utils.exampleLomGird([nc, nc, nc], kwrd)
                self.M = LogicallyOrthogonalMesh([X, Y, Z])
            return 1./nc

    def getError(self):
        """For given h, generate A[h], f and A(f) and return norm of error."""
        return 1.

    def orderTest(self):
        """
        For number of cells specified in meshSizes setup mesh, call getError
        and prints mesh size, error, ratio between current and previous error,
        and estimated order of convergence.


        """
        assert type(self.meshTypes) == list, 'meshTypes must be a list'
        if type(self.tolerance) is not list:
            self.tolerance = np.ones(len(self.meshTypes))*self.tolerance

        # if we just provide one expected order, repeat it for each mesh type
        if type(self.expectedOrders) == float or type(self.expectedOrders) == int:
            self.expectedOrders = [self.expectedOrders for i in self.meshTypes]

        assert type(self.expectedOrders) == list, 'expectedOrders must be a list'
        assert len(self.expectedOrders) == len(self.meshTypes), 'expectedOrders must have the same length as the meshTypes'

        for ii_meshType, meshType in enumerate(self.meshTypes):
            self._meshType = meshType
            self._tolerance = self.tolerance[ii_meshType]
            self._expectedOrder = self.expectedOrders[ii_meshType]

            order = []
            err_old = 0.
            max_h_old = 0.
            for ii, nc in enumerate(self.meshSizes):
                max_h = self.setupMesh(nc)
                err = self.getError()
                if ii == 0:
                    print ''
                    print self._meshType + ':  ' + self.name
                    print '_____________________________________________'
                    print '   h  |    error    | e(i-1)/e(i) |  order'
                    print '~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~'
                    print '%4i  |  %8.2e   |' % (nc, err)
                else:
                    order.append(np.log(err/err_old)/np.log(max_h/max_h_old))
                    print '%4i  |  %8.2e   |   %6.4f    |  %6.4f' % (nc, err, err_old/err, order[-1])
                err_old = err
                max_h_old = max_h
            print '---------------------------------------------'
            passTest = np.mean(np.array(order)) > self._tolerance*self._expectedOrder
            if passTest:
                print happiness[np.random.randint(len(happiness))]
            else:
                print 'Failed to pass test on ' + self._meshType + '.'
                print sadness[np.random.randint(len(sadness))]
            print ''
            self.assertTrue(passTest)

def Rosenbrock(x, return_g=True, return_H=True):
    """Rosenbrock function for testing GaussNewton scheme"""

    f = 100*(x[1]-x[0]**2)**2+(1-x[0])**2
    g = np.array([2*(200*x[0]**3-200*x[0]*x[1]+x[0]-1), 200*(x[1]-x[0]**2)])
    H = np.array([[-400*x[1]+1200*x[0]**2+2, -400*x[0]], [-400*x[0], 200]])

    out = (f,)
    if return_g:
        out += (g,)
    if return_H:
        out += (H,)
    return out

def checkDerivative(fctn, x0, num=7, plotIt=True, dx=None):
    """
        Basic derivative check

        Compares error decay of 0th and 1st order Taylor approximation at point
        x0 for a randomized search direction.

        :param lambda fctn: function handle
        :param numpy.array x0: point at which to check derivative
        :param int num: number of times to reduce step length, h
        :param bool plotIt: if you would like to plot
        :param numpy.array dx: step direction
        :rtype: bool
        :return: did you pass the test?!


        .. plot::
            :include-source:

            from SimPEG.tests import checkDerivative
            from SimPEG.utils import sdiag
            import numpy as np
            def simplePass(x):
                return np.sin(x), sdiag(np.cos(x))
            checkDerivative(simplePass, np.random.randn(5))
    """

    print "%s checkDerivative %s" % ('='*20, '='*20)
    print "iter\th\t\t|J0-Jt|\t\t|J0+h*dJ'*dx-Jt|\tOrder\n%s" % ('-'*57)

    Jc = fctn(x0)

    x0 = mkvc(x0)

    if dx is None:
        dx = np.random.randn(len(x0))

    t  = np.logspace(-1, -num, num)
    E0 = np.ones(t.shape)
    E1 = np.ones(t.shape)

    l2norm = lambda x: np.sqrt(np.inner(x, x))  # because np.norm breaks if they are scalars?
    for i in range(num):
        Jt = fctn(x0+t[i]*dx)
        E0[i] = l2norm(Jt[0]-Jc[0])               # 0th order Taylor
        if inspect.isfunction(Jc[1]):
            E1[i] = l2norm(Jt[0]-Jc[0]-t[i]*Jc[1](dx))  # 1st order Taylor
        else:
            # We assume it is a numpy.ndarray
            E1[i] = l2norm(Jt[0]-Jc[0]-t[i]*Jc[1].dot(dx))  # 1st order Taylor
        order0 = np.log10(E0[:-1]/E0[1:])
        order1 = np.log10(E1[:-1]/E1[1:])
        print "%d\t%1.2e\t%1.3e\t\t%1.3e\t\t%1.3f" % (i, t[i], E0[i], E1[i], np.nan if i == 0 else order1[i-1])

    tolerance = 0.9
    expectedOrder = 2
    eps = 1e-10
    order0 = order0[E0[1:] > eps]
    order1 = order1[E1[1:] > eps]
    belowTol = order1.size == 0 and order0.size > 0
    correctOrder = order1.size > 0 and np.mean(order1) > tolerance * expectedOrder

    passTest = belowTol or correctOrder

    if passTest:
        print "%s PASS! %s" % ('='*25, '='*25)
        print happiness[np.random.randint(len(happiness))]+'\n'
    else:
        print "%s\n%s FAIL! %s\n%s" % ('*'*57, '<'*25, '>'*25, '*'*57)
        print sadness[np.random.randint(len(sadness))]+'\n'


    if plotIt:
        plt.figure()
        plt.clf()
        plt.loglog(t, E0, 'b')
        plt.loglog(t, E1, 'g--')
        plt.title('checkDerivative')
        plt.xlabel('h')
        plt.ylabel('error of Taylor approximation')
        plt.legend(['0th order', '1st order'], loc='upper left')
        plt.show()

    return passTest


if __name__ == '__main__':

    def simplePass(x):
        return np.sin(x), sdiag(np.cos(x))

    def simpleFunction(x):
        return np.sin(x), lambda xi: sdiag(np.cos(x))*xi

    def simpleFail(x):
        return np.sin(x), -sdiag(np.cos(x))

    checkDerivative(simplePass, np.random.randn(5), plotIt=False)
    checkDerivative(simpleFunction, np.random.randn(5), plotIt=False)
    checkDerivative(simpleFail, np.random.randn(5), plotIt=False)
