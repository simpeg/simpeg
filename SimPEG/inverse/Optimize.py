import numpy as np
import matplotlib.pyplot as plt
from SimPEG.utils import mkvc, sdiag
norm = np.linalg.norm


class Minimize(object):
    """docstring for Minimize"""

    name = "GeneralOptimizationAlgorithm"

    maxIter = 20
    maxIterLS = 10
    LSreduction = 1e-4
    LSshorten = 0.5
    tolF = 1e-4
    tolX = 1e-4
    tolG = 1e-4
    eps = 1e-16

    def __init__(self, problem, **kwargs):
        self.problem = problem

        # Set the variables, throw an error if they don't exist.
        for attr in kwargs:
            if hasattr(self, attr):
                setattr(self, attr, kwargs[attr])
            else:
                raise Exception('%s attr is not recognized' % attr)

    def minimize(self, x0):

        self.startup(x0)
        self.printInit()

        while True:
            self.f, self.g, self.H = self.evalFunction(self.xc)
            self.printIter()
            if self.stoppingCriteria(): break
            p = self.findSearchDirection()
            xt, passLS = self.linesearch(p)
            if not passLS:
                xt = self.linesearchBreak(p)
            self.doEndIteration(xt)

        self.printDone()

        return self.xc

    def startup(self, x0):
        self._iter = 0
        self._iterLS = 0
        self._STOP = np.zeros((5,1),dtype=bool)

        self.x0 = x0
        self.xc = x0
        self.xOld = x0

    def printInit(self):
        print "%s %s %s" % ('='*22, self.name, '='*22)
        print "iter\tJc\t\tnorm(dJ)\tLS"
        print "%s" % '-'*57

    def printIter(self):
        print "%3d\t%1.2e\t%1.2e\t%d" % (self._iter, self.f, norm(self.g), self._iterLS)

    def printDone(self):
        print "%s STOP! %s" % ('-'*25,'-'*25)
        print "%d : |fc-fOld| = %1.4e <= tolF*(1+|fStop|) = %1.4e"  % (self._STOP[0], abs(self.f-self.fOld), self.tolF*(1+abs(self.fStop)))
        print "%d : |xc-xOld| = %1.4e <= tolX*(1+|x0|)    = %1.4e"  % (self._STOP[1], norm(self.xc-self.xOld), self.tolX*(1+norm(self.x0)))
        print "%d : |g|       = %1.4e <= tolG*(1+|fStop|) = %1.4e"  % (self._STOP[2], norm(self.g), self.tolG*(1+abs(self.fStop)))
        print "%d : |g|       = %1.4e <= 1e3*eps          = %1.4e"  % (self._STOP[3], norm(self.g), 1e3*self.eps)
        print "%d : iter      = %3d\t <= maxIter\t       = %3d"     % (self._STOP[4], self._iter, self.maxIter)
        print "%s DONE! %s\n" % ('='*25,'='*25)

    def evalFunction(self, x, doDerivative=True):
        f, g, H = self.problem(x)
        return f, g, H

    def findSearchDirection(self):
        return -self.g

    def stoppingCriteria(self):
        if self._iter == 0:
            self.fStop = self.f  # Save this for stopping criteria

        # check stopping rules
        self._STOP[0] = self._iter > 0 and (abs(self.f-self.fOld) <= self.tolF*(1+abs(self.fStop)))
        self._STOP[1] = self._iter > 0 and (norm(self.xc-self.xOld) <= self.tolX*(1+norm(self.x0)))
        self._STOP[2] = norm(self.g) <= self.tolG*(1+abs(self.fStop))
        self._STOP[3] = norm(self.g) <= 1e3*self.eps
        self._STOP[4] = self._iter >= self.maxIter
        return all(self._STOP[0:3]) | any(self._STOP[3:])

    def linesearch(self, p):
        # Armijo linesearch
        descent = np.inner(self.g, p)
        t = 1
        iterLS = 0
        while iterLS < self.maxIterLS:
            xt = self.xc + t*p
            ft, temp, temp = self.evalFunction(xt, doDerivative=False)
            if ft < self.f + t*self.LSreduction*descent:
                break
            iterLS += 1
            t = self.LSshorten*t

        self._iterLS = iterLS
        return xt, iterLS < self.maxIterLS

    def linesearchBreak(self, p):
        raise Exception('The linesearch got broken. Boo.')

    def doEndIteration(self, xt):
        # store old values
        self.fOld = self.f
        self.xOld, self.xc = self.xc, xt
        self._iter += 1


class GaussNewton(Minimize):
    name = 'GaussNewton'
    def findSearchDirection(self):
        return np.linalg.solve(self.H,-self.g)


class SteepestDescent(Minimize):
    name = 'SteepestDescent'
    def findSearchDirection(self):
        return -self.g



def Rosenbrock(x):
    """Rosenbrock function for testing GaussNewton scheme"""

    f = 100*(x[1]-x[0]**2)**2+(1-x[0])**2
    g = np.array([2*(200*x[0]**3-200*x[0]*x[1]+x[0]-1), 200*(x[1]-x[0]**2)])
    H = np.array([[-400*x[1]+1200*x[0]**2+2, -400*x[0]], [-400*x[0], 200]])
    return f, g, H


def checkDerivative(fctn, x0, num=7, plotIt=True, dx=None):
    """
        Basic derivative check

        Compares error decay of 0th and 1st order Taylor approximation at point
        x0 for a randomized search direction.

       Input:
       ------
         fctn  -  function handle
         x0    -  point at which to check derivative
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
        print "%s PASS! %s\n" % ('='*25, '='*25)
    else:
        print "%s\n%s FAIL! %s\n%s" % ('*'*57, '<'*25, '>'*25, '*'*57)

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
    x0 = np.array([2.6, 3.7])
    checkDerivative(Rosenbrock, x0, plotIt=False)
    xOpt = GaussNewton(Rosenbrock, maxIter=20).minimize(x0)
    print "xOpt=[%f, %f]" % (xOpt[0], xOpt[1])
    xOpt = SteepestDescent(Rosenbrock, maxIter=20, maxIterLS=15).minimize(x0)
    print "xOpt=[%f, %f]" % (xOpt[0], xOpt[1])

    def simplePass(x):
        return np.sin(x), sdiag(np.cos(x))

    def simpleFail(x):
        return np.sin(x), -sdiag(np.cos(x))

    checkDerivative(simplePass, np.random.randn(5), plotIt=False)
    checkDerivative(simpleFail, np.random.randn(5), plotIt=False)
