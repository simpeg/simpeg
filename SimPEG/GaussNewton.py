import numpy as np
import matplotlib.pyplot as plt
from pylab import norm

def GaussNewton(fctn, x0,maxIter=20, maxIterLS=10, LSreduction=1e-4, tolJ=1e-3, tolX=1e-3, 
                                                    tolG=1e-3, eps=1e-16, xStop=[]):
    """
        GaussNewton Optimization
        
        Input:
        ------
            fctn - objective Function (lambda function)
            x0   - starting guess
            
        Output:
        -------
            xOpt - numerical optimizer
    """
    # initial output
    print "%s GaussNewton %s" % ('='*22,'='*22)
    print "iter\tJc\t\tnorm(dJ)\tLS" 
    print "%s" % '-'*57
    
    # evaluate stopping criteria
    if xStop==[]:
        xStop=x0
    Jstop = fctn(xStop)
    print "%3d\t%1.2e" % (-1, Jstop[0])

    # initialize
    xc = x0
    STOP = np.zeros((5,1),dtype=bool)
    iterLS=0; iter=0
    
    Jold = Jstop
    xOld=xc
    while 1:
        # evaluate objective function
        Jc,dJ,H = fctn(xc)
        print "%3d\t%1.2e\t%1.2e\t%d" % (iter, Jc[0],norm(dJ),iterLS)
        
        # check stopping rules
        STOP[0] = (iter>0) & (abs(Jc[0]-Jold[0]) <= tolJ*(1+abs(Jstop[0])))
        STOP[1] = (iter>0) & (norm(xc-xOld)      <= tolX*(1+norm(x0)))
        STOP[2] = norm(dJ)                       <= tolG*(1+abs(Jstop[0]))
        STOP[3] = norm(dJ)                       <= 1e3*eps
        STOP[4] = (iter >= maxIter)
        if all(STOP[0:3]) | any(STOP[3:]): 
            break
        
        # get search direction
        dx = np.linalg.solve(H,-dJ)
        
        # Armijo linesearch
        descent = np.dot(dJ.T,dx)
        LS =0; t = 1; iterLS=1
        while  (iterLS<maxIterLS):
            xt = xc + t*dx
            Jt = fctn(xt)
            LS = Jt[0]<Jc[0]+t*LSreduction*descent
            if LS:
                break
            iterLS = iterLS+1
            t = .5*t
            
        # store old values
        Jold = Jc; xOld = xc
        # update 
        xc = xt
        iter = iter +1
        
    print "%s STOP! %s" % ('-'*25,'-'*25)
    print "%d : |Jc-Jold| = %1.4e <= tolJ*(1+|Jstop|) = %1.4e"  % (STOP[0],abs(Jc[0]-Jold[0]),tolJ*(1+abs(Jstop[0])))
    print "%d : |xc-xOld| = %1.4e <= tolX*(1+|x0|)    = %1.4e"  % (STOP[1],norm(xc-xOld),tolX*(1+norm(x0)))
    print "%d : |dJ|      = %1.4e <= tolG*(1+|Jstop|) = %1.4e"  % (STOP[2],norm(dJ),tolG*(1+abs(Jstop[0])))
    print "%d : |dJ|      = %1.4e <= 1e3*eps          = %1.4e"  % (STOP[3],norm(dJ),1e3*eps)
    print "%d : iter      = %3d\t <= maxIter\t       = %3d"     % (STOP[4],iter,maxIter)
    print "%s DONE! %s\n" % ('='*25,'='*25)
    
    return xc
      
def Rosenbrock(x):
    """
        Rosenbrock function for testing GaussNewton scheme
    """
    J   = 100*(x[1]-x[0]**2)**2+(1-x[0])**2 
    dJ  = np.array([2*(200*x[0]**3-200*x[0]*x[1]+x[0]-1),200*(x[1]-x[0]**2)])
    H = np.array([[-400*x[1]+1200*x[0]**2+2, -400*x[0]],[ -400*x[0], 200]],dtype=float);
    
    return J,dJ,H
    
def checkDerivative(fctn,x0):
    """
        Basic derivative check
        
        Compares error decay of 0th and 1st order Taylor approximation at point 
        x0 for a randomized search direction.
        
       Input:
       ------
         fctn  -  function handle
         x0    -  point at which to check derivative   
    """
    
    print "%s checkDerivative %s" % ('='*20,'='*20)
    print "iter\th\t\t|J0-Jt|\t\t|J0+h*dJ'*dx-Jt|"

    Jc,dJ,H = fctn(x0)
    
    dx = np.random.randn(len(x0),1)
    
    t  = np.logspace(-1,-10,10)
    E0 = np.zeros(t.shape)
    E1 = np.zeros(t.shape)
    
    for i in range(0,10):
        Jt = fctn(x0+t[i]*dx)
        E0[i] = norm(Jt[0]-Jc[0])                           # 0th order Taylor 
        E1[i] = norm(Jt[0]-Jc[0]-t[i]*np.dot(dJ.T,dx))      # 1st order Taylor 
        
        print "%d\t%1.2e\t%1.3e\t%1.3e" % (i,t[i],E0[i],E1[i])
        
    print "%s DONE! %s\n" % ('='*25,'='*25)
    plt.figure()
    plt.clf()
    plt.loglog(t,E0,'b')
    plt.loglog(t,E1,'g--')
    plt.title('checkDerivative')
    plt.xlabel('h')
    plt.ylabel('error of Taylor approximation')
    plt.legend(['0th order', '1st order'],loc='upper left')
    plt.show()
    return
    
if __name__ == '__main__':
    x0 = np.array([[2.6],[3.7]])
    fctn = lambda x:Rosenbrock(x)
    checkDerivative(fctn,x0)
    xOpt = GaussNewton(fctn,x0,maxIter=20)
    print "xOpt=[%f,%f]" % (xOpt[0],xOpt[1])
   