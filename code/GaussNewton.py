import numpy as np
from pylab import norm

def GaussNewton(x0, maxIter=20, maxIterLS=10, LSreduction=1e-4, tolJ=1e-3, tolX=1e-3, tolG=1e-3, eps=1e-16, xStop=np.empty):
    """
        GaussNewton optimization for Rosenbrock function (has to be generalized)
    """
    # initial output
    print "%s GaussNewton %s" % ('='*22,'='*22)
    print "iter\tJc\t\tnorm(dJ)\tLS" 
    print "%s" % '-'*57
    
    # evaluate stopping criteria
    if xStop==np.empty:
        xStop=x0
    Jstop = Rosenbrock(xStop)
    print "%3d\t%1.2e" % (-1, Jstop[0])

    # initialize
    xc = x0
    STOP = np.zeros((5,1),dtype=bool)
    iterLS=0; iter=0
    
    Jold = Jstop
    xOld=xc
    while 1:
        # evaluate objective function
        Jc,dJ,H = Rosenbrock(xc)
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
            Jt = Rosenbrock(xt)
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
    print "%d : |Jc-Jold|=%1.4e <= tolJ*(1+|Jstop|)=%1.4e"  % (STOP[0],abs(Jc[0]-Jold[0]),tolJ*(1+abs(Jstop[0])))
    print "%d : |xc-xOld|=%1.4e <= tolX*(1+|x0|)   =%1.4e"  % (STOP[1],norm(xc-xOld),tolX*(1+norm(x0)))
    print "%d : |dJ|     =%1.4e <= tolG*(1+|Jstop|)=%1.4e"  % (STOP[2],norm(dJ),tolG*(1+abs(Jstop[0])))
    print "%d : |dJ|     =%1.4e <= 1e3*eps         =%1.4e"  % (STOP[3],norm(dJ),1e3*eps)
    print "%d : iter     =%d\t\t <= maxIter         =%d"     % (STOP[4],iter,maxIter)
    print "%s DONE! %s\n" % ('='*25,'='*25)
    
    return xc
      
def Rosenbrock(x):
    """
        Rosenbrock function for testing GaussNewton scheme
    """
    J   = 100*(x[1]-x[0]**2)**2+(1-x[0])**2
    dJ  = np.array([-400*(x[1]*x[0]-x[0]**3)-2*(1-x[0]),200*(x[1]-x[0]**2)])
    H = np.array([[-400*x[1]+1200*x[0]**2+2, -400*x[0]],[ -400*x[0], 200]],dtype=float);
    
    return J,dJ,H
    
if __name__ == '__main__':
    x = np.array([[2.6],[3.7]])
    xOpt = GaussNewton(x,maxIter=20)
    print "xOpt=[%f,%f]" % (xOpt[0],xOpt[1])
   