from numpy import *
from utils import diff, ave


def getVolume(X,Y,Z):
 
    # compute edge vectors   
    t1x = ave(ave(diff(X, 1),2),3)
    t1y = ave(ave(diff(Y, 1),2),3)
    t1z = ave(ave(diff(Z, 1),2),3)

    t2x = ave(ave(diff(X, 2),1),3)
    t2y = ave(ave(diff(Y, 2),1),3)
    t2z = ave(ave(diff(Z, 2),1),3)

    t3x = ave(ave(diff(X, 3),1),2)
    t3y = ave(ave(diff(Y, 3),1),2)
    t3z = ave(ave(diff(Z, 3),1),2)
    
    #   v = [t1x  t1y  t1z][i     j    k]
    #                      [t2x  t2y  t2z]
    #                      [t3x  t3y  t3z]
    
    v = t1x*(t2y*t3z - t2z*t3y) - t1y*(t2x*t3z - t2z*t3x) + t1z*(t2x*t3y-t2y*t3x) 
    
    return v
    
    
if __name__ == '__main__':

    X, Y, Z = mgrid[0:4, 0:5, 0:6]
    X = (1.0*X)/2
    v = getVolume(X, Y, Z)
    print v