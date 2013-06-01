from numpy import *

def ndgrid(x,y,z):
    
    n1 = size(x)
    n2 = size(y)
    n3 = size(z)
    X = zeros([n1,n2,n3])
    Y = zeros([n1,n2,n3])
    Z = zeros([n1,n2,n3])
    for i in range(0, n2):
        for j in range(0,n3):
            X[:,i,j] = x
            
    for i in range(0, n1):
        for j in range(0,n3):
            Y[i,:,j] = y
    
    for i in range(0, n1):
        for j in range(0,n2):
            Z[i,j,:] = z
            
                              
    return (X,Y,Z)


if __name__ == '__main__':

    X = ndgrid([1,2,3],[2,4,5,6],[4,6,7,8])
