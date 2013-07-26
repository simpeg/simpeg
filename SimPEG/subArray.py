import numpy as np

def getSubArray(A,ind):
    """subArray"""
    i = ind[0]; j = ind[1]; k = ind[2]
    
    return A[i,:,:][:,j,:][:,:,k]
    