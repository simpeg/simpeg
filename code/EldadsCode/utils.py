from numpy import *
import numpy as np


def diff(A, d):
    if(d == 1):
        return A[1:, :, :] - A[:-1, :, :]
    elif(d == 2):
        return A[:, 1:, :] - A[:, :-1, :]
    else:
        return A[:, :, 1:] - A[:, :, :-1]
    #else:
    #    print('d must be 1,2 or 3')


def diffp(A, d1, d2):
    if(d1 == 1 and d2 == 2):
        return A[1:, 1:, :] - A[:-1, :-1, :]
    elif(d1 == 1 and d2 == 3):
        return A[1:, :, 1:] - A[:-1, :, :-1]
    else:
        return A[:, 1:, 1:] - A[:, :-1, :-1]


def diffm(A, d1, d2):
    if(d1 == 3 and d2 == 2):
        return A[:, :-1, 1:] - A[:, 1:, :-1]
    elif(d1 == 1 and d2 == 3):
        return A[1:, :, :-1] - A[:-1, :, 1:]
    elif(d1 == 2 and d2 == 1):
        return A[:-1, 1:, :] - A[1:, :-1, :]
    else:
        print('d must be 1, 2 or 3')


def ave(A, d):
    if(d == 1):
        return 0.5*(A[1:, :, :] + A[:-1, :, :])
    elif(d == 2):
        return 0.5*(A[:, 1:, :] + A[:, :-1, :])
    elif(d == 3):
        return 0.5*(A[:, :, 1:] + A[:, :, :-1])
    else:
        print('d must be 1,2 or 3')


def mkmat(x):
    return reshape(matrix(x), (size(x), 1), 'F')


def hstack3(a, b, c):
    a = mkvc(a)
    b = mkvc(b)
    c = mkvc(c)
    a = mkmat(a)
    b = mkmat(b)
    c = mkmat(c)
    return hstack((hstack((a, b)), c))


def ind2sub(shape, ind):
    """From the given shape, returns the subscrips of the given index"""
    revshp = []
    revshp.extend(shape)
    mult = [1]
    for i in range(0, len(revshp)-1):
        mult.extend([mult[i]*revshp[i]])
    mult = array(mult).reshape(len(mult))

    sub = []

    for i in range(0, len(shape)):
        sub.extend([math.floor(ind / mult[i])])
        ind = ind - (math.floor(ind/mult[i]) * mult[i])
    return sub


def sub2ind(shape, subs):
    """From the given shape, returns the index of the given subscript"""
    revshp = list(shape)
    mult = [1]
    for i in range(0, len(revshp)-1):
        mult.extend([mult[i]*revshp[i]])
    mult = array(mult).reshape(len(mult), 1)

    idx = dot((subs), (mult))
    return idx
