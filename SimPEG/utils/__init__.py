import matutils
import sputils
import lomutils
import interputils
import ModelBuilder
from matutils import getSubArray, mkvc, ndgrid, ind2sub, sub2ind
from sputils import spzeros, kron3, speye, sdiag
from lomutils import volTetra, faceInfo, inv2X2BlockDiagonal, inv3X3BlockDiagonal, indexCube, exampleLomGird
from interputils import interpmat
from ipythonUtils import easyAnimate as animate
import Solver
from Solver import Solver

def setKwargs(obj, **kwargs):
    """Sets key word arguments (kwargs) that are present in the object, throw an error if they don't exist."""
    for attr in kwargs:
        if hasattr(obj, attr):
            setattr(obj, attr, kwargs[attr])
        else:
            raise Exception('%s attr is not recognized' % attr)

def printTitles(obj, printers, name='Print Titles', pad=''):
    titles = ''
    widths = 0
    for printer in printers:
        titles += ('{:^%i}'%printer['width']).format(printer['title']) + ''
        widths += printer['width']
    print pad + "{0} {1} {0}".format('='*((widths-1-len(name))/2), name)
    print pad + titles
    print pad + "%s" % '-'*widths

def printLine(obj, printers, pad=''):
    values = ''
    for printer in printers:
        values += ('{:^%i}'%printer['width']).format(printer['format'] % printer['value'](obj))
    print pad + values

def checkStoppers(obj, stoppers):
    # check stopping rules
    optimal = []
    critical = []
    for stopper in stoppers:
        l = stopper['left'](obj)
        r = stopper['right'](obj)
        if stopper['stopType'] == 'optimal':
            optimal.append(l <= r)
        if stopper['stopType'] == 'critical':
            critical.append(l <= r)

    if obj.debug: print 'checkStoppers.optimal: ', optimal
    if obj.debug: print 'checkStoppers.critical: ', critical

    return (len(optimal)>0 and all(optimal)) | (len(critical)>0 and any(critical))

def printStoppers(obj, stoppers, pad='', stop='STOP!', done='DONE!'):
    print pad + "%s%s%s" % ('-'*25,stop,'-'*25)
    for stopper in stoppers:
        l = stopper['left'](obj)
        r = stopper['right'](obj)
        print pad + stopper['str'] % (l<=r,l,r)
    print pad + "%s%s%s" % ('-'*25,done,'-'*25)
