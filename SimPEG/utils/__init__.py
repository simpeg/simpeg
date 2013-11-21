import matutils
import sputils
import lomutils
import interputils
import ModelBuilder
from matutils import getSubArray, mkvc, ndgrid, ind2sub, sub2ind
from sputils import spzeros, kron3, speye, sdiag, ddx, av
from lomutils import volTetra, faceInfo, inv2X2BlockDiagonal, inv3X3BlockDiagonal, indexCube, exampleLomGird
from interputils import interpmat
from ipythonUtils import easyAnimate as animate
import Solver
from Solver import Solver
import Geophysics

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


import time
import numpy as np


class Counter(object):
    """
        Counter allows anything that calls it to record iterations and
        timings in a simple way.

        Also has plotting functions that allow quick recalls of data.

        If you want to use this, import *count* or *timeIt* and use them as decorators on class methods.

        .. ::

            class MyClass(object):
                def __init__(self, url):
                    self.counter = Counter()

                @count
                def MyMethod(self):
                    pass

                @timeIt
                def MySecondMethod(self):
                    pass

            c = MyClass('blah')
            for i in range(100): c.MyMethod()
            for i in range(300): c.MySecondMethod()
            c.counter.summary()

    """
    def __init__(self):
        self._countList = {}
        self._timeList = {}

    def count(self, prop):
        """
            Increases the count of the property.
        """
        assert type(prop) is str, 'The property must be a string.'
        if prop not in self._countList:
            self._countList[prop] = 0
        self._countList[prop] += 1

    def countTic(self, prop):
        """
            Times a property call, this is the init call.
        """
        assert type(prop) is str, 'The property must be a string.'
        if prop not in self._timeList:
            self._timeList[prop] = []
        self._timeList[prop].append(-time.time())

    def countToc(self, prop):
        """
            Times a property call, this is the end call.
        """
        assert type(prop) is str, 'The property must be a string.'
        assert prop in self._timeList, 'The property must already be in the dictionary.'
        self._timeList[prop][-1] += time.time()

    def summary(self):
        """
            Provides a text summary of the current counters and timers.
        """
        print 'Counters:'
        for prop in sorted(self._countList):
            print "  {0:<40}: {1:8d}".format(prop,self._countList[prop])
        print '\nTimes:'+' '*40+'mean      sum'
        for prop in sorted(self._timeList):
            l = len(self._timeList[prop])
            a = np.array(self._timeList[prop])
            print "  {0:<40}: {1:4.2e}, {2:4.2e}, {3:4d}x".format(prop,a.mean(),a.sum(),l)

def count(f):
    def wrapper(self,*args,**kwargs):
        counter = getattr(self,'counter',None)
        if type(counter) is Counter: counter.count(self.__class__.__name__+'.'+f.__name__)
        out = f(self,*args,**kwargs)
        return out
    return wrapper

def timeIt(f):
    def wrapper(self,*args,**kwargs):
        counter = getattr(self,'counter',None)
        if type(counter) is Counter: counter.countTic(self.__class__.__name__+'.'+f.__name__)
        out = f(self,*args,**kwargs)
        if type(counter) is Counter: counter.countToc(self.__class__.__name__+'.'+f.__name__)
        return out
    return wrapper
if __name__ == '__main__':
    class MyClass(object):
        def __init__(self, url):
            self.counter = Counter()

        @count
        def MyMethod(self):
            pass

        @timeIt
        def MySecondMethod(self):
            pass

    c = MyClass('blah')
    for i in range(100): c.MyMethod()
    for i in range(300): c.MySecondMethod()
    c.counter.summary()
