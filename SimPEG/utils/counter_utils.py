from __future__ import print_function
from six import string_types
import time
import numpy as np
from functools import wraps


class Counter(object):
    """Class for recording iterations and operation times.
    Counter allows anything that calls it to record iterations and
    timings in a simple way.

    Also has plotting functions that allow quick recalls of data.

    Examples
    --------
    If you want to use this, import *count* or *timeIt* and use them as
    decorators on class methods.


    >>> from SimPEG.utils import Counter, count, timeIt

    >>> class MyClass(object):
    ...     def __init__(self, url):
    ...         self.counter = Counter()
    ...     @count
    ...     def MyMethod(self):
    ...         pass
    ...     @timeIt
    ...     def MySecondMethod(self):
    ...         pass

    >>> c = MyClass('blah')
    >>> for i in range(100): c.MyMethod()
    >>> for i in range(300): c.MySecondMethod()
    >>> c.counter.summary()
    Counters:
      MyClass.MyMethod                        :      100
    Times:                                        mean      sum
      MyClass.MySecondMethod                  : 9.08e-07, 2.72e-04,  300x

    """

    def __init__(self):
        self._countList = {}
        self._timeList = {}

    def count(self, prop):
        """Increases the count of the property.

        Parameters
        ----------
        prop : str
            The property being counted
        """
        assert isinstance(prop, string_types), "The property must be a string."
        if prop not in self._countList:
            self._countList[prop] = 0
        self._countList[prop] += 1

    def countTic(self, prop):
        """Times a property call, this is the init call.

        Parameters
        ----------
        prop : str
            The property being timed
        """
        assert isinstance(prop, string_types), "The property must be a string."
        if prop not in self._timeList:
            self._timeList[prop] = []
        self._timeList[prop].append(-time.time())

    def countToc(self, prop):
        """Times a property call, this is the end call.

        Parameters
        ----------
        prop : str
            The property being timed
        """
        assert isinstance(prop, string_types), "The property must be a string."
        assert prop in self._timeList, "The property must already be in the dictionary."
        self._timeList[prop][-1] += time.time()

    def summary(self):
        """
        Provides a text summary of the current counters and timers.
        """
        print("Counters:")
        for prop in sorted(self._countList):
            print("  {0:<40}: {1:8d}".format(prop, self._countList[prop]))
        print("\nTimes:" + " " * 40 + "mean      sum")
        for prop in sorted(self._timeList):
            l = len(self._timeList[prop])
            a = np.array(self._timeList[prop])
            print(
                "  {0:<40}: {1:4.2e}, {2:4.2e}, {3:4d}x".format(
                    prop, a.mean(), a.sum(), l
                )
            )


def count(f):
    """Count function (**DOCSTRING INCOMPLETE**)

    Returns
    -------
    wrapper
        Returns the wrapper
    """

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        counter = getattr(self, "counter", None)
        if isinstance(counter, Counter):
            counter.count(self.__class__.__name__ + "." + f.__name__)
        out = f(self, *args, **kwargs)
        return out

    return wrapper


def timeIt(f):
    """Timing function (**DOCSTRING INCOMPLETE**)

    Returns
    -------
    wrapper
        Returns the wrapper
    """

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        counter = getattr(self, "counter", None)
        if isinstance(counter, Counter):
            counter.countTic(self.__class__.__name__ + "." + f.__name__)
        out = f(self, *args, **kwargs)
        if isinstance(counter, Counter):
            counter.countToc(self.__class__.__name__ + "." + f.__name__)
        return out

    return wrapper
