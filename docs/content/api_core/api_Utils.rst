Utils
*****

.. automodule:: SimPEG.Utils
    :members:
    :undoc-members:

Version
=======

.. automodule:: SimPEG.Utils.printinfo
    :members:
    :undoc-members:


Matrix Utilities
================

.. automodule:: SimPEG.Utils.matutils
    :members:
    :undoc-members:


Solver Utilities
================

.. automodule:: SimPEG.Utils.SolverUtils
    :members:
    :undoc-members:

Curv Utilities
==============

.. automodule:: SimPEG.Utils.curvutils
    :members:
    :undoc-members:

Mesh Utilities
==============

.. automodule:: SimPEG.Utils.meshutils
    :members:
    :undoc-members:

Model Builder Utilities
=======================

.. automodule:: SimPEG.Utils.ModelBuilder
    :members:
    :undoc-members:

Interpolation Utilities
=======================

.. automodule:: discretize.utils.interputils
    :members:
    :undoc-members:

Counter Utilities
=================

.. code-block:: python
    :linenos:

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


.. code-block:: text
    :linenos:

    Counters:
      MyClass.MyMethod                        :      100

    Times:                                        mean      sum
      MyClass.MySecondMethod                  : 1.70e-06, 5.10e-04,  300x



The API
-------

.. automodule:: SimPEG.Utils.CounterUtils
    :members:
    :undoc-members:

