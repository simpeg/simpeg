Utils
*****

.. automodule:: SimPEG.utils
    :members:
    :undoc-members:


Matrix Utilities
================

.. automodule:: SimPEG.utils.mat_utils
    :members:
    :undoc-members:


Solver Utilities
================

.. automodule:: SimPEG.utils.solver_utils
    :members:
    :undoc-members:

Curv Utilities
==============

.. automodule:: SimPEG.utils.curv_utils
    :members:
    :undoc-members:

Mesh Utilities
==============

.. automodule:: SimPEG.utils.mesh_utils
    :members:
    :undoc-members:

Model Builder Utilities
=======================

.. automodule:: SimPEG.utils.model_builder
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

.. automodule:: SimPEG.utils.counter_utils
    :members:
    :undoc-members:
