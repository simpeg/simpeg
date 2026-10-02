.. _deprecations:

Deprecating code
================

During the development of SimPEG we might run into situations in which we would like to change the way users interact with a particular piece of code, for example:

* renaming a function or class,
* renaming an argument or attribute,
* reorganizing objects into another submodule.

Applying such changes at once would introduce a *breaking change*, i.e. a change that breaks backward compatibility. This means that users' code written with a previous version of SimPEG will fail when they update to the next version.

Encoutering breaking changes can be frustrating. For this reason, any breaking change that removes a piece of code will be introduced through a **deprecation cycle**. Any deprecation cycle consists broadly in the following steps:

1. **Deprecate** the bit of code where the breaking change will be introduced.
2. Wait **two minor releases**.
3. **Remove** the deprecated bit of code (introduce a *breaking change*).

The goal is for users to get warnings on a future breaking change and instructions on how to update their code before the breaking change is finally introduced.

Deprecation
-----------

*Deprecation* is the act of discouraging the use of a particular bit of code, argument, attribute, property, etc.

We keep deprecated code in our code base during a sensible amount of time (usually two minor releases), make sure that it behaves in the same way as before, and keep its tests to guarantee that its behavior won't change until the breaking change is introduced.

We raise a :exc:`warnings.FutureWarning` any time that bit of code is accessed or used. Such warning should:

* warn them about its future removal or change,
* let them know in which future version the breaking change will get introduced,
* provide instructions on how to update their code.

.. hint::

    We use  :exc:`warnings.FutureWarning` to warn users about deprecations and future removals because this type of warnings are shown by default.
    Python also ships a :class:`warnings.DeprecationWarning`, which are intended for Python developers.
    By default, :class:`warnings.DeprecationWarning` are filtered and not shown.


Deprecating functions or classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Python offers built-in utilities to ease the deprecation process.
We can use the :func:`warnings.deprecated` (or :func:`typing_extensions.deprecated` for Python < 3.13) decorator to deprecate functions, classes, and methods.

For example, consider the ``gettopoCC`` function:

.. code:: python

   def gettopoCC(mesh, ind_active, option="top"):
       ...

that we want to rename to a PEP8 compliant name, like ``get_discrete_topography`` that will live in ``simpeg.utils.get_discrete_topography``. We can use the
:func:`~warnings.deprecated` decorator to deprecate it:

.. code:: python

   from warnings import deprecated

   @deprecated(
       "The `gettopoCC` function is deprecated, and will be removed in SimPEG v0.27.0. "
       "This functionality has been replaced by the 'get_discrete_topography' "
       "function, which can be imported from simpeg.utils.",
       category=FutureWarning,
   )
   def gettopoCC(mesh, ind_active, option="top"):
       ...

Anytime that the ``gettopoCC`` is called, we'll get the following warning:

.. code:: python

   mesh, ind_active = ..., ...
   gettopoCC(mesh, ind_active)

.. code::

   FutureWarning: The `gettopoCC` function is deprecated, and will be removed in SimPEG v0.27.0. This functionality has been replaced by the 'get_discrete_topography' function, which can be imported from simpeg.utils.

A similar procedure can be applied to classes:

.. code:: python

   @deprecated(
       "The `Simulation3D` class is deprecatd. "
       "It will be removed in SimPEG v0.28.0. "
       "Please use SimulationMagnetic instead."
   )
   class Simulation3D:

       ...

methods:

.. code:: python

   class SimulationMagnetic:

       @deprecated(
           "The `get_indActive` method is deprecated and will be removed in "
           "SimPEG v0.29.0. "
           "Please use the `active_cells` property instead."
       )
       def get_indActive(self):
           ...

       @property
       def active_cells(self):
           ...

and properties:

.. code:: python

   class SimulationMagnetic:

       @property
       @deprecated(
           "The `indActive` property is deprecated and will be removed in "
           "SimPEG v0.29.0. "
           "Please use the `active_cells` property instead."
       )
       def indActive(self):
           ...

       @indActive.setter
       @deprecated(
           "The `indActive` property is deprecated and will be removed in "
           "SimPEG v0.29.0. "
           "Please use the `active_cells` property instead."
       )
       def indActive(self, value):
           ...

       @property
       def active_cells(self):
           ...

       @active_cells.setter
       def active_cells(self, value):
           ...

Deprecating arguments
~~~~~~~~~~~~~~~~~~~~~

When renaming an argument from a function or method we'll need to ensure to raise a
:exc:`warnings.FutureWarning` within its body. For example, consider an ``InjectActiveCells`` map that takes ``indActive`` as argument for its constructor:

.. code:: python

   from simpeg.maps import IdentityMap

   class InjectActiveCells(IdentityMap):

       def __init__(self, mesh, indActive=None):
         self.indActive = indActive
         ...

We want to rename the ``indActive`` argument in favor of ``active_cells``. We can do so by deprecating the ``indActive`` argument first:

.. code:: python

   import warnings

   from simpeg.maps import IdentityMap

   class InjectActiveCells(IdentityMap):

       def __init__(self, mesh, active_cells=None, **kwargs):
         if "indActive" in kwargs:
             # Warn about deprecation
             msg = (
                 "Argument 'indActive' is deprecated and will be removed in "
                 "SimPEG v0.27.0. Use the 'active_cells' argument instead."
             )
             warnings.warn(msg, category=FutureWarning)

             # Raise error if both arguments are passed
             if active_cells is not None:
                 msg = "Invalid arguments: cannot pass 'active_cells' and 'indActive' simultanously."
                 raise TypeError(msg)

             # Assign active_cells to the passed indActive
             self.active_cells = kwargs.pop("indActive")
         else:
            self.active_cells = active_cells

         # Raise error if users passed any other argument through kwargs
         if kwargs:
             args = "', '".join(list(kwargs.keys()))
             msg = f"Invalid arguments '{args}' passed."
             raise TypeError(msg)


.. important::

   It's recommended to add tests to check the expected behaviour of the function
   to ensure that the warnings and errors are corretly raised, and the arguments
   correctly processed.


Documenting deprecations
~~~~~~~~~~~~~~~~~~~~~~~~

When deprecating a function, class, or argument it's a good practice to document such deprecation. We can use the ``deprecated`` admonition.

For example, we can add such admonition when deprecating a function:

.. code:: python

   @deprecated(
       "The `gettopoCC` function is deprecated, and will be removed in SimPEG v0.27.0. "
       "This functionality has been replaced by the 'get_discrete_topography' "
       "function, which can be imported from simpeg.utils.",
       category=FutureWarning,
   )
   def gettopoCC(mesh, ind_active, option="top"):
       """
       Generate discrete topography locations from mesh and active cells.

       .. deprecated:: 0.25.0

          The ``gettopoCC`` function is deprecated and will be removed in SimPEG v0.27.0. This function has been replaced by the :func:`simpeg.utils.get_discrete_topography` function.

       Parameters
       ----------
       """
       ...


.. note::

   Note that we can add the version in which the deprecation got introduced in the ``.. deprecated::`` line.


The same admonition can be added when deprecating arguments:

.. code:: python

   class InjectActiveCells(IdentityMap):
       """
       Map active cells model to all cell of a mesh.

       Parameters
       ----------
       mesh : discretize.BaseMesh
           A discxretize mesh.
       active_cells : numpy.ndarray
           Active cells array. Can be a boolean ``numpy.ndarray`` of length *mesh.nC*
           or a ``numpy.ndarray`` of ``int`` containing the indices of the active cells.
       ind_active : numpy.ndarray
           Active cells array. Can be a boolean ``numpy.ndarray`` of length *mesh.nC*
           or a ``numpy.ndarray`` of ``int`` containing the indices of the active cells.
           .. deprecated:: 0.25.0

              The ``indActive`` argument is deprecated and will be removed in
              SimPEG v0.27.0. Use the ``active_cells`` argument instead.
       ...
       """

       def __init__(self, mesh, active_cells=None, **kwargs):
           ...

Testing the deprecations
~~~~~~~~~~~~~~~~~~~~~~~~

It's recommended to test the deprecation warnings are raised when deprecating a bit of code.
We can easily check that with the :func:`pytest.raises` context manager.

For example, we could test if the previous ``InjectActiveCells`` map warns when passing ``indActive`` as argument:

.. code:: python

    import re
    import pytest

    def test_deprecated_indactive():
        """Test FutureWarning after passing indActive to InjectActiveCells."""
        mesh = ...
        active_cells = ...
        match = re.escape("Argument 'indActive' is deprecated and will be removed ")
        with pytest.raises(FutureWarning, match=match):
            InjectActiveCells(mesh, indActive=active_cells)

If ``InjectActiveCells.__init__`` throws a :exc:`warnings.FutureWarning` with a message that matches the passed pattern, the tests passes.
Otherwise, the test will fail.


Removals
--------

When removing a bit of code it's usually a good practice to raise an error when the old object is accessed, along with information about in which version it got removed and instructions on how to update their code. This way, even if the user missed the deprecation warning, they will receive some information.

.. important::

    Do not remove deprecated bits of code as part of another Pull Request that implements a new feature or fixes the bug.
    Removals of code should live in their own Pull Request, so it's easy to track down when and how that bit of code was removed.
