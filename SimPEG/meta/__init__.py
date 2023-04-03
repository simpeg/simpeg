"""
========================================================
Meta SimPEG Classes (:mod:`SimPEG.meta`)
========================================================
.. currentmodule:: SimPEG.meta

SimPEG's meta module defines tools for working with simulations representing
many smaller simulations working together to solve a geophysical problem.
A meta simulation is a simulation of simulations!

.. warning::

    These classes are under active development. Their conventions might change
    in the future in backward compatibility breaking ways.


Serial Simulations
==================

These classes implement the serial versions of the ``meta`` simulations.
As they do not have any extra dependencies they will be available
by default. They are also useful to check the setup of the problem before
moving on to the parallel implementations, as the calling structure will
match the parallel versions.

.. autosummary::
  :toctree: generated/

  MetaSimulation
  SumMetaSimulation
  RepeatedSimulation

Parallel Simulations
====================

There will be several version of meta simulations with different flavors
of parallelism used. Other than the multiprocessing based implementation,
they all will require extra packages beyond the standard SimPEG
requirements.

Multiprocessing
---------------
Coming soon!

Dask
----
Coming soon!

MPI
---
Coming soon!

Ray
---
Coming soon!

"""

from .simulation import MetaSimulation, SumMetaSimulation, RepeatedSimulation

from .multiprocessing import MultiprocessingMetaSimulation
