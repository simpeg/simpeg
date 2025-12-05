.. _version-compatibility:

Version compatibility
=====================

SimPEG follows the time-window based policy for support of Python and Numpy
versions introduced in `NEP29
<https://numpy.org/neps/nep-0029-deprecation_policy.html>`_. In summary, SimPEG supports:

- all minor versions of Python released in the **prior 42 months** before
  a SimPEG release, and
- all minor versions of Numpy released in the **prior 24 months** before
  a SimPEG release.

We follow these guidelines conservatively, meaning that we might not drop
support for older versions of our dependencies if they are not causing any
issue. We include notes in the :ref:`release_notes` every time we drop support
for a Python or Numpy version.


Supported Python versions
-------------------------

If you require support for older Python versions, please pin SimPEG to the
following releases to ensure compatibility:


.. list-table::
    :widths: 40 60

    * - **Python version**
      - **Last compatible release**
    * - 3.8
      - 0.22.2
    * - 3.9
      - 0.22.2
    * - 3.10
      - 0.24.0
