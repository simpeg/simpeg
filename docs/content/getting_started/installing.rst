.. _api_installing:

Getting Started with SimPEG
***************************


.. _installing_python:

Prerequisite: Installing Python
===============================

SimPEG is written in Python_!
We highly recommend installing it using Anaconda_ (or the alternative Miniforge_).
It installs `Python <https://www.python.org/>`_,
`Jupyter <https://jupyter.org/>`_ and other core
Python libraries for scientific computing.
If you and Python_ are not yet acquainted, we highly
recommend checking out `Software Carpentry <https://software-carpentry.org/>`_.

.. note::

   As of version 0.11.0, we will no longer ensure compatibility with Python 2.7. Please use
   the latest version of Python 3 with SimPEG. For more information on the transition of the
   Python ecosystem to Python 3, please see the `Python 3 Statement <https://python3statement.org/>`_.

.. image:: https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/220px-Python-logo-notext.svg.png
    :align: right
    :width: 100
    :target: https://www.python.org/

.. _Python: https://www.python.org/
.. _Anaconda: https://www.anaconda.com/products/individual
.. _Miniforge: https://github.com/conda-forge/miniforge


.. _installing_simpeg:

Installing SimPEG
=================

Conda Forge
-----------

SimPEG is available through `conda-forge` and you can install is using the
`conda package manager <https://conda.io/>`_ that comes with the Anaconda_
or Miniforge_ distributions:

.. code::

    conda install SimPEG --channel conda-forge

Installing through `conda` is our recommended method of installation.

.. note::

    Since `version 23.10.0
    <https://docs.conda.io/projects/conda/en/latest/release-notes.html#id33>`_,
    ``conda`` makes use of the ``libmamba`` solver to resolve dependencies. It
    makes creation of environments and installation of new packages much faster
    than when using older versions of ``conda``.

    Since this version, ``conda`` can achieve the same performance as
    ``mamba``, so there's no need to install ``mamba`` if you have an updated
    version of ``conda``.
    If not, either `update conda
    <https://docs.anaconda.com/free/anaconda/install/update-version/>`_, or
    keep using ``mamba`` instead.

PyPi
----

SimPEG is on `pypi <https://pypi.python.org/pypi/SimPEG>`_! First, make sure
your version of pip is up-to-date

.. code::

    pip install --upgrade pip

Then you can install SimPEG

.. code::

    pip install SimPEG


To update SimPEG, you can run

.. code::

    pip install --upgrade SimPEG


Installing from Source
----------------------

First (you need git)::

    git clone https://github.com/simpeg/simpeg

Second (from the root of the SimPEG repository)::

    pip install .

If you are interested in contributing to SimPEG, please check out the page on :ref:`Contributing <contributing>`


Success?
========

If you have been successful at downloading and installing SimPEG, you should
be able to download and run any of the :ref:`examples and tutorials <sphx_glr_content_examples>`.

If not, you can reach out to other people developing and using SimPEG on the
`google forum <https://groups.google.com/forum/#!forum/simpeg>`_ or on
`Mattermost <https://mattermost.softwareunderground.org/simpeg>`_.

Useful Links
============

An enormous amount of information (including tutorials and examples) can be found on the official websites of the packages

* `Python <https://www.python.org/>`_
* `Numpy <https://www.numpy.org/>`_
* `SciPy <https://www.scipy.org/>`_
* `Matplotlib <https://matplotlib.org/>`_

Python for scientific computing
-------------------------------

* `Python for Scientists <https://sites.google.com/site/pythonforscientists/>`_ Links to commonly used packages, Matlab to Python comparison
* `Python Wiki <https://wiki.python.org/moin/NumericAndScientific>`_ Lists packages and resources for scientific computing in Python
* `Jupyter <https://jupyter.org/>`_

Numpy and Matlab
----------------

* `NumPy for Matlab Users <https://numpy.org/doc/stable/user/numpy-for-matlab-users.html>`_
* `Python vs Matlab <https://sites.google.com/site/pythonforscientists/python-vs-matlab>`_

Lessons in Python
-----------------

* `Software Carpentry <https://swcarpentry.github.io/python-novice-inflammation/>`_
* `Introduction to NumPy and Matplotlib <https://www.youtube.com/watch?v=3Fp1zn5ao2M>`_


Editing Python
--------------

There are numerous ways to edit and test Python (see
`PythonWiki <https://wiki.python.org/moin/PythonEditors>`_ for an overview) and
in our group at least the following options are being used:

* `Jupyter <https://jupyter.org/>`_
* `Sublime <https://www.sublimetext.com/>`_
* `PyCharm <https://www.jetbrains.com/pycharm/>`_
