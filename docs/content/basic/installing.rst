.. _api_installing:

Getting Started with SimPEG
***************************


.. _installing_python:

Prerequisite: Installing Python
===============================

We highly recommend installing python using
`Anaconda <https://anaconda.com/download/>`_.
It installs `python <https://www.python.org/>`_,
`Jupyter <http://jupyter.org/>`_ and other core
python libraries for scientific computing.

As of version 0.11.0, we will no longer ensure compatibility with Python 2.7. Please use
the latest version of Python 3 with SimPEG. For more information on the transition of the
Python ecosystem to Python 3, please see the `Python 3 Statement <https://python3statement.org/>`_.


.. _installing_simpeg:

Installing SimPEG
=================

Conda Forge
-----------

You can install SimPEG using the `conda package manager <https://conda.io/>`_ that comes with the Anaconda distribution:

.. code::

    conda install SimPEG --channel conda-forge


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

    python setup.py install

.. attention:: Windows users
	A common error when installing the setup.py is:
	``Missing linker, needs MSC v.1500 (Microsoft Visual C++ 2008) Runtime Library``

	The missing library can be found `here <https://www.microsoft.com/en-ca/download/details.aspx?id=29>`_

If you are interested in contributing to SimPEG, please check out the page on :ref:`Contributing <contributing>`


Success?
========

If you have been successful at downloading and installing SimPEG, you should
be able to download and run any of the `Examples <http://docs.simpeg.xyz/content/examples/index.html>`_.

If not, you can reach out to other people developing and using SimPEG on the
`google forum <https://groups.google.com/forum/#!forum/simpeg>`_ or on
`slack <http://slack.simpeg.xyz>`_.

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
* `Python Wiki <http://wiki.python.org/moin/NumericAndScientific>`_ Lists packages and resources for scientific computing in Python
* `Jupyter <http://jupyter.org/>`_

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

There are numerous ways to edit and test Python (see `PythonWiki <http://wiki.python.org/moin/PythonEditors>`_ for an overview) and in our group at least the following options are being used:

* `Sublime <https://www.sublimetext.com/>`_
* `Jupyter <http://jupyter.org/>`_
