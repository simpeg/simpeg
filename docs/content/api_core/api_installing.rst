.. _api_installing:

Getting Started with SimPEG
***************************

Dependencies
============

- Python 3.5 (or 2.7)
- NumPy 1.8 (or greater)
- SciPy 0.13 (or greater)
- matplotlib 1.3 (or greater)
- Cython 0.20 (or greater)

Installing Python
=================

Python is available on all major operating systems, but if you are getting started with python
it is best to use a package manager such as
`Continuum Anaconda <https://www.continuum.io/downloads>`_ or
`Enthought Canopy <https://www.enthought.com/products/canopy/>`_.
You can download the package manager and use it to install the dependencies above.

.. note::
    When using Continuum Anaconda, make sure to run::

        conda update conda
        conda update anaconda


Installing SimPEG
=================

SimPEG is on pip!!::

    pip install SimPEG


Installing from Source
----------------------

First (you need git)::

    git clone https://github.com/simpeg/simpeg

Second (from the root of the simpeg repository)::

    python setup.py install

.. attention:: Windows users
	A common error when installing the setup.py is:
	``Missing linker, needs MSC v.1500 (Microsoft Visual C++ 2008) Runtime Library``

	The missing library can be found `here <https://www.microsoft.com/en-ca/download/details.aspx?id=29>`

Useful Links
============
An enormous amount of information (including tutorials and examples) can be found on the official websites of the packages

* `Python Website <https://www.python.org/>`_
* `Numpy Website <https://www.numpy.org/>`_
* `SciPy Website <https://www.scipy.org/>`_
* `Matplotlib <https://matplotlib.org/>`_

Python for scientific computing
-------------------------------

* `Python for Scientists <https://sites.google.com/site/pythonforscientists/>`_ Links to commonly used packages, Matlab to Python comparison
* `Python Wiki <http://wiki.python.org/moin/NumericAndScientific>`_ Lists packages and resources for scientific computing in Python

Numpy and Matlab
----------------

* `NumPy for Matlab Users <https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html>`_
* `Python vs Matlab <https://sites.google.com/site/pythonforscientists/python-vs-matlab>`_

Lessons in Python
-----------------

* `Software Carpentry <https://swcarpentry.github.io/python-novice-inflammation/>`_
* `Introduction to NumPy and Matplotlib <https://www.youtube.com/watch?v=3Fp1zn5ao2M>`_

Editing Python
--------------

There are numerous ways to edit and test Python (see `PythonWiki <http://wiki.python.org/moin/PythonEditors>`_ for an overview) and in our group at least the following options are being used:

* `Sublime <https://www.sublimetext.com/>`_
* `iPython Notebook <https://ipython.org/notebook.html>`_
* `iPython <https://ipython.org/>`_
* `Enthought Canopy <https://www.enthought.com/products/canopy/>`_
