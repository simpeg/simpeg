.. _installing:

==========
Installing
==========


.. _installing_python:

Installing Python
=================

SimPEG is written in Python_!
This means we need Python_ in order to run SimPEG.
We highly recommend installing a Python distribution like Miniforge_ that will
install the Python interpreter along with the conda_ package manager.

.. note::

   Miniforge_ is a community-driven alternative to Anaconda_, a well-known
   Python distribution.

   We recommend Miniforge_ over Anaconda_ because it's more lightweight and
   because it makes use of the conda-forge_ community-led channel to download
   packages. Downloading packages from Anaconda_ (usually refered as the
   ``default`` channel) requires us to adhere to their `Terms of Service
   <https://legal.anaconda.com/policies/en/>`_.
   Make sure to read them and their `FAQs
   <https://www.anaconda.com/pricing/terms-of-service-faqs>`_ if you decide to
   still use Anaconda_.

.. seealso::

   If you are starting with Python_ and want to learn more and feel more
   comfortable with the language, we recommend checking out
   `Software Carpentry <https://software-carpentry.org/>`_'s lessons.

.. _Python: https://www.python.org/
.. _Anaconda: https://www.anaconda.com/products/individual
.. _Miniforge: https://github.com/conda-forge/miniforge
.. _conda: https://docs.conda.io/en/latest
.. _conda-forge: https://conda-forge.org/


.. _installing_simpeg:

Installing SimPEG
=================

Conda Forge
-----------

SimPEG is available through conda-forge_ and you can install is using the
`conda package manager <https://conda.io/>`_ that comes with Miniforge_ (or
Anaconda_):

.. code:: bash

    conda install --channel conda-forge simpeg

.. note::

   Installing through ``conda`` is our recommended method of installation.

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

.. code:: bash

    pip install --upgrade pip

Then you can install SimPEG

.. code:: bash

    pip install simpeg


To update SimPEG, you can run

.. code:: bash

    pip install --upgrade simpeg


Installing from Source
----------------------

First (you need git):

.. code:: bash

    git clone https://github.com/simpeg/simpeg

Second (from the root of the SimPEG repository):

.. code:: bash

    pip install .

If you are interested in contributing to SimPEG, please check out the page on :ref:`Contributing <contributing>`


Success?
========

If you have been successful at downloading and installing SimPEG, you should
be able to download and run any of the :ref:`examples and tutorials <sphx_glr_content_examples>`.

If not, you can reach out to other people developing and using SimPEG on our
Mattermost_ channel or in our `Discourse forum`_.

.. _Discourse forum: https://simpeg.discourse.group/
.. _Mattermost: https://mattermost.softwareunderground.org/simpeg

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
