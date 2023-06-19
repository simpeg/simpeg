.. _setting-up-environment:

Setting up your environment
===========================

Install Python
--------------

First you will need to install Python. You can find instructions in
:ref:`installing_python`. We highly encourage to install Anaconda_ or
Mambaforge_.

Create environment
------------------

To get started developing SimPEG we recommend setting up an environment using
the ``conda`` ( or ``mamba``) package manager that mimics the testing
environment used for continuous integration testing. Most of the packages that
we use are available through the ``conda-forge`` project. This will ensure you
have all of the necessary packages to both develop SimPEG and run tests
locally. We provide an ``environment_test.yml`` in the base level directory.

.. code::

    conda env create -f environment_test.yml

.. note::
    If you find yourself wanting a faster package manager than ``conda``
    check out the ``mamba`` project at https://mamba.readthedocs.io/. It
    usually is able to set up environments much quicker than ``conda`` and
    can be used as a drop-in replacement (i.e. replace ``conda`` commands with
    ``mamba``).

Once the environment is successfully created, you can *activate* it with

.. code::

    conda activate simpeg-test


Install SimPEG in developer mode
--------------------------------

There are many options to install SimPEG into this local environment, we
recommend using `pip`. After ensuring that all necessary packages from
`environment_test.yml` are installed, the most robust command you can use,
executed from the base level directory would be ::

.. code::

    pip install --no-deps -e .

This is called an editable mode install (`-e`). This will make a symbolic link
for you to the working ``simpeg`` directory for that Python environment to use
and you can then make use of any changes you have made to the repository
without re-installing it. This command (`--no-deps`) also ensures pip won't
unintentionally re-install a package that was previously installed with conda.
This practice also allows you to uninstall SimPEG if so desired ::

    pip uninstall SimPEG

.. note::

    We no longer recommend modifying your Python path environment variable as
    a way to install SimPEG for developers.

.. _Anaconda: https://www.anaconda.com/products/individual
.. _Mambaforge: https://www.anaconda.com/products/individual

Check your installation
-----------------------

You should be able to open a terminal within SimPEG/tutorials and run an
example, ie.

.. code::

    python 02-linear_inversion/plot_inv_1_inversion_lsq.py

or you can download and run the :ref:`notebook from the docs
<sphx_glr_content_tutorials_02-linear_inversion_plot_inv_1_inversion_lsq.py>`.

.. image:: ../../tutorials/02-linear_inversion/images/sphx_glr_plot_inv_1_inversion_lsq_003.png

You are now set up to SimPEG!

.. note::

   If all is not well, please submit an issue_ and `change this file`_!

.. _issue: https://github.com/simpeg/simpeg/issues
.. _change this file: https://github.com/simpeg/simpeg/edit/main/docs/content/basic/contributing/setting-up-environment.rst


.. _configure-pre-commit:

Configure pre-commit
--------------------

We recommend using pre-commit_ to ensure that your new code follows the code
style of SimPEG. pre-commit will run Black_ and flake8_ before any commit you
make. To configure it, you need to navigate to your cloned SimPEG repo and run:

.. code::

   pre-commit install

.. _pre-commit: https://pre-commit.com/
.. _Black: https://black.readthedocs.io
.. _flake8: https://flake8.pycqa.org
