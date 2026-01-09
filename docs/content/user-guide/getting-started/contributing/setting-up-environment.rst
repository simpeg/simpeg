.. _setting-up-environment:

Setting up your environment
===========================

Install Python
--------------

First you will need to install Python. You can find instructions in
:ref:`installing_python`. We highly encourage to install Miniforge_ (or
Anaconda_).

Create environment
------------------

To get started developing SimPEG we recommend setting up an environment using
the ``conda`` package manager that includes all odthe packages necessary to
both develop SimPEG and run tests locally. Most of the packages that
we use are available through the ``conda-forge`` project.
We provide an ``environment.yml`` in the base level directory.

To create the environment and install all packages needed to run and write code
for SimPEG, navigate to the directory where you :ref:`cloned SimPEG's
repository <working-with-github>` and run:

.. code::

    conda env create -f environment.yml

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


Once the environment is successfully created, you can *activate* it with

.. code::

    conda activate simpeg-dev


Install SimPEG in developer mode
--------------------------------

There are many options to install SimPEG into this local environment, we
recommend using `pip`. After ensuring that all necessary packages from
`environment.yml` are installed, the most robust command you can use,
executed from the base level directory would be:

.. code::

    pip install --no-deps -e .

This is called an editable mode install (`-e`). This will make a symbolic link
for you to the working ``simpeg`` directory for that Python environment to use
and you can then make use of any changes you have made to the repository
without re-installing it. This command (`--no-deps`) also ensures pip won't
unintentionally re-install a package that was previously installed with conda.
This practice also allows you to uninstall SimPEG if so desired:

.. code::

    pip uninstall simpeg

.. note::

    We no longer recommend modifying your Python path environment variable as
    a way to install SimPEG for developers.

.. _Anaconda: https://www.anaconda.com/products/individual
.. _Miniforge: https://github.com/conda-forge/miniforge

Check your installation
-----------------------

You should be able to open a terminal within simpeg/tutorials and run an
example, i.e.

.. code::

    python 02-linear_inversion/plot_inv_1_inversion_lsq.py

or you can download and run the :ref:`notebook from the docs
<sphx_glr_content_tutorials_02-linear_inversion_plot_inv_1_inversion_lsq.py>`.

.. image:: ../../tutorials/02-linear_inversion/images/sphx_glr_plot_inv_1_inversion_lsq_003.png

You are now set up to SimPEG!

.. note::

   If all is not well, please submit an issue_ and `change this file`_!

.. _issue: https://github.com/simpeg/simpeg/issues
.. _change this file: https://github.com/simpeg/simpeg/edit/main/docs/content/getting_started/contributing/setting-up-environment.rst


.. _configure-pre-commit:

Configure pre-commit (optional)
-------------------------------

We recommend using pre-commit_ to ensure that your new code follows the code
style of SimPEG. pre-commit will run Black_ and flake8_ before any commit you
make. To configure it, you need to navigate to your cloned SimPEG repo and run:

.. code::

   pre-commit install

.. note::

   Using ``pre-commit`` is recommended, but not necessary. You can still
   manually run Black_ and flake8_. See our :ref:`code-style` page for more
   details.

If for some reason you want to stop using ``pre-commit`` on SimPEG, you can
permanently configure it to stop running automatically with:

.. code::

   pre-commit uninstall

Alternatively, you can temporarily bypass ``pre-commit`` when committing some changes by running:

.. code::

   git commit --no-verify

This is specially useful if the checks run by ``pre-commit`` are failing, but
you want to commit them nonetheless.


.. _pre-commit: https://pre-commit.com/
.. _Black: https://black.readthedocs.io
.. _flake8: https://flake8.pycqa.org


Update your environment
-----------------------

Every once in a while, the minimum versions of the packages in the
``environment.yml`` file get updated. After this happens, it's better to update
the ``simpeg-dev`` environment we have created. This way we ensure that we are
checking the style and testing our code using those updated versions.

To update our environment we need to navigate to the directory where you
:ref:`cloned SimPEG's repository <working-with-github>` and run:

.. code::

    conda env update -f environment.yml
