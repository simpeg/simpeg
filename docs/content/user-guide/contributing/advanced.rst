.. _advanced:

Advanced: Installing Solvers
----------------------------

Pardiso_ is a direct solver that can be used for solving large(ish)
linear systems of equations. The provided testing environment should install
the necessary solvers for you. If you wish to modify pymatsolver_ as well
follow the instructions to download and install pymatsolver_.

.. _Pardiso: https://www.pardiso-project.org

.. _pymatsolver: https://github.com/simpeg/pymatsolver

If you open a `Jupyter notebook`_ and are able to run:

.. code:: python

    from pymatsolver import Pardiso

.. _Jupyter notebook: https://jupyter.org/

then you have succeeded! Otherwise, make an `issue in pymatsolver`_.

.. _issue in pymatsolver: https://github.com/simpeg/pymatsolver/issues
