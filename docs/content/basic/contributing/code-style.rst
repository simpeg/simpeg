.. _code-style:

Code style
==========

Consistency makes code more readable and easier for collaborators to jump in.
SimPEG uses Black_ to autoformat its codebase, and flake8_ to lint its code and
enforce style rules. Black_ can automatically format SimPEG's codebase to
ensure it complies with Black code style. flake8_ performs style checks, raises
warnings on code that could lead towards bugs, performs checks on consistent
documentation formatting, and identifies poor coding practices.

.. important::

   If you :ref:`configure-pre-commit`, pre-commit_ will automatically run
   Black and flake8 on every commit.

Alternatively, one could run them manually at anytime.
Run ``black`` on SimPEG directories that contain Python source files:

.. code::

   black SimPEG examples tutorials tests

Run ``flake8`` on the whole project with:

.. code::

   flake8

.. note::

   SimPEG is currently not `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_
   compliant and is not following all flake8 rules, but we are working towards
   it and would appreciate contributions that do too!

.. _Black: https://black.readthedocs.io/
.. _flake8: https://flake8.pycqa.org/
.. _pre-commit: https://pre-commit.com/
