.. _code-style:

Code style
==========

Consistency makes code more readable and easier for collaborators to jump in. SimPEG uses Black_ to autoformat its codebase, and flake8_ and Ruff_ to lint its code and enforce style rules.
Black_ can automatically format SimPEG's codebase to ensure it complies with Black code style.
flake8_ and Ruff_ perform style checks, raise warnings on code that could lead towards bugs, performs checks on consistent documentation formatting, and identifies poor coding practices.

.. hint::

   If you :ref:`configure pre-commit <configure-pre-commit>`, it will
   automatically run Black, flake8, and Ruff on every commit.

One can manually run Black_, flake8_, and Ruff_ anytime.

- Run ``black`` on SimPEG directories that contain Python source files:

  .. code:: bash

     black .

- Run ``flake8`` on the whole project with:

  .. code:: bash

     flake8

- Run ``ruff`` on the whole project with:

  .. code:: bash

     ruff check

.. important::

   Following code style rules can be challenging for new contributors. These
   rules are meant to ease the development process, not to generate an obstacle
   to contribute. Please, don't hesistate to **ask for help** if your
   contribution raises some flake8 errors. And **feel free to push** code that
   **don't follow our code style 100%** in :ref:`pull-requests`. Other
   developers will be there to help you solve them.

.. note::

   SimPEG is currently not `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_
   compliant and is not following all flake8 rules, but we are working towards
   it and would appreciate contributions that do too!

.. hint::

   Configurations for Black_, flake8_, and Ruff_ live inside the ``pyproject.toml`` file, under the ``tool`` section.
   Refer to their respective documentation pages to learn how each tool can be configured, and what each different rule checks for.


.. _Black: https://black.readthedocs.io/
.. _flake8: https://flake8.pycqa.org/
.. _Ruff: https://docs.astral.sh/
.. _pre-commit: https://pre-commit.com/
