.. _api_getting_started_developers:

Getting Started: for Developers
===============================

- **Purpose:** To download and set up your environment for using and developing within SimPEG.


.. _getting_started_installing_python:

Installing Python
-----------------

.. image:: https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/220px-Python-logo-notext.svg.png
    :align: right
    :width: 100
    :target: https://www.python.org/

SimPEG is written in Python_! To install and maintain your Python_ 2.7
environment, Anaconda_ is a package manager that you can use. SimPEG
requires Python_ 2.7. If you and Python_ are not yet acquainted, we highly
recommend checking out `Software Carpentry <http://software-carpentry.org/>`_.

.. _Python: https://www.python.org/

.. _Anaconda: https://www.continuum.io/downloads/


.. _getting_started_working_with_git_and_github:

Working with Git and GitHub
---------------------------

.. image:: https://assets-cdn.github.com/images/modules/logos_page/Octocat.png
    :align: right
    :width: 100
    :target: http://github.com


To keep track of your code changes and contribute back to SimPEG, you will
need a github_ account and fork the `SimPEG repository <http://github.com/simpeg/simpeg>`_
(`How to fork a repo <https://help.github.com/articles/fork-a-repo/>`_). Software


.. _github: http://github.com

Next, clone your fork so that you have a local copy. We recommend setting up a
directory called :code:`git` in your home directory to put your version-
controlled repositories. There are two ways you can clone a repository: (1)
from a terminal (checkout: https://try.github.io for an tutorial)::

    git clone https://github.com/YOUR-USERNAME/SimPEG

or (2) using a desktop client such as SourceTree_.

.. _SourceTree: https://www.sourcetreeapp.com/

.. image:: ../images/sourceTreeSimPEG.png
    :align: center
    :width: 400
    :target: https://www.sourcetreeapp.com/

If this is your first time managing a github_ repository through SourceTree_,
it is also handy to set up the remote account so it remembers your github_
user name and password

.. image:: ../images/sourceTreeRemote.png
    :align: center
    :width: 400

For managing your copy of SimPEG and contributing back to the main
repository, have a look at the article: `A successful git branching model
<http://nvie.com/posts/a-successful-git-branching-model/>`_


.. _getting_started_setting_up_your_environment:

Setting up your environment
---------------------------

So that you can access SimPEG from anywhere on your computer, you need to add
it to your path. This can be done using symlinks. In your :code:`git` directory,
create a directory called :code:`python_symlinks`.

.. image:: ../images/gitfolders.png
    :align: center
    :width: 40%

Open a terminal in this directory and create a symlink for SimPEG ::

    ln -s ../SimPEG/SimPEG .

Then, in your shell, you need to add a :code:`PYTHONPATH` variable. For Mac and
Linux, if you are using Z shell (`Oh My Zsh <http://ohmyz.sh/>`_ is used by a
lot of SimPEG developers) or bash open the config in a text editor, ie::

    nano ~/.zshrc

or::

    nano ~/.bash_profile

and add a :code:`PYTHONPATH` variable::

    export PYTHONPATH="$PYTHONPATH:/Users/USER/git/python_symlinks"

and save and close. If you then restart the terminal, and run::

    echo $PYTHONPATH

the output should be::

    /Users/USER/git/python_symlinks


.. _getting_started_text_editors:

Text Editors
------------

Sublime_ is a text editor used by many SimPEG developers.

.. _Sublime: https://www.sublimetext.com/

You can configure the Sublime so that you can use the sublime
build (Tools / Build) to run Python_ code.

Open your user settings

.. image:: ../images/sublimeSettings.png
    :align: center
    :width: 400

and edit them to include the path to your :code:`python_symlinks`::

    {
    "added_words":
    [
        "electromagnetics"
    ],
    "ensure_newline_at_eof_on_save": true,
    "extra_paths":
    [
        "/Users/USER/git/python_symlinks/"
    ],
    "font_size": 11,
    "ignored_packages":
    [
        "Vintage"
    ],
    "translate_tabs_to_spaces": true,
    "trim_trailing_white_space_on_save": true,
    "word_wrap": false
    }

There are a few other things configured here. In particular you will want to
ensure that :code:`"translate_tabs_to_spaces": true` is configured (Python_ is
sensitive to tabs and spaces), that
:code:`"trim_trailing_white_space_on_save": true` so that your git flow does
not get cluttered with extra spaces that are not actually changes to code and
that :code:`"ensure_newline_at_eof_on_save": true`, so that there is a blank
line at the end of all saved documents. The rest are up to you.

.. _getting_started_jupyter_notebook:

Jupyter Notebook
----------------

.. image:: http://blog.jupyter.org/content/images/2015/02/jupyter-sq-text.png
    :align: right
    :width: 100

The SimPEG team loves the `Jupyter notebook`_. It is an interactive
development environment. It is installed it you used Anaconda_ and can be
launched from a terminal using::

    jupyter notebook


.. _getting_started_if_all_is_well:

If all is well ...
------------------

You should be able to open a terminal within SimPEG/SimPEG/Examples and run an example, ie.::

    python Inversion_Linear.py

and open a Jupyter Notebook, and run the linear inversion

.. image:: ../images/SimPEGInversionLinearNotebook.png
    :align: center
    :width: 350

and see

.. plot::

    from SimPEG.Examples import Inversion_Linear
    Inversion_Linear.run()
    plt.show()

You are now set up to SimPEG!

.. note::

    you likely got a message that said::

        Efficiency Warning: Interpolation will be slow, use setup.py!
            python setup.py build_ext --inplace

    This is because we use Cython_ to speed up interpolation. To set this up open
    up a command prompt in :code:`git/simpeg`, and run::

        python setup.py build_ext --inplace

    .. _Cython: http://cython.org/

    Which might output a bunch of warnings, but so long as there are no
    errors, you should be good to go. To check, re-run the example and see if
    the efficiency warning still appears.


If all is not well ...
----------------------

Submit an issue_ and `change this file`_!

.. _issue: https://github.com/simpeg/simpeg/issues

.. _change this file: https://github.com/simpeg/simpeg/edit/master/docs/content/api_getting_started_developers.rst

Advanced: Installing Mumps
--------------------------

Mumps_ is a direct solver that can be used for solving large(ish) [#f1]_
linear systems of equations. To install, follow the instructions to download
and install pymatsolver_.

.. _Mumps: http://mumps.enseeiht.fr/

.. _pymatsolver: https://github.com/rowanc1/pymatsolver

- **Disclaimer for Windows users**: we have not figured out a stable way to install
  and connect Mumps for Windows Machines. If you have one, please `change this file`_!

If you open a `Jupyter notebook`_ and are able to run::

    from pymatsolver import MumpsSolver

.. _Jupyter notebook: http://jupyter.org/

then you have succeeded! Otherwise, make an `issue in pymatsolver`_.

.. _issue in pymatsolver: https://github.com/rowanc1/pymatsolver/issues

.. rubric:: Footnotes

.. [#f1] These instructions are for serial (not parallel) Mumps_ installation. The definition of large also depends on the size of your computer
