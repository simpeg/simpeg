"""
`printinfo` -- Tools to print date, time, and version information
=================================================================

Print or return date, time, and package version information in any environment
(Jupyter notebook, IPython console, Python console, QT console), either as
html-table (notebook) or as plain text (anywhere).

This script is an adapted version of ``empymod.printversions``
(https://empymod.github.io), which itself was heavily inspired by

- ``ipynbtools.py`` from https://github.com/qutip, and
- ``watermark.py`` from https://github.com/rasbt/watermark,

Always shown are the OS, number of CPU(s), ``numpy``, ``scipy``, ``SimPEG``,
``cython``, ``properties``, ``vectormath``, ``discretize``, ``pymatsolver``,
``sys.version``, and time/date.

Additionally shown are, if they can be imported, ``IPython``, ``matplotlib``,
and ``ipywidgets``. It also shows MKL information, if available.

All modules provided in ``add_pckg`` are also shown. They have to be imported
before ``versions`` is called.

"""
from __future__ import print_function

# Mandatory modules
import sys
import time
import numpy
import scipy
import textwrap
import platform
import multiprocessing

# Required packages
import SimPEG
import cython
import properties
import vectormath
import discretize
import pymatsolver

# Optional modules
try:
    import IPython
    from IPython.display import HTML, Pretty
except ImportError:
    IPython = False
try:
    import matplotlib
except ImportError:
    matplotlib = False
try:
    import ipywidgets
except ImportError:
    ipywidgets = False
try:
    import numexpr
except ImportError:
    numexpr = False
try:
    import mkl
except ImportError:
    mkl = False

# Get mkl info from numexpr or mkl, if available
if mkl:
    mklinfo = mkl.get_version_string()
elif numexpr:
    mklinfo = numexpr.get_vml_version()
else:
    mklinfo = False

__all__ = ['versions', 'versions_html', 'versions_text']


def versions(mode='print', add_pckg=None, ncol=4):
    """Return date, time, and version information.

    Print or return date, time, and package version information in any
    environment (Jupyter notebook, IPython console, Python console, QT
    console), either as html-table (notebook) or as plain text (anywhere).

    This script is an adapted version of ``empymod.printversions``,
    (https://empymod.github.io) which itself was heavily inspired by

        - ipynbtools.py from qutip https://github.com/qutip
        - watermark.py from https://github.com/rasbt/watermark

    This is a wrapper for ``versions_html`` and ``versions_text``.

    **Parameters**

    mode : string, optional; {<'print'>, 'HTML', 'Pretty', 'plain', 'html'}
        Defaults to 'print':
            - 'print': Prints text-version to stdout, nothing returned.
            - 'HTML': Returns html-version as IPython.display.HTML(html).
            - 'html': Returns html-version as plain text.
            - 'Pretty': Returns text-version as IPython.display.Pretty(text).
            - 'plain': Returns text-version as plain text.

        'HTML' and 'Pretty' require IPython.

    add_pckg : packages, optional
        Package or list of packages to add to output information (must be
        imported beforehand).

    ncol : int, optional
        Number of package-columns in html table; only has effect if
        ``mode='HTML'`` or ``mode='html'``. Defaults to 3.


    **Returns**

    Depending on ``mode`` (HTML-instance; plain text; html as plain text; or
    nothing, only printing to stdout).


    **Examples**

    >>> import pytest
    >>> import dateutil
    >>> from SimPEG import versions
    >>> versions()                 # Default values
    >>> versions('plain', pytest)  # Provide additional package
    >>> versions('HTML', [pytest, dateutil], ncol=5)  # HTML

    """
    if mode == 'html':
        return versions_html(add_pckg, ncol)
    elif mode == 'plain':
        return versions_text(add_pckg)
    elif mode == 'Pretty' and IPython:
        return Pretty(versions_text(add_pckg))
    elif mode == 'HTML' and IPython:
        return HTML(versions_html(add_pckg, ncol))
    else:
        print(versions_text(add_pckg))


def versions_html(add_pckg=None, ncol=4):
    """HTML version.

    See ``versions`` for details.
    """

    # Check ncol
    ncol = int(ncol)

    # Define html-styles
    border = "border: 2px solid #fff;'"

    def colspan(html, txt, ncol, nrow):
        """Print txt in a row spanning whole table."""
        html += "  <tr>\n"
        html += "     <td style='text-align: center; "
        if nrow == 0:
            html += "font-weight: bold; font-size: 1.2em; "
        elif nrow % 2 == 0:
            html += "background-color: #ddd;"
        html += border + " colspan='"
        html += str(2*ncol)+"'>%s</td>\n" % txt
        html += "  </tr>\n"
        return html

    def cols(html, version, name, ncol, i):
        """Print package information in two cells."""

        # Check if we have to start a new row
        if i > 0 and i % ncol == 0:
            html += "  </tr>\n"
            html += "  <tr>\n"

        html += "    <td style='text-align: right; background-color: #ccc; "
        html += border + ">%s</td>\n" % version

        html += "    <td style='text-align: left; "
        html += border + ">%s</td>\n" % name

        return html, i+1

    # Start html-table
    html = "<table style='border: 3px solid #ddd;'>\n"

    # Date and time info as title
    html = colspan(html, time.strftime('%a %b %d %H:%M:%S %Y %Z'), ncol, 0)

    # OS and CPUs
    html += "  <tr>\n"
    html, i = cols(html, platform.system(), 'OS', ncol, 0)
    html, i = cols(html, multiprocessing.cpu_count(), 'CPU(s)', ncol, i)

    # Loop over packages
    for pckg in _get_packages(add_pckg):
        html, i = cols(html, pckg.__version__, pckg.__name__, ncol, i)
    # Fill up the row
    while i % ncol != 0:
        html += "    <td style= " + border + "></td>\n"
        html += "    <td style= " + border + "></td>\n"
        i += 1
    # Finish row
    html += "  </tr>\n"

    # sys.version
    html = colspan(html, sys.version, ncol, 1)

    # mkl version
    if mklinfo:
        html = colspan(html, mklinfo, ncol, 2)

    # Finish table
    html += "</table>"

    return html


def versions_text(add_pckg=None):
    """Plain-text version.

    See ``versions`` for details.
    """

    # Width for text-version
    n = 54
    text = u'\n' + ''.join(n*['-']) + '\n'

    # Date and time info as title
    text += time.strftime('  %a %b %d %H:%M:%S %Y %Z\n\n')

    # OS and CPUs
    text += '{:>15}'.format(platform.system())+' : OS\n'
    text += '{:>15}'.format(multiprocessing.cpu_count())+' : CPU(s)\n'

    # Loop over packages
    for pckg in _get_packages(add_pckg):
        text += '{:>15} : {}\n'.format(pckg.__version__, pckg.__name__)

    # sys.version
    text += '\n'
    for txt in textwrap.wrap(sys.version, n-4):
        text += '  '+txt+'\n'

    # mkl version
    if mklinfo:
        text += '\n'
        for txt in textwrap.wrap(mklinfo, n-4):
            text += '  '+txt+'\n'

    # Finish
    text += ''.join(n*['-'])
    return text


def _get_packages(add_pckg):
    """Create list of packages."""

    # Mandatory packages
    pckgs = [numpy, scipy, SimPEG, cython, properties, vectormath, discretize,
             pymatsolver]

    # Optional packages
    for module in [IPython, ipywidgets, matplotlib]:
        if module:
            pckgs += [module]

    # Cast and add add_pckg
    if add_pckg is not None:

        # Cast add_pckg
        if isinstance(add_pckg, tuple):
            add_pckg = list(add_pckg)

        if not isinstance(add_pckg, list):
            add_pckg = [add_pckg, ]

        # Add add_pckg
        pckgs += add_pckg

    return pckgs
