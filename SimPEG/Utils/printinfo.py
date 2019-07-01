import scooby

class Versions(scooby.Report):
    """Print date, time, and version information.

    Use scooby to print date, time, and package version information in any
    environment (Jupyter notebook, IPython console, Python console, QT
    console), either as html-table (notebook) or as plain text (anywhere).

    Always shown are the OS, number of CPU(s), ``numpy``, ``scipy``,
    ``SimPEG``, ``cython``, ``properties``, ``vectormath``, ``discretize``,
    ``pymatsolver``, ``sys.version``, and time/date.

    Additionally shown are, if they can be imported, ``IPython``,
    ``matplotlib``, and ``ipywidgets``. It also shows MKL information, if
    available.

    All modules provided in ``add_pckg`` are also shown.


    Parameters
    ----------
    add_pckg : packages, optional
        Package or list of packages to add to output information (must be
        imported beforehand).

    ncol : int, optional
        Number of package-columns in html table; only has effect if
        ``mode='HTML'`` or ``mode='html'``. Defaults to 3.

    **kwargs : dict
        Passed through to scooby.


    Examples
    --------

    >>> import pytest
    >>> import dateutil
    >>> from SimPEG import Versions
    >>> Versions()                            # Default values
    >>> Versions(pytest)                      # Provide additional package
    >>> Versions([pytest, dateutil], ncol=5)  # Define nr of columns

    """

    def __init__(self, add_pckg=None, ncol=4, **kwargs):
        """Initiate a scooby.Report instance."""

        # Mandatory packages.
        core = ['SimPEG', 'discretize', 'pymatsolver', 'vectormath',
                'properties', 'numpy', 'scipy', 'cython']

        # Optional packages.
        optional = ['IPython', 'matplotlib', 'ipywidgets']

        super().__init__(add_pckg, core, optional, ncol=ncol, **kwargs)
