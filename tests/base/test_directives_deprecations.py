"""
Test deprecation of public directives submodules.
"""

import pytest
import importlib

REGEX = r"The `simpeg\.directives\.[a-z_]+` submodule has been deprecated, "
DEPRECATED_SUBMODULES = ("directives", "pgi_directives", "sim_directives")


@pytest.mark.parametrize("submodule", DEPRECATED_SUBMODULES)
def test_deprecations(submodule):
    """
    Test FutureWarning when trying to import the deprecated modules.
    """
    with pytest.warns(FutureWarning, match=REGEX):
        importlib.import_module(f"simpeg.directives.{submodule}")
