"""
Test deprecation of public directives submodules.
"""

import pytest
import importlib

REGEX = r"No module named 'simpeg\.directives\.[a-z_]+'"
DEPRECATED_SUBMODULES = ("directives", "pgi_directives", "sim_directives")


@pytest.mark.parametrize("submodule", DEPRECATED_SUBMODULES)
def test_removals(submodule):
    """
    Test if error is raised when trying to import the removed modules.
    """
    with pytest.raises(ModuleNotFoundError, match=REGEX):
        importlib.import_module(f"simpeg.directives.{submodule}")
