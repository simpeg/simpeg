import importlib

import pytest


def test_legacy_module_removed():
    with pytest.raises(ModuleNotFoundError, match="^No module named 'SimPEG'$"):
        importlib.import_module("SimPEG")
