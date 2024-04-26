import pytest


def test_SimPEG_import():
    with pytest.warns(
        FutureWarning,
        match="Importing `SimPEG` is deprecated. please import from `simpeg`.",
    ):
        from SimPEG import data
    import SimPEG
    import simpeg

    assert SimPEG is simpeg
    assert data.__file__.endswith("simpeg/data.py")
