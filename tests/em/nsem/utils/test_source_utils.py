"""
Extra test for source utils.
"""

import re
import pytest
import numpy as np
from discretize import TensorMesh
from simpeg.electromagnetics.natural_source.utils.source_utils import (
    project_1d_fields_to_mesh_edges,
)


def test_project_1d_fields_invalid_size():
    """
    Test error after invalid size of 1D solution.
    """
    mesh = TensorMesh(h=[[(1.0, 5)]])
    invalid_u_1d = np.ones(2)
    msg = re.escape("Found invalid 'u_1d' with '2' elements. ")
    with pytest.raises(ValueError, match=msg):
        project_1d_fields_to_mesh_edges(mesh, invalid_u_1d)
