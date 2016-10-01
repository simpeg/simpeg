from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np

from SimPEG import Mesh
from SimPEG import Maps


class TestWires(unittest.TestCase):

    def test_basic(self):
        mesh = Mesh.TensorMesh([10, 10, 10])

        wires = Maps.Wires(
            ('sigma', mesh.nCz),
            ('mu_casing', 1),
        )

        model = np.arange(mesh.nCz + 1)

        assert isinstance(wires.sigma, Maps.WireMap)
        assert wires.nP == mesh.nCz + 1

        named_model = wires * model

        named_model.sigma == model[:mesh.nCz]
        named_model.mu_casing = 2


if __name__ == '__main__':
    unittest.main()
