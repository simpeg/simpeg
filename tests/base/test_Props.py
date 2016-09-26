from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np

from SimPEG import Mesh
from SimPEG import Maps
from SimPEG import Utils
from SimPEG import Props


class SimpleExample(Props.BaseSimPEG):

    sigmaMap = Props.Mapping(
        "Mapping to the inversion model."
    )

    sigma = Props.PhysicalProperty(
        "Electrical conductivity (S/m)",
        mapping=sigmaMap
    )

    sigmaDeriv = Props.Derivative(
        "Derivative of sigma wrt the model.",
        physical_property=sigma
    )


class ShortcutExample(Props.BaseSimPEG):

    sigma, sigmaMap, sigmaDeriv = Props.Invertible(
        "Electrical conductivity (S/m)"
    )


class ReciprocalExample(Props.BaseSimPEG):

    sigma, sigmaMap, sigmaDeriv = Props.Invertible(
        "Electrical conductivity (S/m)"
    )

    rho, rhoMap, rhoDeriv = Props.Invertible(
        "Electrical resistivity (Ohm m)"
    )

    Props.Reciprocal(sigma, rho)


class TestPropMaps(unittest.TestCase):

    def setUp(self):
        pass

    def test_setup(self):
        expMap = Maps.ExpMap(Mesh.TensorMesh((3,)))
        assert expMap.nP == 3

        for Example in [SimpleExample, ShortcutExample]:

            PM = Example(sigmaMap=expMap)
            assert PM.sigmaMap is not None
            assert PM.sigmaMap is expMap

            # There is currently no model, so sigma, which is mapped, fails
            self.assertRaises(AttributeError, getattr, PM, 'sigma')

            PM.model = np.r_[1., 2., 3.]
            assert np.all(PM.sigma == np.exp(np.r_[1., 2., 3.]))
            assert np.all(
                PM.sigmaDeriv.todense() ==
                Utils.sdiag(np.exp(np.r_[1., 2., 3.])).todense()
            )

            # If we set sigma, we should delete the mapping
            PM.sigma = np.r_[1., 2., 3.]
            assert np.all(PM.sigma == np.r_[1., 2., 3.])
            assert PM.sigmaMap is None
            assert PM.sigmaDeriv == 0

            del PM.model
            # sigma is not changed
            assert np.all(PM.sigma == np.r_[1., 2., 3.])

    def test_reciprocal(self):
        expMap = Maps.ExpMap(Mesh.TensorMesh((3,)))

        PM = ReciprocalExample(sigmaMap=expMap)

        PM.model = np.r_[1., 2., 3.]
        assert np.all(PM.sigma == np.exp(np.r_[1., 2., 3.]))
        assert np.all(PM.rho == 1.0 / np.exp(np.r_[1., 2., 3.]))

        PM.rho = np.r_[1., 2., 3.]
        assert PM.rhoMap is None
        assert PM.sigmaMap is None
        assert PM.rhoDeriv == 0
        assert PM.sigmaDeriv == 0
        assert np.all(PM.sigma == 1.0 / np.r_[1., 2., 3.])

        PM.sigmaMap = expMap
        # change your mind?
        PM.rhoMap = expMap
        assert PM._get('sigmaMap') is None
        assert len(PM.rhoMap) == 1
        assert len(PM.sigmaMap) == 2
        assert np.all(PM.rho == np.exp(np.r_[1., 2., 3.]))
        assert np.all(PM.sigma == 1.0 / np.exp(np.r_[1., 2., 3.]))
        assert isinstance(PM.sigmaDeriv.todense(), np.ndarray)


if __name__ == '__main__':
    unittest.main()
