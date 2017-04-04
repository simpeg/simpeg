from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
import pickle
import properties

from SimPEG import Mesh
from SimPEG import Maps
from SimPEG import Utils
from SimPEG import Props


class SimpleExample(Props.HasModel):

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


class ShortcutExample(Props.HasModel):

    sigma, sigmaMap, sigmaDeriv = Props.Invertible(
        "Electrical conductivity (S/m)"
    )


class ReciprocalMappingExample(Props.HasModel):

    sigma, sigmaMap, sigmaDeriv = Props.Invertible(
        "Electrical conductivity (S/m)"
    )

    rho, rhoMap, rhoDeriv = Props.Invertible(
        "Electrical resistivity (Ohm m)"
    )

    Props.Reciprocal(sigma, rho)


class ReciprocalExample(Props.HasModel):

    sigma, sigmaMap, sigmaDeriv = Props.Invertible(
        "Electrical conductivity (S/m)"
    )

    rho = Props.PhysicalProperty(
        "Electrical resistivity (Ohm m)"
    )

    Props.Reciprocal(sigma, rho)


class ReciprocalPropExample(Props.HasModel):

    sigma = Props.PhysicalProperty(
        "Electrical conductivity (S/m)"
    )

    rho = Props.PhysicalProperty(
        "Electrical resistivity (Ohm m)"
    )

    Props.Reciprocal(sigma, rho)


class ReciprocalPropExampleDefaults(Props.HasModel):

    sigma = Props.PhysicalProperty(
        "Electrical conductivity (S/m)",
        default=np.r_[1., 2., 3.]
    )

    rho = Props.PhysicalProperty(
        "Electrical resistivity (Ohm m)"
    )

    Props.Reciprocal(sigma, rho)


class ComplicatedInversion(Props.HasModel):

    Ks, KsMap, KsDeriv = Props.Invertible(
        "Saturated hydraulic conductivity",
        default=24.96
    )

    A, AMap, ADeriv = Props.Invertible(
        "fitting parameter",
        default=1.175e+06
    )

    gamma, gammaMap, gammaDeriv = Props.Invertible(
        "fitting parameter",
        default=4.74
    )


class NestedModels(Props.HasModel):
    complicated = properties.Instance(
        "Nested models",
        ComplicatedInversion
    )


class TestPropMaps(unittest.TestCase):

    def setUp(self):
        pass

    def test_basic(self):
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
            PM = pickle.loads(pickle.dumps(PM))
            assert np.all(
                PM.sigmaDeriv.todense() ==
                Utils.sdiag(np.exp(np.r_[1., 2., 3.])).todense()
            )

            # If we set sigma, we should delete the mapping
            PM.sigma = np.r_[1., 2., 3.]
            assert np.all(PM.sigma == np.r_[1., 2., 3.])
            PM = pickle.loads(pickle.dumps(PM))
            assert PM.sigmaMap is None
            assert PM.sigmaDeriv == 0

            del PM.model
            # sigma is not changed
            assert np.all(PM.sigma == np.r_[1., 2., 3.])

    def test_reciprocal(self):
        expMap = Maps.ExpMap(Mesh.TensorMesh((3,)))

        PM = ReciprocalMappingExample()

        self.assertRaises(AttributeError, getattr, PM, 'sigma')
        PM.sigmaMap = expMap
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
        PM = pickle.loads(pickle.dumps(PM))
        PM.rhoMap = expMap
        assert PM._get('sigmaMap') is None
        assert len(PM.rhoMap) == 1
        assert len(PM.sigmaMap) == 2
        PM = pickle.loads(pickle.dumps(PM))
        assert np.all(PM.rho == np.exp(np.r_[1., 2., 3.]))
        assert np.all(PM.sigma == 1.0 / np.exp(np.r_[1., 2., 3.]))
        PM = pickle.loads(pickle.dumps(PM))
        assert isinstance(PM.sigmaDeriv.todense(), np.ndarray)

    def test_reciprocal_no_map(self):
        expMap = Maps.ExpMap(Mesh.TensorMesh((3,)))

        PM = ReciprocalExample()
        self.assertRaises(AttributeError, getattr, PM, 'sigma')

        PM.sigmaMap = expMap
        PM = pickle.loads(pickle.dumps(PM))
        PM.model = np.r_[1., 2., 3.]
        assert np.all(PM.sigma == np.exp(np.r_[1., 2., 3.]))
        assert np.all(PM.rho == 1.0 / np.exp(np.r_[1., 2., 3.]))

        PM.rho = np.r_[1., 2., 3.]
        assert PM.sigmaMap is None
        assert PM.sigmaDeriv == 0
        assert np.all(PM.sigma == 1.0 / np.r_[1., 2., 3.])

        PM.sigmaMap = expMap
        assert len(PM.sigmaMap) == 1
        PM = pickle.loads(pickle.dumps(PM))
        assert np.all(PM.rho == 1.0 / np.exp(np.r_[1., 2., 3.]))
        assert np.all(PM.sigma == np.exp(np.r_[1., 2., 3.]))
        assert isinstance(PM.sigmaDeriv.todense(), np.ndarray)

    def test_reciprocal_no_maps(self):

        PM = ReciprocalPropExample()
        self.assertRaises(AttributeError, getattr, PM, 'sigma')

        PM = pickle.loads(pickle.dumps(PM))
        PM.sigma = np.r_[1., 2., 3.]
        PM = pickle.loads(pickle.dumps(PM))

        assert np.all(PM.sigma == np.r_[1., 2., 3.])
        PM = pickle.loads(pickle.dumps(PM))
        assert np.all(PM.rho == 1.0 / np.r_[1., 2., 3.])

        PM.rho = np.r_[1., 2., 3.]
        assert np.all(PM.sigma == 1.0 / np.r_[1., 2., 3.])

    def test_reciprocal_defaults(self):

        PM = ReciprocalPropExampleDefaults()
        assert np.all(PM.sigma == np.r_[1., 2., 3.])
        assert np.all(PM.rho == 1.0 / np.r_[1., 2., 3.])

        rho = np.r_[2., 4., 6.]
        PM.rho = rho
        assert np.all(PM.rho == rho)
        assert np.all(PM.sigma == 1./rho)

    def test_multi_parameter_inversion(self):
        """The setup of the defaults should not invalidated the
        mappings or other defaults.
        """
        PM = ComplicatedInversion()

        assert PM.Ks == PM._props['Ks'].default
        assert PM.gamma == PM._props['gamma'].default
        assert PM.A == PM._props['A'].default

    def test_summary_validate(self):

        PM = ComplicatedInversion()
        PM.summary()
        PM.validate()
        with self.assertRaises(ValueError):
            PM.model = np.ones(2)
            PM.summary()
            PM.validate()
        PM.AMap = Maps.ExpMap(nP=3)
        with self.assertRaises(ValueError):
            PM.model = np.ones(2)
            PM.summary()
            PM.validate()
        PM.gammaMap = Maps.ExpMap(nP=2)
        with self.assertRaises(ValueError):
            # maps are mismatching sizes
            PM.model = np.ones(2)
            PM.summary()
            PM.validate()
        PM.AMap = Maps.ExpMap(nP=2)
        PM.model = np.ones(2)
        PM.summary()
        PM.validate()
        assert PM.KsDeriv == 0

    def test_nested(self):
        PM = NestedModels()
        assert PM._has_nested_models is True


if __name__ == '__main__':
    unittest.main()
