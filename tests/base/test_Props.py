from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
import pickle
import properties

import discretize

from SimPEG import maps
from SimPEG import utils
from SimPEG import props


class SimpleExample(props.HasModel):

    sigmaMap = props.Mapping("Mapping to the inversion model.")

    sigma = props.PhysicalProperty("Electrical conductivity (S/m)", mapping=sigmaMap)

    sigmaDeriv = props.Derivative(
        "Derivative of sigma wrt the model.", physical_property=sigma
    )


class ShortcutExample(props.HasModel):

    sigma, sigmaMap, sigmaDeriv = props.Invertible("Electrical conductivity (S/m)")


class ReciprocalMappingExample(props.HasModel):

    sigma, sigmaMap, sigmaDeriv = props.Invertible("Electrical conductivity (S/m)")

    rho, rhoMap, rhoDeriv = props.Invertible("Electrical resistivity (Ohm m)")

    props.Reciprocal(sigma, rho)


class ReciprocalExample(props.HasModel):

    sigma, sigmaMap, sigmaDeriv = props.Invertible("Electrical conductivity (S/m)")

    rho = props.PhysicalProperty("Electrical resistivity (Ohm m)")

    props.Reciprocal(sigma, rho)


class ReciprocalPropExample(props.HasModel):

    sigma = props.PhysicalProperty("Electrical conductivity (S/m)")

    rho = props.PhysicalProperty("Electrical resistivity (Ohm m)")

    props.Reciprocal(sigma, rho)


class ReciprocalPropExampleDefaults(props.HasModel):

    sigma = props.PhysicalProperty(
        "Electrical conductivity (S/m)", default=np.r_[1.0, 2.0, 3.0]
    )

    rho = props.PhysicalProperty("Electrical resistivity (Ohm m)")

    props.Reciprocal(sigma, rho)


class ComplicatedInversion(props.HasModel):

    Ks, KsMap, KsDeriv = props.Invertible(
        "Saturated hydraulic conductivity", default=24.96
    )

    A, AMap, ADeriv = props.Invertible("fitting parameter", default=1.175e06)

    gamma, gammaMap, gammaDeriv = props.Invertible("fitting parameter", default=4.74)


class NestedModels(props.HasModel):
    complicated = properties.Instance("Nested models", ComplicatedInversion)


class TestPropMaps(unittest.TestCase):
    def setUp(self):
        pass

    def test_basic(self):
        expMap = maps.ExpMap(discretize.TensorMesh((3,)))
        assert expMap.nP == 3

        for Example in [SimpleExample, ShortcutExample]:

            PM = Example(sigmaMap=expMap)
            assert PM.sigmaMap is not None
            assert PM.sigmaMap is expMap

            # There is currently no model, so sigma, which is mapped, fails
            self.assertRaises(AttributeError, getattr, PM, "sigma")

            PM.model = np.r_[1.0, 2.0, 3.0]
            assert np.all(PM.sigma == np.exp(np.r_[1.0, 2.0, 3.0]))
            # PM = pickle.loads(pickle.dumps(PM))
            # PM = maps.ExpMap.deserialize(PM.serialize())

            assert np.all(
                PM.sigmaDeriv.todense()
                == utils.sdiag(np.exp(np.r_[1.0, 2.0, 3.0])).todense()
            )

            # If we set sigma, we should delete the mapping
            PM.sigma = np.r_[1.0, 2.0, 3.0]
            assert np.all(PM.sigma == np.r_[1.0, 2.0, 3.0])
            # PM = pickle.loads(pickle.dumps(PM))
            assert PM.sigmaMap is None
            assert PM.sigmaDeriv == 0

            del PM.model
            # sigma is not changed
            assert np.all(PM.sigma == np.r_[1.0, 2.0, 3.0])

    def test_reciprocal(self):
        expMap = maps.ExpMap(discretize.TensorMesh((3,)))

        PM = ReciprocalMappingExample()

        self.assertRaises(AttributeError, getattr, PM, "sigma")
        PM.sigmaMap = expMap
        PM.model = np.r_[1.0, 2.0, 3.0]
        assert np.all(PM.sigma == np.exp(np.r_[1.0, 2.0, 3.0]))
        assert np.all(PM.rho == 1.0 / np.exp(np.r_[1.0, 2.0, 3.0]))

        PM.rho = np.r_[1.0, 2.0, 3.0]
        assert PM.rhoMap is None
        assert PM.sigmaMap is None
        assert PM.rhoDeriv == 0
        assert PM.sigmaDeriv == 0
        assert np.all(PM.sigma == 1.0 / np.r_[1.0, 2.0, 3.0])

        PM.sigmaMap = expMap
        # change your mind?
        # PM = pickle.loads(pickle.dumps(PM))
        PM.rhoMap = expMap
        assert PM._get("sigmaMap") is None
        assert len(PM.rhoMap) == 1
        assert len(PM.sigmaMap) == 2
        # PM = pickle.loads(pickle.dumps(PM))
        assert np.all(PM.rho == np.exp(np.r_[1.0, 2.0, 3.0]))
        assert np.all(PM.sigma == 1.0 / np.exp(np.r_[1.0, 2.0, 3.0]))
        # PM = pickle.loads(pickle.dumps(PM))
        assert isinstance(PM.sigmaDeriv.todense(), np.ndarray)

    def test_reciprocal_no_map(self):
        expMap = maps.ExpMap(discretize.TensorMesh((3,)))

        PM = ReciprocalExample()
        self.assertRaises(AttributeError, getattr, PM, "sigma")

        PM.sigmaMap = expMap
        # PM = pickle.loads(pickle.dumps(PM))
        PM.model = np.r_[1.0, 2.0, 3.0]
        assert np.all(PM.sigma == np.exp(np.r_[1.0, 2.0, 3.0]))
        assert np.all(PM.rho == 1.0 / np.exp(np.r_[1.0, 2.0, 3.0]))

        PM.rho = np.r_[1.0, 2.0, 3.0]
        assert PM.sigmaMap is None
        assert PM.sigmaDeriv == 0
        assert np.all(PM.sigma == 1.0 / np.r_[1.0, 2.0, 3.0])

        PM.sigmaMap = expMap
        assert len(PM.sigmaMap) == 1
        # PM = pickle.loads(pickle.dumps(PM))
        assert np.all(PM.rho == 1.0 / np.exp(np.r_[1.0, 2.0, 3.0]))
        assert np.all(PM.sigma == np.exp(np.r_[1.0, 2.0, 3.0]))
        assert isinstance(PM.sigmaDeriv.todense(), np.ndarray)

    def test_reciprocal_no_maps(self):

        PM = ReciprocalPropExample()
        self.assertRaises(AttributeError, getattr, PM, "sigma")

        # PM = pickle.loads(pickle.dumps(PM))
        PM.sigma = np.r_[1.0, 2.0, 3.0]
        # PM = pickle.loads(pickle.dumps(PM))

        assert np.all(PM.sigma == np.r_[1.0, 2.0, 3.0])
        # PM = pickle.loads(pickle.dumps(PM))
        assert np.all(PM.rho == 1.0 / np.r_[1.0, 2.0, 3.0])

        PM.rho = np.r_[1.0, 2.0, 3.0]
        assert np.all(PM.sigma == 1.0 / np.r_[1.0, 2.0, 3.0])

    def test_reciprocal_defaults(self):

        PM = ReciprocalPropExampleDefaults()
        assert np.all(PM.sigma == np.r_[1.0, 2.0, 3.0])
        assert np.all(PM.rho == 1.0 / np.r_[1.0, 2.0, 3.0])

        rho = np.r_[2.0, 4.0, 6.0]
        PM.rho = rho
        assert np.all(PM.rho == rho)
        assert np.all(PM.sigma == 1.0 / rho)

    def test_multi_parameter_inversion(self):
        """The setup of the defaults should not invalidated the
        mappings or other defaults.
        """
        PM = ComplicatedInversion()

        assert PM.Ks == PM._props["Ks"].default
        assert PM.gamma == PM._props["gamma"].default
        assert PM.A == PM._props["A"].default

    def test_summary_validate(self):

        PM = ComplicatedInversion()
        PM.summary()
        PM.validate()
        with self.assertRaises(ValueError):
            PM.model = np.ones(2)
            PM.summary()
            PM.validate()
        PM.AMap = maps.ExpMap(nP=3)
        with self.assertRaises(ValueError):
            PM.model = np.ones(2)
            PM.summary()
            PM.validate()
        PM.gammaMap = maps.ExpMap(nP=2)
        with self.assertRaises(ValueError):
            # maps are mismatching sizes
            PM.model = np.ones(2)
            PM.summary()
            PM.validate()
        PM.AMap = maps.ExpMap(nP=2)
        PM.model = np.ones(2)
        PM.summary()
        PM.validate()
        assert PM.KsDeriv == 0

    def test_nested(self):
        PM = NestedModels()
        assert PM._has_nested_models is True


if __name__ == "__main__":
    unittest.main()
