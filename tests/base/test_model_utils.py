
import unittest

import numpy as np

from discretize import TensorMesh

from SimPEG.potential_fields import magnetics
from SimPEG import (
    maps,
    utils,
)

np.random.seed(100)
decimal_digit = 2
norm = np.linalg.norm

class DepthWeightingTest(unittest.TestCase):

    def setUp(self):

        # Mesh
        dh = 5.0
        hx = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
        hy = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
        hz = [(dh, 15)]
        mesh = TensorMesh([hx, hy, hz], "CCN")
        
        actv = [True for i in range(mesh.nC)]
        nC = mesh.nC

        # Data
        x = np.linspace(-100, 100, 20)
        y = np.linspace(-100, 100, 20)
        x, y = np.meshgrid(x, y)
        x, y = utils.mkvc(x.T), utils.mkvc(y.T)
        z = np.ones(len(x))*0.1
        xyzLoc = np.c_[x, y, z]

        # Model
        model = np.zeros(mesh.nC)
        model_3d = np.reshape(model, [mesh.nCx, mesh.nCy, mesh.nCz], order="F")
        model_3d[
            int(mesh.nCx/2)-2 : int(mesh.nCx/2)+2, 
            int(mesh.nCy/2)-2 : int(mesh.nCy/2)+2,
            mesh.nCz-3 : mesh.nCz-1,
        ] = 0.5
        
        model = utils.mkvc(model_3d)

        # construct the survey
        components = ["tmi"]
        inclination = 90
        declination = 0
        strength = 50000
        inducing_field = (strength, inclination, declination)
        
        receiver_list = magnetics.receivers.Point(xyzLoc, components=components)
        
        source_field = magnetics.sources.SourceField(
            receiver_list=[receiver_list],
            parameters=inducing_field
        )
        
        survey = magnetics.survey.Survey(source_field)
        
        # Create reduced identity map
        model_map = maps.IdentityMap(nP=nC)
        
        # Create the forward model operator
        simulation = magnetics.simulation.Simulation3DIntegral(
            survey=survey, mesh=mesh, modelType="susceptibility", 
            chiMap=model_map, actInd=actv, 
        )
        
        self.simulation = simulation
        self.actv = actv
        self.mesh = mesh

    def test_depth_weighting(self):
        
        # Depth weighting
        wz = utils.depth_weighting(self.mesh, 0.1, indActive=self.actv, exponent=5, threshold=0)
        wz /= np.nanmax(wz)
        
        # Sensitivity weighting
        kernel = np.sum(self.simulation.G**2., axis=0)**0.5
        kernel /= np.nanmax(kernel)
        
        # Randomly select a few cells for the comparision of depth and sensitivity weighting 
        ind = [np.random.randint(0, self.mesh.nC) for i in range(self.mesh.nCz)]
        
        self.assertAlmostEqual(
            norm(wz[ind]), 
            norm(kernel[ind]), 
            places=decimal_digit
            )


if __name__ == "__main__":
    unittest.main()