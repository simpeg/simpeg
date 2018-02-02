from __future__ import print_function
import unittest
from SimPEG import Mesh
from SimPEG import Utils
from SimPEG import Maps
from SimPEG import Regularization
from SimPEG import DataMisfit
from SimPEG import Optimization
from SimPEG import InvProblem
from SimPEG import Directives
from SimPEG import Inversion
import numpy as np
import SimPEG.PF as PF


class MagInvLinProblemTest(unittest.TestCase):

    def setUp(self):

        np.random.seed(0)

        # Define the inducing field parameter
        H0 = (50000, 90, 0)

        # Create a mesh
        dx = 5.

        hxind = [(dx, 5, -1.3), (dx, 5), (dx, 5, 1.3)]
        hyind = [(dx, 5, -1.3), (dx, 5), (dx, 5, 1.3)]
        hzind = [(dx, 5, -1.3), (dx, 6)]

        mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CCC')

        # Get index of the center
        midx = int(mesh.nCx/2)
        midy = int(mesh.nCy/2)

        # Lets create a simple Gaussian topo and set the active cells
        [xx, yy] = np.meshgrid(mesh.vectorNx, mesh.vectorNy)
        zz = -np.exp((xx**2 + yy**2) / 75**2) + mesh.vectorNz[-1]

        # Go from topo to actv cells
        topo = np.c_[Utils.mkvc(xx), Utils.mkvc(yy), Utils.mkvc(zz)]
        actv = Utils.surface2ind_topo(mesh, topo, 'N')
        actv = np.asarray([inds for inds, elem in enumerate(actv, 1)
                          if elem], dtype=int) - 1

        # Create active map to go from reduce space to full
        actvMap = Maps.InjectActiveCells(mesh, actv, -100)
        nC = len(actv)

        # Create and array of observation points
        xr = np.linspace(-20., 20., 20)
        yr = np.linspace(-20., 20., 20)
        X, Y = np.meshgrid(xr, yr)

        # Move the observation points 5m above the topo
        Z = -np.exp((X**2 + Y**2) / 75**2) + mesh.vectorNz[-1] + 5.

        # Create a MAGsurvey
        rxLoc = np.c_[Utils.mkvc(X.T), Utils.mkvc(Y.T), Utils.mkvc(Z.T)]
        rxLoc = PF.BaseMag.RxObs(rxLoc)
        srcField = PF.BaseMag.SrcField([rxLoc], param=H0)
        survey = PF.BaseMag.LinearSurvey(srcField)

        # We can now create a susceptibility model and generate data
        # Here a simple block in half-space
        model = np.zeros((mesh.nCx, mesh.nCy, mesh.nCz))
        model[(midx-2):(midx+2), (midy-2):(midy+2), -6:-2] = 0.02
        model = Utils.mkvc(model)
        self.model = model[actv]

        # Create active map to go from reduce set to full
        actvMap = Maps.InjectActiveCells(mesh, actv, -100)

        # Creat reduced identity map
        idenMap = Maps.IdentityMap(nP=nC)

        # Create the forward model operator
        prob = PF.Magnetics.MagneticIntegral(mesh, chiMap=idenMap,
                                             actInd=actv)

        # Pair the survey and problem
        survey.pair(prob)

        # Compute linear forward operator and compute some data
        d = prob.fields(self.model)

        # Add noise and uncertainties (1nT)
        data = d + np.random.randn(len(d))
        wd = np.ones(len(data))*1.

        survey.dobs = data
        survey.std = wd

        # Create sensitivity weights from our linear forward operator
        wr = np.sum(prob.G**2., axis=0)**0.5
        wr = (wr/np.max(wr))

        # Create a regularization
        reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
        reg.cell_weights = wr
        reg.norms = [0, 1, 1, 1]
        reg.eps_p, reg.eps_q = 1e-3, 1e-3

        # Data misfit function
        dmis = DataMisfit.l2_DataMisfit(survey)
        dmis.W = 1/wd

        # Add directives to the inversion
        opt = Optimization.ProjectedGNCG(maxIter=100, lower=0., upper=1.,
                                         maxIterLS=20, maxIterCG=10,
                                         tolCG=1e-3)

        invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
        betaest = Directives.BetaEstimate_ByEig()

        # Here is where the norms are applied
        IRLS = Directives.Update_IRLS(f_min_change=1e-3,
                                      minGNiter=3)
        update_Jacobi = Directives.UpdatePreconditioner()
        self.inv = Inversion.BaseInversion(invProb,
                                           directiveList=[IRLS, betaest,
                                                          update_Jacobi])

    def test_mag_inverse(self):

        # Run the inversion
        mrec = self.inv.run(self.model)

        residual = np.linalg.norm(mrec-self.model) / np.linalg.norm(self.model)
        print(residual)
        self.assertTrue(residual < 0.05)

if __name__ == '__main__':
    unittest.main()
