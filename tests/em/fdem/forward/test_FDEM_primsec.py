from SimPEG import *
from SimPEG.EM import mu_0, FDEM, Analytics
from SimPEG.EM.Utils import omega
try:
    from pymatsolver import PardisoSolver as Solver
except ImportError:
    Solver = SolverLU
import matplotlib.pyplot as plt
import time
import os
from SimPEG.Utils.io_utils import remoteDownload
import unittest
from SimPEG import Tests

TOL = 1e-2 # relative tolerance for prim-sec comparison

np.random.seed(2016)

# To test the primary secondary-source, we look at make sure doing primary
# secondary for a simple model gives comprable results to just solving a 3D
# problem

# Also run a sensitivity test, adjoint test


class PrimSecFDEMSrcTest_Cyl2Cart(unittest.TestCase):

    # physical properties
    sigmaback = 1e-1
    sigmablock = 1

    block_x = np.r_[100., 200.]
    block_y = np.r_[-50., 50.]
    block_z = np.r_[-50., 50.]

    # source
    src_loc = np.r_[0., 0., 0.]
    freq = 10

    # receivers
    rx_x = np.linspace(-175., 175., 8)
    rx_y = rx_x.copy()
    rx_z = np.r_[175.]
    rx_locs = Utils.ndgrid(rx_x, rx_y, rx_z)

    # mesh
    csx, ncx, npadx = 25., 16, 12
    csz, ncz, npadz = 25., 8, 12
    pf = 1.3

    @property
    def model(self):
        # this is our model!
        return np.r_[np.log(self.sigmaback),
                     np.log(self.sigmablock),
                     np.mean(self.block_z),
                     np.diff(self.block_z),
                     np.mean(self.block_x),
                     np.diff(self.block_x),
                     np.mean(self.block_y),
                     np.diff(self.block_y)]

    @property
    def rxlist(self):
        if getattr(self, '_rxlist', None) is None:
            rxlist = []
            for rxtype in ['b', 'e']:
                rx = getattr(FDEM.Rx, 'Point_{}'.format(rxtype))
                for orientation in ['x', 'y', 'z']:
                    for comp in ['real', 'imag']:
                        rxlist.append(rx(self.rx_locs, component=comp,
                                         orientation=orientation))
            self._rxlist = rxlist
        return self._rxlist

    # --------------------- Primary --------------------- #
    @property
    def meshp(self):
        if getattr(self, '_meshp', None) is None:
            hx = [(self.csx, self.ncx), (self.csx, self.npadx, self.pf)]
            hz = [(self.csz, self.npadz, -self.pf), (self.csz, self.ncz),
                  (self.csz, self.npadz, self.pf)]
            self._meshp = Mesh.CylMesh([hx, 1., hz], x0='0CC' )
        return self._meshp

    @property
    def primaryMapping(self):
        if getattr(self, '_primaryMapping', None) is None:
            self._primaryMapping = (Maps.ExpMap(self.meshp) *
                                    Maps.SurjectFull(self.meshp) *
                                    Maps.Projection([0], [0], (1, 8)))
        return self._primaryMapping

    # --------------------- 3D Problem --------------------- #
    @property
    def meshs(self):
        if getattr(self, '_meshs', None) is None:
            h = [(self.csz, self.npadz-2, -self.pf),
                 (self.csz, self.ncz),
                 (self.csz, self.npadz-2, self.pf)]
            self._meshs = Mesh.TensorMesh(3*[h], x0 = 'CCC')
        return self._meshs

    @property
    def mapping(self):
        if getattr(self, '_mapping', None) is None:
            self._mapping = (
                Maps.ExpMap(self.meshs) *
                Maps.ParametrizedBlockInLayer(self.meshs) *
                Maps.Projection(np.arange(0, 9), np.hstack([np.r_[0],
                                np.arange(0, 8)]), (9, 8))
                )
        return self._mapping

    # --------------------- Secondary --------------------- #
    @property
    def primaryMap2Meshs(self):
        if getattr(self, '_primaryMap2Meshs', None) is None:
            self._primaryMap2Meshs = (
                Maps.ExpMap(self.meshs) *
                Maps.SurjectFull(self.meshs) *
                Maps.Projection([0], [0], (1, 8))
            )
        return self._primaryMap2Meshs


    # --------------------- SetUp the test --------------------- #
    def setUp(self):

        print('\n---------------- Testing Prim-Sec Source ----------------\n')

        # Primary Problem
        self.primarySrc = FDEM.Src.MagDipole(self.rxlist, freq=self.freq,
                                             loc=self.src_loc)
        self.primarySurvey = FDEM.Survey([self.primarySrc])
        self.primaryProblem = FDEM.Problem3D_b(self.meshp,
                                               mapping=self.primaryMapping)
        self.primaryProblem.solver = Solver

        # Secondary Problem
        self.secondaryProblem = FDEM.Problem3D_b(self.meshs,
                                                 mapping=self.mapping)
        self.secondaryProblem.Solver = Solver
        self.secondarySrc = FDEM.Src.PrimSecMappedSigma(
            self.rxlist, self.freq, self.primaryProblem, self.primarySurvey,
            self.primaryMap2Meshs)
        self.secondarySurvey = FDEM.Survey([self.secondarySrc])
        self.secondaryProblem.pair(self.secondarySurvey)

        # Full 3D problem to compare with
        self.problem3D = FDEM.Problem3D_b(self.meshs, mapping=self.mapping)
        self.problem3D.Solver = Solver
        self.survey3D = FDEM.Survey([self.primarySrc])
        self.problem3D.pair(self.survey3D)

        # solve for the fields
        print('Setting up ... ')
        print('   solving primary - secondary')
        self.fields_primsec = self.secondaryProblem.fields(self.model)
        print('   ... done')

        print('   solving 3D')
        self.fields_3D = self.problem3D.fields(self.model)
        print('   ... done')
        print('done setting up \n')

        return None

    # --------------------- Run some tests! --------------------- #
    def test_data(self):
        dpred_primsec = self.secondarySurvey.dpred(self.model, f=self.fields_primsec)
        dpred_3D = self.survey3D.dpred(self.model, f=self.fields_3D)

        nrx_locs = self.rx_locs.shape[0]
        dpred_primsec = dpred_primsec.reshape(nrx_locs, len(self.rxlist))
        dpred_3D = dpred_3D.reshape(nrx_locs, len(self.rxlist))

        for i in range(len(self.rxlist)):
            rx = self.rxlist[i]
            normps = np.linalg.norm(dpred_primsec[:, i])
            norm3D = np.linalg.norm(dpred_3D[:, i])
            normdiff = np.linalg.norm(dpred_primsec[:, i] - dpred_3D[:, i])
            passed = normdiff < TOL * np.mean([normps, norm3D])
            print(
                '  Testing {rxfield}{rxorient} {rxcomp}...   '
                'prim-sec: {normps:25.10e}, 3D: {norm3D}, diff: {diff} '
                ' passed? {passed}'.format(
                    rxfield=rx.projField, rxorient=rx.projComp,
                    rxcomp=rx.component, normps=normps, norm3D=norm3D,
                    diff=normdiff, passed=passed
                ) )

            self.assertTrue(passed)

        return True







if __name__ == '__main__':
    unittest.main()
