import unittest
from SimPEG import *
from SimPEG import EM

TOL = 1e-5
FLR = 1e-20

np.random.seed(seed=25) # set a seed so that the same conductivity model is used for all runs


def setUp_TDEM(prbtype='b', rxcomp='bz'):
    cs = 5.
    ncx = 20
    ncy = 15
    npad = 20
    hx = [(cs, ncx), (cs, npad, 1.3)]
    hy = [(cs, npad, -1.3), (cs, ncy), (cs, npad, 1.3)]
    mesh = Mesh.CylMesh([hx, 1, hy], '00C')
#
    active = mesh.vectorCCz <  0.
    activeMap = Maps.InjectActiveCells(mesh, active, np.log(1e-8), nC=mesh.nCz)
    mapping = Maps.ExpMap(mesh) * Maps.SurjectVertical1D(mesh) * activeMap

    rxOffset = 10.
    rx = EM.TDEM.Rx(np.array([[rxOffset, 0., -1e-2]]), np.logspace(-4, -3, 20), rxcomp)
    src = EM.TDEM.Src.MagDipole([rx], loc=np.array([0., 0., 0.]))

    survey = EM.TDEM.Survey([src])

    if prbtype == 'b':
        prb = EM.TDEM.Problem_b(mesh, mapping=mapping)
    elif prbtype == 'e':
        prb = EM.TDEM.Problem_e(mesh, mapping=mapping)
    else:
        raise NotImplementedError()

    prb.timeSteps = [(1e-05, 10), (5e-05, 10), (2.5e-4, 10)]
    # prb.timeSteps = [(1e-05, 10), (1e-05, 50), (1e-05, 50) ] #, (2.5e-4, 10)]

    try:
        from pymatsolver import MumpsSolver
        prb.Solver = MumpsSolver
    except ImportError, e:
        prb.Solver = SolverLU

    m = np.log(1e-1)*np.ones(prb.mapping.nP) + 1e-2*np.random.rand(prb.mapping.nP)

    prb.pair(survey)
    mesh = mesh

    return prb, m, mesh

def CrossCheck(prbtype1='b', prbtype2='e', rxcomp='bz'):

    prb1, m1, mesh1 = setUp_TDEM(prbtype1, rxcomp)
    prb2, _, mesh2  = setUp_TDEM(prbtype2, rxcomp)

    d1 = prb1.survey.dpred(m1)
    d2 = prb2.survey.dpred(m1)


    check = np.linalg.norm(d1 - d2)
    tol   = 0.5 * (np.linalg.norm(d1) + np.linalg.norm(d2)) * TOL
    passed = check < tol

    print 'Checking %s, %s for %s data'%(prbtype1, prbtype2, rxcomp)
    print '  ', np.linalg.norm(d1), np.linalg.norm(d2), np.linalg.norm(check), tol, passed

    assert passed


class TDEM_cross_check_EB(unittest.TestCase):
    def test_EB_ey(self):
        CrossCheck(prbtype1='b', prbtype2='e', rxcomp='ey')


if __name__ == '__main__':
    unittest.main()

