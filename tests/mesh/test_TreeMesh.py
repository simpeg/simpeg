from __future__ import print_function
from SimPEG import Mesh, Tests
from SimPEG.Mesh.TreeMesh import CellLookUpException
import numpy as np
import matplotlib.pyplot as plt
import unittest

TOL = 1e-8


class TestSimpleQuadTree(unittest.TestCase):

    def test_counts(self):
        nc = 8
        h1 = np.random.rand(nc)*nc*0.5 + nc*0.5
        h2 = np.random.rand(nc)*nc*0.5 + nc*0.5
        h = [hi/np.sum(hi) for hi in [h1, h2]]  # normalize
        M = Mesh.TreeMesh(h)
        M._refineCell([0,0,0])
        M._refineCell([0,0,1])
        M.number()
        # M.plotGrid(showIt=True)
        print(M)
        assert M.nhFx == 2
        assert M.nFx == 9

        assert np.allclose(M.vol.sum(), 1.0)

        assert np.allclose(np.r_[M._areaFxFull, M._areaFyFull], M._deflationMatrix('F') * M.area)

    def test_getitem(self):
        M = Mesh.TreeMesh([4,4])
        M.refine(1)
        assert M.nC == 4
        assert len(M) == M.nC
        assert np.allclose(M[0].center, [0.25,0.25])
        actual = [[0,0],[0.5,0],[0,0.5],[0.5,0.5]]
        for i, n in enumerate(M[0].nodes):
            assert np.allclose(M._gridN[n,:], actual[i])

    def test_getitem3D(self):
        M = Mesh.TreeMesh([4,4,4])
        M.refine(1)
        assert M.nC == 8
        assert len(M) == M.nC
        assert np.allclose(M[0].center, [0.25,0.25,0.25])
        actual = [[0,0,0],[0.5,0,0],[0,0.5,0],[0.5,0.5,0],
                  [0,0,0.5],[0.5,0,0.5],[0,0.5,0.5],[0.5,0.5,0.5]]
        for i, n in enumerate(M[0].nodes):
            assert np.allclose(M._gridN[n,:], actual[i])

    def test_refine(self):
        M = Mesh.TreeMesh([4,4,4])
        M.refine(1)
        assert M.nC == 8
        M.refine(0)
        assert M.nC == 8
        M.corsen(0)
        assert M.nC == 1

    def test_corsen(self):
        nc = 8
        h1 = np.random.rand(nc)*nc*0.5 + nc*0.5
        h2 = np.random.rand(nc)*nc*0.5 + nc*0.5
        h = [hi/np.sum(hi) for hi in [h1, h2]]  # normalize
        M = Mesh.TreeMesh(h)
        M._refineCell([0,0,0])
        M._refineCell([0,0,1])
        self.assertRaises(CellLookUpException, M._refineCell, [0,0,1])
        assert M._index([0,0,1]) not in M
        assert M._index([0,0,2]) in M
        assert M._index([2,0,2]) in M
        assert M._index([0,2,2]) in M
        assert M._index([2,2,2]) in M

        self.assertRaises(CellLookUpException, M._corsenCell, [0,0,1])
        M._corsenCell([0,0,2])
        assert M._index([0,0,1]) in M
        assert M._index([0,0,2]) not in M
        assert M._index([2,0,2]) not in M
        assert M._index([0,2,2]) not in M
        assert M._index([2,2,2]) not in M
        M._refineCell([0,0,1])

        self.assertRaises(CellLookUpException, M._corsenCell, [0,0,1])
        M._corsenCell([2,0,2])
        assert M._index([0,0,1]) in M
        assert M._index([0,0,2]) not in M
        assert M._index([2,0,2]) not in M
        assert M._index([0,2,2]) not in M
        assert M._index([2,2,2]) not in M
        M._refineCell([0,0,1])

        self.assertRaises(CellLookUpException, M._corsenCell, [0,0,1])
        M._corsenCell([0,2,2])
        assert M._index([0,0,1]) in M
        assert M._index([0,0,2]) not in M
        assert M._index([2,0,2]) not in M
        assert M._index([0,2,2]) not in M
        assert M._index([2,2,2]) not in M
        M._refineCell([0,0,1])

        self.assertRaises(CellLookUpException, M._corsenCell, [0,0,1])
        M._corsenCell([2,2,2])
        assert M._index([0,0,1]) in M
        assert M._index([0,0,2]) not in M
        assert M._index([2,0,2]) not in M
        assert M._index([0,2,2]) not in M
        assert M._index([2,2,2]) not in M

    def test_faceDiv(self):

        hx, hy = np.r_[1.,2,3,4], np.r_[5.,6,7,8]
        T = Mesh.TreeMesh([hx, hy], levels=2)
        T.refine(lambda xc:2)
        # T.plotGrid(showIt=True)
        M = Mesh.TensorMesh([hx, hy])
        assert M.nC == T.nC
        assert M.nF == T.nF
        assert M.nFx == T.nFx
        assert M.nFy == T.nFy
        assert M.nE == T.nE
        assert M.nEx == T.nEx
        assert M.nEy == T.nEy
        assert np.allclose(M.area, T.permuteF*T.area)
        assert np.allclose(M.edge, T.permuteE*T.edge)
        assert np.allclose(M.vol, T.permuteCC*T.vol)

        # plt.subplot(211).spy(M.faceDiv)
        # plt.subplot(212).spy(T.permuteCC*T.faceDiv*T.permuteF.T)
        # plt.show()

        assert (M.faceDiv - T.permuteCC*T.faceDiv*T.permuteF.T).nnz == 0


class TestOcTree(unittest.TestCase):

    def test_counts(self):
        nc = 8
        h1 = np.random.rand(nc)*nc*0.5 + nc*0.5
        h2 = np.random.rand(nc)*nc*0.5 + nc*0.5
        h3 = np.random.rand(nc)*nc*0.5 + nc*0.5
        h = [hi/np.sum(hi) for hi in [h1, h2, h3]]  # normalize
        M = Mesh.TreeMesh(h, levels=3)
        M._refineCell([0,0,0,0])
        M._refineCell([0,0,0,1])
        M.number()
        # M.plotGrid(showIt=True)
        # assert M.nhFx == 2
        # assert M.nFx == 9

        assert np.allclose(M.vol.sum(), 1.0)

        # assert np.allclose(M._areaFxFull, (M._deflationMatrix('F') * M.area)[:M.ntFx])
        # assert np.allclose(M._areaFyFull, (M._deflationMatrix('F') * M.area)[M.ntFx:(M.ntFx+M.ntFy)])
        # assert np.allclose(M._areaFzFull, (M._deflationMatrix('F') * M.area)[(M.ntFx+M.ntFy):])

        # assert np.allclose(M._edgeExFull, (M._deflationMatrix('E') * M.edge)[:M.ntEx])
        # assert np.allclose(M._edgeEyFull, (M._deflationMatrix('E') * M.edge)[M.ntEx:(M.ntEx+M.ntEy)])
        # assert np.allclose(M._edgeEzFull, (M._deflationMatrix('E') * M.edge)[(M.ntEx+M.ntEy):])

    def test_faceDiv(self):

        hx, hy, hz = np.r_[1.,2,3,4], np.r_[5.,6,7,8], np.r_[9.,10,11,12]
        M = Mesh.TreeMesh([hx, hy, hz], levels=2)
        M.refine(lambda xc:2)
        # M.plotGrid(showIt=True)
        Mr = Mesh.TensorMesh([hx, hy, hz])
        assert M.nC == Mr.nC
        assert M.nF == Mr.nF
        assert M.nFx == Mr.nFx
        assert M.nFy == Mr.nFy
        assert M.nE == Mr.nE
        assert M.nEx == Mr.nEx
        assert M.nEy == Mr.nEy
        assert np.allclose(Mr.area, M.permuteF*M.area)
        assert np.allclose(Mr.edge, M.permuteE*M.edge)
        assert np.allclose(Mr.vol, M.permuteCC*M.vol)

        # plt.subplot(211).spy(Mr.faceDiv)
        # plt.subplot(212).spy(M.permuteCC*M.faceDiv*M.permuteF.T)
        # plt.show()

        assert (Mr.faceDiv - M.permuteCC*M.faceDiv*M.permuteF.T).nnz == 0


    def test_edgeCurl(self):

        hx, hy, hz = np.r_[1.,2,3,4], np.r_[5.,6,7,8], np.r_[9.,10,11,12]
        M = Mesh.TreeMesh([hx, hy, hz], levels=2)
        M.refine(lambda xc:2)
        # M.plotGrid(showIt=True)
        Mr = Mesh.TensorMesh([hx, hy, hz])

        # plt.subplot(211).spy(Mr.faceDiv)
        # plt.subplot(212).spy(M.permuteCC.T*M.faceDiv*M.permuteF)
        # plt.show()

        assert (Mr.edgeCurl - M.permuteF*M.edgeCurl*M.permuteE.T).nnz == 0

    def test_faceInnerProduct(self):

        hx, hy, hz = np.r_[1.,2,3,4], np.r_[5.,6,7,8], np.r_[9.,10,11,12]
        # hx, hy, hz = [[(1,4)], [(1,4)], [(1,4)]]

        M = Mesh.TreeMesh([hx, hy, hz], levels=2)
        M.refine(lambda xc:2)
        # M.plotGrid(showIt=True)
        Mr = Mesh.TensorMesh([hx, hy, hz])

        # plt.subplot(211).spy(Mr.getFaceInnerProduct())
        # plt.subplot(212).spy(M.getFaceInnerProduct())
        # plt.show()

        # print(M.nC, M.nF, M.getFaceInnerProduct().shape, M.permuteF.shape)

        assert np.allclose(Mr.getFaceInnerProduct().todense(), (M.permuteF * M.getFaceInnerProduct() * M.permuteF.T).todense())
        assert np.allclose(Mr.getEdgeInnerProduct().todense(), (M.permuteE * M.getEdgeInnerProduct() * M.permuteE.T).todense())

    def test_VectorIdenties(self):
        hx, hy, hz = [[(1,4)], [(1,4)], [(1,4)]]

        M = Mesh.TreeMesh([hx, hy, hz], levels=2)
        Mr = Mesh.TensorMesh([hx, hy, hz])

        assert (M.faceDiv * M.edgeCurl).nnz == 0
        assert (Mr.faceDiv * Mr.edgeCurl).nnz == 0

        hx, hy, hz = np.r_[1.,2,3,4], np.r_[5.,6,7,8], np.r_[9.,10,11,12]

        M = Mesh.TreeMesh([hx, hy, hz], levels=2)
        Mr = Mesh.TensorMesh([hx, hy, hz])

        assert np.max(np.abs((M.faceDiv * M.edgeCurl).todense().flatten())) < TOL
        assert np.max(np.abs((Mr.faceDiv * Mr.edgeCurl).todense().flatten())) < TOL

class Test2DInterpolation(unittest.TestCase):

    def setUp(self):
        def topo(x):
            return np.sin(x*(2.*np.pi))*0.3 + 0.5

        def function(cell):
            r = cell.center - np.array([0.5]*len(cell.center))
            dist1 = np.sqrt(r.dot(r)) - 0.08
            dist2 = np.abs(cell.center[-1] - topo(cell.center[0]))

            dist = min([dist1,dist2])
            # if dist < 0.05:
            #     return 5
            if dist < 0.05:
                return 6
            if dist < 0.2:
                return 5
            if dist < 0.3:
                return 4
            if dist < 1.0:
                return 3
            else:
                return 0

        M = Mesh.TreeMesh([64,64],levels=6)
        M.refine(function)
        self.M = M

    def test_fx(self):
        r = np.random.rand(self.M.nFx)
        P = self.M.getInterpolationMat(self.M.gridFx, 'Fx')
        assert np.abs(P[:,:self.M.nFx]*r - r).max() < TOL

    def test_fy(self):
        r = np.random.rand(self.M.nFy)
        P = self.M.getInterpolationMat(self.M.gridFy, 'Fy')
        assert np.abs(P[:,self.M.nFx:]*r - r).max() < TOL


class Test3DInterpolation(unittest.TestCase):

    def setUp(self):
        def function(cell):
            r = cell.center - np.array([0.5]*len(cell.center))
            dist = np.sqrt(r.dot(r))
            if dist < 0.2:
                return 4
            if dist < 0.3:
                return 3
            if dist < 1.0:
                return 2
            else:
                return 0

        M = Mesh.TreeMesh([16,16,16],levels=4)
        M.refine(function)
        # M.plotGrid(showIt=True)
        self.M = M

    def test_Fx(self):
        r = np.random.rand(self.M.nFx)
        P = self.M.getInterpolationMat(self.M.gridFx, 'Fx')
        assert np.abs(P[:,:self.M.nFx]*r - r).max() < TOL

    def test_Fy(self):
        r = np.random.rand(self.M.nFy)
        P = self.M.getInterpolationMat(self.M.gridFy, 'Fy')
        assert np.abs(P[:,self.M.nFx:(self.M.nFx+self.M.nFy)]*r - r).max() < TOL

    def test_Fz(self):
        r = np.random.rand(self.M.nFz)
        P = self.M.getInterpolationMat(self.M.gridFz, 'Fz')
        assert np.abs(P[:,(self.M.nFx+self.M.nFy):]*r - r).max() < TOL

    def test_Ex(self):
        r = np.random.rand(self.M.nEx)
        P = self.M.getInterpolationMat(self.M.gridEx, 'Ex')
        assert np.abs(P[:,:self.M.nEx]*r - r).max() < TOL

    def test_Ey(self):
        r = np.random.rand(self.M.nEy)
        P = self.M.getInterpolationMat(self.M.gridEy, 'Ey')
        assert np.abs(P[:,self.M.nEx:(self.M.nEx+self.M.nEy)]*r - r).max() < TOL

    def test_Ez(self):
        r = np.random.rand(self.M.nEz)
        P = self.M.getInterpolationMat(self.M.gridEz, 'Ez')
        assert np.abs(P[:,(self.M.nEx+self.M.nEy):]*r - r).max() < TOL


if __name__ == '__main__':
    unittest.main()
