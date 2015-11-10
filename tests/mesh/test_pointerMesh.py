from SimPEG import Mesh
from SimPEG.Mesh.PointerTree import Tree
import numpy as np
import matplotlib.pyplot as plt
import unittest

TOL = 1e-10



class TestSimpleQuadTree(unittest.TestCase):

    def test_counts(self):
        nc = 8
        h1 = np.random.rand(nc)*nc*0.5 + nc*0.5
        h2 = np.random.rand(nc)*nc*0.5 + nc*0.5
        h = [hi/np.sum(hi) for hi in [h1, h2]]  # normalize
        M = Tree(h)
        M._refineCell([0,0,0])
        M._refineCell([0,0,1])
        M.number()
        # M.plotGrid(showIt=True)
        assert M.nhFx == 2
        assert M.nFx == 9

        assert np.allclose(M.vol.sum(), 1.0)

        assert np.allclose(np.r_[M._areaFxFull, M._areaFyFull], M._deflationMatrix('F') * M.area)


    # def test_connectivity(self):
    #     T = Tree([8,8])
    #     T._refineCell([0,0,0])
    #     T._refineCell([4,4,1])
    #     T._refineCell([0,0,1])
    #     T._refineCell([2,2,2])
    #     T.number()
    #     assert T._getNextCell([4,0,1]) is None
    #     assert T._getNextCell([0,4,1]) == [T._index([4,4,2]), T._index([4,6,2])]
    #     assert T._getNextCell([0,2,2]) == [T._index([2,2,3]), T._index([2,3,3])]
    #     assert T._getNextCell([4,4,2]) == T._index([6,4,2])
    #     assert T._getNextCell([6,4,2]) is None
    #     assert T._getNextCell([2,0,2]) == T._index([4,0,1])
    #     assert T._getNextCell([4,0,1], positive=False) == [T._index([2,0,2]), [T._index([3,2,3]), T._index([3,3,3])]]
    #     assert T._getNextCell([3,3,3]) == T._index([4,0,1])
    #     assert T._getNextCell([3,2,3]) == T._index([4,0,1])
    #     assert T._getNextCell([2,2,3]) == T._index([3,2,3])
    #     assert T._getNextCell([3,2,3], positive=False) == T._index([2,2,3])


    #     assert T._getNextCell([0,0,2], direction=1) == T._index([0,2,2])
    #     assert T._getNextCell([0,2,2], direction=1, positive=False) == T._index([0,0,2])
    #     assert T._getNextCell([0,2,2], direction=1) == T._index([0,4,1])
    #     assert T._getNextCell([0,4,1], direction=1, positive=False) ==  [T._index([0,2,2]), [T._index([2,3,3]), T._index([3,3,3])]]

    def test_faceDiv(self):

        hx, hy = np.r_[1.,2,3,4], np.r_[5.,6,7,8]
        T = Tree([hx, hy], levels=2)
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
        M = Tree(h, levels=3)
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
        M = Tree([hx, hy, hz], levels=2)
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
        M = Tree([hx, hy, hz], levels=2)
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

        M = Tree([hx, hy, hz], levels=2)
        M.refine(lambda xc:2)
        # M.plotGrid(showIt=True)
        Mr = Mesh.TensorMesh([hx, hy, hz])

        # plt.subplot(211).spy(Mr.getFaceInnerProduct())
        # plt.subplot(212).spy(M.getFaceInnerProduct())
        # plt.show()

        # print M.nC, M.nF, M.getFaceInnerProduct().shape, M.permuteF.shape

        assert np.allclose(Mr.getFaceInnerProduct().todense(), (M.permuteF * M.getFaceInnerProduct() * M.permuteF.T).todense())
        assert np.allclose(Mr.getEdgeInnerProduct().todense(), (M.permuteE * M.getEdgeInnerProduct() * M.permuteE.T).todense())

    def test_VectorIdenties(self):
        hx, hy, hz = [[(1,4)], [(1,4)], [(1,4)]]

        M = Tree([hx, hy, hz], levels=2)
        Mr = Mesh.TensorMesh([hx, hy, hz])

        assert (M.faceDiv * M.edgeCurl).nnz == 0
        assert (Mr.faceDiv * Mr.edgeCurl).nnz == 0

        hx, hy, hz = np.r_[1.,2,3,4], np.r_[5.,6,7,8], np.r_[9.,10,11,12]

        M = Tree([hx, hy, hz], levels=2)
        Mr = Mesh.TensorMesh([hx, hy, hz])

        assert np.max(np.abs((M.faceDiv * M.edgeCurl).todense().flatten())) < TOL
        assert np.max(np.abs((Mr.faceDiv * Mr.edgeCurl).todense().flatten())) < TOL



if __name__ == '__main__':
    unittest.main()
