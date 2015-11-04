from SimPEG.Mesh.PointerTree import Tree
import numpy as np
import unittest

TOL = 1e-10

class TestOcTreeObjects(unittest.TestCase):

    def test_counts(self):

        T = Tree([8,8])
        T._refineCell([0,0,0])
        T._refineCell([4,4,1])
        T._refineCell([0,0,1])
        T._refineCell([2,2,2])
        T.number()
        assert sorted(T._treeInds) == [2, 34, 66, 99, 107, 115, 123, 129, 257, 386, 418, 450, 482]
        assert len(T._hangingFacesX) == 7
        assert T.nFx == 18
        assert T.vol.sum() == 1.0


    def test_connectivity(self):
        T = Tree([8,8])
        T._refineCell([0,0,0])
        T._refineCell([4,4,1])
        T._refineCell([0,0,1])
        T._refineCell([2,2,2])
        T.number()
        assert T._getNextCell([4,0,1]) is None
        assert T._getNextCell([0,4,1]) == [T._index([4,4,2]), T._index([4,6,2])]
        assert T._getNextCell([0,2,2]) == [T._index([2,2,3]), T._index([2,3,3])]
        assert T._getNextCell([4,4,2]) == T._index([6,4,2])
        assert T._getNextCell([6,4,2]) is None
        assert T._getNextCell([2,0,2]) == T._index([4,0,1])
        assert T._getNextCell([4,0,1], positive=False) == [T._index([2,0,2]), [T._index([3,2,3]), T._index([3,3,3])]]
        assert T._getNextCell([3,3,3]) == T._index([4,0,1])
        assert T._getNextCell([3,2,3]) == T._index([4,0,1])
        assert T._getNextCell([2,2,3]) == T._index([3,2,3])
        assert T._getNextCell([3,2,3], positive=False) == T._index([2,2,3])


        assert T._getNextCell([0,0,2], direction=1) == T._index([0,2,2])
        assert T._getNextCell([0,2,2], direction=1, positive=False) == T._index([0,0,2])
        assert T._getNextCell([0,2,2], direction=1) == T._index([0,4,1])
        assert T._getNextCell([0,4,1], direction=1, positive=False) ==  [T._index([0,2,2]), [T._index([2,3,3]), T._index([3,3,3])]]




if __name__ == '__main__':
    unittest.main()
