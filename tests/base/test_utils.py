from __future__ import print_function
import unittest
import numpy as np
import scipy.sparse as sp
import os
import shutil
from SimPEG.Utils import (
    sdiag, sub2ind, ndgrid, mkvc, inv2X2BlockDiagonal,
    inv3X3BlockDiagonal, invPropertyTensor, makePropertyTensor, indexCube,
    ind2sub, asArray_N_x_Dim, TensorType, diagEst, count, timeIt, Counter,
    download, surface2ind_topo
)
from SimPEG import Mesh
from SimPEG.Tests import checkDerivative


TOL = 1e-8


class TestCheckDerivative(unittest.TestCase):

    def test_simplePass(self):
        def simplePass(x):
            return np.sin(x), sdiag(np.cos(x))
        passed = checkDerivative(simplePass, np.random.randn(5), plotIt=False)
        self.assertTrue(passed, True)

    def test_simpleFunction(self):
        def simpleFunction(x):
            return np.sin(x), lambda xi: sdiag(np.cos(x))*xi
        passed = checkDerivative(
            simpleFunction, np.random.randn(5), plotIt=False
        )
        self.assertTrue(passed, True)

    def test_simpleFail(self):
        def simpleFail(x):
            return np.sin(x), -sdiag(np.cos(x))
        passed = checkDerivative(simpleFail, np.random.randn(5), plotIt=False)
        self.assertTrue(not passed, True)


class TestCounter(unittest.TestCase):
    def test_simpleFail(self):
        class MyClass(object):
            def __init__(self, url):
                self.counter = Counter()

            @count
            def MyMethod(self):
                pass

            @timeIt
            def MySecondMethod(self):
                pass

        c = MyClass('blah')
        for i in range(100):
            c.MyMethod()
        for i in range(300):
            c.MySecondMethod()
        c.counter.summary()
        self.assertTrue(True)


class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.a = np.array([1, 2, 3])
        self.b = np.array([1, 2])
        self.c = np.array([1, 2, 3, 4])

    def test_mkvc1(self):
        x = mkvc(self.a)
        self.assertTrue(x.shape, (3,))

    def test_mkvc2(self):
        x = mkvc(self.a, 2)
        self.assertTrue(x.shape, (3, 1))

    def test_mkvc3(self):
        x = mkvc(self.a, 3)
        self.assertTrue(x.shape, (3, 1, 1))

    def test_ndgrid_2D(self):
        XY = ndgrid([self.a, self.b])

        X1_test = np.array([1, 2, 3, 1, 2, 3])
        X2_test = np.array([1, 1, 1, 2, 2, 2])

        self.assertTrue(np.all(XY[:, 0] == X1_test))
        self.assertTrue(np.all(XY[:, 1] == X2_test))

    def test_ndgrid_3D(self):
        XYZ = ndgrid([self.a, self.b, self.c])

        X1_test = np.array([
            1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1,
            2, 3
        ])
        X2_test = np.array([
            1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2,
            2, 2
        ])
        X3_test = np.array([
            1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4,
            4, 4
        ])

        self.assertTrue(np.all(XYZ[:, 0] == X1_test))
        self.assertTrue(np.all(XYZ[:, 1] == X2_test))
        self.assertTrue(np.all(XYZ[:, 2] == X3_test))

    def test_sub2ind(self):
        x = np.ones((5, 2))
        self.assertTrue(np.all(sub2ind(x.shape, [0, 0]) == [0]))
        self.assertTrue(np.all(sub2ind(x.shape, [4, 0]) == [4]))
        self.assertTrue(np.all(sub2ind(x.shape, [0, 1]) == [5]))
        self.assertTrue(np.all(sub2ind(x.shape, [4, 1]) == [9]))
        self.assertTrue(np.all(sub2ind(x.shape, [[4, 1]]) == [9]))
        self.assertTrue(
            np.all(sub2ind(
                x.shape, [[0, 0], [4, 0], [0, 1], [4, 1]]) == [0, 4, 5, 9]
            )
        )

    def test_ind2sub(self):
        x = np.ones((5, 2))
        self.assertTrue(
            np.all(ind2sub(x.shape, [0, 4, 5, 9])[0] == [0, 4, 0, 4])
        )
        self.assertTrue(
            np.all(ind2sub(x.shape, [0, 4, 5, 9])[1] == [0, 0, 1, 1])
        )

    def test_indexCube_2D(self):
        nN = np.array([3, 3])
        self.assertTrue(np.all(indexCube('A', nN) == np.array([0, 1, 3, 4])))
        self.assertTrue(np.all(indexCube('B', nN) == np.array([3, 4, 6, 7])))
        self.assertTrue(np.all(indexCube('C', nN) == np.array([4, 5, 7, 8])))
        self.assertTrue(np.all(indexCube('D', nN) == np.array([1, 2, 4, 5])))

    def test_indexCube_3D(self):
        nN = np.array([3, 3, 3])
        self.assertTrue(np.all(
            indexCube('A', nN) == np.array([0, 1, 3, 4, 9, 10, 12, 13])
        ))
        self.assertTrue(np.all(
            indexCube('B', nN) == np.array([3, 4, 6, 7, 12, 13, 15, 16])
        ))
        self.assertTrue(np.all(
            indexCube('C', nN) == np.array([4, 5, 7, 8, 13, 14, 16, 17])
        ))
        self.assertTrue(np.all(
            indexCube('D', nN) == np.array([1, 2, 4, 5, 10, 11, 13, 14])
        ))
        self.assertTrue(np.all(
            indexCube('E', nN) == np.array([9, 10, 12, 13, 18, 19, 21, 22])
        ))
        self.assertTrue(np.all(
            indexCube('F', nN) == np.array([12, 13, 15, 16, 21, 22, 24, 25])
        ))
        self.assertTrue(np.all(
            indexCube('G', nN) == np.array([13, 14, 16, 17, 22, 23, 25, 26])
        ))
        self.assertTrue(np.all(
            indexCube('H', nN) == np.array([10, 11, 13, 14, 19, 20, 22, 23])
        ))

    def test_invXXXBlockDiagonal(self):
        a = [np.random.rand(5, 1) for i in range(4)]

        B = inv2X2BlockDiagonal(*a)

        A = sp.vstack((sp.hstack((sdiag(a[0]), sdiag(a[1]))),
                       sp.hstack((sdiag(a[2]), sdiag(a[3])))))

        Z2 = B*A - sp.identity(10)
        self.assertTrue(np.linalg.norm(Z2.todense().ravel(), 2) < TOL)

        a = [np.random.rand(5, 1) for i in range(9)]
        B = inv3X3BlockDiagonal(*a)

        A = sp.vstack((sp.hstack((sdiag(a[0]), sdiag(a[1]),  sdiag(a[2]))),
                       sp.hstack((sdiag(a[3]), sdiag(a[4]),  sdiag(a[5]))),
                       sp.hstack((sdiag(a[6]), sdiag(a[7]),  sdiag(a[8])))))

        Z3 = B*A - sp.identity(15)

        self.assertTrue(np.linalg.norm(Z3.todense().ravel(), 2) < TOL)

    def test_invPropertyTensor2D(self):
        M = Mesh.TensorMesh([6, 6])
        a1 = np.random.rand(M.nC)
        a2 = np.random.rand(M.nC)
        a3 = np.random.rand(M.nC)
        prop1 = a1
        prop2 = np.c_[a1, a2]
        prop3 = np.c_[a1, a2, a3]

        for prop in [4, prop1, prop2, prop3]:
            b = invPropertyTensor(M, prop)
            A = makePropertyTensor(M, prop)
            B1 = makePropertyTensor(M, b)
            B2 = invPropertyTensor(M, prop, returnMatrix=True)

            Z = B1*A - sp.identity(M.nC*2)
            self.assertTrue(np.linalg.norm(Z.todense().ravel(), 2) < TOL)
            Z = B2*A - sp.identity(M.nC*2)
            self.assertTrue(np.linalg.norm(Z.todense().ravel(), 2) < TOL)

    def test_TensorType2D(self):
        M = Mesh.TensorMesh([6, 6])
        a1 = np.random.rand(M.nC)
        a2 = np.random.rand(M.nC)
        a3 = np.random.rand(M.nC)
        prop1 = a1
        prop2 = np.c_[a1, a2]
        prop3 = np.c_[a1, a2, a3]

        for ii, prop in enumerate([4, prop1, prop2, prop3]):
            self.assertTrue(TensorType(M, prop) == ii)

        self.assertRaises(Exception, TensorType, M, np.c_[a1, a2, a3, a3])
        self.assertTrue(TensorType(M, None) == -1)

    def test_TensorType3D(self):
        M = Mesh.TensorMesh([6, 6, 7])
        a1 = np.random.rand(M.nC)
        a2 = np.random.rand(M.nC)
        a3 = np.random.rand(M.nC)
        a4 = np.random.rand(M.nC)
        a5 = np.random.rand(M.nC)
        a6 = np.random.rand(M.nC)
        prop1 = a1
        prop2 = np.c_[a1, a2, a3]
        prop3 = np.c_[a1, a2, a3, a4, a5, a6]

        for ii, prop in enumerate([4, prop1, prop2, prop3]):
            self.assertTrue(TensorType(M, prop) == ii)

        self.assertRaises(Exception, TensorType, M, np.c_[a1, a2, a3, a3])
        self.assertTrue(TensorType(M, None) == -1)

    def test_invPropertyTensor3D(self):
        M = Mesh.TensorMesh([6, 6, 6])
        a1 = np.random.rand(M.nC)
        a2 = np.random.rand(M.nC)
        a3 = np.random.rand(M.nC)
        a4 = np.random.rand(M.nC)
        a5 = np.random.rand(M.nC)
        a6 = np.random.rand(M.nC)
        prop1 = a1
        prop2 = np.c_[a1, a2, a3]
        prop3 = np.c_[a1, a2, a3, a4, a5, a6]

        for prop in [4, prop1, prop2, prop3]:
            b = invPropertyTensor(M, prop)
            A = makePropertyTensor(M, prop)
            B1 = makePropertyTensor(M, b)
            B2 = invPropertyTensor(M, prop, returnMatrix=True)

            Z = B1*A - sp.identity(M.nC*3)
            self.assertTrue(np.linalg.norm(Z.todense().ravel(), 2) < TOL)
            Z = B2*A - sp.identity(M.nC*3)
            self.assertTrue(np.linalg.norm(Z.todense().ravel(), 2) < TOL)

    def test_asArray_N_x_Dim(self):

        true = np.array([[1, 2, 3]])

        listArray = asArray_N_x_Dim([1, 2, 3], 3)
        self.assertTrue(np.all(true == listArray))
        self.assertTrue(true.shape == listArray.shape)

        listArray = asArray_N_x_Dim(np.r_[1, 2, 3], 3)
        self.assertTrue(np.all(true == listArray))
        self.assertTrue(true.shape == listArray.shape)

        listArray = asArray_N_x_Dim(np.array([[1, 2, 3.]]), 3)
        self.assertTrue(np.all(true == listArray))
        self.assertTrue(true.shape == listArray.shape)

        true = np.array([[1, 2], [4, 5]])

        listArray = asArray_N_x_Dim([[1, 2], [4, 5]], 2)
        self.assertTrue(np.all(true == listArray))
        self.assertTrue(true.shape == listArray.shape)

    def test_surface2ind_topo(self):
        file_url = "https://storage.googleapis.com/simpeg/tests/utils/vancouver_topo.xyz"
        file2load = download(file_url)
        vancouver_topo = np.loadtxt(file2load)
        mesh_topo = Mesh.TensorMesh([
            [(500., 24)],
            [(500., 20)],
            [(10., 30)]
            ],
            x0='CCC')

        indtopoCC = surface2ind_topo(mesh_topo, vancouver_topo, gridLoc='CC', method='nearest')
        indtopoN = surface2ind_topo(mesh_topo, vancouver_topo, gridLoc='N', method='nearest')

        assert len(np.where(indtopoCC)[0]) == 8729
        assert len(np.where(indtopoN)[0]) == 8212


class TestDiagEst(unittest.TestCase):

    def setUp(self):
        self.n = 1000
        self.A = np.random.rand(self.n, self.n)
        self.Adiag = np.diagonal(self.A)

    def getTest(self, testType):
        Adiagtest = diagEst(self.A, self.n, self.n, testType)
        r = np.abs(Adiagtest-self.Adiag)
        err = r.dot(r)
        return err

    def testProbing(self):
        err = self.getTest('probing')
        print('Testing probing. {}'.format(err))
        self.assertTrue(err < TOL)


class TestDownload(unittest.TestCase):
    def test_downloads(self):
        url = "https://storage.googleapis.com/simpeg/Chile_GRAV_4_Miller/"
        cloudfiles = [
            'LdM_grav_obs.grv', 'LdM_mesh.mesh',
            'LdM_topo.topo', 'LdM_input_file.inp'
        ]

        url1 = url + cloudfiles[0]
        url2 = url + cloudfiles[1]

        file_names = download(
            [url1, url2], folder='./test_urls', overwrite=True
        )
        # or
        file_name = download(url1, folder='./test_url', overwrite=True)
        # where
        assert isinstance(file_names, list)
        assert len(file_names) == 2
        assert isinstance(file_name, str)

        # clean up
        shutil.rmtree(os.path.expanduser('./test_urls'))
        shutil.rmtree(os.path.expanduser('./test_url'))

if __name__ == '__main__':
    unittest.main()
