from SimPEG import Mesh
import numpy as np
import sympy
from sympy.abc import r, t, z
import unittest

TOL = 1e-1

class CylInnerProducts_Test(unittest.TestCase):

    def test_FaceInnerProduct(self):
        # Here we will make up some j vectors that vary in space
        # j = [j_r, j_z] - to test face inner products
        j = sympy.Matrix([
            r**2 * z,
            r * z**2
        ])

        # Create an isotropic sigma vector
        Sig = sympy.Matrix([
            [540/sympy.pi*(r*z)**2, 0       ],
            [     0  , 540/sympy.pi*(r*z)**4],
        ])

        # Do the inner product! - we are in cyl coordinates!
        jTSj = j.T*Sig*j
        ans  = sympy.integrate(
                               sympy.integrate(
                                               sympy.integrate(r * jTSj, (r,0,1)), # we are in cyl coordinates
                                               (t,0,2.*sympy.pi)),
                                (z,0,1))[0] # The `[0]` is to make it an int.

        def get_vectors(mesh):
            """ Get Vectors sig, sr. jx from sympy"""

            f_jr = sympy.lambdify((r,z), j[0], 'numpy')
            f_jz = sympy.lambdify((r,z), j[1], 'numpy')
            f_sigr = sympy.lambdify((r,z), Sig[0], 'numpy')
            f_sigz = sympy.lambdify((r,z), Sig[1], 'numpy')

            jr = f_jr(mesh.gridFx[:,0], mesh.gridFx[:,2])
            jz = f_jz(mesh.gridFz[:,0], mesh.gridFz[:,2])
            sigr = f_sigr(mesh.gridCC[:,0], mesh.gridCC[:,2])

            return sigr, np.r_[jr, jz]


        n = 100.
        mesh = Mesh.CylMesh([n, 1, n])

        sig, jv = get_vectors(mesh)
        MfSig = mesh.getFaceInnerProduct(sig)
        numeric_ans = jv.T.dot(MfSig.dot(jv))

        print('------ Testing Face Inner Product-----------')
        print(' Analytic: {analytic}, Numeric: {numeric}'.format(analytic=ans, numeric=numeric_ans))
        assert(np.abs(ans-numeric_ans) < TOL)

if __name__ == '__main__':
    unittest.main()
