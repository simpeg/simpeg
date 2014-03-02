import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
from Utils.matutils import mkvc, sdiag
import warnings

DEFAULTS = {'direct':'scipy', 'iter':'scipy', 'triangular':'fortran', 'diagonal':'python'}
OPTIONS = {'direct':['scipy'], 'iter':['scipy'], 'triangular':['python'], 'diagonal':['python']}

try:
    import Utils.TriSolve as TriSolve
    OPTIONS['triangular'].append('fortran')
except Exception, e:
    print 'Warning: Python backend is being used for solver. Run setup.py from the command line.'
    DEFAULTS['triangular'] = 'python'

try:
    import mumps
    OPTIONS['direct'].append('mumps')
except Exception, e:
    print 'Warning: mumps solver not available.'

class Solver(object):
    """
        Solver is a light wrapper on the various types of
        linear solvers available in python.

        :param scipy.sparse A: Matrix
        :param bool doDirect: if you want a direct solver
        :param string flag: Matrix type flag for special solves: [None, 'L', 'U', 'D']
        :param dict options: options which are passed to each sub solver, see each for details.
        :rtype: Solver
        :return: Solver

        To use for direct solvers::

            solve = Solver(A, doDirect=True, flag=None, options={'factorize':True,'backend':'scipy'})
            x = solve.solve(rhs)

        Or in one line::

            x = Solver(A).solve(rhs)

        The flag can be set to None, 'L', 'U', or 'D', for general, lower, upper, and diagonal matrices, respectively.

    """
    def __init__(self, A, doDirect=True, flag=None, options={}):
        assert type(doDirect) is bool, 'doDirect must be a boolean'
        assert flag in [None, 'L', 'U', 'D'], "flag must be set to None, 'L', 'U', or 'D'"
        assert type(options) is dict, 'options must be a dictionary object'
        self.A = A

        self.dsolve = None
        self.doDirect = doDirect
        self.flag = flag
        self.options = options
        if doDirect: return

        # Now deal with iterative stuff only
        if 'M' not in options:
            warnings.warn("You should provide a preconditioner, M.", UserWarning)
            return
        M = options['M']
        if isinstance(M, sp.linalg.LinearOperator):
            return
        PreconditionerList = ['J','GS']
        if type(M) is str:
            assert M in PreconditionerList, "M must be in the known preconditioner list. ['J','GS']"
            M = (M,A) # use A as the base for the preconditioner.
        if type(M) is tuple:
            assert type(M[0]) is str and M[0] in PreconditionerList, "M as a tuple must be (str, Matrix) where str is in ['J','GS']: e.g. ('J', WtW) where J stands for Jacobi, and WtW is a sparse matrix."
            if M[0] is 'J':
                Jacobi = sdiag(1.0/M[1].diagonal())
                options['M'] = Jacobi
            elif M[0] is 'GS':
                DD = sdiag(M[1].diagonal())
                Uinv = Solver(M[1], flag='U')
                Linv = Solver(M[1], flag='L')
                def GS(f):
                    return Uinv.solve(DD*Linv.solve(f))
                options['M'] = sp.linalg.LinearOperator( A.shape, GS, dtype=A.dtype )

        else:
            raise Exception('M must be a LinearOperator or a tuple')


    def solve(self, b):
        """
            Solves the linear system.

            .. math::

                Ax=b

            :param numpy.ndarray b: the right hand side
            :rtype: numpy.ndarray
            :return: x
        """
        if self.flag is None and self.doDirect:
            return self.solveDirect(b, **self.options)
        elif self.flag is None and not self.doDirect:
            return self.solveIter(b, **self.options)
        elif self.flag == 'U':
            return self.solveBackward(b, **self.options)
        elif self.flag == 'L':
            return self.solveForward(b, **self.options)
        elif self.flag == 'D':
            return self.solveDiagonal(b, **self.options)
        else:
            raise Exception('Unknown flag.')
        pass

    def clean(self):
        """Cleans up the memory"""
        if self.options.has_key('backend'):
            if self.options['backend'] == 'mumps':
                self.mctx.destroy()
        del self.dsolve
        self.dsolve = None

    def solveDirect(self, b, factorize=False, backend=None):
        """
            Use solve instead of this interface.

            :param numpy.ndarray b: the right hand side
            :param bool factorize: if you want to factorize and store factors
            :param str backend: which backend to use. Default is scipy
            :rtype: numpy.ndarray
            :return: x
        """
        if backend is None: backend = DEFAULTS['direct']

        assert np.shape(self.A)[1] == np.shape(b)[0], 'Dimension mismatch'

        if backend == 'scipy':
            X = self.solveDirect_scipy(b, factorize)
        elif backend == 'mumps':
            X = self.solveDirect_mumps(b, factorize)

        return X

    def solveDirect_scipy(self, b, factorize):
        """
            Use solve instead of this interface.

            :param numpy.ndarray b: the right hand side
            :param bool factorize: if you want to factorize and store factors
            :rtype: numpy.ndarray
            :return: x
        """
        if factorize and self.dsolve is None:
            self.A = self.A.tocsc()  # for efficiency
            self.dsolve = linalg.factorized(self.A)

        if len(b.shape) == 1 or b.shape[1] == 1:
            # Just one RHS
            if factorize:
                return self.dsolve(b.flatten())
            else:
                return linalg.dsolve.spsolve(self.A, b)

        # Multiple RHSs
        X = np.empty_like(b)
        for i in range(b.shape[1]):
            if factorize:
                X[:,i] = self.dsolve(b[:,i])
            else:
                X[:,i] = linalg.dsolve.spsolve(self.A,b[:,i])

        return X

    def solveDirect_mumps(self, b, factorize):
        """
            Use solve instead of this interface.

            :param numpy.ndarray b: the right hand side
            :param bool factorize: if you want to factorize and store factors
            :rtype: numpy.ndarray
            :return: x
        """
        if factorize and self.dsolve is None:
            self.mctx = mumps.DMumpsContext()
            self.mctx.set_icntl(14, 60)
            # self.mctx.set_silent()
            self.mctx.set_centralized_sparse(self.A)
            self.mctx.run(job=4)

            def mdsolve(rhs):
                x = rhs.copy()
                self.mctx.set_rhs(x)
                self.mctx.run(job=3)
                return x

            self.dsolve = mdsolve

        if len(b.shape) == 1 or b.shape[1] == 1:
            # Just one RHS
            if factorize:
                X = self.dsolve(b)
            else:
                X = mumps.spsolve(self.A, b)

        else:
            # Multiple RHSs
            X = np.empty_like(b)
            for i in range(b.shape[1]):
                if factorize:
                    X[:,i] = self.dsolve(b[:,i])
                else:
                    X[:,i] = mumps.spsolve(self.A,b[:,i])

        return X

    def solveIter(self, b, backend=None, M=None, iterSolver='CG', tol=1e-6, maxIter=50):
        if backend is None: backend = DEFAULTS['iter']

        algorithms = {'CG':sp.linalg.cg}
        assert iterSolver in algorithms, "iterSolver must be 'CG', or implement it yourself and add it here!"
        alg = algorithms[iterSolver]

        if len(b.shape) == 1 or b.shape[1] == 1:
            x, self.info = alg(self.A, b, M=M, tol=tol, maxiter=maxIter)
        else:
            x = np.empty_like(b)
            for i in range(b.shape[1]):
                x[:,i], self.info = alg(self.A, b[:,i], M=M, tol=tol, maxiter=maxIter)
        return x

    def solveBackward(self, b, backend=None):
        """
            Use solve instead of this interface.

            Perform a backwards solve with upper triangular A in CSR format (best, if not, it will be converted).

            :param str backend: which backend to use. Default is python.
            :rtype: numpy.ndarray
            :return: x
        """
        if backend is None: backend = DEFAULTS['triangular']
        if backend not in OPTIONS['triangular']:
            print 'Warning: %s-backend not being used, %s-default will be used instead.'%(backend,DEFAULTS['triangular'])
            backend = DEFAULTS['triangular']
        if type(self.A) is not sp.csr.csr_matrix:
            self.A = sp.csr_matrix(self.A)
        vals = self.A.data
        rowptr = self.A.indptr
        colind = self.A.indices
        if backend == 'fortran':
            if len(b.shape) == 1 or b.shape[1] == 1:
                x = TriSolve.backward(vals, rowptr, colind, b, self.A.data.size, b.size, 1)
                x = mkvc(x)
            else:
                x = TriSolve.backward(vals, rowptr, colind, b, self.A.data.size, b.shape[0], b.shape[1])
        elif backend == 'python':
            x = np.empty_like(b)   # empty() is faster than zeros().
            for i in reversed(xrange(self.A.shape[0])):
                ith_row = vals[rowptr[i] : rowptr[i+1]]
                cols = colind[rowptr[i] : rowptr[i+1]]
                x_vals = x[cols]
                x[i] = (b[i] - np.dot(ith_row[1:], x_vals[1:])) / ith_row[0]
        return x

    def solveForward(self, b, backend=None):
        """
            Use solve instead of this interface.

            Perform a forward solve with lower triangular A in CSR format (best, if not, it will be converted).

            :param str backend: which backend to use. Default is python.
            :rtype: numpy.ndarray
            :return: x
        """
        if backend is None: backend = DEFAULTS['triangular']
        if backend not in OPTIONS['triangular']:
            print 'Warning: %s-backend not being used, %s-default will be used instead.'%(backend,DEFAULTS['triangular'])
            backend = DEFAULTS['triangular']
        if type(self.A) is not sp.csr.csr_matrix:
            from scipy.sparse import csr_matrix
            self.A = csr_matrix(self.A)
        vals = self.A.data
        rowptr = self.A.indptr
        colind = self.A.indices
        if backend == 'fortran':
            if len(b.shape) == 1 or b.shape[1] == 1:
                x = TriSolve.forward(vals, rowptr, colind, b, self.A.data.size, b.size, 1)
                x = mkvc(x)
            else:
                x = TriSolve.forward(vals, rowptr, colind, b, self.A.data.size, b.shape[0], b.shape[1])
        elif backend == 'python':
            x = np.empty_like(b)   # empty() is faster than zeros().
            for i in xrange(self.A.shape[0]):
                ith_row = vals[rowptr[i] : rowptr[i+1]]
                cols = colind[rowptr[i] : rowptr[i+1]]
                x_vals = x[cols]
                x[i] = (b[i] - np.dot(ith_row[:-1], x_vals[:-1])) / ith_row[-1]
        return x

    def solveDiagonal(self, b, backend=None):
        """
            Use solve instead of this interface.

            Perform a diagonal solve with diagonal matrix A.

            :param str backend: which backend to use. Default is python.
            :rtype: numpy.ndarray
            :return: x
        """
        if backend is None: backend = DEFAULTS['diagonal']

        diagA = self.A.diagonal()
        if len(b.shape) == 1 or b.shape[1] == 1:
            # Just one RHS
            return b/diagA
        # Multiple RHSs
        X = np.empty_like(b)
        for i in range(b.shape[1]):
            X[:,i] = b[:,i]/diagA
        return X


if __name__ == '__main__':
    from SimPEG.Mesh import TensorMesh
    from time import time
    h1 = np.ones(20)*100.
    h2 = np.ones(20)*100.
    h3 = np.ones(20)*100.

    h = [h1,h2,h3]

    M = TensorMesh(h)

    D = M.faceDiv
    G = M.cellGrad
    Msig = M.getFaceMass()
    A = D*Msig*G
    A[0,0] *= 10 # remove the constant null space from the matrix

    e = np.ones(M.nC)
    rhs = A.dot(e)

    tic = time()
    solve = Solver(A, options={'factorize':True})
    x = solve.solve(rhs)
    print 'Factorized', time() - tic
    print np.linalg.norm(e-x,np.inf)
    tic = time()
    solve = Solver(A, options={'factorize':False})
    x = solve.solve(rhs)
    print 'spsolve', time() - tic
    print np.linalg.norm(e-x,np.inf)


    n = 600
    A_dense = np.random.random((n,n))
    L = np.tril(np.dot(A_dense, A_dense))  # Positive definite is better conditioned.
    e = np.ones(n)
    b = np.dot(L, e)

    A = sp.csr_matrix(L)
    pSolve = Solver(A,flag='L',options={'backend':'python'});
    fSolve = Solver(A,flag='L',options={'backend':'fortran'})
    tic = time()
    x = pSolve.solve(b)
    toc = time() - tic
    print 'Error Forward Python  = ', np.linalg.norm(x-e, np.inf), 'Time: ', toc
    tic = time()
    x = fSolve.solve(b)
    toc = time() - tic
    print 'Error Forward Fortran = ', np.linalg.norm(x-e, np.inf), 'Time: ', toc



    A = -D*D.T
    A[0,0] *= 10 # remove the constant null space from the matrix
    e = np.ones(M.nC)
    b = A.dot(e)

    iSolve = Solver(A, doDirect=False,options={'M':('GS',A)})
    tic = time()
    x = iSolve.solve(b)
    toc = time() - tic
    print x
    print 'Error CG  = ', np.linalg.norm(x-e, np.inf), 'Time: ', toc, 'Info: ', iSolve.info
