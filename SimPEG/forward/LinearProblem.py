import numpy as np
from SimPEG.mesh import TensorMesh
from SimPEG.forward import Problem
from SimPEG.regularization import Regularization
from SimPEG.inverse import *
import matplotlib.pyplot as plt


class LinearProblem(Problem):
    """docstring for LinearProblem"""

    def dpred(self, m, u=None):
        return self.G.dot(m)

    def J(self, m, v, u=None):
        return G.dot(v)

    def Jt(self, m, v, u=None):
        return G.T.dot(v)

if __name__ == '__main__':
    N = 100
    h = np.ones(N)/N
    M = TensorMesh([h])

    nk = 20
    jk = np.linspace(1.,20.,nk)
    p = -0.25
    q = 0.25



    g = lambda k: np.exp(p*jk[k]*M.vectorCCx)*np.cos(2*np.pi*q*jk[k]*M.vectorCCx)

    G = np.empty((nk, M.nC))

    for i in range(nk):
        G[i,:] = g(i)



    plt.figure(1)
    for i in range(nk):
        plt.plot(G[i,:])


    m_true = np.zeros(M.nC)
    m_true[M.vectorCCx > 0.3] = 1.
    m_true[M.vectorCCx > 0.45] = -0.5
    m_true[M.vectorCCx > 0.6] = 0


    d_true = G.dot(m_true)
    noise = 0.1 * np.random.rand(d_true.size)

    d_obs = d_true + noise

    # plt.figure(3)
    # plt.plot(d_true,'-o')
    # plt.plot(d_obs,'r-o')





    prob = LinearProblem(M)
    prob.G = G
    prob.dobs = d_obs
    prob.std = np.ones_like(d_obs)*0.1

    reg = Regularization(M)

    opt = InexactGaussNewton(maxIter=20)

    inv = Inversion(prob,reg,opt,beta0=1e-4)

    m0 = np.zeros_like(m_true)

    mrec = inv.run(m0)


    plt.figure(2)

    plt.plot(M.vectorCCx, m_true, 'b-')
    plt.plot(M.vectorCCx, mrec, 'r-')



    plt.show()
