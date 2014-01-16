from SimPEG import Mesh, Model, Problem, Data, Inverse, np
import matplotlib.pyplot as plt


class LinearProblem(Problem.BaseProblem):
    """docstring for LinearProblem"""

    def __init__(self, *args, **kwargs):
        problem.BaseProblem.__init__(self, *args, **kwargs)

    def dpred(self, m, u=None):
        return self.G.dot(m)

    def J(self, m, v, u=None):
        return self.G.dot(v)

    def Jt(self, m, v, u=None):
        return self.G.T.dot(v)


def example(N):
    h = np.ones(N)/N
    M = mesh.TensorMesh([h])

    nk = 20
    jk = np.linspace(1.,20.,nk)
    p = -0.25
    q = 0.25

    g = lambda k: np.exp(p*jk[k]*M.vectorCCx)*np.cos(2*np.pi*q*jk[k]*M.vectorCCx)

    G = np.empty((nk, M.nC))

    for i in range(nk):
        G[i,:] = g(i)

    mtrue = np.zeros(M.nC)
    mtrue[M.vectorCCx > 0.3] = 1.
    mtrue[M.vectorCCx > 0.45] = -0.5
    mtrue[M.vectorCCx > 0.6] = 0



    prob = LinearProblem(M, None)
    prob.G = G
    data = prob.createSyntheticData(mtrue, std=0.01)

    return prob, data


if __name__ == '__main__':

    prob, data = example(100)
    M = prob.mesh

    reg = inverse.Regularization(M)
    opt = inverse.InexactGaussNewton(maxIter=20)
    inv = inverse.Inversion(prob,reg,opt,data)
    m0 = np.zeros_like(data.mtrue)

    mrec = inv.run(m0)

    plt.figure(1)
    for i in range(prob.G.shape[0]):
        plt.plot(prob.G[i,:])

    plt.figure(2)

    plt.plot(M.vectorCCx, data.mtrue, 'b-')
    plt.plot(M.vectorCCx, mrec, 'r-')

    plt.show()
