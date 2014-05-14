from SimPEG import *

class LinearSurvey(Survey.BaseSurvey):
    def projectFields(self, u):
        return u

class LinearProblem(Problem.BaseProblem):
    """docstring for LinearProblem"""

    surveyPair = LinearSurvey

    def __init__(self, model, G, **kwargs):
        Problem.BaseProblem.__init__(self, model, **kwargs)
        self.G = G

    def fields(self, m, u=None):
        return self.G.dot(m)

    def Jvec(self, m, v, u=None):
        return self.G.dot(v)

    def Jtvec(self, m, v, u=None):
        return self.G.T.dot(v)


def example(N):
    mesh = Mesh.TensorMesh([N])

    nk = 20
    jk = np.linspace(1.,20.,nk)
    p = -0.25
    q = 0.25

    g = lambda k: np.exp(p*jk[k]*mesh.vectorCCx)*np.cos(2*np.pi*q*jk[k]*mesh.vectorCCx)

    G = np.empty((nk, mesh.nC))

    for i in range(nk):
        G[i,:] = g(i)

    mtrue = np.zeros(mesh.nC)
    mtrue[mesh.vectorCCx > 0.3] = 1.
    mtrue[mesh.vectorCCx > 0.45] = -0.5
    mtrue[mesh.vectorCCx > 0.6] = 0

    prob = LinearProblem(mesh, G)
    survey = prob.createSyntheticSurvey(mtrue, std=0.01)

    return prob, survey, mesh


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    prob, survey, mesh = example(100)
    M = prob.mesh

    reg = Regularization.Tikhonov(mesh)
    objFunc = ObjFunction.BaseObjFunction(survey, reg)
    opt = Optimization.InexactGaussNewton(maxIter=20)
    inv = Inversion.BaseInversion(objFunc, opt)
    beta = Rules.BetaSchedule()
    betaest = Rules.BetaEstimate_ByEig()
    inv.ruleList = Rules.RuleList(betaest, beta)
    m0 = np.zeros_like(survey.mtrue)

    mrec = inv.run(m0)

    plt.figure(1)
    for i in range(prob.G.shape[0]):
        plt.plot(prob.G[i,:])

    plt.figure(2)
    plt.plot(M.vectorCCx, survey.mtrue, 'b-')
    plt.plot(M.vectorCCx, mrec, 'r-')

    plt.show()
