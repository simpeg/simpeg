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
    M = Mesh.TensorMesh([N])

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



    model = Model.BaseModel(M)
    prob = LinearProblem(model, G)
    survey = prob.createSyntheticSurvey(mtrue, std=0.01)

    return prob, survey, model


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    prob, survey, model = example(100)
    M = prob.mesh

    reg = Regularization.Tikhonov(model)
    beta = Parameters.BetaSchedule()
    objFunc = ObjFunction.BaseObjFunction(survey, reg, beta=beta)
    opt = Optimization.InexactGaussNewton(maxIter=20)
    inv = Inversion.BaseInversion(objFunc, opt)
    m0 = np.zeros_like(survey.mtrue)

    mrec = inv.run(m0)

    plt.figure(1)
    for i in range(prob.G.shape[0]):
        plt.plot(prob.G[i,:])

    plt.figure(2)

    plt.plot(M.vectorCCx, survey.mtrue, 'b-')
    plt.plot(M.vectorCCx, mrec, 'r-')

    plt.show()
