import SimPEG
# from SimPEG.EM.Base import BaseEMSurvey
from SimPEG.Utils import Zero, closestPoints

class BaseSrc(SimPEG.Survey.BaseSrc):

    current = 1
    loc = None

    def __init__(self, rxList, **kwargs):
        SimPEG.Survey.BaseSrc.__init__(self, rxList, **kwargs)

    def eval(self, prob):
        raise NotImplementedError

    def evalDeriv(self, prob):
        Zero()


class Dipole(BaseSrc):

    def __init__(self, rxList, locA, locB, **kwargs):
        assert locA.shape == locB.shape, 'Shape of locA and locB should be the same'
        self.loc = [locA, locB]
        BaseSrc.__init__(self, rxList, **kwargs)

    def eval(self, prob):
        if prob._formulation == 'HJ':
            inds = closestPoints(prob.mesh, self.loc)
            q = np.zeros(prob.mesh.nC)
            q[inds] = self.current * np.r_[1., -1.]
        elif prob._formulation == 'EB':
            # TODO: there is probably a faster way to do this
            # Utils.cellNodes , Utils.cellFaces, Utils.cellEdges
            raise NotImplementedError
        return q

    # def bc_contribution


# How to treat boundary conditions here

class Pole(BaseSrc):

    def __init__(self, rxList, loc, **kwargs):
        BaseSrc.__init__(self, rxList, loc=loc, **kwargs)

    def eval(self, prob):
        if prob._formulation == 'HJ':
            inds = closestPoints(prob.mesh, self.loc)
            q = np.zeros(prob.mesh.nC)
            q[inds] = self.current * np.r_[1.]
        elif prob._formulation == 'EB':
            # TODO: there is probably a faster way to do this
            # Utils.cellNodes , Utils.cellFaces, Utils.cellEdges
            raise NotImplementedError
        return q

    # def bc_contribution

