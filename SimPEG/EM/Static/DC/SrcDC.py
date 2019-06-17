import numpy as np

from .... import survey
from ....utils import Zero, closestPoints, mkvc


class BaseSrc(survey.BaseSrc):
    """
    Base DC source
    """

    current = 1.0
    loc = None
    _q = None

    def __init__(self, rxList, **kwargs):
        survey.BaseSrc.__init__(self, rxList, **kwargs)

    def eval(self, prob):
        raise NotImplementedError

    def evalDeriv(self, prob):
        return Zero()


class Dipole(BaseSrc):
    """
    Dipole source
    """

    def __init__(self, rxList, locA, locB, **kwargs):
        assert locA.shape == locB.shape, ('Shape of locA and locB should be '
                                          'the same')
        self.loc = [locA, locB]
        BaseSrc.__init__(self, rxList, **kwargs)

    def eval(self, prob):
        if self._q is not None:
            return self._q
        else:
            if prob._formulation == 'HJ':
                inds = closestPoints(prob.mesh, self.loc, gridLoc='CC')
                self._q = np.zeros(prob.mesh.nC)
                self._q[inds] = self.current * np.r_[1., -1.]
            elif prob._formulation == 'EB':
                qa = prob.mesh.getInterpolationMat(self.loc[0],
                                                   locType='N').toarray()
                qb = -prob.mesh.getInterpolationMat(self.loc[1],
                                                    locType='N').toarray()
                self._q = self.current * (qa+qb)
            return self._q


class Pole(BaseSrc):

    def __init__(self, rxList, loc, **kwargs):
        BaseSrc.__init__(self, rxList, loc=loc, **kwargs)

    def eval(self, prob):
        if self._q is not None:
            return self._q
        else:
            if prob._formulation == 'HJ':
                inds = closestPoints(prob.mesh, self.loc)
                self._q = np.zeros(prob.mesh.nC)
                self._q[inds] = self.current * np.r_[1.]
            elif prob._formulation == 'EB':
                q = prob.mesh.getInterpolationMat(
                    self.loc, locType='N'
                )
                self._q = self.current * q.toarray()
            return self._q
