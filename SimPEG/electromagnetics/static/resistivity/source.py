import numpy as np
import properties

from .... import survey
from ....utils import Zero, closestPoints, mkvc


class BaseSrc(survey.BaseSrc):
    """
    Base DC source
    """

    current = properties.Float(
        "amplitude of the current", default=1.
    )

    _q = None

    def __init__(self, rxList, **kwargs):
        super(BaseSrc, self).__init__(rxList, **kwargs)

    def eval(self, prob):
        raise NotImplementedError

    def evalDeriv(self, prob):
        return Zero()


class Dipole(BaseSrc):
    """
    Dipole source
    """

    location = properties.List(
        "location of the source electrodes",
        survey.SourceLocationArray("location of electrode")
    )

    def __init__(self, rxList, locA, locB, **kwargs):
        if locA.shape != locB.shape:
            raise Exception('Shape of locA and locB should be the same')
        self.location = [locA, locB]
        super(Dipole, self).__init__(rxList, **kwargs)

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

    def __init__(self, rxList, location, **kwargs):
        BaseSrc.__init__(self, rxList, location=location, **kwargs)

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
