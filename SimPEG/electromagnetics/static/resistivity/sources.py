import numpy as np
import properties

from .... import survey
from ....utils import Zero, closestPoints
from ....utils.code_utils import deprecate_property


class BaseSrc(survey.BaseSrc):
    """
    Base DC source
    """

    current = properties.Float("amplitude of the source current", default=1.)

    _q = None

    def __init__(self, receiver_list, **kwargs):
        super(BaseSrc, self).__init__(receiver_list, **kwargs)

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
    loc = deprecate_property(location, 'loc', removal_version='0.15.0')

    def __init__(self, receiver_list, locationA, locationB, **kwargs):
        if locationA.shape != locationB.shape:
            raise Exception('Shape of locationA and locationB should be the same')
        super(Dipole, self).__init__(receiver_list, **kwargs)
        self.location = [locationA, locationB]

    def eval(self, prob):
        if self._q is not None:
            return self._q
        else:
            if prob._formulation == 'HJ':
                inds = closestPoints(prob.mesh, self.location, gridLoc='CC')
                self._q = np.zeros(prob.mesh.nC)
                self._q[inds] = self.current * np.r_[1., -1.]
            elif prob._formulation == 'EB':
                qa = prob.mesh.getInterpolationMat(
                    self.location[0], locType='N'
                ).toarray()
                qb = -prob.mesh.getInterpolationMat(
                    self.location[1], locType='N'
                ).toarray()
                self._q = self.current * (qa+qb)
            return self._q


class Pole(BaseSrc):

    def __init__(self, receiver_list, location, **kwargs):
        super(Pole, self).__init__(receiver_list, location=location, **kwargs)

    def eval(self, prob):
        if self._q is not None:
            return self._q
        else:
            if prob._formulation == 'HJ':
                inds = closestPoints(prob.mesh, self.location)
                self._q = np.zeros(prob.mesh.nC)
                self._q[inds] = self.current * np.r_[1.]
            elif prob._formulation == 'EB':
                q = prob.mesh.getInterpolationMat(
                    self.location, locType='N'
                )
                self._q = self.current * q.toarray()
            return self._q
