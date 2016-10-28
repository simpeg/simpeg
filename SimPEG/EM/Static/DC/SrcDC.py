from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import SimPEG
from SimPEG.Utils import Zero, closestPoints, mkvc
import numpy as np


class BaseSrc(SimPEG.Survey.BaseSrc):
    """
    Base DC source
    """

    current = 1.0
    loc = None

    def __init__(self, rxList, **kwargs):
        SimPEG.Survey.BaseSrc.__init__(self, rxList, **kwargs)

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
        if prob._formulation == 'HJ':
            inds = closestPoints(prob.mesh, self.loc, gridLoc='CC')
            q = np.zeros(prob.mesh.nC)
            q[inds] = self.current * np.r_[1., -1.]
        elif prob._formulation == 'EB':
            qa = prob.mesh.getInterpolationMat(self.loc[0],
                                               locType='N').todense()
            qb = -prob.mesh.getInterpolationMat(self.loc[1],
                                                locType='N').todense()
            q = self.current * mkvc(qa+qb)
        return q


class Pole(BaseSrc):

    def __init__(self, rxList, loc, **kwargs):
        BaseSrc.__init__(self, rxList, loc=loc, **kwargs)

    def eval(self, prob):
        if prob._formulation == 'HJ':
            inds = closestPoints(prob.mesh, self.loc)
            q = np.zeros(prob.mesh.nC)
            q[inds] = self.current * np.r_[1.]
        elif prob._formulation == 'EB':
            q = prob.mesh.getInterpolationMat(self.loc, locType='N').todense()
            q = self.current * mkvc(q)
        return q


# class Dipole_ky(BaseSrc):

#     def __init__(self, rxList, locA, locB, **kwargs):
#         assert locA.shape == locB.shape, 'Shape of locA and locB should be the same'
#         self.loc = [locA[[0,2]], locB[[0,2]]]
#         BaseSrc.__init__(self, rxList, **kwargs)

#     def eval(self, prob):
#         if prob._formulation == 'HJ':
#             inds = closestPoints(prob.mesh, self.loc, gridLoc='CC')
#             q = np.zeros(prob.mesh.nC)
#             q[inds] = self.current * np.r_[1., -1.]
#         elif prob._formulation == 'EB':
#             qa = prob.mesh.getInterpolationMat(self.loc[0], locType='N').todense()
#             qb = -prob.mesh.getInterpolationMat(self.loc[1], locType='N').todense()
#             q = self.current * mkvc(qa+qb)
#         return q

# class Pole_ky(BaseSrc):

#     def __init__(self, rxList, loc, **kwargs):
#         BaseSrc.__init__(self, rxList, loc=loc, **kwargs)

#     def eval(self, prob):
#         if prob._formulation == 'HJ':
#             inds = closestPoints(prob.mesh, self.loc[[0,2]])
#             q = np.zeros(prob.mesh.nC)
#             q[inds] = self.current * np.r_[1.]
#         elif prob._formulation == 'EB':
#             q = prob.mesh.getInterpolationMat(self.loc[[0,2]], locType='N').todense()
#             q = self.current * mkvc(q)
#         return q
