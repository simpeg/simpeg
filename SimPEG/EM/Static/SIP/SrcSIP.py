from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import SimPEG
from SimPEG.Utils import Zero, closestPoints, mkvc
import numpy as np


class BaseSrc(SimPEG.Survey.BaseSrc):

    current = 1.0
    loc = None

    def __init__(self, rxList, **kwargs):
        SimPEG.Survey.BaseSrc.__init__(self, rxList, **kwargs)

    def eval(self, prob):
        raise NotImplementedError

    def evalDeriv(self, prob):
        return Zero()

    @property
    def nD(self):
        """Number of data"""
        return self.vnD.sum()

    @property
    def vnD(self):
        """Vector number of data"""
        return np.array([rx.nD*len(rx.times) for rx in self.rxList])


class Dipole(BaseSrc):

    def __init__(self, rxList, locA, locB, **kwargs):
        assert locA.shape == locB.shape, 'Shape of locA and locB should be the same'
        self.loc = [locA, locB]
        BaseSrc.__init__(self, rxList, **kwargs)

    def eval(self, prob):
        if prob._formulation == 'HJ':
            inds = closestPoints(prob.mesh, self.loc, gridLoc='CC')
            q = np.zeros(prob.mesh.nC)
            q[inds] = self.current * np.r_[1., -1.]
        elif prob._formulation == 'EB':
            qa = prob.mesh.getInterpolationMat(self.loc[0], locType='N').todense()
            qb = -prob.mesh.getInterpolationMat(self.loc[1], locType='N').todense()
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
