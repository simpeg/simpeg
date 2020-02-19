import numpy as np
import properties

from .... import survey
from ....utils import Zero, closestPoints, mkvc


class BaseSrc(survey.BaseSrc):

    current = properties.Float("Source current", default=1.0)

    def __init__(self, receiver_list, **kwargs):
        super(BaseSrc, self).__init__(receiver_list, **kwargs)

    def eval(self, simulation):
        raise NotImplementedError

    def evalDeriv(self, simulation):
        return Zero()

    @property
    def nD(self):
        """Number of data"""
        return self.vnD.sum()

    @property
    def vnD(self):
        """Vector number of data"""
        return np.array([rx.nD*len(rx.times) for rx in self.receiver_list])


class Dipole(BaseSrc):
    """
    Dipole source
    """

    location = properties.List(
        "location of the source electrodes",
        survey.SourceLocationArray("location of electrode")
    )

    def __init__(self, receiver_list, locationA, locationB, **kwargs):
        if locationA.shape != locationB.shape:
            raise Exception('Shape of locationA and locationB should be the same')
        super(Dipole, self).__init__(receiver_list, **kwargs)
        self.location = [locationA, locationB]


    def eval(self, simulation):
        if simulation._formulation == 'HJ':
            inds = closestPoints(simulation.mesh, self.location, gridLoc='CC')
            q = np.zeros(simulation.mesh.nC)
            q[inds] = self.current * np.r_[1., -1.]
        elif simulation._formulation == 'EB':
            qa = simulation.mesh.getInterpolationMat(
                    self.location[0], locType='N'
                ).todense()
            qb = -simulation.mesh.getInterpolationMat(
                self.location[1], locType='N'
            ).todense()
            q = self.current * mkvc(qa+qb)
        return q


class Pole(BaseSrc):
    """
    Pole source
    """

    def __init__(self, receiver_list, location, **kwargs):
        super(Pole, self).__init__(receiver_list, location=location, **kwargs)

    def eval(self, simulation):
        if simulation._formulation == 'HJ':
            inds = closestPoints(simulation.mesh, self.location)
            q = np.zeros(simulation.mesh.nC)
            q[inds] = self.current * np.r_[1.]
        elif simulation._formulation == 'EB':
            q = simulation.mesh.getInterpolationMat(self.location, locType='N').todense()
            q = self.current * mkvc(q)
        return q
