from SimPEG import *


class Magnetics(object):
    """docstring for Magnetics"""
    def __init__(self, arg):
        super(Magnetics, self).__init__()
        self.arg = arg

    def getA(self, m):
        """
        GetA creates and returns the A matrix for the Magnetics problem

        The A matrix has the form:

        .. math::

            \mathbf{A} = \mathbf{D}\mu\mathbf{G}u



        """
        return self.mesh.faceDiv



