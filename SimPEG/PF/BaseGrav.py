from SimPEG import Survey, Maps


class LinearSurvey(Survey.BaseSurvey):
    """Base Magnetics Survey"""

    rxLoc = None  #: receiver locations
    rxType = None  #: receiver type

    def __init__(self, srcField, **kwargs):
        self.srcField = srcField
        Survey.BaseSurvey.__init__(self, **kwargs)

    def eval(self, u):
        return u

    @property
    def nD(self):
        return self.prob.G.shape[0]

    @property
    def rxLoc(self):
        return self.srcField.rxList[0].locs

    @property
    def nRx(self):
        return self.srcField.rxList[0].locs.shape[0]

    @property
    def Qfx(self):
        if getattr(self, '_Qfx', None) is None:
            self._Qfx = self.prob.mesh.getInterpolationMat(self.rxLoc, 'Fx')
        return self._Qfx

    @property
    def Qfy(self):
        if getattr(self, '_Qfy', None) is None:
            self._Qfy = self.prob.mesh.getInterpolationMat(self.rxLoc, 'Fy')
        return self._Qfy

    @property
    def Qfz(self):
        if getattr(self, '_Qfz', None) is None:
            self._Qfz = self.prob.mesh.getInterpolationMat(self.rxLoc, 'Fz')
        return self._Qfz

    def projectFields(self, u):
        """
            This function projects the fields onto the data space.

            First we project our B on to data location

            .. math::

                \mathbf{B}_{rec} = \mathbf{P} \mathbf{B}

            then we take the dot product between B and b_0

            .. math ::

                \\text{TMI} = \\vec{B}_s \cdot \hat{B}_0

        """
        # TODO: There can be some different tyes of data like |B| or B

        gfx = self.Qfx*u['G']
        gfy = self.Qfy*u['G']
        gfz = self.Qfz*u['G']

        fields = {'gx': gfx, 'gy': gfy, 'gz': gfz}
        return fields

class SrcField(Survey.BaseSrc):
    """ Define the inducing field """

    param = None  #: Inducing field param (Amp, Incl, Decl)

    def __init__(self, rxList, **kwargs):
        super(SrcField, self).__init__(rxList, **kwargs)


class RxObs(Survey.BaseRx):
    """A station location must have be located in 3-D"""
    def __init__(self, locsXYZ, **kwargs):
        locs = locsXYZ
        assert locsXYZ.shape[1] == 3, 'locs must in 3-D (x,y,z).'
        super(RxObs, self).__init__(locs, 'tmi',
                                    storeProjections=False, **kwargs)

    @property
    def nD(self):
        """Number of data in the receiver."""
        return self.locs[0].shape[0]

class BaseGravMap(Maps.IdentityMap):
    """BaseGravMap"""

    def __init__(self, mesh, **kwargs):
        Maps.IdentityMap.__init__(self, mesh)

    def _transform(self, m):

        return m

    def deriv(self, m):

        return sp.identity(self.nP)
