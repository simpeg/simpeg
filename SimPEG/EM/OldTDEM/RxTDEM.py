import SimPEG
from SimPEG import Utils
import scipy.sparse as sp


class BaseRx(SimPEG.Survey.BaseTimeRx):
    """
    Time domain receiver base class

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param numpy.ndarray times: times
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    """

    def __init__(self, locs, times, orientation=None):
        assert(orientation in ['x', 'y', 'z']), (
            "Orientation {0!s} not known. Orientation must be in "
            "'x', 'y', 'z'. Arbitrary orientations have not yet been "
            "implemented.".format(orientation)
        )
        self.projComp = orientation
        SimPEG.Survey.BaseTimeRx.__init__(self, locs, times, rxType=None) #TODO: remove rxType from baseRx

    def projGLoc(self, f):
        """Grid Location projection (e.g. Ex Fy ...)"""
        return f._GLoc(self.projField) + self.projComp

    def projTLoc(self, f):
        """Time Location projection (e.g. CC N)"""
        return f._TLoc(self.projField)

    def getSpatialP(self, mesh, f):
        """
            Returns the spatial projection matrix.

            .. note::

                This is not stored in memory, but is created on demand.
        """
        return mesh.getInterpolationMat(self.locs, self.projGLoc(f))

    def getTimeP(self, timeMesh, f):
        """
            Returns the time projection matrix.

            .. note::

                This is not stored in memory, but is created on demand.
        """
        return timeMesh.getInterpolationMat(self.times, self.projTLoc(f))

    def getP(self, mesh, timeMesh, f):
        """
            Returns the projection matrices as a
            list for all components collected by
            the receivers.

            .. note::

                Projection matrices are stored as a dictionary (mesh, timeMesh) if storeProjections is True
        """
        if (mesh, timeMesh) in self._Ps:
            return self._Ps[(mesh, timeMesh)]

        Ps = self.getSpatialP(mesh, f)
        Pt = self.getTimeP(timeMesh, f)
        P = sp.kron(Pt, Ps)

        if self.storeProjections:
            self._Ps[(mesh, timeMesh)] = P

        return P

    def getTimeP(self, timeMesh, f):
        """
            Returns the time projection matrix.

            .. note::

                This is not stored in memory, but is created on demand.
        """
        # if self.projField == 'dbdt':
        #     return timeMesh.getInterpolationMat(
        #         self.times, self.projTLoc(f)
        #     )*timeMesh.faceDiv
        # else:
        return timeMesh.getInterpolationMat(self.times, self.projTLoc(f))

    def eval(self, src, mesh, timeMesh, f):
        """
        Project fields to receivers to get data.

        :param SimPEG.EM.TDEM.SrcTDEM.BaseSrc src: TDEM source
        :param BaseMesh mesh: mesh used
        :param Fields f: fields object
        :rtype: numpy.ndarray
        :return: fields projected to recievers
        """

        P = self.getP(mesh, timeMesh, f)
        f_part = Utils.mkvc(f[src, self.projField, :])
        return P*f_part

    def evalDeriv(self, src, mesh, timeMesh, f, v, adjoint=False):
        """
        Derivative of projected fields with respect to the inversion model times a vector.

        :param SimPEG.EM.TDEM.SrcTDEM.BaseSrc src: TDEM source
        :param BaseMesh mesh: mesh used
        :param BaseMesh timeMesh: time mesh
        :param Fields f: fields object
        :param numpy.ndarray v: vector to multiply
        :rtype: numpy.ndarray
        :return: fields projected to recievers
        """

        P = self.getP(mesh, timeMesh, f)
        if not adjoint:
            return P * v # Utils.mkvc(v[src, self.projField+'Deriv', :])
        elif adjoint:
            # dP_dF_T = P.T * v #[src, self]
            # newshape = (len(dP_dF_T)/timeMesh.nN, timeMesh.nN )
            return P.T * v # np.reshape(dP_dF_T, newshape, order='F')


class Point_e(BaseRx):
    """
    Electric field TDEM receiver

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param numpy.ndarray times: times
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    """

    def __init__(self, locs, times, orientation=None):
        self.projField = 'e'
        super(Point_e, self).__init__(locs, times, orientation)


class Point_b(BaseRx):
    """
    Magnetic flux TDEM receiver

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param numpy.ndarray times: times
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    """

    def __init__(self, locs, times, orientation=None):
        self.projField = 'b'
        super(Point_b, self).__init__(locs, times, orientation)


class Point_dbdt(BaseRx):
    """
    dbdt TDEM receiver

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param numpy.ndarray times: times
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    """

    def __init__(self, locs, times, orientation=None):
        self.projField = 'dbdt'
        super(Point_dbdt, self).__init__(locs, times, orientation)

    def eval(self, src, mesh, timeMesh, f):

        if self.projField in f.aliasFields:
            return super(Point_dbdt, self).eval(src, mesh, timeMesh, f)

        P = self.getP(mesh, timeMesh, f)
        f_part = Utils.mkvc(f[src, 'b', :])
        return P*f_part

    def projGLoc(self, f):
        """Grid Location projection (e.g. Ex Fy ...)"""
        if self.projField in f.aliasFields:
            return super(Point_dbdt, self).projGLoc(f)
        return f._GLoc(self.projField) + self.projComp

    def getTimeP(self, timeMesh, f):
        """
            Returns the time projection matrix.

            .. note::

                This is not stored in memory, but is created on demand.
        """
        if self.projField in f.aliasFields:
            return super(Point_dbdt, self).getTimeP(timeMesh, f)

        return timeMesh.getInterpolationMat(
            self.times, 'CC'
        )*timeMesh.faceDiv


class Point_h(BaseRx):
    """
    Magnetic field TDEM receiver

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param numpy.ndarray times: times
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    """

    def __init__(self, locs, times, orientation=None):
        self.projField = 'h'
        super(Point_h, self).__init__(locs, times, orientation)


class Point_j(BaseRx):
    """
    Current density TDEM receiver

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param numpy.ndarray times: times
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    """

    def __init__(self, locs, times, orientation=None):
        self.projField = 'j'
        super(Point_j, self).__init__(locs, times, orientation)

class Point_dhdt(BaseRx):

    def __init__(self, locs, times, orientation=None):
        self.projField = 'dhdt'
        super(Point_dhdt, self).__init__(locs, times, orientation)
