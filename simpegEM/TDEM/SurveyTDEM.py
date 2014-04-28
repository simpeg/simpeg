from SimPEG import Utils, Survey, np
from SimPEG.Survey import BaseSurvey
from simpegEM.Utils import Sources


class RxTDEM(Survey.BaseTimeRx):

    knownRxTypes = {
                    'ex':['e', 'Ex'],
                    'ey':['e', 'Ey'],
                    'ez':['e', 'Ez'],

                    'bx':['b', 'Fx'],
                    'by':['b', 'Fy'],
                    'bz':['b', 'Fz'],
                   }

    def __init__(self, locs, times, rxType):
        Survey.BaseTimeRx.__init__(self, locs, times, rxType)

    @property
    def projField(self):
        """Field Type projection (e.g. e b ...)"""
        return self.knownRxTypes[self.rxType][0]

    @property
    def projGLoc(self):
        """Grid Location projection (e.g. Ex Fy ...)"""
        return self.knownRxTypes[self.rxType][1]

    def projectFields(self, tx, mesh, timeMesh, u):
        P = self.getP(mesh, timeMesh)
        u_part = Utils.mkvc(u[tx, self.projField, :])
        return P*u_part

    def projectFieldsDeriv(self, tx, mesh, timeMesh, u, v, adjoint=False):
        P = self.getP(mesh, timeMesh)

        if not adjoint:
            return P * Utils.mkvc(v[tx, self.projField, :])
        elif adjoint:
            return P.T * v[tx, self]


class FieldsTDEM(Survey.TimeFields):
    """Fancy Field Storage for a TDEM survey."""
    knownFields = {'b': 'F', 'e': 'E'}

    def tovec(self):
        nTx, nF, nE = self.survey.nTx, self.mesh.nF, self.mesh.nE
        u = np.empty(0 if nTx == 1 else (0, nTx))

        for i in range(self.survey.prob.nT):
            if 'b' in self:
                b = self[:,'b',i+1]
            else:
                b = np.zeros(nF if nTx == 1 else (nF, nTx))

            if 'e' in self:
                e = self[:,'e',i+1]
            else:
                e = np.zeros(nE if nTx == 1 else (nE, nTx))
            u = np.r_[u, b, e]
        return u

class TxTDEM(Survey.BaseTx):
    rxPair = RxTDEM
    knownTxTypes = ['VMD_MVP']

    def getInitialFields(self, mesh):
        F0 = getattr(self, '_getInitialFields_' + self.txType)(mesh)
        return F0

    def _getInitialFields_VMD_MVP(self, mesh):
        """Vertical magnetic dipole, magnetic vector potential"""
        if mesh._meshType is 'CYL':
            if mesh.isSymmetric:
                MVP = Sources.MagneticDipoleVectorPotential(self.loc, mesh.gridEy, 'y')
            else:
                raise NotImplementedError('Non-symmetric cyl mesh not implemented yet!')
        elif mesh._meshType is 'TENSOR':
            MVPx = Sources.MagneticDipoleVectorPotential(self.loc, mesh.gridEx, 'x')
            MVPy = Sources.MagneticDipoleVectorPotential(self.loc, mesh.gridEy, 'y')
            MVPz = Sources.MagneticDipoleVectorPotential(self.loc, mesh.gridEz, 'z')
            MVP = np.concatenate((MVPx, MVPy, MVPz))
        else:
            raise Exception('Unknown mesh for VMD')

        return {"b": mesh.edgeCurl*MVP}

    def getJs(self, time):
        return None

class SurveyTDEM(Survey.BaseSurvey):
    """
        docstring for SurveyTDEM
    """

    txPair = TxTDEM

    def __init__(self, txList, **kwargs):
        # Sort these by frequency
        self.txList = txList
        Survey.BaseSurvey.__init__(self, **kwargs)

    def projectFields(self, u):
        data = Survey.Data(self)
        for tx in self.txList:
            for rx in tx.rxList:
                data[tx, rx] = rx.projectFields(tx, self.mesh, self.prob.timeMesh, u)
        return data

    def projectFieldsDeriv(self, u, v=None, adjoint=False):
        assert v is not None, 'v to multiply must be provided.'

        if not adjoint:
            data = Survey.Data(self)
            for tx in self.txList:
                for rx in tx.rxList:
                    data[tx, rx] = rx.projectFieldsDeriv(tx, self.mesh, self.prob.timeMesh, u, v)
            return data
        else:
            f = FieldsTDEM(self.mesh, self)
            for tx in self.txList:
                for rx in tx.rxList:
                    Ptv = rx.projectFieldsDeriv(tx, self.mesh, self.prob.timeMesh, u, v, adjoint=True)
                    Ptv = Ptv.reshape((-1, 1, self.prob.timeMesh.nN), order='F')
                    f[tx, rx.projField, :] = Ptv
            return f



# class SurveyTDEM1D(BaseSurvey):
#     """
#         docstring for SurveyTDEM1D
#     """

#     txLoc = None #: txLoc
#     txType = None #: txType
#     rxLoc = None #: rxLoc
#     rxType = None #: rxType
#     timeCh = None #: timeCh
#     nTx    = 1 #: Number of transmitters

#     @property
#     def nTimeCh(self):
#         """Number of time channels"""
#         return self.timeCh.size

#     def __init__(self, **kwargs):
#         BaseSurvey.__init__(self, **kwargs)
#         Utils.setKwargs(self, **kwargs)

#     def projectFields(self, u):
#         #TODO: this is hardcoded to 1Tx
#         return self.Qrx.dot(u.b[:,:,0].T).T

#     def projectFieldsAdjoint(self, d):
#         # TODO: make the following self.nTimeCh
#         d = d.reshape((self.prob.nT, self.nTx), order='F')
#         #TODO: *Qtime.T need to multiply by a time projection. (outside for loop??)
#         ii = 0
#         F = FieldsTDEM(self.prob.mesh, self.nTx, self.prob.nT, 'b')
#         for ii in range(self.prob.nT):
#             b = self.Qrx.T*d[ii,:]
#             F.set_b(b, ii)
#             F.set_e(np.zeros((self.prob.mesh.nE,self.nTx)), ii)
#         return F

#     ####################################################
#     # Interpolation Matrices
#     ####################################################

#     @property
#     def Qrx(self):
#         if self._Qrx is None:
#             if self.rxType == 'bz':
#                 locType = 'Fz'
#             self._Qrx = self.prob.mesh.getInterpolationMat(self.rxLoc, locType=locType)
#         return self._Qrx
#     _Qrx = None


# class FieldsTDEM_OLD(object):
#     """docstring for FieldsTDEM"""

#     phi0 = None #: Initial electric potential
#     A0 = None #: Initial magnetic vector potential
#     e0 = None #: Initial electric field
#     b0 = None #: Initial magnetic flux density
#     j0 = None #: Initial current density
#     h0 = None #: Initial magnetic field

#     phi = None #: Electric potential
#     A = None #: Magnetic vector potential
#     e = None #: Electric field
#     b = None #: Magnetic flux density
#     j = None #: Current density
#     h = None #: Magnetic field

#     def __init__(self, mesh, nTx, nT, store='b'):

#         self.nT = nT #: Number of times
#         self.nTx = nTx #: Number of transmitters
#         self.mesh = mesh

#     def update(self, newFields, tInd):
#         self.set_b(newFields['b'], tInd)
#         self.set_e(newFields['e'], tInd)

#     def fieldVec(self):
#         u = np.ndarray((0, self.nTx))
#         for i in range(self.nT):
#             u = np.r_[u, self.get_b(i), self.get_e(i)]
#         if self.nTx == 1:
#             u = u.flatten()
#         return u

#     ####################################################
#     # Get Methods
#     ####################################################

#     def get_b(self, ind):
#         if ind == -1:
#             return self.b0
#         else:
#             return self.b[ind,:,:]

#     def get_e(self, ind):
#         if ind == -1:
#             return self.e0
#         else:
#             return self.e[ind,:,:]

#     ####################################################
#     # Set Methods
#     ####################################################

#     def set_b(self, b, ind):
#         if self.b is None:
#             self.b = np.zeros((self.nT, np.sum(self.mesh.nF), self.nTx))
#             self.b[:] = np.nan
#         if len(b.shape) == 1:
#             b = b[:, np.newaxis]
#         self.b[ind,:,:] = b

#     def set_e(self, e, ind):
#         if self.e is None:
#             self.e = np.zeros((self.nT, np.sum(self.mesh.nE), self.nTx))
#             self.e[:] = np.nan
#         if len(e.shape) == 1:
#             e = e[:, np.newaxis]
#         self.e[ind,:,:] = e


#     def __contains__(self, key):
#         return key in self.children
