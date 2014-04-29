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
            u = np.concatenate((u, b, e))
        return Utils.mkvc(u)

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
                    if rx.projField not in f: # first time we are projecting
                        Ptv = Ptv.reshape((-1, 1, self.prob.timeMesh.nN), order='F')
                        f[tx, rx.projField, :] = Ptv
                    else:
                        Ptv = Ptv.reshape((-1, self.prob.timeMesh.nN), order='F')
                        addedPtv = f[tx, rx.projField, :]  + Ptv
                        addedPtv = addedPtv.reshape((-1, 1, self.prob.timeMesh.nN), order='F')
                        f[tx, rx.projField, :] = addedPtv
            return f


