from SimPEG import Utils, Survey, np
from SimPEG.Survey import BaseSurvey
from SimPEG.EM.Utils import *
from BaseTDEM import FieldsTDEM


class RxTDEM(Survey.BaseTimeRx):

    knownRxTypes = {
                    'ex':['e', 'Ex', 'N'],
                    'ey':['e', 'Ey', 'N'],
                    'ez':['e', 'Ez', 'N'],

                    'bx':['b', 'Fx', 'N'],
                    'by':['b', 'Fy', 'N'],
                    'bz':['b', 'Fz', 'N'],

                    'dbxdt':['b', 'Fx', 'CC'],
                    'dbydt':['b', 'Fy', 'CC'],
                    'dbzdt':['b', 'Fz', 'CC'],
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

    @property
    def projTLoc(self):
        """Time Location projection (e.g. CC N)"""
        return self.knownRxTypes[self.rxType][2]

    def getTimeP(self, timeMesh):
        """
            Returns the time projection matrix.

            .. note::

                This is not stored in memory, but is created on demand.
        """
        if self.rxType in ['dbxdt','dbydt','dbzdt']:
            return timeMesh.getInterpolationMat(self.times, self.projTLoc)*timeMesh.faceDiv
        else:
            return timeMesh.getInterpolationMat(self.times, self.projTLoc)

    def eval(self, src, mesh, timeMesh, u):
        P = self.getP(mesh, timeMesh)
        u_part = Utils.mkvc(u[src, self.projField, :])
        return P*u_part

    def evalDeriv(self, src, mesh, timeMesh, u, v, adjoint=False):
        P = self.getP(mesh, timeMesh)

        if not adjoint:
            return P * Utils.mkvc(v[src, self.projField, :])
        elif adjoint:
            return P.T * v[src, self]


class SrcTDEM(Survey.BaseSrc):
    rxPair = RxTDEM
    radius = None

    def getInitialFields(self, mesh):
        F0 = getattr(self, '_getInitialFields_' + self.srcType)(mesh)
        return F0

    def getJs(self, mesh, time):
        return None


class SrcTDEM_VMD_MVP(SrcTDEM):

    def __init__(self,rxList,loc,waveformType="STEPOFF"):
        self.loc = loc
        self.waveformType = waveformType
        SrcTDEM.__init__(self,rxList)

    def getInitialFields(self, mesh):
        """Vertical magnetic dipole, magnetic vector potential"""
        if self.waveformType == "STEPOFF":
            print ">> Step waveform: Non-zero initial condition"        
            if mesh._meshType is 'CYL':
                if mesh.isSymmetric:
                    MVP = MagneticDipoleVectorPotential(self.loc, mesh, 'Ey')
                else:
                    raise NotImplementedError('Non-symmetric cyl mesh not implemented yet!')
            elif mesh._meshType is 'TENSOR':
                MVP = MagneticDipoleVectorPotential(self.loc, mesh, ['Ex','Ey','Ez'])
            else:
                raise Exception('Unknown mesh for VMD') 
            return {"b": mesh.edgeCurl*MVP}               
        elif self.waveformType == "GENERAL":
            print ">> General waveform: Zero initial condition"
            return {"b": np.zeros(mesh.nF)}
        else:
            raise NotImplementedError("Only use STEPOFF or GENERAL")

    def getMeS(self, mesh, MfMui):
        if mesh._meshType is 'CYL':
            if mesh.isSymmetric:
                MVP = MagneticDipoleVectorPotential(self.loc, mesh, 'Ey')
            else:
                raise NotImplementedError('Non-symmetric cyl mesh not implemented yet!')
        elif mesh._meshType is 'TENSOR':
            MVP = MagneticDipoleVectorPotential(self.loc, mesh, ['Ex','Ey','Ez'])
        else:
            raise Exception('Unknown mesh for VMD')            
        return mesh.edgeCurl.T*MfMui*mesh.edgeCurl*MVP


class SrcTDEM_CircularLoop_MVP(SrcTDEM):
    def __init__(self,rxList,loc,radius,waveformType="STEPOFF"):
        self.loc = loc
        self.radius = radius
        self.waveformType = waveformType
        SrcTDEM.__init__(self,rxList)        

    def getInitialFields(self, mesh):
        """Circular Loop, magnetic vector potential"""
        if self.waveformType == "STEPOFF":
            print ">> Step waveform: Non-zero initial condition"
            if mesh._meshType is 'CYL':
                if mesh.isSymmetric:
                    MVP = MagneticLoopVectorPotential(self.loc, mesh, 'Ey', self.radius)
                else:
                    raise NotImplementedError('Non-symmetric cyl mesh not implemented yet!')
            elif mesh._meshType is 'TENSOR':
                MVP = MagneticLoopVectorPotential(self.loc, mesh, ['Ex','Ey','Ez'], self.radius)
            else:
                raise Exception('Unknown mesh for CircularLoop')
            return {"b": mesh.edgeCurl*MVP}
        elif self.waveformType == "GENERAL":
            print ">> General waveform: Zero initial condition"
            return {"b": np.zeros(mesh.nF)}
        else:
            raise NotImplementedError("Only use STEPOFF or GENERAL")

    def getMeS(self, mesh, MfMui):
        if mesh._meshType is 'CYL':
            if mesh.isSymmetric:
                MVP = MagneticLoopVectorPotential(self.loc, mesh, 'Ey', self.radius)
            else:
                raise NotImplementedError('Non-symmetric cyl mesh not implemented yet!')
        elif mesh._meshType is 'TENSOR':
            MVP = MagneticLoopVectorPotential(self.loc, mesh, ['Ex','Ey','Ez'], self.radius)
        else:
            raise Exception('Unknown mesh for CircularLoop')            
        return mesh.edgeCurl.T*MfMui*mesh.edgeCurl*MVP


class SurveyTDEM(Survey.BaseSurvey):
    """
        docstring for SurveyTDEM
    """
    srcPair = SrcTDEM

    def __init__(self, srcList, **kwargs):
        # Sort these by frequency
        self.srcList = srcList
        Survey.BaseSurvey.__init__(self, **kwargs)

    def eval(self, u):
        data = Survey.Data(self)
        for src in self.srcList:
            for rx in src.rxList:
                data[src, rx] = rx.eval(src, self.mesh, self.prob.timeMesh, u)
        return data

    def evalDeriv(self, u, v=None, adjoint=False):
        assert v is not None, 'v to multiply must be provided.'

        if not adjoint:
            data = Survey.Data(self)
            for src in self.srcList:
                for rx in src.rxList:
                    data[src, rx] = rx.evalDeriv(src, self.mesh, self.prob.timeMesh, u, v)
            return data
        else:
            f = FieldsTDEM(self.mesh, self)
            for src in self.srcList:
                for rx in src.rxList:
                    Ptv = rx.evalDeriv(src, self.mesh, self.prob.timeMesh, u, v, adjoint=True)
                    Ptv = Ptv.reshape((-1, self.prob.timeMesh.nN), order='F')
                    if rx.projField not in f: # first time we are projecting
                        f[src, rx.projField, :] = Ptv
                    else: # there are already fields, so let's add to them!
                        f[src, rx.projField, :] += Ptv
            return f


