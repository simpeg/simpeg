from __future__ import print_function
from SimPEG import *
from SimPEG.FLOW.Richards.Empirical import RichardsMap
import time


class RichardsRx(Survey.BaseTimeRx):
    """Richards Receiver Object"""

    knownRxTypes = ['saturation','pressureHead']

    def eval(self, U, m, mapping, mesh, timeMesh):

        if self.rxType == 'pressureHead':
            u = np.concatenate(U)
        elif self.rxType == 'saturation':
            u = np.concatenate([mapping.theta(ui, m) for ui in U])

        return self.getP(mesh, timeMesh) * u

    def evalDeriv(self, U, m, mapping, mesh, timeMesh):

        P = self.getP(mesh, timeMesh)
        if self.rxType == 'pressureHead':
            return P
        elif self.rxType == 'saturation':
            #TODO: if m is a parameter in the theta
            #      distribution, we may need to do
            #      some more chain rule here.
            dT = sp.block_diag([mapping.thetaDerivU(ui, m) for ui in U])
            return P*dT


class RichardsSurvey(Survey.BaseSurvey):
    """docstring for RichardsSurvey"""

    rxList = None

    def __init__(self, rxList, **kwargs):
        self.rxList = rxList
        Survey.BaseSurvey.__init__(self, **kwargs)

    @property
    def nD(self):
        return np.array([rx.nD for rx in self.rxList]).sum()

    @Utils.count
    @Utils.requires('prob')
    def dpred(self, m, f=None):
        """
            Create the projected data from a model.
            The field, f, (if provided) will be used for the predicted data
            instead of recalculating the fields (which may be expensive!).

            .. math::
                d_\\text{pred} = P(f(m), m)

            Where P is a projection of the fields onto the data space.
        """
        if f is None: f = self.prob.fields(m)
        return Utils.mkvc(self.eval(f, m))

    @Utils.requires('prob')
    def eval(self, U, m):
        Ds = list(range(len(self.rxList)))
        for ii, rx in enumerate(self.rxList):
            Ds[ii] = rx.eval(U, m,
                                self.prob.mapping,
                                self.prob.mesh,
                                self.prob.timeMesh)

        return np.concatenate(Ds)

    @Utils.requires('prob')
    def evalDeriv(self, U, m):
        """The Derivative with respect to the fields."""
        Ds = list(range(len(self.rxList)))
        for ii, rx in enumerate(self.rxList):
            Ds[ii] = rx.evalDeriv(U, m,
                                self.prob.mapping,
                                self.prob.mesh,
                                self.prob.timeMesh)

        return sp.vstack(Ds)

class RichardsProblem(Problem.BaseTimeProblem):
    """docstring for RichardsProblem"""

    boundaryConditions = None
    initialConditions  = None

    surveyPair  = RichardsSurvey
    mapPair = RichardsMap

    debug=True

    Solver = Solver
    solverOpts = {}

    def __init__(self, mesh, mapping=None, **kwargs):
        Problem.BaseTimeProblem.__init__(self, mesh, mapping=mapping, **kwargs)

    def getBoundaryConditions(self, ii, u_ii):
        if type(self.boundaryConditions) is np.ndarray:
            return self.boundaryConditions

        time = self.timeMesh.vectorCCx[ii]

        return self.boundaryConditions(time, u_ii)

    @property
    def method(self):
        """Method must be either 'mixed' or 'head'. See notes in Celia et al., 1990."""
        return getattr(self, '_method', 'mixed')
    @method.setter
    def method(self, value):
        assert value in ['mixed','head'], "method must be 'mixed' or 'head'."
        self._method = value

    # Setting doNewton will clear the rootFinder, which will be reinitialized when called
    doNewton = Utils.dependentProperty('_doNewton', False, ['_rootFinder'],
                "Do a Newton iteration. If False, a Picard iteration will be completed.")

    maxIterRootFinder = Utils.dependentProperty('_maxIterRootFinder', 30, ['_rootFinder'],
                "Maximum iterations for rootFinder iteration.")
    tolRootFinder = Utils.dependentProperty('_tolRootFinder', 1e-4, ['_rootFinder'],
                "Maximum iterations for rootFinder iteration.")

    @property
    def rootFinder(self):
        """Root-finding Algorithm"""
        if getattr(self, '_rootFinder', None) is None:
            self._rootFinder = Optimization.NewtonRoot(doLS=self.doNewton, maxIter=self.maxIterRootFinder, tol=self.tolRootFinder, Solver=self.Solver)
        return self._rootFinder

    @Utils.timeIt
    def fields(self, m):
        tic = time.time()
        u = list(range(self.nT+1))
        u[0] = self.initialConditions
        for ii, dt in enumerate(self.timeSteps):
            bc = self.getBoundaryConditions(ii, u[ii])
            u[ii+1] = self.rootFinder.root(lambda hn1m, return_g=True: self.getResidual(m, u[ii], hn1m, dt, bc, return_g=return_g), u[ii])
            if self.debug: print("Solving Fields ({0:4d}/{1:d} - {2:3.1f}% Done) {3:d} Iterations, {4:4.2f} seconds".format(ii+1, self.nT, 100.0*(ii+1)/self.nT, self.rootFinder.iter, time.time() - tic))
        return u

    @Utils.timeIt
    def diagsJacobian(self, m, hn, hn1, dt, bc):

        DIV  = self.mesh.faceDiv
        GRAD = self.mesh.cellGrad
        BC   = self.mesh.cellGradBC
        AV   = self.mesh.aveF2CC.T
        if self.mesh.dim == 1:
            Dz = self.mesh.faceDivx
        elif self.mesh.dim == 2:
            Dz = sp.hstack((Utils.spzeros(self.mesh.nC,self.mesh.vnF[0]), self.mesh.faceDivy),format='csr')
        elif self.mesh.dim == 3:
            Dz = sp.hstack((Utils.spzeros(self.mesh.nC,self.mesh.vnF[0]+self.mesh.vnF[1]), self.mesh.faceDivz),format='csr')

        dT   = self.mapping.thetaDerivU(hn, m)
        dT1  = self.mapping.thetaDerivU(hn1, m)
        K1   = self.mapping.k(hn1, m)
        dK1  = self.mapping.kDerivU(hn1, m)
        dKm1 = self.mapping.kDerivM(hn1, m)

        # Compute part of the derivative of:
        #
        #       DIV*diag(GRAD*hn1+BC*bc)*(AV*(1.0/K))^-1

        DdiagGh1 = DIV*Utils.sdiag(GRAD*hn1+BC*bc)
        diagAVk2_AVdiagK2 = Utils.sdiag((AV*(1./K1))**(-2)) * AV*Utils.sdiag(K1**(-2))

        # The matrix that we are computing has the form:
        #
        #   -                                      -   -  -     -  -
        #  |  Adiag                                 | | h1 |   | b1 |
        #  |   Asub    Adiag                        | | h2 |   | b2 |
        #  |            Asub    Adiag               | | h3 | = | b3 |
        #  |                 ...     ...            | | .. |   | .. |
        #  |                         Asub    Adiag  | | hn |   | bn |
        #   -                                      -   -  -     -  -

        Asub = (-1.0/dt)*dT

        Adiag = (
                  (1.0/dt)*dT1
                 -DdiagGh1*diagAVk2_AVdiagK2*dK1
                 -DIV*Utils.sdiag(1./(AV*(1./K1)))*GRAD
                 -Dz*diagAVk2_AVdiagK2*dK1
                )

        B = DdiagGh1*diagAVk2_AVdiagK2*dKm1 + Dz*diagAVk2_AVdiagK2*dKm1

        return Asub, Adiag, B

    @Utils.timeIt
    def getResidual(self, m, hn, h, dt, bc, return_g=True):
        """
            Where h is the proposed value for the next time iterate (h_{n+1})
        """
        DIV  = self.mesh.faceDiv
        GRAD = self.mesh.cellGrad
        BC   = self.mesh.cellGradBC
        AV   = self.mesh.aveF2CC.T
        if self.mesh.dim == 1:
            Dz = self.mesh.faceDivx
        elif self.mesh.dim == 2:
            Dz = sp.hstack((Utils.spzeros(self.mesh.nC,self.mesh.vnF[0]), self.mesh.faceDivy),format='csr')
        elif self.mesh.dim == 3:
            Dz = sp.hstack((Utils.spzeros(self.mesh.nC,self.mesh.vnF[0]+self.mesh.vnF[1]), self.mesh.faceDivz),format='csr')

        T  = self.mapping.theta(h, m)
        dT = self.mapping.thetaDerivU(h, m)
        Tn = self.mapping.theta(hn, m)
        K  = self.mapping.k(h, m)
        dK = self.mapping.kDerivU(h, m)

        aveK = 1./(AV*(1./K))

        RHS = DIV*Utils.sdiag(aveK)*(GRAD*h+BC*bc) + Dz*aveK
        if self.method == 'mixed':
            r = (T-Tn)/dt - RHS
        elif self.method == 'head':
            r = dT*(h - hn)/dt - RHS

        if not return_g: return r

        J = dT/dt - DIV*Utils.sdiag(aveK)*GRAD
        if self.doNewton:
            DDharmAve = Utils.sdiag(aveK**2)*AV*Utils.sdiag(K**(-2)) * dK
            J = J - DIV*Utils.sdiag(GRAD*h + BC*bc)*DDharmAve - Dz*DDharmAve

        return r, J

    @Utils.timeIt
    def Jfull(self, m, f=None):
        if f is None:
            f = self.fields(m)

        nn = len(f)-1
        Asubs, Adiags, Bs = list(range(nn)), list(range(nn)), list(range(nn))
        for ii in range(nn):
            dt = self.timeSteps[ii]
            bc = self.getBoundaryConditions(ii, f[ii])
            Asubs[ii], Adiags[ii], Bs[ii] = self.diagsJacobian(m, f[ii], f[ii+1], dt, bc)
        Ad = sp.block_diag(Adiags)
        zRight = Utils.spzeros((len(Asubs)-1)*Asubs[0].shape[0],Adiags[0].shape[1])
        zTop = Utils.spzeros(Adiags[0].shape[0], len(Adiags)*Adiags[0].shape[1])
        As = sp.vstack((zTop,sp.hstack((sp.block_diag(Asubs[1:]),zRight))))
        A = As + Ad
        B = np.array(sp.vstack(Bs).todense())

        Ainv = self.Solver(A, **self.solverOpts)
        P = self.survey.evalDeriv(f, m)
        AinvB = Ainv * B
        z = np.zeros((self.mesh.nC, B.shape[1]))
        zAinvB = np.vstack((z, AinvB))
        J = P * zAinvB
        return J

    @Utils.timeIt
    def Jvec(self, m, v, f=None):
        if f is None:
            f = self.fields(m)

        JvC = list(range(len(f)-1)) # Cell to hold each row of the long vector.

        # This is done via forward substitution.
        bc = self.getBoundaryConditions(0, f[0])
        temp, Adiag, B = self.diagsJacobian(m, f[0], f[1], self.timeSteps[0], bc)
        Adiaginv = self.Solver(Adiag, **self.solverOpts)
        JvC[0] = Adiaginv * (B*v)

        for ii in range(1,len(f)-1):
            bc = self.getBoundaryConditions(ii, f[ii])
            Asub, Adiag, B = self.diagsJacobian(m, f[ii], f[ii+1], self.timeSteps[ii], bc)
            Adiaginv = self.Solver(Adiag, **self.solverOpts)
            JvC[ii] = Adiaginv * (B*v - Asub*JvC[ii-1])

        P = self.survey.evalDeriv(f, m)
        return P * np.concatenate([np.zeros(self.mesh.nC)] + JvC)

    @Utils.timeIt
    def Jtvec(self, m, v, f=None):
        if f is None:
            f = self.field(m)

        P = self.survey.evalDeriv(f, m)
        PTv = P.T*v

        # This is done via backward substitution.
        minus = 0
        BJtv = 0
        for ii in range(len(f)-1,0,-1):
            bc = self.getBoundaryConditions(ii-1, f[ii-1])
            Asub, Adiag, B = self.diagsJacobian(m, f[ii-1], f[ii], self.timeSteps[ii-1], bc)
            #select the correct part of v
            vpart = list(range((ii)*Adiag.shape[0], (ii+1)*Adiag.shape[0]))
            AdiaginvT = self.Solver(Adiag.T, **self.solverOpts)
            JTvC = AdiaginvT * (PTv[vpart] - minus)
            minus = Asub.T*JTvC  # this is now the super diagonal.
            BJtv = BJtv + B.T*JTvC

        return BJtv
