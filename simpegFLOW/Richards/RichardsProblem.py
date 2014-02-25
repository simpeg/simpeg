from SimPEG import *
from BaseRichards import RichardsModel

class RichardsData(Data.BaseData):
    """docstring for RichardsData"""

    P = None

    def __init__(self, **kwargs):
        Data.BaseData.__init__(self, **kwargs)

    @property
    def dataType(self):
        """Choose how your data is collected, must be 'saturation' or 'pressureHead'."""
        return getattr(self, '_dataType', 'pressureHead')
    @dataType.setter
    def dataType(self, value):
        assert value in ['saturation','pressureHead'], "dataType must be 'saturation' or 'pressureHead'."
        self._dataType = value

    def projectFields(self, u):
        u = np.concatenate(u[1:])
        if self.dataType == 'saturation':
            #TODO: Fix this:
            u = self.prob.model.theta(MODEL, u)
        return self.P*u


class RichardsProblem(Problem.BaseProblem):
    """docstring for RichardsProblem"""

    timeEnd  = None
    boundaryConditions = None
    initialConditions  = None

    dataPair = RichardsData
    modelPair = RichardsModel

    def __init__(self, mesh, model, **kwargs):
        self.doNewton = False  # This also sets the rootFinder algorithm.
        Problem.BaseProblem.__init__(self, mesh, model, **kwargs)

    @property
    def timeStep(self):
        """The time between steps."""
        return getattr(self, '_timeStep', None)
    @timeStep.setter
    def timeStep(self, value):
        self._timeStep = float(value) # Because integers suck.

    @property
    def numIts(self):
        """The number of iterations in the time domain problem."""
        return int(self.timeEnd/self.timeStep)

    @property
    def method(self):
        """Method must be either 'mixed' or 'head'. See notes in Celia et al., 1990."""
        return getattr(self, '_method', 'mixed')
    @method.setter
    def method(self, value):
        assert value in ['mixed','head'], "method must be 'mixed' or 'head'."
        self._method = value

    @property
    def doNewton(self):
        """Do a Newton iteration. If False, a Picard iteration will be completed."""
        return self._doNewton
    @doNewton.setter
    def doNewton(self, value):
        value = bool(value)
        self.rootFinder = Optimization.NewtonRoot(doLS=value)
        self._doNewton = value

    def fields(self, m):
        Hs = range(self.numIts+1)
        Hs[0] = self.initialConditions
        for ii in range(self.numIts):
            Hs[ii+1] = self.rootFinder.root(lambda hn1m, return_g=True: self.getResidual(m, Hs[ii], hn1m, return_g=return_g), Hs[ii])
        return Hs

    def diagsJacobian(self, m, hn, hn1):

        DIV  = self.mesh.faceDiv
        GRAD = self.mesh.cellGrad
        BC   = self.mesh.cellGradBC
        AV   = self.mesh.aveCC2F
        if self.mesh.dim == 1:
            Dz = self.mesh.faceDivx
        elif self.mesh.dim == 2:
            Dz = sp.hstack((Utils.spzeros(self.mesh.nC,self.mesh.nFv[0]), self.mesh.faceDivy),format='csr')
        elif self.mesh.dim == 3:
            Dz = sp.hstack((Utils.spzeros(self.mesh.nC,self.mesh.nFv[0]+self.mesh.nFv[1]), self.mesh.faceDivz),format='csr')

        bc   = self.boundaryConditions
        dt   = self.timeStep

        dT   = self.model.thetaDerivU(hn, m)
        dT1  = self.model.thetaDerivU(hn1, m)
        K1   = self.model.k(hn1, m)
        dK1  = self.model.kDerivU(hn1, m)
        dKa1 = self.model.kDerivM(hn1, m)

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

        B = DdiagGh1*diagAVk2_AVdiagK2*dKa1 + Dz*diagAVk2_AVdiagK2*dKa1

        return Asub, Adiag, B

    def getResidual(self, m, hn, h, return_g=True):
        """
            Where h is the proposed value for the next time iterate (h_{n+1})
        """
        DIV  = self.mesh.faceDiv
        GRAD = self.mesh.cellGrad
        BC   = self.mesh.cellGradBC
        AV   = self.mesh.aveCC2F
        if self.mesh.dim == 1:
            Dz = self.mesh.faceDivx
        elif self.mesh.dim == 2:
            Dz = sp.hstack((Utils.spzeros(self.mesh.nC,self.mesh.nFv[0]), self.mesh.faceDivy),format='csr')
        elif self.mesh.dim == 3:
            Dz = sp.hstack((Utils.spzeros(self.mesh.nC,self.mesh.nFv[0]+self.mesh.nFv[1]), self.mesh.faceDivz),format='csr')

        bc = self.boundaryConditions
        dt = self.timeStep

        T  = self.model.theta(h, m)
        dT = self.model.thetaDerivU(h, m)
        Tn = self.model.theta(hn, m)
        K  = self.model.k(h, m)
        dK = self.model.kDerivU(h, m)

        aveK = 1./(AV*(1./K));

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

    def fullJ(self, m, u=None):
        if u is None:
            u = self.field(m)
        Hs = u
        nn = len(Hs)-1
        Asubs, Adiags, Bs = range(nn), range(nn), range(nn)
        for ii in range(nn):
            Asubs[ii], Adiags[ii], Bs[ii] = self.diagsJacobian(m, Hs[ii],Hs[ii+1])
        Ad = sp.block_diag(Adiags)
        zRight = Utils.spzeros((len(Asubs)-1)*Asubs[0].shape[0],Adiags[0].shape[1])
        zTop = Utils.spzeros(Adiags[0].shape[0], len(Adiags)*Adiags[0].shape[1])
        As = sp.vstack((zTop,sp.hstack((sp.block_diag(Asubs[1:]),zRight))))
        A = As + Ad
        B = np.array(sp.vstack(Bs).todense())

        Ainv = Solver(A)
        J = Ainv.solve(B)
        return J


    def Jvec(self, m, v, u=None):
        if u is None:
            u = self.field(m)
        Hs = u
        JvC = range(len(Hs)-1) # Cell to hold each row of the long vector.

        # This is done via forward substitution.
        temp, Adiag, B = self.diagsJacobian(m, Hs[0],Hs[1])
        Adiaginv = Solver(Adiag)
        JvC[0] = Adiaginv.solve(B*v)

        # M = @(x) tril(Adiag)\(diag(Adiag).*(triu(Adiag)\x));
        # JvC{1} = bicgstab(Adiag,(B*v),tolbcg,500,M);

        for ii in range(1,len(Hs)-1):
            Asub, Adiag, B = self.diagsJacobian(m, Hs[ii],Hs[ii+1])
            Adiaginv = Solver(Adiag)
            JvC[ii] = Adiaginv.solve(B*v - Asub*JvC[ii-1])

        if self.dataType == 'pressureHead':
            Jv = self.P*np.concatenate(JvC)
        elif self.dataType == 'saturation':
            dT = self.model.thetaDerivU(np.concatenate(Hs[1:]), m)
            Jv = self.P*dT*np.concatenate(JvC)

        return Jv

    def Jtvec(self, m, v, u=None):
        if u is None:
            u = self.field(m)
        Hs = u

        if self.dataType == 'pressureHead':
            PTv = self.P.T*v;
        elif self.dataType == 'saturation':
            dT = self.model.thetaDerivU(np.concatenate(Hs[1:]), m)
            PTv = dT.T*self.P.T*v

        # This is done via backward substitution.
        minus = 0
        BJtv = 0
        for ii in range(len(Hs)-1,0,-1):
            Asub, Adiag, B = self.diagsJacobian(m, Hs[ii-1], Hs[ii])
            #select the correct part of v
            vpart = range((ii-1)*Adiag.shape[0], (ii)*Adiag.shape[0])
            AdiaginvT = Solver(Adiag.T)
            JTvC = AdiaginvT.solve(PTv[vpart] - minus)
            minus = Asub.T*JTvC  # this is now the super diagonal.
            BJtv = BJtv + B.T*JTvC

        return BJtv



if __name__ == '__main__':
    from SimPEG import *
    import Richards
    import matplotlib.pyplot as plt
    M = Mesh.TensorMesh([np.ones(40)])
    Ks = 9.4400e-03
    E = Richards.Haverkamp(Ks=np.log(Ks), A=1.1750e+06, gamma=4.74, alpha=1.6110e+06, theta_s=0.287, theta_r=0.075, beta=3.96)
    bc = np.array([-61.5,-20.7])
    h = np.zeros(M.nC) + bc[0]

    # data = R
    prob = Richards.RichardsProblem(M,E, timeStep=10, timeEnd=100, boundaryConditions=bc, initialConditions=h, doNewton=False, method='mixed')

    q = sp.csr_matrix((np.ones(4),(np.arange(4),np.array([20, 30, 35, 38]))), shape=(4,M.nCx))
    P = sp.kron(sp.identity(prob.numIts),q)

    prob.dataType = 'pressureHead'
    mTrue = np.ones(M.nC)*np.log(Ks)
    stdev = 0.01  # The standard deviation for the noise
    data = prob.createSyntheticData(mTrue,std=stdev, P=P)
    p = plt.plot(data.dobs.reshape((-1,4)))
    plt.show()
    # opt = Optimization.InexactGaussNewton(maxIterLS=20, maxIter=10, tolF=1e-6, tolX=1e-6, tolG=1e-6, maxIterCG=6)
    # reg = Regularization.Tikhonov(model)
    # inv = Inversion.BaseInversion(prob, reg, opt, beta0=1e4)
    # derChk = lambda m: [inv.dataObj(m), inv.dataObjDeriv(m)]
    # print inv.dataObj(mTrue*0+np.log(1e-5))
    # print inv.dataObj(mTrue)
    # tests.checkDerivative(derChk, mTrue, plotIt=False)

