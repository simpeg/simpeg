from SimPEG import *
import BaseMag
from scipy.constants import mu_0
from MagAnalytics import spheremodel, CongruousMagBC

class MagneticIntegral(Problem.BaseProblem):
    """
         approach using IE
    """


class MagneticsDiffSecondary(Problem.BaseProblem):
    """
        Secondary field approach using differential equations!
    """

    surveyPair = BaseMag.BaseMagSurvey
    modelPair = BaseMag.BaseMagMap

    def __init__(self, model, mapping=None, **kwargs):
        Problem.BaseProblem.__init__(self, model, mapping=mapping, **kwargs)

        Pbc, Pin, self._Pout = \
            self.mesh.getBCProjWF('neumann', discretization='CC')

        Dface = self.mesh.faceDiv
        Mc = Utils.sdiag(self.mesh.vol)
        self._Div = Mc*Dface*Pin.T*Pin

    @property
    def MfMuI(self): return self._MfMuI

    @property
    def MfMui(self): return self._MfMui

    @property
    def MfMu0(self): return self._MfMu0

    def makeMassMatrices(self, m):
        mu = self.mapping*m
        self._MfMui = self.mesh.getFaceInnerProduct(1./mu)/self.mesh.dim
        # self._MfMui = self.mesh.getFaceInnerProduct(1./mu)
        #TODO: this will break if tensor mu
        self._MfMuI = Utils.sdiag(1./self._MfMui.diagonal())
        self._MfMu0 = self.mesh.getFaceInnerProduct(1./mu_0)/self.mesh.dim
        # self._MfMu0 = self.mesh.getFaceInnerProduct(1/mu_0)

    @Utils.requires('survey')
    def getB0(self):
        b0 = self.survey.B0
        B0 = np.r_[b0[0]*np.ones(self.mesh.nFx),
                   b0[1]*np.ones(self.mesh.nFy),
                   b0[2]*np.ones(self.mesh.nFz)]
        return B0

    def getRHS(self, m):
        """

        .. math ::

            \mathbf{rhs} =  \Div(\MfMui)^{-1}\mathbf{M}^f_{\mu_0^{-1}}\mathbf{B}_0 - \Div\mathbf{B}_0+\diag(v)\mathbf{D} \mathbf{P}_{out}^T \mathbf{B}_{sBC}

        """
        B0 = self.getB0()
        Dface = self.mesh.faceDiv
        Mc = Utils.sdiag(self.mesh.vol)

        mu = self.mapping*m
        chi = mu/mu_0-1


        #temporary fix
        Bbc, Bbc_const = CongruousMagBC(self.mesh, self.survey.B0, chi)
        self.Bbc = Bbc
        self.Bbc_const = Bbc_const
        # return self._Div*self.MfMuI*self.MfMu0*B0 - self._Div*B0 + Mc*Dface*self._Pout.T*Bbc
        return self._Div*self.MfMuI*self.MfMu0*B0 - self._Div*B0

    def getA(self, m):
        """
        GetA creates and returns the A matrix for the Magnetics problem

        The A matrix has the form:

        .. math ::

            \mathbf{A} =  \Div(\MfMui)^{-1}\Div^{T}

        """
        return self._Div*self.MfMuI*self._Div.T


    def fields(self, m):
        """
            Return magnetic potential (u) and flux (B)
            u: defined on the cell center [nC x 1]
            B: defined on the cell center [nF x 1]

            After we compute u, then we update B.

            .. math ::

                \mathbf{B}_s = (\MfMui)^{-1}\mathbf{M}^f_{\mu_0^{-1}}\mathbf{B}_0-\mathbf{B}_0 -(\MfMui)^{-1}\Div^T \mathbf{u}

        """
        self.makeMassMatrices(m)
        A = self.getA(m)
        rhs = self.getRHS(m)
        m1 = sp.linalg.interface.aslinearoperator(Utils.sdiag(1/A.diagonal()))
        u, info = sp.linalg.bicgstab(A, rhs, tol=1e-6, maxiter=1000, M=m1)
        B0 = self.getB0()
        B = self.MfMuI*self.MfMu0*B0-B0-self.MfMuI*self._Div.T*u

        return {'B': B, 'u': u}

    @Utils.timeIt
    def Jvec(self, m, v, u=None):
        """
            Computing Jacobian multiplied by vector

            By setting our problem as

            .. math ::

                \mathbf{C}(\mathbf{m}, \mathbf{u}) = \mathbf{A}\mathbf{u} - \mathbf{rhs} = 0

            And taking derivative w.r.t m

            .. math ::

                \\nabla \mathbf{C}(\mathbf{m}, \mathbf{u}) = \\nabla_m \mathbf{C}(\mathbf{m}) \delta \mathbf{m} +
                                                             \\nabla_u \mathbf{C}(\mathbf{u}) \delta \mathbf{u} = 0

                \\frac{\delta \mathbf{u}}{\delta \mathbf{m}} = - [\\nabla_u \mathbf{C}(\mathbf{u})]^{-1}\\nabla_m \mathbf{C}(\mathbf{m})

            With some linear algebra we can have

            .. math ::

                \\nabla_u \mathbf{C}(\mathbf{u}) = \mathbf{A}

                \\nabla_m \mathbf{C}(\mathbf{m}) =
                \\frac{\partial \mathbf{A}}{\partial \mathbf{m}}(\mathbf{m})\mathbf{u} - \\frac{\partial \mathbf{rhs}(\mathbf{m})}{\partial \mathbf{m}}

            .. math ::

                \\frac{\partial \mathbf{A}}{\partial \mathbf{m}}(\mathbf{m})\mathbf{u} =
                \\frac{\partial \mathbf{\mu}}{\partial \mathbf{m}} \left[\Div \diag (\Div^T \mathbf{u}) \dMfMuI \\right]

                \dMfMuI = \diag(\MfMui)^{-1}_{vec} \mathbf{Av}_{F2CC}^T\diag(\mathbf{v})\diag(\\frac{1}{\mu^2})

                \\frac{\partial \mathbf{rhs}(\mathbf{m})}{\partial \mathbf{m}} =  \\frac{\partial \mathbf{\mu}}{\partial \mathbf{m}} \left[
                \Div \diag(\M^f_{\mu_{0}^{-1}}\mathbf{B}_0) \dMfMuI \\right] - \diag(\mathbf{v})\mathbf{D} \mathbf{P}_{out}^T\\frac{\partial B_{sBC}}{\partial \mathbf{m}}

            In the end,

            .. math ::

                \\frac{\delta \mathbf{u}}{\delta \mathbf{m}} =
                - [ \mathbf{A} ]^{-1}\left[ \\frac{\partial \mathbf{A}}{\partial \mathbf{m}}(\mathbf{m})\mathbf{u}
                - \\frac{\partial \mathbf{rhs}(\mathbf{m})}{\partial \mathbf{m}} \\right]

            A little tricky point here is we are not interested in potential (u), but interested in magnetic flux (B).
            Thus, we need sensitivity for B. Now we take derivative of B w.r.t m and have

            .. math ::

                \\frac{\delta \mathbf{B}} {\delta \mathbf{m}} = \\frac{\partial \mathbf{\mu} } {\partial \mathbf{m} }
                \left[
                \diag(\M^f_{\mu_{0}^{-1} } \mathbf{B}_0) \dMfMuI  \\
                 -  \diag (\Div^T\mathbf{u})\dMfMuI
                \\right ]

                 -  (\MfMui)^{-1}\Div^T\\frac{\delta\mathbf{u}}{\delta \mathbf{m}}

            Finally we evaluate the above, but we should remember that

            .. note ::

                We only want to evalute

                .. math ::

                    \mathbf{J}\mathbf{v} = \\frac{\delta \mathbf{P}\mathbf{B}} {\delta \mathbf{m}}\mathbf{v}

                Since forming sensitivity matrix is very expensive in that this monster is "big" and "dense" matrix!!


        """
        if u is None:
            u = self.fields(m)

        B, u = u['B'], u['u']
        mu = self.mapping*(m)
        dmudm = self.mapping.deriv(m)
        dchidmu = Utils.sdiag(1/mu_0*np.ones(self.mesh.nC))

        vol = self.mesh.vol
        Div = self._Div
        Dface = self.mesh.faceDiv
        P = self.survey.projectFieldsDeriv(B)                 # Projection matrix
        B0 = self.getB0()

        MfMuIvec = 1/self.MfMui.diagonal()
        dMfMuI = Utils.sdiag(MfMuIvec**2)*self.mesh.aveF2CC.T*Utils.sdiag(vol*1./mu**2)

        # A = self._Div*self.MfMuI*self._Div.T
        # RHS = Div*MfMuI*MfMu0*B0 - Div*B0 + Mc*Dface*Pout.T*Bbc
        # C(m,u) = A*m-rhs
        # dudm = -(dCdu)^(-1)dCdm

        dCdu = self.getA(m)
        dCdm_A = Div * ( Utils.sdiag( Div.T * u )* dMfMuI *dmudm  )
        dCdm_RHS1 = Div * (Utils.sdiag( self.MfMu0*B0  ) * dMfMuI)
        temp1 = (Dface*(self._Pout.T*self.Bbc_const*self.Bbc))
        dCdm_RHS2v  = (Utils.sdiag(vol)*temp1)*np.inner(vol, dchidmu*dmudm*v)
        #temporary fix
        # dCdm_RHSv =  dCdm_RHS1*(dmudm*v) +  dCdm_RHS2v
        dCdm_RHSv =  dCdm_RHS1*(dmudm*v)
        dCdm_v = dCdm_A*v - dCdm_RHSv

        m1 = sp.linalg.interface.aslinearoperator(Utils.sdiag(1/dCdu.diagonal()))
        sol, info = sp.linalg.bicgstab(dCdu, dCdm_v, tol=1e-6, maxiter=1000, M=m1)

        if info > 0:
            print "Iterative solver did not work well (Jvec)"
            # raise Exception ("Iterative solver did not work well")

        # B = self.MfMuI*self.MfMu0*B0-B0-self.MfMuI*self._Div.T*u
        # dBdm = d\mudm*dBd\mu

        dudm = -sol
        dBdmv =     (  Utils.sdiag(self.MfMu0*B0)*(dMfMuI * (dmudm*v)) \
                     - Utils.sdiag(Div.T*u)*(dMfMuI* (dmudm*v)) \
                     - self.MfMuI*(Div.T* (dudm)) )

        return Utils.mkvc(P*dBdmv)

    @Utils.timeIt
    def Jtvec(self, m, v, u=None):
        """
            Computing Jacobian^T multiplied by vector.

        .. math ::

            (\\frac{\delta \mathbf{P}\mathbf{B}} {\delta \mathbf{m}})^{T} = \left[ \mathbf{P}_{deriv}\\frac{\partial \mathbf{\mu} } {\partial \mathbf{m} }
            \left[
            \diag(\M^f_{\mu_{0}^{-1} } \mathbf{B}_0) \dMfMuI  \\
             -  \diag (\Div^T\mathbf{u})\dMfMuI
            \\right ]\\right]^{T}

             -  \left[\mathbf{P}_{deriv}(\MfMui)^{-1}\Div^T\\frac{\delta\mathbf{u}}{\delta \mathbf{m}} \\right]^{T}

        where

        .. math ::

            \mathbf{P}_{derv} = \\frac{\partial \mathbf{P}}{\partial\mathbf{B}}

        .. note ::

            Here we only want to compute

            .. math ::

                \mathbf{J}^{T}\mathbf{v} = (\\frac{\delta \mathbf{P}\mathbf{B}} {\delta \mathbf{m}})^{T} \mathbf{v}

        """
        if u is None:
            u = self.fields(m)

        B, u = u['B'], u['u']
        mu = self.mapping*(m)
        dmudm = self.mapping.deriv(m)
        dchidmu = Utils.sdiag(1/mu_0*np.ones(self.mesh.nC))

        vol = self.mesh.vol
        Div = self._Div
        Dface = self.mesh.faceDiv
        P = self.survey.projectFieldsDeriv(B)                 # Projection matrix
        B0 = self.getB0()

        MfMuIvec = 1/self.MfMui.diagonal()
        dMfMuI = Utils.sdiag(MfMuIvec**2)*self.mesh.aveF2CC.T*Utils.sdiag(vol*1./mu**2)

        # A = self._Div*self.MfMuI*self._Div.T
        # RHS = Div*MfMuI*MfMu0*B0 - Div*B0 + Mc*Dface*Pout.T*Bbc
        # C(m,u) = A*m-rhs
        # dudm = -(dCdu)^(-1)dCdm

        dCdu = self.getA(m)
        s = Div * ( self.MfMuI.T * ( P.T*v ) )

        m1 = sp.linalg.interface.aslinearoperator(Utils.sdiag(1/(dCdu.T).diagonal()))
        sol, info = sp.linalg.bicgstab(dCdu.T, s, tol=1e-6, maxiter=1000, M=m1)

        if info > 0:
            print "Iterative solver did not work well (Jtvec)"
            # raise Exception ("Iterative solver did not work well")


        # dCdm_A = Div * ( Utils.sdiag( Div.T * u )* dMfMuI *dmudm  )
        # dCdm_Atsol = ( dMfMuI.T*( Utils.sdiag( Div.T * u ) * (Div.T * dmudm)) ) * sol
        dCdm_Atsol = ( dmudm.T * dMfMuI.T*( Utils.sdiag( Div.T * u ) * Div.T ) ) * sol

        # dCdm_RHS1 = Div * (Utils.sdiag( self.MfMu0*B0  ) * dMfMuI)
        # dCdm_RHS1tsol = (dMfMuI.T*( Utils.sdiag( self.MfMu0*B0  ) ) * Div.T * dmudm) * sol
        dCdm_RHS1tsol = ( dmudm.T * dMfMuI.T*( Utils.sdiag( self.MfMu0*B0  ) ) * Div.T ) * sol


        # temp1 = (Dface*(self._Pout.T*self.Bbc_const*self.Bbc))
        temp1sol = ( Dface.T*( Utils.sdiag(vol)*sol ) )
        temp2 = self.Bbc_const*(self._Pout.T*self.Bbc).T
        # dCdm_RHS2v  = (Utils.sdiag(vol)*temp1)*np.inner(vol, dchidmu*dmudm*v)
        dCdm_RHS2tsol  = (dmudm.T*dchidmu.T*vol)*np.inner(temp2, temp1sol)

        # dCdm_RHSv =  dCdm_RHS1*(dmudm*v) +  dCdm_RHS2v

        #temporary fix
        # dCdm_RHStsol = dCdm_RHS1tsol - dCdm_RHS2tsol
        dCdm_RHStsol = dCdm_RHS1tsol

        # dCdm_RHSv =  dCdm_RHS1*(dmudm*v) +  dCdm_RHS2v
        # dCdm_v = dCdm_A*v - dCdm_RHSv

        Ctv = dCdm_Atsol - dCdm_RHStsol

        # B = self.MfMuI*self.MfMu0*B0-B0-self.MfMuI*self._Div.T*u
        # dBdm = d\mudm*dBd\mu
        # dPBdm^T*v = Atemp^T*P^T*v - Btemp^T*P^T*v - Ctv

        Atemp = Utils.sdiag(self.MfMu0*B0)*(dMfMuI * (dmudm))
        Btemp = Utils.sdiag(Div.T*u)*(dMfMuI* (dmudm))
        Jtv = Atemp.T*(P.T*v) - Btemp.T*(P.T*v) - Ctv

        return Utils.mkvc(Jtv)

def MagneticsDiffSecondaryInv(mesh, model, data, **kwargs):

    """
        Inversion module for MagneticsDiffSecondary

    """
    from SimPEG import Optimization, Regularization, Parameters, ObjFunction, Inversion
    prob = MagneticsDiffSecondary(mesh, model)

    miter = kwargs.get('maxIter', 10)

    if prob.ispaired:
        prob.unpair()
    if data.ispaired:
        data.unpair()
    prob.pair(data)

    # Create an optimization program
    opt = Optimization.InexactGaussNewton(maxIter=miter)
    opt.bfgsH0 = Solver(sp.identity(model.nP),flag='D')
    # Create a regularization program
    reg = Regularization.Tikhonov(model)
    # Create an objective function
    beta = Parameters.BetaSchedule(beta0=1e0)
    obj = ObjFunction.BaseObjFunction(data, reg, beta=beta)
    # Create an inversion object
    inv = Inversion.BaseInversion(obj, opt)

    return inv, reg



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    hxind = ((5,25,1.3),(41, 12.5),(5,25,1.3))
    hyind = ((5,25,1.3),(41, 12.5),(5,25,1.3))
    hzind = ((5,25,1.3),(40, 12.5),(5,25,1.3))
    hx, hy, hz = Utils.meshTensors(hxind, hyind, hzind)
    mesh = Mesh.TensorMesh([hx, hy, hz], [-hx.sum()/2,-hy.sum()/2,-hz.sum()/2])

    chibkg = 0.
    chiblk = 0.01
    chi = np.ones(mesh.nC)*chibkg
    sph_ind = spheremodel(mesh, 0., 0., 0., 100)
    chi[sph_ind] = chiblk
    model = BaseMag.BaseMagModel(mesh)
    # mu = (1.+chi)*mu_0

    data = BaseMag.BaseMagData()
    Inc = 90.
    Dec = 0.
    Btot = 51000

    data.setBackgroundField(Inc, Dec, Btot)

    xr = np.linspace(-300, 300, 41)
    yr = np.linspace(-300, 300, 41)
    X, Y = np.meshgrid(xr, yr)
    Z = np.ones((xr.size, yr.size))*150
    rxLoc = np.c_[Utils.mkvc(X), Utils.mkvc(Y), Utils.mkvc(Z)]
    data.rxLoc = rxLoc

    prob = MagneticsDiffSecondary(mesh, model)
    prob.pair(data)

    dpred = data.dpred(chi)

    # fig = plt.figure( figsize = (8,5) )
    # ax = plt.subplot(111)
    # dat = plt.imshow(np.reshape(dpred, (xr.size, yr.size), order='F'), extent=[min(xr), max(xr), min(yr), max(yr)])
    # plt.colorbar(dat, ax = ax)
    # plt.show()


def Intgrl_Fwr_Data(mesh,B,M,rxLoc,model,actv,flag):
    """
    Forward model magnetic data using integral equation

    INPUT:
    mesh        = SimPEG.TensorMesh
    B           = Inducing field parameter [Binc, Bdecl, B0]
    M           = Magnetization matrix [Minc, Mdecl] -90:90, 0:360
    rxLox       = Observation location informat [obsx, obsy, obsz]
    model       = Model associated with mesh
    actv        = Active cells from topo (from getActiveTopo)
    flag        = Data type "tmi" | "xyz"

    OUTPUT:
    dobs        =Observation array in format [obsx, obsy, obsz, data]

    Created on Oct 7, 2015

    @author: dominiquef
     """

    inds = np.nonzero(actv)[0]
    P = sp.csr_matrix((np.ones(inds.size),(inds, range(inds.size))), shape=(mesh.nC, len(inds)))

    xn = mesh.vectorNx;
    yn = mesh.vectorNy;
    zn = mesh.vectorNz;

    yn2,xn2,zn2 = np.meshgrid(yn[1:], xn[1:], zn[1:])
    yn1,xn1,zn1 = np.meshgrid(yn[0:-1], xn[0:-1], zn[0:-1])

    Yn = P.T*np.c_[mkvc(yn1), mkvc(yn2)]
    Xn = P.T*np.c_[mkvc(xn1), mkvc(xn2)]
    Zn = P.T*np.c_[mkvc(zn1), mkvc(zn2)]

    mcell = len(inds)

    ndata = rxLoc.shape[0]

    # Convert declination from north to cartesian
    Md = (450.-float(M[1]))%360.

    # Create magnetization matrix
    mx = np.cos(np.deg2rad(M[0])) * np.cos(np.deg2rad(Md))
    my = np.cos(np.deg2rad(M[0])) * np.sin(np.deg2rad(Md))
    mz = np.sin(np.deg2rad(M[0]))

    Mx = Utils.sdiag(np.ones([mcell])*mx*B[2])
    My = Utils.sdiag(np.ones([mcell])*my*B[2])
    Mz = Utils.sdiag(np.ones([mcell])*mz*B[2])

    #matplotlib.pyplot.spy(scipy.sparse.csr_matrix(Mx))
    #plt.show()
    Mxyz = sp.vstack((Mx,My,Mz));

    #%% Create TMI projector

    # Convert Bdecination from north to cartesian
    D = (450.-float(B[1]))%360.


    if flag=='tmi':
        Ptmi = mkvc(np.r_[np.cos(np.deg2rad(B[0]))*np.cos(np.deg2rad(D)),np.cos(np.deg2rad(B[0]))*np.sin(np.deg2rad(D)),np.sin(np.deg2rad(B[0]))],2).T;
        d = np.zeros(ndata)

    elif flag=='xyz':
        d = np.zeros(int(3*ndata))

    # Loop through all observations and create forward operator (ndata-by-mcell)
    print "Begin forward modeling " +str(int(ndata)) + " data points..."

    # Add counter to dsiplay progress.
    count = -1

    for ii in range(ndata):

        tx, ty, tz = get_T_mat(Xn,Yn,Zn,rxLoc[ii,:])
        Gxyz = np.vstack((tx,ty,tz))*Mxyz

        # Remove non-active cells
        if flag=='xyz':
            d[ii::ndata] = mkvc(Gxyz.dot(P.T*model))

        elif flag=='tmi':
            d[ii] = Ptmi.dot(Gxyz.dot(P.T*model))

        # Display progress
        count = progress(ii,count,ndata)


    print "Done 100% ...forward modeling completed!!\n"

    return d


def Intrgl_Fwr_Op(mesh,B,M,rxLoc,actv,flag):
    """ 

    Magnetic forward operator in integral form

    INPUT:
    mesh        = Mesh in SimPEG format
    B           = Inducing field parameter [Binc, Bdecl, B0]
    M           = Magnetization information
    [OPTIONS]
      1- [Minc, Mdecl] : Assumes uniform magnetization orientation
      2- [mx1,mx2,..., my1,...,mz1] : cell-based defined magnetization direction
      3- diag(M): Block diagonal matrix with [Mx, My, Mz] along the diagonal

    rxLox       = Observation location informat [obsx, obsy, obsz]

    flag        = 'tmi' | 'xyz' | 'full'
    [OPTIONS]
      1- tmi : Magnetization direction used and data are projected onto the
                inducing field direction F.shape([ndata, nc])

      2- xyz : Magnetization direction used and data are given in 3-components
                F.shape([3*ndata, nc])

      3- full: Full tensor matrix stored with shape([3*ndata, 3*nc])

    OUTPUT:
    F        = Linear forward modeling operation

    Created on Dec, 20th 2015

    @author: dominiquef

     """    
    # Find non-zero cells
    inds = np.nonzero(actv)[0]
    
    # Create active cell projector
    P = sp.csr_matrix((np.ones(inds.size),(inds, range(inds.size))),
                      shape=(mesh.nC, len(inds)))
    
    mcell = len(inds)
    
    # Create vectors of nodal location (lower and upper coners for each cell)
    xn = mesh.vectorNx;
    yn = mesh.vectorNy;
    zn = mesh.vectorNz;
    
    yn2,xn2,zn2 = np.meshgrid(yn[1:], xn[1:], zn[1:])
    yn1,xn1,zn1 = np.meshgrid(yn[0:-1], xn[0:-1], zn[0:-1])
    
    Yn = P.T*np.c_[mkvc(yn1), mkvc(yn2)]
    Xn = P.T*np.c_[mkvc(xn1), mkvc(xn2)]
    Zn = P.T*np.c_[mkvc(zn1), mkvc(zn2)]
          
    ndata = rxLoc.shape[0]    

    # Convert Bdecination from north to cartesian
    D = (450.-float(B[1]))%360.


    # Pre-allocate space and create magnetization matrix if required
    if (flag=='tmi') | (flag == 'xyz'):
        # If assumes uniform magnetization direction
        if M.shape != (mcell,3):

            print 'Magnetization vector must be Nc x 3'
            return


        Mx = Utils.sdiag(M[:,0]*B[2])
        My = Utils.sdiag(M[:,1]*B[2])
        Mz = Utils.sdiag(M[:,2]*B[2])

        Mxyz = sp.vstack((Mx,My,Mz))



        if flag == 'tmi':
            F = np.zeros((ndata, mcell))

            # Projection matrix
            Ptmi = mkvc(np.r_[np.cos(np.deg2rad(B[0]))*np.cos(np.deg2rad(D)),
                      np.cos(np.deg2rad(B[0]))*np.sin(np.deg2rad(D)),
                        np.sin(np.deg2rad(B[0]))],2).T;

        elif flag == 'xyz':

            F = np.zeros((int(3*ndata), mcell))
        
    elif flag == 'full':       
        F = np.zeros((int(3*ndata), int(3*mcell)))
        

    else:
        print """Flag must be either 'tmi' | 'xyz' | 'full', please revised"""
        return


    # Loop through all observations and create forward operator (ndata-by-mcell)
    print "Begin calculation of forward operator: " + flag

    # Add counter to dsiplay progress. Good for large problems
    count = -1;
    for ii in range(ndata):

    
        tx, ty, tz = get_T_mat(Xn,Yn,Zn,rxLoc[ii,:])  
        
        if flag=='tmi':
            F[ii,:] = Ptmi.dot(np.vstack((tx,ty,tz)))*Mxyz

        elif flag == 'xyz':
            F[ii,:] = tx*Mxyz
            F[ii+ndata,:] = ty*Mxyz
            F[ii+2*ndata,:] = tz*Mxyz

        elif flag == 'full':
            F[ii,:] = tx
            F[ii+ndata,:] = ty
            F[ii+2*ndata,:] = tz


    # Display progress
        count = progress(ii,count,ndata)

    print "Done 100% ...forward operator completed!!\n"
    
    return F

def get_T_mat(Xn,Yn,Zn,rxLoc):
    """
    Load in the active nodes of a tensor mesh and computes the magnetic tensor
    for a given observation location rxLoc[obsx, obsy, obsz]

    INPUT:
    Xn, Yn, Zn: Node location matrix for the lower and upper most corners of
                all cells in the mesh shape[nC,2]
    M
    OUTPUT:
    Tx = [Txx Txy Txz]
    Ty = [Tyx Tyy Tyz]
    Tz = [Tzx Tzy Tzz]

    where each elements have dimension 1-by-mcell.
    Only the upper half 5 elements have to be computed since symetric.
    Currently done as for-loops but will eventually be changed to vector
    indexing, once the topography has been figured out.

    Created on Oct, 20th 2015

    @author: dominiquef

     """ 
    
    eps = 1e-10 # add a small value to the locations to avoid /0
    
    mcell = Xn.shape[0]

    # Pre-allocate space for 1D array
    Tx = np.zeros((1,3*mcell))
    Ty = np.zeros((1,3*mcell))
    Tz = np.zeros((1,3*mcell))

    
    dz2 = rxLoc[2] - Zn[:,0] + eps
    dz1 = rxLoc[2] - Zn[:,1] + eps
         
    dy2 = Yn[:,1] - rxLoc[1] + eps
    dy1 = Yn[:,0] - rxLoc[1] + eps
        
    dx2 = Xn[:,1] - rxLoc[0] + eps
    dx1 = Xn[:,0] - rxLoc[0] + eps
    

    R1 = ( dy2**2 + dx2**2 )
    R2 = ( dy2**2 + dx1**2 )
    R3 = ( dy1**2 + dx2**2 )
    R4 = ( dy1**2 + dx1**2 )


    arg1 = np.sqrt( dz2**2 + R2 )
    arg2 = np.sqrt( dz2**2 + R1 )
    arg3 = np.sqrt( dz1**2 + R1 )
    arg4 = np.sqrt( dz1**2 + R2 )
    arg5 = np.sqrt( dz2**2 + R3 )
    arg6 = np.sqrt( dz2**2 + R4 )
    arg7 = np.sqrt( dz1**2 + R4 )
    arg8 = np.sqrt( dz1**2 + R3 )



    Tx[0,0:mcell] = np.arctan2( dy1 * dz2 , ( dx2 * arg5 ) ) +\
                - np.arctan2( dy2 * dz2 , ( dx2 * arg2 ) ) +\
                np.arctan2( dy2 * dz1 , ( dx2 * arg3 ) ) +\
                - np.arctan2( dy1 * dz1 , ( dx2 * arg8 ) ) +\
                np.arctan2( dy2 * dz2  , ( dx1 * arg1 ) ) +\
                - np.arctan2( dy1 * dz2 , ( dx1 * arg6 ) ) +\
                np.arctan2( dy1 * dz1 , ( dx1 * arg7 ) ) +\
                - np.arctan2( dy2 * dz1 , ( dx1 * arg4 ) );


    Ty[0,0:mcell] = np.log( ( dz2 + arg2 ) / (dz1 + arg3 ) ) +\
                -np.log( ( dz2 + arg1 ) / (dz1 + arg4 ) ) +\
                np.log( ( dz2 + arg6 ) / (dz1 + arg7 ) ) +\
                -np.log( ( dz2 + arg5 ) / (dz1 + arg8 ) );

    Ty[0,mcell:2*mcell] = np.arctan2( dx1 * dz2 , ( dy2 * arg1 ) ) +\
                    - np.arctan2( dx2 * dz2 , ( dy2 * arg2 ) ) +\
                    np.arctan2( dx2 * dz1 , ( dy2 * arg3 ) ) +\
                    - np.arctan2( dx1 * dz1 , ( dy2 * arg4 ) ) +\
                    np.arctan2( dx2 * dz2 , ( dy1 * arg5 ) ) +\
                    - np.arctan2( dx1 * dz2 , ( dy1 * arg6 ) ) +\
                    np.arctan2( dx1 * dz1 , ( dy1 * arg7 ) ) +\
                    - np.arctan2( dx2 * dz1 , ( dy1 * arg8 ) );

    R1 = (dy2**2 + dz1**2);
    R2 = (dy2**2 + dz2**2);
    R3 = (dy1**2 + dz1**2);
    R4 = (dy1**2 + dz2**2);

    Ty[0,2*mcell:] = np.log( ( dx1 + np.sqrt( dx1**2 + R1 ) ) / (dx2 + np.sqrt( dx2**2 + R1 ) ) ) +\
                        -np.log( ( dx1 + np.sqrt( dx1**2 + R2 ) ) / (dx2 + np.sqrt( dx2**2 + R2 ) ) ) +\
                        np.log( ( dx1 + np.sqrt( dx1**2 + R4 ) ) / (dx2 + np.sqrt( dx2**2 + R4 ) ) ) +\
                        -np.log( ( dx1 + np.sqrt( dx1**2 + R3 ) ) / (dx2 + np.sqrt( dx2**2 + R3 ) ) );

    R1 = (dx2**2 + dz1**2);
    R2 = (dx2**2 + dz2**2);
    R3 = (dx1**2 + dz1**2);
    R4 = (dx1**2 + dz2**2);

    Tx[0,2*mcell:] = np.log( ( dy1 + np.sqrt( dy1**2 + R1 ) ) / (dy2 + np.sqrt( dy2**2 + R1 ) ) ) +\
                        -np.log( ( dy1 + np.sqrt( dy1**2 + R2 ) ) / (dy2 + np.sqrt( dy2**2 + R2 ) ) ) +\
                        np.log( ( dy1 + np.sqrt( dy1**2 + R4 ) ) / (dy2 + np.sqrt( dy2**2 + R4 ) ) ) +\
                        -np.log( ( dy1 + np.sqrt( dy1**2 + R3 ) ) / (dy2 + np.sqrt( dy2**2 + R3 ) ) );

    Tz[0,2*mcell:] = -( Ty[0,mcell:2*mcell] + Tx[0,0:mcell] );
    Tz[0,mcell:2*mcell] = Ty[0,2*mcell:];
    Tx[0,mcell:2*mcell] = Ty[0,0:mcell];
    Tz[0,0:mcell] = Tx[0,2*mcell:];



    Tx = Tx/(4*np.pi);
    Ty = Ty/(4*np.pi);
    Tz = Tz/(4*np.pi);


    return Tx,Ty,Tz

def progress(iter,prog,final):
    """
    progress(iter,prog,final)

    Function measuring the progress of a process and print to screen the %.
    Useful to estimate the remaining runtime of a large problem.

    Created on Dec, 20th 2015

    @author: dominiquef
    """
    arg = np.floor(float(iter)/float(final)*10.);

    if  arg > prog:

        strg = "Done " + str(arg*10) + " %"
        print strg
        prog = arg;

    return prog

def dipazm_2_xyz(dip,azm_N):
    """
    dipazm_2_xyz(dip,azm_N)

    Function converting degree angles for dip and azimuth from north to a
    3-components in cartesian coordinates.

    INPUT
    dip     : Value or vector of dip from horizontal in DEGREE
    azm_N   : Value or vector of azimuth from north in DEGREE

    OUTPUT
    M       : [n-by-3] Array of xyz components of a unit vector in cartesian

    Created on Dec, 20th 2015

    @author: dominiquef
    """
    mcell = len(azm_N)

    M = np.zeros((mcell,3))

    # Modify azimuth from North to Cartesian-X
    azm_X = (450.- azm_N) % 360.

    D = np.deg2rad(dip)
    I = np.deg2rad(azm_X)

    M[:,0] = np.cos(D) * np.cos(I) ;
    M[:,1] = np.cos(D) * np.sin(I) ;
    M[:,2] = np.sin(D) ;

    return M

def get_dist_wgt(mesh,rxLoc,R,R0):
    """
    get_dist_wgt(xn,yn,zn,rxLoc,R,R0)

    Function creating a distance weighting function required for the magnetic
    inverse problem.

    INPUT
    xn, yn, zn : Node location
    rxLoc       : Observation locations [obsx, obsy, obsz]
    R           : Decay factor (mag=3, grav =2)
    R0          : Small factor added (default=dx/4)

    OUTPUT
    wr       : [mcell] Vector of distance weighting

    Created on Dec, 20th 2015

    @author: dominiquef
    """

    p = 1/np.sqrt(3);

    # Create cell center location
    Ym,Xm,Zm = np.meshgrid(mesh.vectorCCy, mesh.vectorCCx, mesh.vectorCCz)
    hY,hX,hZ = np.meshgrid(mesh.hy, mesh.hx, mesh.hz)

    V = np.reshape(mesh.vol,hY.shape)
    wr = np.zeros(hY.shape)

    ndata = rxLoc.shape[0]
    count = -1;
    print "Begin calculation of distance weighting for R= " + str(R)

    for dd in range(ndata):

        nx1 = (Xm - hX * p - rxLoc[dd,0])**2
        nx2 = (Xm + hX * p - rxLoc[dd,0])**2

        ny1 = (Ym - hY * p - rxLoc[dd,1])**2
        ny2 = (Ym + hY * p - rxLoc[dd,1])**2

        nz1 = (Zm - hZ * p - rxLoc[dd,2])**2
        nz2 = (Zm + hZ * p - rxLoc[dd,2])**2

        R1 = np.sqrt(nx1 + ny1 + nz1)
        R2 = np.sqrt(nx1 + ny1 + nz2)
        R3 = np.sqrt(nx2 + ny1 + nz1)
        R4 = np.sqrt(nx2 + ny1 + nz2)
        R5 = np.sqrt(nx1 + ny2 + nz1)
        R6 = np.sqrt(nx1 + ny2 + nz2)
        R7 = np.sqrt(nx2 + ny2 + nz1)
        R8 = np.sqrt(nx2 + ny2 + nz2)

        temp = (R1 + R0)**-R  + (R2 + R0)**-R  + (R3 + R0)**-R  + (R4 + R0)**-R  + (R5 + R0)**-R  + (R6 + R0)**-R  + (R7 + R0)**-R  + (R8 + R0)**-R

        wr = wr + (V*temp/8.)**2

        count = progress(dd,count,ndata)


    wr = np.sqrt(wr)/V
    wr = mkvc(wr)
    wr = np.sqrt(wr/(np.max(wr)))

    return wr

def writeUBCobs(filename,B,M,rxLoc,d,wd):
    """
    writeUBCobs(filename,B,M,rxLoc,d,wd)

    Function writing an observation file in UBC-MAG3D format.

    INPUT
    filename    : Name of out file including directory
    B           : Inducing field parameters [Inc, Decl, Intensity]
    M           : Magnetization orientation [Inc, Decl, dtype]
    rxLoc       : Observation locations [obsx, obsy, obsz]
    d           : Data vector
    wd          : Uncertainty vector

    OUTPUT
    Obsfile

    Created on Dec, 27th 2015

    @author: dominiquef
    """

    data = np.c_[rxLoc , d , wd]

    with file(filename,'w') as fid:
        fid.write('%6.2f %6.2f %6.2f\n' %(B[0], B[1], B[2]) )
        fid.write('%6.2f %6.2f %6.2f\n' %(M[0], M[1], 1) )
        fid.write('%i\n' %len(d) )
        np.savetxt(fid, data, fmt='%e',delimiter=' ',newline='\n')


    print "Observation file saved to: " + filename

def getActiveTopo(mesh,topo,flag):
    """
    getActiveTopo(mesh,topo)

    Function creates an active cell model from topography

    INPUT
    mesh        : Mesh in SimPEG format
    topo        : Scatter points defining topography [x,y,z]

    OUTPUT
    actv        : Active cell model

    Created on Dec, 27th 2015

    @author: dominiquef
    """
    import scipy.interpolate as interpolation

    if (flag=='N'):
        Zn      = np.zeros((mesh.nNx,mesh.nNy))
        # wght    = np.zeros((mesh.nNx,mesh.nNy))
        cx      = mesh.vectorNx
        cy      = mesh.vectorNy

    F = interpolation.NearestNDInterpolator(topo[:,0:2],topo[:,2])
    [Y,X] = np.meshgrid(cy,cx)

    Zn = F(X,Y)

    actv = np.zeros((mesh.nCx, mesh.nCy, mesh.nCz))

    if (flag=='N'):
        Nz = mesh.vectorNz[1:]


    for jj in range(mesh.nCy):

        for ii in range(mesh.nCx):

            temp = [kk for kk in range(len(Nz)) if np.all(Zn[ii:(ii+2),jj:(jj+2)] > Nz[kk]) ]
            actv[ii,jj,temp] = 1


    actv = mkvc(actv)

    return actv


