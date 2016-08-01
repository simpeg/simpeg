from SimPEG import *
import SimPEG.PF as PF


def run(plotIt=True):
    """
        PF: Magnetics: Inversion Linear
        ===============================

        Create a synthetic block model and invert
        with a compact norm

    """

    # First we need to define the direction of the inducing field
    # As a simple case, we pick a vertical inducing field of magnitude 50,000nT.
    # From old convention, field orientation is given as an azimuth from North
    # (positive clockwise) and dip from the horizontal (positive downward).
    H0 = (60000.,90.,0.)


    # Create a mesh
    dx    = 5.

    hxind = [(dx,5,-1.3), (dx, 10), (dx,5,1.3)]
    hyind = [(dx,5,-1.3), (dx, 10), (dx,5,1.3)]
    hzind = [(dx,5,-1.3),(dx, 10)]

    mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CCC')

    # Get index of the center
    midx = int(mesh.nCx/2)
    midy = int(mesh.nCy/2)

    # Lets create a simple Gaussian topo and set the active cells
    [xx,yy] = np.meshgrid(mesh.vectorNx,mesh.vectorNy)
    zz = -np.exp( ( xx**2 + yy**2 )/ 75**2 ) + mesh.vectorNz[-1]

    topo = np.c_[mkvc(xx),mkvc(yy),mkvc(zz)] # We would usually load a topofile

    actv = Utils.surface2ind_topo(mesh,topo,'N') # Go from topo to actv cells
    actv = np.asarray([inds for inds, elem in enumerate(actv, 1) if elem], dtype = int) - 1
    #nC   = mesh.nC
    #actv = np.asarray(range(mesh.nC))

    # Create active map to go from reduce space to full
    actvMap = Maps.ActiveCells(mesh, actv, -100)
    nC = len(actv)

    # Create and array of observation points
    xr = np.linspace(-20., 20., 20)
    yr = np.linspace(-20., 20., 20)
    X, Y = np.meshgrid(xr, yr)

    # Move the observation points 5m above the topo
    Z = -np.exp( ( X**2 + Y**2 )/ 75**2 ) + mesh.vectorNz[-1] + 5.

    # Create a MAGsurvey
    rxLoc = np.c_[Utils.mkvc(X.T), Utils.mkvc(Y.T), Utils.mkvc(Z.T)]
    rxLoc = PF.BaseMag.RxObs(rxLoc)
    srcField = PF.BaseMag.SrcField([rxLoc],param = H0)
    survey = PF.BaseMag.LinearSurvey(srcField)

    # We can now create a susceptibility model and generate data
    # Here a simple block in half-space
    model = np.zeros((mesh.nCx,mesh.nCy,mesh.nCz))
    model[(midx-2):(midx+2),(midy-2):(midy+2),-6:-2] = 0.02
    model = mkvc(model)
    model = model[actv]

    # Create active map to go from reduce set to full
    actvMap = Maps.InjectActiveCells(mesh, actv, -100)

    # Store the true model for later
    m_true = actvMap * model

    # Creat reduced identity map
    idenMap = Maps.IdentityMap(nP = nC)

    # Create the forward model operator
    prob = PF.Magnetics.MagneticIntegral(mesh, mapping = idenMap, actInd = actv)

    # Pair the survey and problem
    survey.pair(prob)

    # Compute linear forward operator and compute some data
    d = prob.fields(model)

    # Add noise and uncertainties
    data = d + np.random.randn(len(d)) # We add some random Gaussian noise (1nT)
    wd = np.ones(len(data))*1. # Assign flat uncertainties

    survey.dobs= data
    survey.std = wd
    survey.mtrue = model

    # Create sensitivity weights from our linear forward operator
    wr = np.sum(prob.G**2.,axis=0)**0.5
    wr = ( wr/np.max(wr) )

    # Create a regularization
    reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
    reg.cell_weights = wr

    # Data misfit function
    dmis = DataMisfit.l2_DataMisfit(survey)
    dmis.Wd = 1/wd

    # Add directives to the inversion
    opt = Optimization.ProjectedGNCG(maxIter=100 ,lower=0.,upper=1., maxIterLS = 20, maxIterCG= 10, tolCG = 1e-3)
    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
    betaest = Directives.BetaEstimate_ByEig()

    # Here is where the norms are applied
    # Use pick a treshold parameter empirically based on the distribution of model
    # parameters (run last cell to see the histogram before and after IRLS)
    IRLS = Directives.Update_IRLS( norms=([0,1,1,1]),  eps=None, f_min_change = 1e-3, minGNiter=3)
    update_Jacobi = Directives.Update_lin_PreCond()
    inv = Inversion.BaseInversion(invProb, directiveList=[IRLS,betaest,update_Jacobi])

    # Run the inversion
    m0 = np.ones(nC)*1e-4 # Starting model
    mrec = inv.run(m0)

    if plotIt:
        import matplotlib.pyplot as plt
        # Here is the recovered susceptibility model
        ypanel = midx
        zpanel = -4
        m_l2 = actvMap * reg.l2model
        m_l2[m_l2==-100] = np.nan

        m_lp = actvMap * mrec
        m_lp[m_lp==-100] = np.nan

        m_true = actvMap * model
        m_true[m_true==-100] = np.nan

        plt.figure()

        #Plot L2 model
        ax = plt.subplot(321)
        mesh.plotSlice(m_l2, ax = ax, normal = 'Z', ind=zpanel, grid=True, clim = (0., model.max()))
        plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([mesh.vectorCCy[ypanel],mesh.vectorCCy[ypanel]]),Color='w')
        plt.title('Plan l2-model.')
        plt.gca().set_aspect('equal')
        plt.ylabel('y')
        ax.xaxis.set_visible(False)
        plt.gca().set_aspect('equal', adjustable='box')

        # Vertica section
        ax = plt.subplot(322)
        mesh.plotSlice(m_l2, ax = ax, normal = 'Y', ind=midx, grid=True, clim = (0., model.max()))
        plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([mesh.vectorCCz[zpanel],mesh.vectorCCz[zpanel]]),Color='w')
        plt.title('E-W l2-model.')
        plt.gca().set_aspect('equal')
        ax.xaxis.set_visible(False)
        plt.ylabel('z')
        plt.gca().set_aspect('equal', adjustable='box')

        #Plot Lp model
        ax = plt.subplot(323)
        mesh.plotSlice(m_lp, ax = ax, normal = 'Z', ind=zpanel, grid=True, clim = (0., model.max()))
        plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([mesh.vectorCCy[ypanel],mesh.vectorCCy[ypanel]]),Color='w')
        plt.title('Plan lp-model.')
        plt.gca().set_aspect('equal')
        ax.xaxis.set_visible(False)
        plt.ylabel('y')
        plt.gca().set_aspect('equal', adjustable='box')


        # Vertical section
        ax = plt.subplot(324)
        mesh.plotSlice(m_lp, ax = ax, normal = 'Y', ind=midx, grid=True, clim = (0., model.max()))
        plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([mesh.vectorCCz[zpanel],mesh.vectorCCz[zpanel]]),Color='w')
        plt.title('E-W lp-model.')
        plt.gca().set_aspect('equal')
        ax.xaxis.set_visible(False)
        plt.ylabel('z')
        plt.gca().set_aspect('equal', adjustable='box')

        #Plot True model
        ax = plt.subplot(325)
        mesh.plotSlice(m_true, ax = ax, normal = 'Z', ind=zpanel, grid=True, clim = (0., model.max()))
        plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([mesh.vectorCCy[ypanel],mesh.vectorCCy[ypanel]]),Color='w')
        plt.title('Plan true model.')
        plt.gca().set_aspect('equal')
        plt.xlabel('x');plt.ylabel('y')
        plt.gca().set_aspect('equal', adjustable='box')


        # Vertical section
        ax = plt.subplot(326)
        mesh.plotSlice(m_true, ax = ax, normal = 'Y', ind=midx, grid=True, clim = (0., model.max()))
        plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([mesh.vectorCCz[zpanel],mesh.vectorCCz[zpanel]]),Color='w')
        plt.title('E-W true model.')
        plt.gca().set_aspect('equal')
        plt.xlabel('x');plt.ylabel('z')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

if __name__ == '__main__':
    run()
