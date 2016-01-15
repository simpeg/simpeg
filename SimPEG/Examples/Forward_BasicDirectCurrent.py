from SimPEG import Mesh, Utils, np, SolverLU

## 2D DC forward modeling example with Tensor and Curvilinear Meshes

def run(plotIt=True):
    # Step1: Generate Tensor and Curvilinear Mesh
    sz = [40,40]
    # Tensor Mesh
    tM = Mesh.TensorMesh(sz)
    # Curvilinear Mesh
    rM = Mesh.CurvilinearMesh(Utils.meshutils.exampleLrmGrid(sz,'rotate'))
    # Step2: Direct Current (DC) operator
    def DCfun(mesh, pts):
        D = mesh.faceDiv
        G = D.T
        sigma = 1e-2*np.ones(mesh.nC)
        Msigi = mesh.getFaceInnerProduct(1./sigma)
        MsigI = Utils.sdInv(Msigi)
        A = D*MsigI*G
        A[-1,-1] /= mesh.vol[-1] # Remove null space
        rhs = np.zeros(mesh.nC)
        txind = Utils.meshutils.closestPoints(mesh, pts)
        rhs[txind] = np.r_[1,-1]
        return A, rhs

    pts = np.vstack((np.r_[0.25, 0.5], np.r_[0.75, 0.5]))

    #Step3: Solve DC problem (LU solver)
    AtM, rhstM = DCfun(tM, pts)
    AinvtM = SolverLU(AtM)
    phitM = AinvtM*rhstM

    ArM, rhsrM = DCfun(rM, pts)
    AinvrM = SolverLU(ArM)
    phirM = AinvrM*rhsrM

    if not plotIt: return

    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.mlab import griddata

    #Step4: Making Figure
    fig, axes = plt.subplots(1,2,figsize=(12*1.2,4*1.2))
    label = ["(a)", "(b)"]
    opts = {}
    vmin, vmax = phitM.min(), phitM.max()
    dat = tM.plotImage(phitM, ax=axes[0], clim=(vmin, vmax), grid=True)

    #TODO: At the moment Curvilinear Mesh do not have plotimage

    Xi = tM.gridCC[:,0].reshape(sz[0], sz[1], order='F')
    Yi = tM.gridCC[:,1].reshape(sz[0], sz[1], order='F')
    PHIrM = griddata(rM.gridCC[:,0], rM.gridCC[:,1], phirM, Xi, Yi, interp='linear')
    axes[1].contourf(Xi, Yi, PHIrM, 100, vmin=vmin, vmax=vmax)

    cb = plt.colorbar(dat[0], ax=axes[0]); cb.set_label("Voltage (V)")
    cb = plt.colorbar(dat[0], ax=axes[1]); cb.set_label("Voltage (V)")

    tM.plotGrid(ax=axes[0], **opts)
    axes[0].set_title('TensorMesh')
    rM.plotGrid(ax=axes[1], **opts)
    axes[1].set_title('CurvilinearMesh')
    for i in range(2):
        axes[i].set_xlim(0.025, 0.975)
        axes[i].set_ylim(0.025, 0.975)
        axes[i].text(0., 1.0, label[i], fontsize=20)
        if i==0:
            axes[i].set_ylabel("y")
        else:
            axes[i].set_ylabel(" ")
        axes[i].set_xlabel("x")
    plt.show()


if __name__ == '__main__':
    run()
