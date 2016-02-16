from SimPEG import *
import SimPEG.DCIP as DC

def run(plotIt=False):
    cs = 25.
    hx = [(cs,7, -1.3),(cs,21),(cs,7, 1.3)]
    hy = [(cs,7, -1.3),(cs,21),(cs,7, 1.3)]
    hz = [(cs,7, -1.3),(cs,20)]
    mesh = Mesh.TensorMesh([hx, hy, hz], 'CCN')
    sighalf = 1e-2
    sigma = np.ones(mesh.nC)*sighalf
    xtemp = np.linspace(-150, 150, 21)
    ytemp = np.linspace(-150, 150, 21)
    xyz_rxP = Utils.ndgrid(xtemp-10., ytemp, np.r_[0.])
    xyz_rxN = Utils.ndgrid(xtemp+10., ytemp, np.r_[0.])
    xyz_rxM = Utils.ndgrid(xtemp, ytemp, np.r_[0.])

    # if plotIt:
    #     fig, ax = plt.subplots(1,1, figsize = (5,5))
    #     mesh.plotSlice(sigma, grid=True, ax = ax)
    #     ax.plot(xyz_rxP[:,0],xyz_rxP[:,1], 'w.')
    #     ax.plot(xyz_rxN[:,0],xyz_rxN[:,1], 'r.', ms = 3)

    rx = DC.RxDipole(xyz_rxP, xyz_rxN)
    src = DC.SrcDipole([rx], [-200, 0, -12.5], [+200, 0, -12.5])
    survey = DC.SurveyDC([src])
    problem = DC.ProblemDC_CC(mesh)
    problem.pair(survey)
    try:
        from pymatsolver import MumpsSolver
        problem.Solver = MumpsSolver
    except Exception, e:
        pass
    data = survey.dpred(sigma)

    def DChalf(srclocP, srclocN, rxloc, sigma, I=1.):
        rp = (srclocP.reshape([1,-1])).repeat(rxloc.shape[0], axis = 0)
        rn = (srclocN.reshape([1,-1])).repeat(rxloc.shape[0], axis = 0)
        rP = np.sqrt(((rxloc-rp)**2).sum(axis=1))
        rN = np.sqrt(((rxloc-rn)**2).sum(axis=1))
        return I/(sigma*2.*np.pi)*(1/rP-1/rN)

    data_anaP = DChalf(np.r_[-200, 0, 0.],np.r_[+200, 0, 0.], xyz_rxP, sighalf)
    data_anaN = DChalf(np.r_[-200, 0, 0.],np.r_[+200, 0, 0.], xyz_rxN, sighalf)
    data_ana = data_anaP-data_anaN
    Data_ana = data_ana.reshape((21, 21), order = 'F')
    Data = data.reshape((21, 21), order = 'F')
    X = xyz_rxM[:,0].reshape((21, 21), order = 'F')
    Y = xyz_rxM[:,1].reshape((21, 21), order = 'F')

    if plotIt:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,2, figsize = (12, 5))
        vmin = np.r_[data, data_ana].min()
        vmax = np.r_[data, data_ana].max()
        dat1 = ax[1].contourf(X, Y, Data, 60, vmin = vmin, vmax = vmax)
        dat0 = ax[0].contourf(X, Y, Data_ana, 60, vmin = vmin, vmax = vmax)
        cb0 = plt.colorbar(dat1, orientation = 'horizontal', ax = ax[0])
        cb1 = plt.colorbar(dat1, orientation = 'horizontal', ax = ax[1])
        ax[1].set_title('Analytic')
        ax[0].set_title('Computed')
        plt.show()

    return np.linalg.norm(data-data_ana)/np.linalg.norm(data_ana)


if __name__ == '__main__':
    print run(plotIt=True)
