from SimPEG import *
import simpegDC as DC
import matplotlib.pyplot as plt

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

    rx = DC.DipoleRx(xyz_rxP, xyz_rxN)
    tx = DC.DipoleTx([-200, 0, -12.5],[+200, 0, -12.5], [rx])
    survey = DC.SurveyDC([tx])
    problem = DC.ProblemDC(mesh)
    problem.pair(survey)
    data = survey.dpred(sigma)

    def DChalf(txlocP, txlocN, rxloc, sigma, I=1.):
        rp = (txlocP.reshape([1,-1])).repeat(rxloc.shape[0], axis = 0)
        rn = (txlocN.reshape([1,-1])).repeat(rxloc.shape[0], axis = 0)
        rP = np.sqrt(((rxloc-rp)**2).sum(axis=1))
        rN = np.sqrt(((rxloc-rn)**2).sum(axis=1))
        return I/(sigma*2.*np.pi)*(1/rP-1/rN)

    data_analP = DChalf(np.r_[-200, 0, 0.],np.r_[+200, 0, 0.], xyz_rxP, sighalf)
    data_analN = DChalf(np.r_[-200, 0, 0.],np.r_[+200, 0, 0.], xyz_rxN, sighalf)
    data_anal = data_analP-data_analN
    Data_anal = data_anal.reshape((21, 21), order = 'F')
    Data = data.reshape((21, 21), order = 'F')
    X = xyz_rxM[:,0].reshape((21, 21), order = 'F')
    Y = xyz_rxM[:,1].reshape((21, 21), order = 'F')

    if plotIt:
        fig, ax = plt.subplots(1,2, figsize = (12, 5))
        vmin = np.r_[data, data_anal].min()
        vmax = np.r_[data, data_anal].max()
        dat1 = ax[1].contourf(X, Y, Data, 60, vmin = vmin, vmax = vmax)
        dat0 = ax[0].contourf(X, Y, Data_anal, 60, vmin = vmin, vmax = vmax)
        cb0 = plt.colorbar(dat1, orientation = 'horizontal', ax = ax[0])
        cb1 = plt.colorbar(dat1, orientation = 'horizontal', ax = ax[1])
        ax[1].set_title('Analytic')
        ax[0].set_title('Computed')
        plt.show()

    return np.linalg.norm(data-data_anal)/np.linalg.norm(data_anal)


if __name__ == '__main__':
    print run(plotIt=True)
