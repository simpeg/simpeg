"""
Predict Response from a Conductive and Magnetically Viscous Earth
=================================================================

Here, we predict the vertical db/dt response over a conductive and
magnetically viscous Earth for a small coincident loop system. Following
the theory, the total response is approximately equal to the sum of the
inductive and VRM responses modelled separately. The SimPEG.VRM module is
used to model the VRM response while an analytic solution for a conductive
half-space is used to model the inductive response.
"""
import SimPEG.VRM as VRM
import numpy as np
from SimPEG import mkvc, Mesh, Maps, DataMisfit, Directives, Optimization, Regularization, InvProblem, Inversion
import matplotlib.pyplot as plt


def run(plotIt=True):

    # CREATE MESH
    cs, ncx, ncy, ncz, npad = 2., 35, 35, 20, 5
    hx = [(cs, npad, -1.3), (cs, ncx), (cs, npad, 1.3)]
    hy = [(cs, npad, -1.3), (cs, ncy), (cs, npad, 1.3)]
    hz = [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
    mesh = Mesh.TensorMesh([hx, hy, hz], 'CCC')

    # SET MAPPING AND ACTIVE CELLS
    topoCells = mesh.gridCC[:, 2] < 0.

    # CREATE MODEL (XI: THE AMALGAMATED MAGNETIC PROPERTY)
    xyzc = mesh.gridCC[topoCells, :]
    C = 2*np.pi*8**2
    xi_true = (
        4e-4*np.exp(-(xyzc[:, 0]-50)**2/(3*C))*np.exp(-(xyzc[:, 1])**2/(20*C))*np.exp(-(xyzc[:, 2])**2/C) +
        4e-4*np.exp(-(xyzc[:, 0]+50)**2/(3*C))*np.exp(-(xyzc[:, 1])**2/(20*C))*np.exp(-(xyzc[:, 2])**2/C) +
        4e-4*np.exp(-(xyzc[:, 0]+40)**2/(3*C))*np.exp(-(xyzc[:, 1]-40)**2/C)*np.exp(-(xyzc[:, 2])**2/C) +
        6e-4*np.exp(-(xyzc[:, 0]+20)**2/C)*np.exp(-(xyzc[:, 1]-10)**2/C)*np.exp(-(xyzc[:, 2])**2/C) +
        8e-4*np.exp(-(xyzc[:, 0]+15)**2/(3*C))*np.exp(-(xyzc[:, 1]+20)**2/(0.4*C))*np.exp(-(xyzc[:, 2])**2/C) +
        6e-4*np.exp(-(xyzc[:, 0]-20)**2/(0.5*C))*np.exp(-(xyzc[:, 1]-15)**2/(0.5*C))*np.exp(-(xyzc[:, 2])**2/C) +
        8e-4*np.exp(-(xyzc[:, 0]+10)**2/(0.1*C))*np.exp(-(xyzc[:, 1])**2/(0.1*C))*np.exp(-(xyzc[:, 2])**2/C) +
        8e-4*np.exp(-(xyzc[:, 0]-25)**2/(0.1*C))*np.exp(-(xyzc[:, 1])**2/(0.4*C))*np.exp(-(xyzc[:, 2])**2/C) +
        1e-5
        )

    # SET THE TRANSMITTER WAVEFORM
    waveform = VRM.WaveformVRM.StepOff()

    # CREATE SURVEY
    times = np.logspace(-5, -2, 31)  # Observation times
    x, y = np.meshgrid(np.linspace(-30, 30, 21), np.linspace(-30, 30, 21))
    z = 0.5*np.ones(x.shape)
    loc = np.c_[mkvc(x), mkvc(y), mkvc(z)]  # Src and Rx Locations

    srcListVRM = []

    for pp in range(0, loc.shape[0]):

        loc_pp = np.reshape(loc[pp, :], (1, 3))
        rxListVRM = [VRM.Rx.Point(loc_pp, times, 'dbdt', 'z')]

        srcListVRM.append(VRM.Src.MagDipole(rxListVRM, mkvc(loc[pp, :]), [0., 0., 0.01], waveform))

    SurveyVRM = VRM.Survey(srcListVRM)

    # DEFINE THE VRM PROBLEM
    ProblemVRM = VRM.Problem_Linear(mesh, indActive=topoCells, refFact=3, refRadius=[1.25, 2.5, 3.75])
    ProblemVRM.pair(SurveyVRM)

    # PREDICT THE FIELDS
    FieldsVRM = SurveyVRM.dpred(xi_true)

    n_times = len(times)
    n_loc = loc.shape[0]

    sig = 1e-1
    mu0 = 4*np.pi*1e-7
    FieldsTEM = -sig**1.5*mu0**2.5*times**-2.5/(20*np.pi**1.5)
    FieldsTEM = np.kron(np.ones(n_loc), FieldsTEM)
    C = (
       np.exp(-(loc[:, 0]-10)**2/(25**2))*np.exp(-(loc[:, 1]-20)**2/(35**2)) +
       np.exp(-(loc[:, 0]+20)**2/(20**2))*np.exp(-(loc[:, 1]+20)**2/(40**2)) +
       1.5*np.exp(-(loc[:, 0]-25)**2/(10**2))*np.exp(-(loc[:, 1]+25)**2/(10**2)) +
       0.25
       )

    C = np.kron(C, np.ones(n_times))
    FieldsTEM = C*FieldsTEM

    # TOTAL OBSERCED FIELD WITH NOISE
    FieldsTOT = FieldsTEM + FieldsVRM
    FieldsTOT = FieldsTOT + 0.05*np.abs(FieldsTOT)*np.random.normal(size=FieldsTOT.shape)

    ##########################################
    # INVERT LATE TIMES

    # CREATE NEW PROBLEM
    SurveyINV = VRM.Survey(srcListVRM)
    actCells = (mesh.gridCC[:, 2] < 0.) & (mesh.gridCC[:, 2] > -2.)
    ProblemINV = VRM.Problem_Linear(mesh, indActive=actCells, refFact=3, refRadius=[1.25, 2.5, 3.75])
    ProblemINV.pair(SurveyINV)
    SurveyINV.ActiveTimeInterval = [1e-3, 1e-2]
    SurveyINV.dobs = FieldsTOT[SurveyINV.tActive]
    SurveyINV.std = 0.05*np.abs(FieldsTOT[SurveyINV.tActive])
    SurveyINV.eps = 1e-11

    # SET INVERSION
    dmis = DataMisfit.l2_DataMisfit(SurveyINV)
    reg = Regularization.Simple(mesh=mesh, indActive=actCells, alpha_s=1)
    opt = Optimization.ProjectedGNCG(maxIter=20, lower=0., upper=1e-2, maxIterLS=20, tolCG=1e-4)
    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
    directives = [
        Directives.BetaSchedule(coolingFactor=2, coolingRate=1),
        Directives.TargetMisfit()
    ]
    inv = Inversion.BaseInversion(invProb, directiveList=directives)

    xi_0 = 1e-6*np.ones(actCells.sum())
    xi_rec = inv.run(xi_0)

    ############################################
    # REMOVE VRM RESPONSE

    SurveyINV.ActiveTimeInterval = [0., 1.]
    FieldsPRE = SurveyINV.dpred(xi_rec)

    ################################
    # PLOTTING

    FieldsTOT = np.reshape(FieldsTOT, (n_loc, n_times))
    FieldsVRM = np.reshape(FieldsVRM, (n_loc, n_times))
    FieldsTEM = np.reshape(FieldsTEM, (n_loc, n_times))
    FieldsPRE = np.reshape(FieldsPRE, (n_loc, n_times))

    Fig = plt.figure(figsize=(11, 11))
    Ax12 = Fig.add_axes([0.22, 0.66, 0.26, 0.25])
    Ax11 = Fig.add_axes([0.58, 0.66, 0.30, 0.25])

    Ax21 = Fig.add_axes([0.1, 0.33, 0.4, 0.25])
    Ax22 = Fig.add_axes([0.6, 0.33, 0.4, 0.25])

    Ax31 = Fig.add_axes([0.05, 0.05, 0.25, 0.21])
    Ax32 = Fig.add_axes([0.4, 0.05, 0.25, 0.21])
    Ax33 = Fig.add_axes([0.75, 0.05, 0.25, 0.21])
    FS = 12
    N = x.shape[0]

    # PLOT MODEL
    invMap = Maps.InjectActiveCells(mesh, actCells, 0.)  # Maps to mesh
    topoMap = Maps.InjectActiveCells(mesh, topoCells, 0.)

    Cplot11 = mesh.plotSlice(invMap*xi_rec, ind=int((ncz+2*npad)/2-1), ax=Ax11, grid=True, pcolorOpts={'cmap': 'gist_heat_r'})
    cbar11 = plt.colorbar(Cplot11[0], ax=Ax11, pad=0.02, format='%.2e')
    cbar11.set_label('[SI]', rotation=270, labelpad=15, size=FS)
    cbar11.set_clim((0., np.max(xi_true)))
    cbar11.ax.tick_params(labelsize=FS-2)
    Ax11.set_xlabel('X [m]', fontsize=FS)
    Ax11.set_ylabel('Y [m]', fontsize=FS, labelpad=-10)
    Ax11.tick_params(labelsize=FS-2)
    titlestr11 = "Recovered Surface Model"
    Ax11.set_title(titlestr11, fontsize=FS+2)

    Cplot12 = mesh.plotSlice(topoMap*xi_true, ind=int((ncz+2*npad)/2-1), ax=Ax12, grid=True, pcolorOpts={'cmap': 'gist_heat_r'})
    Cplot12[0].set_clim((0., np.max(xi_true)))
    Ax12.set_xlabel('X [m]', fontsize=FS)
    Ax12.set_ylabel('Y [m]', fontsize=FS, labelpad=-10)
    Ax12.tick_params(labelsize=FS-2)
    titlestr12 = "True Model (z = 0 m)"
    Ax12.set_title(titlestr12, fontsize=FS+2)

    # PLOT DECAY
    j1 = int((N**2-1)/2 - 3*N)
    di_tot = mkvc(np.abs(FieldsTOT[j1, :]))
    di_pre = mkvc(np.abs(FieldsVRM[j1, :]))
    di_tem = mkvc(np.abs(FieldsTEM[j1, :]))
    Ax21.loglog(times, di_tot, 'k.-')
    Ax21.loglog(times, di_tem, 'r.-')
    Ax21.loglog(times, di_pre, 'b.-')
    Ax21.loglog(times, np.abs(di_tot-di_pre), 'g.-')
    Ax21.set_xlabel('t [s]', fontsize=FS, labelpad=-10)
    Ax21.set_ylabel('|dBz/dt| [T/s]', fontsize=FS)
    Ax21.tick_params(labelsize=FS-2)
    Ax21.set_xbound(np.min(times), np.max(times))
    Ax21.set_ybound(1.2*np.max(di_tot),1e-5*np.max(di_tot))
    titlestr21 = "Decay at X = " + '{:.2f}'.format(loc[j1, 0]) + " m and Y = " + '{:.2f}'.format(loc[j1, 1]) + " m"
    Ax21.set_title(titlestr21, fontsize=FS+2)
    Ax21.text(1.2e-5, 54*np.max(di_tot)/1e5, "Observed", fontsize=FS, color='k')
    Ax21.text(1.2e-5, 18*np.max(di_tot)/1e5, "True TEM", fontsize=FS, color='r')
    Ax21.text(1.2e-5, 6*np.max(di_tot)/1e5, "Predicted VRM", fontsize=FS, color='b')
    Ax21.text(1.2e-5, 2*np.max(di_tot)/1e5, "Recovered TEM", fontsize=FS, color='g')

    j2 = int((N**2-1)/2 + 3*N)
    di_tot = mkvc(np.abs(FieldsTOT[j1, :]))
    di_pre = mkvc(np.abs(FieldsVRM[j1, :]))
    di_tem = mkvc(np.abs(FieldsTEM[j1, :]))
    Ax22.loglog(times, di_tot, 'k.-')
    Ax22.loglog(times, di_tem, 'r.-')
    Ax22.loglog(times, di_pre, 'b.-')
    Ax22.loglog(times, np.abs(di_tot-di_pre), 'g.-')
    Ax22.set_xlabel('t [s]', fontsize=FS, labelpad=-10)
    Ax22.set_ylabel('|dBz/dt| [T/s]', fontsize=FS)
    Ax22.tick_params(labelsize=FS-2)
    Ax22.set_xbound(np.min(times), np.max(times))
    Ax22.set_ybound(1.2*np.max(di_tot),1e-5*np.max(di_tot))
    titlestr22 = "Decay at X = " + '{:.2f}'.format(loc[j2, 0]) + " m and Y = " + '{:.2f}'.format(loc[j1, 1]) + " m"
    Ax22.set_title(titlestr22, fontsize=FS+2)

    # PLOT ANOMALIES
    d2 = np.reshape(np.abs(FieldsTOT[:, 10]), (N, N))
    d3 = np.reshape(np.abs(FieldsTEM[:, 10]), (N, N))
    d4 = np.reshape(np.abs(FieldsTOT[:, 10]-FieldsPRE[:, 10]), (N, N))

    Cplot31 = Ax31.contourf(x, y, d2.T, 40, cmap='magma_r')
    cbar31 = plt.colorbar(Cplot31, ax=Ax31, pad=0.02, format='%.2e')
    cbar31.set_label('[T/s]', rotation=270, labelpad=12, size=FS)
    cbar31.ax.tick_params(labelsize=FS-2)
    Ax31.set_xlabel('X [m]', fontsize=FS)
    Ax31.set_ylabel('Y [m]', fontsize=FS, labelpad=-12)
    Ax31.tick_params(labelsize=FS-2)
    Ax31.scatter(x, y, color=(0, 0, 0), s=4)
    Ax31.set_xbound(np.min(x), np.max(x))
    Ax31.set_ybound(np.min(y), np.max(y))
    titlestr31 = "Obs at t=" + '{:.1e}'.format(times[10]) + " s"
    Ax31.set_title(titlestr31, fontsize=FS+2)

    Cplot32 = Ax32.contourf(x, y, d3.T, 40, cmap='magma_r')
    cbar32 = plt.colorbar(Cplot32, ax=Ax32, pad=0.02, format='%.2e')
    cbar32.set_label('[T/s]', rotation=270, labelpad=12, size=FS)
    cbar32.ax.tick_params(labelsize=FS-2)
    Ax32.set_xlabel('X [m]', fontsize=FS)
    Ax32.set_ylabel('Y [m]', fontsize=FS, labelpad=-12)
    Ax32.tick_params(labelsize=FS-2)
    Ax32.set_xbound(np.min(x), np.max(x))
    Ax32.set_ybound(np.min(y), np.max(y))
    titlestr32 = "True TEM at t=" + '{:.1e}'.format(times[10]) + " s"
    Ax32.set_title(titlestr32, fontsize=FS+2)

    Cplot33 = Ax33.contourf(x, y, d4.T, 40, cmap='magma_r')
    cbar33 = plt.colorbar(Cplot33, ax=Ax33, pad=0.02, format='%.2e')
    cbar33.set_label('[T/s]', rotation=270, labelpad=12, size=FS)
    cbar33.ax.tick_params(labelsize=FS-2)
    Ax33.set_xlabel('X [m]', fontsize=FS)
    Ax33.set_ylabel('Y [m]', fontsize=FS, labelpad=-12)
    Ax33.tick_params(labelsize=FS-2)
    Ax33.set_xbound(np.min(x), np.max(x))
    Ax33.set_ybound(np.min(y), np.max(y))
    titlestr33 = "Rec TEM at t=" + '{:.1e}'.format(times[10]) + " s"
    Ax33.set_title(titlestr33, fontsize=FS+2)

if __name__ == '__main__':
    run()
    plt.show()
