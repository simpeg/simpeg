"""
Method of Equivalent Sources for Removing VRM Responses
=======================================================

Here, we use an equivalent source inversion to remove the VRM response from TEM
data collected by a small coincident loop system. The data being inverted are
the same as in the forward modeling example. To remove the VRM signal we:

    1. invert the late time data to recover an equivalent source surface layer of cells.
    2. use the recovered model to predict the VRM response at all times
    3. subtract the predicted VRM response from the observed data
"""
import SimPEG.VRM as VRM
import numpy as np
from SimPEG import (
    mkvc, Mesh, Maps, DataMisfit, Directives, Optimization, Regularization,
    InvProblem, Inversion
    )
import matplotlib.pyplot as plt
import matplotlib as mpl


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
    pc = np.r_[4e-4, 4e-4, 4e-4, 6e-4, 8e-4, 6e-4, 8e-4, 8e-4]
    x_0 = np.r_[50., -50., -40., -20., -15., 20., -10., 25.]
    y_0 = np.r_[0., 0., 40., 10., -20., 15., 0., 0.]
    z_0 = np.r_[0., 0., 0., 0., 0., 0., 0., 0.]
    var_x = C*np.r_[3., 3., 3., 1., 3., 0.5, 0.1, 0.1]
    var_y = C*np.r_[20., 20., 1., 1., 0.4, 0.5, 0.1, 0.4]
    var_z = C*np.r_[1., 1., 1., 1., 1., 1., 1., 1.]

    xi_true = np.zeros(np.shape(xyzc[:, 0]))

    for ii in range(0, 8):
        xi_true += (
            pc[ii]*np.exp(-(xyzc[:, 0]-x_0[ii])**2/var_x[ii]) *
            np.exp(-(xyzc[:, 1]-y_0[ii])**2/var_y[ii]) *
            np.exp(-(xyzc[:, 2]-z_0[ii])**2/var_z[ii])
            )

    xi_true += 1e-5

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
        rxListVRM = [VRM.Rx.Point(loc_pp, times=times, fieldType='dbdt', fieldComp='z')]

        srcListVRM.append(
            VRM.Src.MagDipole(rxListVRM, mkvc(loc[pp, :]), [0., 0., 0.01], waveform))

    SurveyVRM = VRM.Survey(srcListVRM)

    # DEFINE THE VRM PROBLEM
    ProblemVRM = VRM.Problem_Linear(
        mesh, indActive=topoCells, ref_factor=3, ref_radius=[1.25, 2.5, 3.75])
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

    FieldsTOT = FieldsTEM + FieldsVRM
    FieldsTOT = FieldsTOT + 0.05*np.abs(FieldsTOT)*np.random.normal(size=FieldsTOT.shape)

    ##########################################
    # INVERT LATE TIMES

    # CREATE NEW PROBLEM
    SurveyINV = VRM.Survey(srcListVRM)
    actCells = (mesh.gridCC[:, 2] < 0.) & (mesh.gridCC[:, 2] > -2.)
    ProblemINV = VRM.Problem_Linear(
        mesh, indActive=actCells, ref_factor=3, ref_radius=[1.25, 2.5, 3.75])
    ProblemINV.pair(SurveyINV)
    SurveyINV.set_active_interval(1e-3, 1e-2)
    SurveyINV.dobs = FieldsTOT[SurveyINV.t_active]
    SurveyINV.std = 0.05*np.abs(FieldsTOT[SurveyINV.t_active])
    SurveyINV.eps = 1e-11

    # SET INVERSION
    dmis = DataMisfit.l2_DataMisfit(SurveyINV)
    W = mkvc((np.sum(np.array(ProblemINV.A)**2, axis=0)))**0.5
    W = W/np.max(W)
    reg = Regularization.Simple(mesh=mesh, indActive=actCells, alpha_s=0.25,  cell_weights=W)
    opt = Optimization.ProjectedGNCG(maxIter=20, lower=0., upper=1e-2, maxIterLS=20, tolCG=1e-4)
    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
    directives = [
        Directives.BetaSchedule(coolingFactor=2, coolingRate=1),
        Directives.TargetMisfit()
    ]
    inv = Inversion.BaseInversion(invProb, directiveList=directives)

    xi_0 = 1e-3*np.ones(actCells.sum())
    xi_rec = inv.run(xi_0)

    ############################################
    # REMOVE VRM RESPONSE

    SurveyINV.set_active_interval(0., 1.)
    FieldsPRE = SurveyINV.dpred(xi_rec)

    ################################
    # PLOTTING

    FieldsTOT = np.reshape(FieldsTOT, (n_loc, n_times))
    FieldsVRM = np.reshape(FieldsVRM, (n_loc, n_times))
    FieldsTEM = np.reshape(FieldsTEM, (n_loc, n_times))
    FieldsPRE = np.reshape(FieldsPRE, (n_loc, n_times))

    # PLOT MODEL
    if plotIt:

        Fig = plt.figure(figsize=(10, 10))
        FS = 12
        
        # PLOT MODELS
        invMap = Maps.InjectActiveCells(mesh, actCells, 0.)  # Maps to mesh
        topoMap = Maps.InjectActiveCells(mesh, topoCells, 0.)
        MAX = np.max(np.r_[xi_true, xi_rec])
        Ax1 = 3*[None]
        Cplot1 = 2*[None]
        XI = [xi_true, xi_rec]
        MAP = [topoMap, invMap]
        titlestr1 = [
            "True Model (z = 0 m)",
            "Equivalent Source Model"
            ]
        
        for qq in range(0,2):
            Ax1[qq] = Fig.add_axes([0.15+0.35*qq, 0.7, 0.25, 0.25])
            Cplot1[qq] = mesh.plotSlice(
                MAP[qq]*XI[qq], ind=int((ncz+2*npad)/2-1),
                ax=Ax1[qq], grid=True, pcolorOpts={'cmap': 'gist_heat_r'})
            Cplot1[qq][0].set_clim((0., MAX))
            Ax1[qq].set_xlabel('X [m]', fontsize=FS)
            Ax1[qq].set_ylabel('Y [m]', fontsize=FS, labelpad=-5)
            Ax1[qq].tick_params(labelsize=FS-2)
            Ax1[qq].set_title(titlestr1[qq], fontsize=FS+2)
        
        Ax1[2] = Fig.add_axes([0.78, 0.7, 0.01, 0.25])
        norm = mpl.colors.Normalize(vmin=0., vmax=MAX)
        cbar14 = mpl.colorbar.ColorbarBase(
            Ax1[2], cmap='gist_heat_r', norm=norm, orientation='vertical')
        cbar14.set_label(
            '$\Delta \chi /$ln$(\lambda_2 / \lambda_1 )$ [SI]',
            rotation=270, labelpad=15, size=FS)
        
        # PLOT DECAYS
        N = x.shape[0]
        Ax2 = 2*[None]
        for qq in range(0,2):
            Ax2[qq] = Fig.add_axes([0.1+0.45*qq, 0.36, 0.35, 0.26])
            k = int((N**2-1)/2 - 3*N*(-1)**qq)
            di_tot = mkvc(np.abs(FieldsTOT[k, :]))
            di_pre = mkvc(np.abs(FieldsVRM[k, :]))
            di_tem = mkvc(np.abs(FieldsTEM[k, :]))
            Ax2[qq].loglog(times, di_tot, 'k.-')
            Ax2[qq].loglog(times, di_tem, 'r.-')
            Ax2[qq].loglog(times, di_pre, 'b.-')
            Ax2[qq].loglog(times, np.abs(di_tot-di_pre), 'g.-')
            Ax2[qq].set_xlabel('t [s]', fontsize=FS, labelpad=-10)
            if qq == 0:
                Ax2[qq].set_ylabel('|dBz/dt| [T/s]', fontsize=FS)
            else:
                Ax2[qq].axes.get_yaxis().set_visible(False)
            Ax2[qq].tick_params(labelsize=FS-2)
            Ax2[qq].set_xbound(np.min(times), np.max(times))
            Ax2[qq].set_ybound(1.2*np.max(di_tot), 1e-5*np.max(di_tot))
            titlestr2 = (
                "Decay at X = " + '{:.2f}'.format(loc[k, 0]) +
                " m and Y = " + '{:.2f}'.format(loc[k, 1]) + " m")
            Ax2[qq].set_title(titlestr2, fontsize=FS+2)
            if qq == 0:
                Ax2[qq].text(1.2e-5, 54*np.max(di_tot)/1e5, "Observed", fontsize=FS, color='k')
                Ax2[qq].text(1.2e-5, 18*np.max(di_tot)/1e5, "True TEM", fontsize=FS, color='r')
                Ax2[qq].text(1.2e-5, 6*np.max(di_tot)/1e5, "Predicted VRM", fontsize=FS, color='b')
                Ax2[qq].text(1.2e-5, 2*np.max(di_tot)/1e5, "Recovered TEM", fontsize=FS, color='g')
        
        # PLOT ANOMALIES
        d = [
            np.reshape(np.abs(FieldsTOT[:, 10]), (N, N)),
            np.reshape(np.abs(FieldsTEM[:, 10]), (N, N)),
            np.reshape(np.abs(FieldsTOT[:, 10]-FieldsPRE[:, 10]), (N, N))
            ]

        MIN = np.min(np.r_[d[0], d[1], d[2]])
        MAX = np.max(np.r_[d[0], d[1], d[2]])
        
        Ax3 = 4*[None]
        Cplot3 = 3*[None]
        STR = [
            "Observed at t=",
            "True TEM at t=",
            "Recov. TEM at t="
            ]
        
        for qq in range(0,3):
            Ax3[qq] = Fig.add_axes([0.07+0.28*qq, 0.05, 0.24, 0.24])
            Cplot3[qq] = Ax3[qq].contourf(x, y, d[qq].T, 40, cmap='magma_r')
            Ax3[qq].set_xticks(np.linspace(-30, 30, 7))
            Ax3[qq].set_xlabel('X [m]', fontsize=FS)
            if qq == 0:
                Ax3[qq].scatter(x, y, color=(0, 0, 0), s=4)
                Ax3[qq].set_ylabel('Y [m]', fontsize=FS, labelpad=-12)
            else:
                Ax3[qq].axes.get_yaxis().set_visible(False)
            Ax3[qq].tick_params(labelsize=FS-2)
            Ax3[qq].set_xbound(np.min(x), np.max(x))
            Ax3[qq].set_ybound(np.min(y), np.max(y))
            titlestr3 = STR[qq] + '{:.1e}'.format(times[10]) + " s"
            Ax3[qq].set_title(titlestr3, fontsize=FS+2)
        
        Ax3[3] = Fig.add_axes([0.88, 0.05, 0.01, 0.24])
        norm = mpl.colors.Normalize(vmin=MIN, vmax=MAX)
        cbar34 = mpl.colorbar.ColorbarBase(
            Ax3[3], cmap='magma_r', norm=norm, orientation='vertical', format='%.1e')
        cbar34.set_label('dBz/dt [T/s]', rotation=270, size=FS, labelpad=15)
        
        

#        Ax11 = Fig.add_axes([0.15, 0.7, 0.25, 0.25])
#        Ax12 = Fig.add_axes([0.5, 0.7, 0.25, 0.25])
#        Ax13 = Fig.add_axes([0.78, 0.7, 0.01, 0.25])
#
#        Ax21 = Fig.add_axes([0.1, 0.36, 0.35, 0.26])
#        Ax22 = Fig.add_axes([0.55, 0.36, 0.35, 0.26])
#
#        Ax31 = Fig.add_axes([0.07, 0.05, 0.24, 0.24])
#        Ax32 = Fig.add_axes([0.35, 0.05, 0.24, 0.24])
#        Ax33 = Fig.add_axes([0.63, 0.05, 0.24, 0.24])
#        Ax34 = Fig.add_axes([0.88, 0.05, 0.01, 0.24])

        

#        invMap = Maps.InjectActiveCells(mesh, actCells, 0.)  # Maps to mesh
#        topoMap = Maps.InjectActiveCells(mesh, topoCells, 0.)
#
#        MAX = np.max(np.r_[xi_true, xi_rec])
#        Cplot11 = mesh.plotSlice(
#            topoMap*xi_true, ind=int((ncz+2*npad)/2-1),
#            ax=Ax11, grid=True, pcolorOpts={'cmap': 'gist_heat_r'})
#        Cplot11[0].set_clim((0., MAX))
#        Ax11.set_xlabel('X [m]', fontsize=FS)
#        Ax11.set_ylabel('Y [m]', fontsize=FS, labelpad=-5)
#        Ax11.tick_params(labelsize=FS-2)
#        titlestr11 = "True Model (z = 0 m)"
#        Ax11.set_title(titlestr11, fontsize=FS+2)
#
#        Cplot12 = mesh.plotSlice(
#            invMap*xi_rec, ind=int((ncz+2*npad)/2-1),
#            ax=Ax12, grid=True, pcolorOpts={'cmap': 'gist_heat_r'})
#        Cplot12[0].set_clim((0., MAX))
#        Ax12.set_xlabel('X [m]', fontsize=FS)
#        Ax12.axes.get_yaxis().set_visible(False)
#        Ax12.tick_params(labelsize=FS-2)
#        titlestr12 = "Equivalent Source Model"
#        Ax12.set_title(titlestr12, fontsize=FS+2)
#
#        norm = mpl.colors.Normalize(vmin=0., vmax=MAX)
#        cbar14 = mpl.colorbar.ColorbarBase(
#            Ax13, cmap='gist_heat_r', norm=norm, orientation='vertical')
#        cbar14.set_label(
#            '$\Delta \chi /$ln$(\lambda_2 / \lambda_1 )$ [SI]',
#            rotation=270, labelpad=15, size=FS)

        
#        # PLOT DECAY
#        j1 = int((N**2-1)/2 - 3*N)
#        di_tot = mkvc(np.abs(FieldsTOT[j1, :]))
#        di_pre = mkvc(np.abs(FieldsVRM[j1, :]))
#        di_tem = mkvc(np.abs(FieldsTEM[j1, :]))
#        Ax21.loglog(times, di_tot, 'k.-')
#        Ax21.loglog(times, di_tem, 'r.-')
#        Ax21.loglog(times, di_pre, 'b.-')
#        Ax21.loglog(times, np.abs(di_tot-di_pre), 'g.-')
#        Ax21.set_xlabel('t [s]', fontsize=FS, labelpad=-10)
#        Ax21.set_ylabel('|dBz/dt| [T/s]', fontsize=FS)
#        Ax21.tick_params(labelsize=FS-2)
#        Ax21.set_xbound(np.min(times), np.max(times))
#        Ax21.set_ybound(1.2*np.max(di_tot), 1e-5*np.max(di_tot))
#        titlestr21 = (
#            "Decay at X = " + '{:.2f}'.format(loc[j1, 0]) +
#            " m and Y = " + '{:.2f}'.format(loc[j1, 1]) + " m")
#        Ax21.set_title(titlestr21, fontsize=FS+2)
#        Ax21.text(1.2e-5, 54*np.max(di_tot)/1e5, "Observed", fontsize=FS, color='k')
#        Ax21.text(1.2e-5, 18*np.max(di_tot)/1e5, "True TEM", fontsize=FS, color='r')
#        Ax21.text(1.2e-5, 6*np.max(di_tot)/1e5, "Predicted VRM", fontsize=FS, color='b')
#        Ax21.text(1.2e-5, 2*np.max(di_tot)/1e5, "Recovered TEM", fontsize=FS, color='g')
#
#        j2 = int((N**2-1)/2 + 3*N)
#        di_tot = mkvc(np.abs(FieldsTOT[j1, :]))
#        di_pre = mkvc(np.abs(FieldsVRM[j1, :]))
#        di_tem = mkvc(np.abs(FieldsTEM[j1, :]))
#        Ax22.loglog(times, di_tot, 'k.-')
#        Ax22.loglog(times, di_tem, 'r.-')
#        Ax22.loglog(times, di_pre, 'b.-')
#        Ax22.loglog(times, np.abs(di_tot-di_pre), 'g.-')
#        Ax22.set_xlabel('t [s]', fontsize=FS, labelpad=-10)
#        Ax22.axes.get_yaxis().set_visible(False)
#        Ax22.tick_params(labelsize=FS-2)
#        Ax22.set_xbound(np.min(times), np.max(times))
#        Ax22.set_ybound(1.2*np.max(di_tot), 1e-5*np.max(di_tot))
#        titlestr22 = (
#            "Decay at X = " + '{:.2f}'.format(loc[j2, 0]) +
#            " m and Y = " + '{:.2f}'.format(loc[j1, 1]) + " m")
#        Ax22.set_title(titlestr22, fontsize=FS+2)

        # PLOT ANOMALIES
#        d2 = np.reshape(np.abs(FieldsTOT[:, 10]), (N, N))
#        d3 = np.reshape(np.abs(FieldsTEM[:, 10]), (N, N))
#        d4 = np.reshape(np.abs(FieldsTOT[:, 10]-FieldsPRE[:, 10]), (N, N))
#
#        MIN = np.min(np.r_[d2, d3, d4])
#        MAX = np.max(np.r_[d2, d3, d4])
#
#        Cplot31 = Ax31.contourf(x, y, d2.T, 40, cmap='magma_r')
#        Ax31.set_xticks(np.linspace(-30, 30, 7))
#        Ax31.set_xlabel('X [m]', fontsize=FS)
#        Ax31.set_ylabel('Y [m]', fontsize=FS, labelpad=-12)
#        Ax31.tick_params(labelsize=FS-2)
#        Ax31.scatter(x, y, color=(0, 0, 0), s=4)
#        Ax31.set_xbound(np.min(x), np.max(x))
#        Ax31.set_ybound(np.min(y), np.max(y))
#        titlestr31 = "Observed at t=" + '{:.1e}'.format(times[10]) + " s"
#        Ax31.set_title(titlestr31, fontsize=FS+2)
#
#        Cplot32 = Ax32.contourf(x, y, d3.T, 40, cmap='magma_r')
#        Ax32.set_xticks(np.linspace(-30, 30, 7))
#        Ax32.set_xlabel('X [m]', fontsize=FS)
#        Ax32.axes.get_yaxis().set_visible(False)
#        Ax32.tick_params(labelsize=FS-2)
#        Ax32.set_xbound(np.min(x), np.max(x))
#        Ax32.set_ybound(np.min(y), np.max(y))
#        titlestr32 = "True TEM at t=" + '{:.1e}'.format(times[10]) + " s"
#        Ax32.set_title(titlestr32, fontsize=FS+2)
#
#        Cplot33 = Ax33.contourf(x, y, d4.T, 40, cmap='magma_r')
#        Ax33.set_xticks(np.linspace(-30, 30, 7))
#        Ax33.set_xlabel('X [m]', fontsize=FS)
#        Ax33.axes.get_yaxis().set_visible(False)
#        Ax33.tick_params(labelsize=FS-2)
#        Ax33.set_xbound(np.min(x), np.max(x))
#        Ax33.set_ybound(np.min(y), np.max(y))
#        titlestr33 = "Recov. TEM at t=" + '{:.1e}'.format(times[10]) + " s"
#        Ax33.set_title(titlestr33, fontsize=FS+2)
#
#        norm = mpl.colors.Normalize(vmin=MIN, vmax=MAX)
#        cbar34 = mpl.colorbar.ColorbarBase(
#            Ax34, cmap='magma_r', norm=norm, orientation='vertical', format='%.1e')
#        cbar34.set_label('dBz/dt [T/s]', rotation=270, size=FS, labelpad=15)

if __name__ == '__main__':
    run()
    plt.show()
