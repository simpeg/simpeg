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
from SimPEG import mkvc, Mesh, Maps
import matplotlib.pyplot as plt
import matplotlib as mpl


def run(plotIt=True):

    # CREATE MESH
    # Works for 3D tensor and 3D tree meshes
    cs, ncx, ncy, ncz, npad = 2., 35, 35, 20, 5
    hx = [(cs, npad, -1.3), (cs, ncx), (cs, npad, 1.3)]
    hy = [(cs, npad, -1.3), (cs, ncy), (cs, npad, 1.3)]
    hz = [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
    mesh = Mesh.TensorMesh([hx, hy, hz], 'CCC')

    # SET MAPPING AND ACTIVE CELLS
    # Only cells below the surface (z=0) are being modeled.
    topoCells = mesh.gridCC[:, 2] < 0.

    # CREATE MODEL (XI: THE AMALGAMATED MAGNETIC PROPERTY)
    # The model is made by summing a set of 3D Gaussian distributions.
    # Only active cells have a model value.
    xyzc = mesh.gridCC[topoCells, :]
    C = 2*np.pi*8**2
    xi_true = (
        4e-4*np.exp(-(xyzc[:, 0]-50)**2/(3*C)) *
        np.exp(-(xyzc[:, 1])**2/(20*C)) *
        np.exp(-(xyzc[:, 2])**2/C) +
        4e-4*np.exp(-(xyzc[:, 0]+50)**2/(3*C)) *
        np.exp(-(xyzc[:, 1])**2/(20*C)) *
        np.exp(-(xyzc[:, 2])**2/C) +
        4e-4*np.exp(-(xyzc[:, 0]+40)**2/(3*C)) *
        np.exp(-(xyzc[:, 1]-40)**2/C) *
        np.exp(-(xyzc[:, 2])**2/C) +
        6e-4*np.exp(-(xyzc[:, 0]+20)**2/C) *
        np.exp(-(xyzc[:, 1]-10)**2/C) *
        np.exp(-(xyzc[:, 2])**2/C) +
        8e-4*np.exp(-(xyzc[:, 0]+15)**2/(3*C)) *
        np.exp(-(xyzc[:, 1]+20)**2/(0.4*C)) *
        np.exp(-(xyzc[:, 2])**2/C) +
        6e-4*np.exp(-(xyzc[:, 0]-20)**2/(0.5*C)) *
        np.exp(-(xyzc[:, 1]-15)**2/(0.5*C)) *
        np.exp(-(xyzc[:, 2])**2/C) +
        8e-4*np.exp(-(xyzc[:, 0]+10)**2/(0.1*C)) *
        np.exp(-(xyzc[:, 1])**2/(0.1*C)) *
        np.exp(-(xyzc[:, 2])**2/C) +
        8e-4*np.exp(-(xyzc[:, 0]-25)**2/(0.1*C)) *
        np.exp(-(xyzc[:, 1])**2/(0.4*C)) *
        np.exp(-(xyzc[:, 2])**2/C) +
        1e-5
        )

    # SET THE TRANSMITTER WAVEFORM
    # This determines the off-time decay behaviour of the VRM response
    waveform = VRM.WaveformVRM.StepOff()

    # CREATE SURVEY
    # Similar to an EM-63 survey by all 3 components of the field are measured
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

    # DEFINE THE PROBLEM
    ProblemVRM = VRM.Problem_Linear(
        mesh, indActive=topoCells, ref_factor=3, ref_radius=[1.25, 2.5, 3.75])
    ProblemVRM.pair(SurveyVRM)

    # PREDICT THE FIELDS
    FieldsVRM = ProblemVRM.fields(xi_true)

    n_times = len(times)
    n_loc = loc.shape[0]
    FieldsVRM = np.reshape(FieldsVRM, (n_loc, n_times))

    ################################
    # ARTIFICIAL TEM RESPONSE
    # An analytic solution for the response near the surface of a conductive
    # half-space (Nabighian, 1979) is scaled at each location to provide
    # lateral variability in the TEM response.

    sig = 1e-1
    mu0 = 4*np.pi*1e-7
    FieldsTEM = -sig**1.5*mu0**2.5*times**-2.5/(20*np.pi**1.5)
    FieldsTEM = np.kron(np.ones((n_loc, 1)), np.reshape(FieldsTEM, (1, n_times)))
    C = (
       np.exp(-(loc[:, 0]-10)**2/(25**2))*np.exp(-(loc[:, 1]-20)**2/(35**2)) +
       np.exp(-(loc[:, 0]+20)**2/(20**2))*np.exp(-(loc[:, 1]+20)**2/(40**2)) +
       1.5*np.exp(-(loc[:, 0]-25)**2/(10**2))*np.exp(-(loc[:, 1]+25)**2/(10**2)) +
       0.25
       )

    C = np.kron(np.reshape(C, (len(C), 1)), np.ones((1, n_times)))
    FieldsTEM = C*FieldsTEM

    # PLOT MODEL
    if plotIt:

        Fig = plt.figure(figsize=(10, 10))
        Ax11 = Fig.add_axes([0.07, 0.7, 0.23, 0.23])
        Ax12 = Fig.add_axes([0.36, 0.7, 0.23, 0.23])
        Ax13 = Fig.add_axes([0.65, 0.7, 0.23, 0.23])
        Ax14 = Fig.add_axes([0.89, 0.7, 0.01, 0.24])

        Ax21 = Fig.add_axes([0.1, 0.335, 0.38, 0.29])
        Ax22 = Fig.add_axes([0.57, 0.335, 0.38, 0.29])

        Ax31 = Fig.add_axes([0.07, 0.05, 0.24, 0.21])
        Ax32 = Fig.add_axes([0.38, 0.05, 0.24, 0.21])
        Ax33 = Fig.add_axes([0.69, 0.05, 0.24, 0.21])
        FS = 12
        N = x.shape[0]

        plotMap = Maps.InjectActiveCells(mesh, topoCells, 0.)  # Maps to mesh

        Cplot11 = mesh.plotSlice(
            plotMap*xi_true, normal='X', ind=int((ncx+2*npad)/2-6),
            ax=Ax11, grid=True, pcolorOpts={'cmap': 'gist_heat_r'})
        Cplot11[0].set_clim((0., np.max(xi_true)))
        Ax11.set_xlabel('Y [m]', fontsize=FS)
        Ax11.set_ylabel('Z [m]', fontsize=FS, labelpad=-10)
        Ax11.tick_params(labelsize=FS-2)
        titlestr11 = "True Model (x = -12 m)"
        Ax11.set_title(titlestr11, fontsize=FS+2)

        Cplot12 = mesh.plotSlice(
            plotMap*xi_true, normal='Y', ind=int((ncy+2*npad)/2),
            ax=Ax12, grid=True, pcolorOpts={'cmap': 'gist_heat_r'})
        Cplot12[0].set_clim((0., np.max(xi_true)))
        Ax12.set_xlabel('X [m]', fontsize=FS)
        Ax12.set_ylabel('Z [m]', fontsize=FS, labelpad=-10)
        Ax12.tick_params(labelsize=FS-2)
        titlestr12 = "True Model (y = 0 m)"
        Ax12.set_title(titlestr12, fontsize=FS+2)

        Cplot13 = mesh.plotSlice(
            plotMap*xi_true, normal='Z', ind=int((ncz+2*npad)/2-1),
            ax=Ax13, grid=True, pcolorOpts={'cmap': 'gist_heat_r'})
        Cplot13[0].set_clim((0., np.max(xi_true)))
        Ax13.set_xlabel('X [m]', fontsize=FS)
        Ax13.set_ylabel('Y [m]', fontsize=FS, labelpad=-10)
        Ax13.tick_params(labelsize=FS-2)
        titlestr13 = "True Model (z = 0 m)"
        Ax13.set_title(titlestr13, fontsize=FS+2)

        norm = mpl.colors.Normalize(vmin=0., vmax=np.max(xi_true))
        cbar14 = mpl.colorbar.ColorbarBase(
            Ax14, cmap='gist_heat_r', norm=norm, orientation='vertical')
        cbar14.set_label(
            '$\Delta \chi /$ln$(\lambda_2 / \lambda_1 )$ [SI]',
            rotation=270, labelpad=15, size=FS)

        # PLOT DECAY
        j1 = int((N**2-1)/2 - 3*N)
        di_vrm = mkvc(np.abs(FieldsVRM[j1, :]))
        di_tem = mkvc(np.abs(FieldsTEM[j1, :]))
        Ax21.loglog(times, di_tem, 'r.-')
        Ax21.loglog(times, di_vrm, 'b.-')
        Ax21.loglog(times, di_tem+di_vrm, 'k.-')
        Ax21.set_xlabel('t [s]', fontsize=FS)
        Ax21.set_ylabel('|dBz/dt| [T/s]', fontsize=FS)
        Ax21.tick_params(labelsize=FS-2)
        Ax21.set_xbound(np.min(times), np.max(times))
        Ax21.set_ybound(1.2*np.max(di_tem+di_vrm), 1e-5*np.max(di_tem+di_vrm))
        titlestr21 = (
            "Decay at X = " + '{:.2f}'.format(loc[j1, 0]) +
            " m and Y = " + '{:.2f}'.format(loc[j1, 1]) + " m")
        Ax21.set_title(titlestr21, fontsize=FS+2)
        Ax21.text(1.2e-5, 18*np.max(di_tem)/1e5, "TEM", fontsize=FS, color='r')
        Ax21.text(1.2e-5, 6*np.max(di_tem)/1e5, "VRM", fontsize=FS, color='b')
        Ax21.text(1.2e-5, 2*np.max(di_tem)/1e5, "TEM + VRM", fontsize=FS, color='k')

        j2 = int((N**2-1)/2 + 3*N)
        di_vrm = mkvc(np.abs(FieldsVRM[j2, :]))
        di_tem = mkvc(np.abs(FieldsTEM[j2, :]))
        Ax22.loglog(times, di_tem, 'r.-')
        Ax22.loglog(times, di_vrm, 'b.-')
        Ax22.loglog(times, di_tem+di_vrm, 'k.-')
        Ax22.set_xlabel('t [s]', fontsize=FS)
        Ax22.axes.get_yaxis().set_visible(False)
        Ax22.tick_params(labelsize=FS-2)
        Ax22.set_xbound(np.min(times), np.max(times))
        Ax22.set_ybound(1.2*np.max(di_tem+di_vrm), 1e-5*np.max(di_tem+di_vrm))
        titlestr22 = (
            "Decay at X = " + '{:.2f}'.format(loc[j2, 0]) +
            " m and Y = " + '{:.2f}'.format(loc[j2, 1]) + " m")
        Ax22.set_title(titlestr22, fontsize=FS+2)

        # PLOT ANOMALIES
        d2 = np.reshape(np.abs(FieldsTEM[:, 0]+FieldsVRM[:, 0]), (N, N))
        d3 = np.reshape(np.abs(FieldsTEM[:, 10]+FieldsVRM[:, 10]), (N, N))
        d4 = np.reshape(np.abs(FieldsTEM[:, 20]+FieldsVRM[:, 20]), (N, N))

        Cplot31 = Ax31.contourf(x, y, d2.T, 40, cmap='magma_r')
        cbar31 = plt.colorbar(Cplot31, ax=Ax31, pad=0.02, format='%.2e')
        cbar31.set_label('[T/s]', rotation=270, labelpad=12, size=FS)
        cbar31.ax.tick_params(labelsize=FS-2)
        Ax31.set_xlabel('X [m]', fontsize=FS)
        Ax31.set_ylabel('Y [m]', fontsize=FS, labelpad=-8)
        Ax31.tick_params(labelsize=FS-2)
        Ax31.scatter(x, y, color=(0, 0, 0), s=4)
        Ax31.set_xbound(np.min(x), np.max(x))
        Ax31.set_ybound(np.min(y), np.max(y))
        titlestr31 = "dBz/dt at t=" + '{:.1e}'.format(times[0]) + " s"
        Ax31.set_title(titlestr31, fontsize=FS+2)

        Cplot32 = Ax32.contourf(x, y, d3.T, 40, cmap='magma_r')
        cbar32 = plt.colorbar(Cplot32, ax=Ax32, pad=0.02, format='%.2e')
        cbar32.set_label('[T/s]', rotation=270, labelpad=12, size=FS)
        cbar32.ax.tick_params(labelsize=FS-2)
        Ax32.set_xlabel('X [m]', fontsize=FS)
        Ax32.axes.get_yaxis().set_visible(False)
        Ax32.tick_params(labelsize=FS-2)
        Ax32.set_xbound(np.min(x), np.max(x))
        Ax32.set_ybound(np.min(y), np.max(y))
        titlestr32 = "dBz/dt at t=" + '{:.1e}'.format(times[10]) + " s"
        Ax32.set_title(titlestr32, fontsize=FS+2)

        Cplot33 = Ax33.contourf(x, y, d4.T, 40, cmap='magma_r')
        cbar33 = plt.colorbar(Cplot33, ax=Ax33, pad=0.02, format='%.2e')
        cbar33.set_label('[T/s]', rotation=270, labelpad=12, size=FS)
        cbar33.ax.tick_params(labelsize=FS-2)
        Ax33.set_xlabel('X [m]', fontsize=FS)
        Ax33.axes.get_yaxis().set_visible(False)
        Ax33.tick_params(labelsize=FS-2)
        Ax33.set_xbound(np.min(x), np.max(x))
        Ax33.set_ybound(np.min(y), np.max(y))
        titlestr33 = "dBz/dt at t=" + '{:.1e}'.format(times[20]) + " s"
        Ax33.set_title(titlestr33, fontsize=FS+2)

if __name__ == '__main__':
    run()
    plt.show()
