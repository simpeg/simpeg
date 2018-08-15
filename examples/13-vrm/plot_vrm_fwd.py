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
    # This determines the off-time decay behaviour of the VRM response
    waveform = VRM.WaveformVRM.StepOff()

    # CREATE SURVEY
    # Similar to an EM-63 survey
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
        mesh, ind_active=topoCells, ref_factor=3, ref_radius=[1.25, 2.5, 3.75])
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

    if plotIt:

        Fig = plt.figure(figsize=(10, 10))
        FS = 12

        # PLOT MODEL
        plotMap = Maps.InjectActiveCells(mesh, topoCells, 0.)  # Maps to mesh
        Ax1 = 4*[None]
        Cplot1 = 3*[None]
        view_str = ['X', 'Y', 'Z']
        param_1 = [ncx, ncy, ncz]
        param_2 = [6, 0, 1]
        param_3 = [-12, 0, 0]

        for qq in range(0, 3):
            Ax1[qq] = Fig.add_axes([0.07+qq*0.29, 0.7, 0.23, 0.23])
            Cplot1[qq] = mesh.plotSlice(
                plotMap*xi_true, normal=view_str[qq],
                ind=int((param_1[qq]+2*npad)/2-param_2[qq]),
                ax=Ax1[qq], grid=True, pcolorOpts={'cmap': 'gist_heat_r'})
            Cplot1[qq][0].set_clim((0., np.max(xi_true)))
            Ax1[qq].set_xlabel('Y [m]', fontsize=FS)
            Ax1[qq].set_ylabel('Z [m]', fontsize=FS, labelpad=-10)
            Ax1[qq].tick_params(labelsize=FS-2)
            Ax1[qq].set_title('True Model (x = {} m)'.format(
                param_3[qq]), fontsize=FS+2
            )

        Ax1[3] = Fig.add_axes([0.89, 0.7, 0.01, 0.24])
        norm = mpl.colors.Normalize(vmin=0., vmax=np.max(xi_true))
        cbar14 = mpl.colorbar.ColorbarBase(
            Ax1[3], cmap='gist_heat_r', norm=norm, orientation='vertical')
        cbar14.set_label(
            '$\Delta \chi /$ln$(\lambda_2 / \lambda_1 )$ [SI]',
            rotation=270, labelpad=15, size=FS)

        # PLOT DECAY
        Ax2 = 2*[None]
        N = x.shape[0]
        for qq in range(0, 2):
            Ax2[qq] = Fig.add_axes([0.1+0.47*qq, 0.335, 0.38, 0.29])
            k = int((N**2-1)/2 - 3*N*(-1)**qq)
            di_vrm = mkvc(np.abs(FieldsVRM[k, :]))
            di_tem = mkvc(np.abs(FieldsTEM[k, :]))
            Ax2[qq].loglog(times, di_tem, 'r.-')
            Ax2[qq].loglog(times, di_vrm, 'b.-')
            Ax2[qq].loglog(times, di_tem+di_vrm, 'k.-')
            Ax2[qq].set_xlabel('t [s]', fontsize=FS)
            if qq == 0:
                Ax2[qq].set_ylabel('|dBz/dt| [T/s]', fontsize=FS)
            else:
                Ax2[qq].axes.get_yaxis().set_visible(False)
            Ax2[qq].tick_params(labelsize=FS-2)
            Ax2[qq].set_xbound(np.min(times), np.max(times))
            Ax2[qq].set_ybound(1.2*np.max(di_tem+di_vrm), 1e-5*np.max(di_tem+di_vrm))
            titlestr2 = (
                "Decay at X = " + '{:.2f}'.format(loc[k, 0]) +
                " m and Y = " + '{:.2f}'.format(loc[k, 1]) + " m")
            Ax2[qq].set_title(titlestr2, fontsize=FS+2)
            if qq == 0:
                Ax2[qq].text(
                    1.2e-5, 18*np.max(di_tem)/1e5, "TEM", fontsize=FS, color='r'
                    )
                Ax2[qq].text(
                    1.2e-5, 6*np.max(di_tem)/1e5, "VRM", fontsize=FS, color='b'
                    )
                Ax2[qq].text(
                    1.2e-5, 2*np.max(di_tem)/1e5, "TEM + VRM", fontsize=FS, color='k'
                    )

        # PLOT ANOMALIES
        Ax3 = 3*[None]
        Cplot3 = 3*[None]
        cbar3 = 3*[None]
        for qq in range(0, 3):
            Ax3[qq] = Fig.add_axes([0.07+0.31*qq, 0.05, 0.24, 0.21])
            d = np.reshape(np.abs(FieldsTEM[:, 10*qq]+FieldsVRM[:, 10*qq]), (N, N))
            Cplot3[qq] = Ax3[qq].contourf(x, y, d.T, 40, cmap='magma_r')
            cbar3[qq] = plt.colorbar(Cplot3[qq], ax=Ax3[qq], pad=0.02, format='%.2e')
            cbar3[qq].set_label('[T/s]', rotation=270, labelpad=12, size=FS)
            cbar3[qq].ax.tick_params(labelsize=FS-2)
            Ax3[qq].set_xlabel('X [m]', fontsize=FS)
            if qq == 0:
                Ax3[qq].scatter(x, y, color=(0, 0, 0), s=4)
                Ax3[qq].set_ylabel('Y [m]', fontsize=FS, labelpad=-8)
            else:
                Ax3[qq].axes.get_yaxis().set_visible(False)
            Ax3[qq].tick_params(labelsize=FS-2)
            Ax3[qq].set_xbound(np.min(x), np.max(x))
            Ax3[qq].set_ybound(np.min(y), np.max(y))
            titlestr3 = "dBz/dt at t=" + '{:.1e}'.format(times[10*qq]) + " s"
            Ax3[qq].set_title(titlestr3, fontsize=FS+2)

if __name__ == '__main__':
    run()
    plt.show()
