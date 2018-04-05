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
import numpy as np
from SimPEG import (Mesh, Maps, Utils)
import SimPEG.VRM as VRM
import SimPEG.EM.TDEM as TDEM
import numpy as np
from SimPEG import mkvc, Mesh, Maps
import matplotlib.pyplot as plt
from pymatsolver import Pardiso


def run(plotIt=True):

    # CREATE MESH
    # Works for 3D tensor and 3D tree meshes
    cs, ncx, ncy, ncz, npad = 2., 35, 35, 20, 5
    hx = [(cs, npad, -1.3), (cs, ncx), (cs, npad, 1.3)]
    hy = [(cs, npad, -1.3), (cs, ncy), (cs, npad, 1.3)]
    hz = [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
    mesh = Mesh.TensorMesh([hx, hy, hz], 'CCC')

    # SET MAPPING AND ACTIVE CELLS
    # Only cells below the surface (z=0) are active.
    # All other cells set to to 0 for plotting on mesh.
    actCells = mesh.gridCC[:, 2] < 0.
    actMap = Maps.InjectActiveCells(mesh, actCells, 0.)

    # CREATE MODEL (XI: THE AMALGAMATED MAGNETIC PROPERTY)
    # The model is made by summing a set of 3D Gaussian distributions.
    # Only active cells have a model value.
    xyzc = mesh.gridCC[actCells, :]
    C = 2*np.pi*8**2
    xi = (
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

    mesh.plotSlice(actMap*xi, ind=0)

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
        rxListVRM = [VRM.Rx.Point(loc_pp, times, 'dbdt', 'z')]

        srcListVRM.append(VRM.Src.MagDipole(rxListVRM, mkvc(loc[pp, :]), [0., 0., 0.01], waveform))

    SurveyVRM = VRM.Survey(srcListVRM)

    # DEFINE THE PROBLEM
    ProblemVRM = VRM.Problem_Linear(mesh, xiMap=actMap, refFact=3, refRadius=[1.25, 2.5, 3.75])
    ProblemVRM.pair(SurveyVRM)

    # PREDICT THE FIELDS
    FieldsVRM = ProblemVRM.fields(xi)

    n_times = len(times)
    n_loc = loc.shape[0]
    FieldsVRM = np.reshape(FieldsVRM, (n_loc, n_times))

    ################################
    # PREDICT TEM RESPONSE
    # Here, we use the analytic solution derived by Nabighian (1979) to predict
    # the inductive response near the surface of a conductive half-space

    sig = 1e-1
    mu0 = 4*np.pi*1e-7
    FieldsTEM = -sig**1.5*mu0**2.5*times**-2.5/(20*np.pi**1.5)
    FieldsTEM = np.kron(np.ones((n_loc, 1)), np.reshape(FieldsTEM, (1, n_times)))

    #################################
    # ADD NOISE
    FieldsVRM = FieldsVRM + 0.05*np.abs(FieldsVRM)*np.random.normal(size=FieldsVRM.shape)
    FieldsTEM = FieldsTEM + 0.05*np.abs(FieldsTEM)*np.random.normal(size=FieldsTEM.shape)

    #################################
    # PLOTTING

    Fig = plt.figure(figsize=(13.5, 8.5))
    Ax1a = Fig.add_axes([0.1, 0.5, 0.4, 0.35])
    Ax1b = Fig.add_axes([0.6, 0.5, 0.4, 0.35])
    Ax2 = Fig.add_axes([0.05, 0.05, 0.25, 0.35])
    Ax3 = Fig.add_axes([0.4, 0.05, 0.25, 0.35])
    Ax4 = Fig.add_axes([0.75, 0.05, 0.25, 0.35])
    FS = 14
    N = x.shape[0]

    # PLOT DECAYS
    j1 = int((N**2-1)/2 - 3*N)
    di_vrm = mkvc(np.abs(FieldsVRM[j1, :]))
    di_tem = mkvc(np.abs(FieldsTEM[j1, :]))
    Ax1a.loglog(times, di_tem, 'r.-')
    Ax1a.loglog(times, di_vrm, 'b.-')
    Ax1a.loglog(times, di_tem+di_vrm, 'k.-')
    Ax1a.set_xlabel('t [s]', fontsize=FS)
    Ax1a.set_ylabel('|dB/dt| [T/s]', fontsize=FS)
    Ax1a.tick_params(labelsize=FS-2)
    Ax1a.set_xbound(np.min(times), np.max(times))
    Ax1a.set_ybound(1.2*np.max(di_tem+di_vrm), 1e-5*np.max(di_tem+di_vrm))
    titlestr1a = "Decay at X = " + '{:.2f}'.format(loc[j1, 0]) + " m and Y = " + '{:.2f}'.format(loc[j1, 1]) + " m"
    Ax1a.set_title(titlestr1a, fontsize=FS+2)

    j2 = int((N**2-1)/2 + 3*N)
    di_vrm = mkvc(np.abs(FieldsVRM[j2, :]))
    di_tem = mkvc(np.abs(FieldsTEM[j2, :]))
    Ax1b.loglog(times, di_tem, 'r.-')
    Ax1b.loglog(times, di_vrm, 'b.-')
    Ax1b.loglog(times, di_tem+di_vrm, 'k.-')
    Ax1b.set_xlabel('t [s]', fontsize=FS)
    Ax1b.set_ylabel('|dB/dt| [T/s]', fontsize=FS)
    Ax1b.tick_params(labelsize=FS-2)
    Ax1b.set_xbound(np.min(times), np.max(times))
    Ax1b.set_ybound(1.2*np.max(di_tem+di_vrm), 1e-5*np.max(di_tem+di_vrm))
    titlestr1b = "Decay at X = " + '{:.2f}'.format(loc[j2, 0]) + " m and Y = " + '{:.2f}'.format(loc[j2, 1]) + " m"
    Ax1b.set_title(titlestr1b, fontsize=FS+2)

    # PLOT ANOMALIES
    d2 = np.reshape(np.abs(FieldsTEM[:, 0]+FieldsVRM[:, 0]), (N, N))
    d3 = np.reshape(np.abs(FieldsTEM[:, 10]+FieldsVRM[:, 10]), (N, N))
    d4 = np.reshape(np.abs(FieldsTEM[:, 20]+FieldsVRM[:, 20]), (N, N))

    Cplot2 = Ax2.contourf(x, y, d2.T, 40, cmap='jet')
    cbar2 = plt.colorbar(Cplot2, ax=Ax2, pad=0.02, format='%.2e')
    cbar2.set_label('[T/s]', rotation=270, labelpad=15, size=FS)
    cbar2.ax.tick_params(labelsize=FS-2)
    Ax2.set_xlabel('X [m]', fontsize=FS)
    Ax2.set_ylabel('Y [m]', fontsize=FS, labelpad=-10)
    Ax2.tick_params(labelsize=FS-2)
    Ax2.scatter(x, y, color=(0, 0, 0), s=4)
    Ax2.set_xbound(np.min(x), np.max(x))
    Ax2.set_ybound(np.min(y), np.max(y))
    titlestr2 = "dBzdt = " + '{:.1e}'.format(times[0]) + " s"
    Ax2.set_title(titlestr2, fontsize=FS+2)

    Cplot3 = Ax3.contourf(x, y, d3.T, 40, cmap='jet')
    cbar3 = plt.colorbar(Cplot3, ax=Ax3, pad=0.02, format='%.2e')
    cbar3.set_label('[T/s]', rotation=270, labelpad=15, size=FS)
    cbar3.ax.tick_params(labelsize=FS-2)
    Ax3.set_xlabel('X [m]', fontsize=FS)
    Ax3.set_ylabel('Y [m]', fontsize=FS, labelpad=-10)
    Ax3.tick_params(labelsize=FS-2)
    Ax3.set_xbound(np.min(x), np.max(x))
    Ax3.set_ybound(np.min(y), np.max(y))
    titlestr3 = "dBzdt = " + '{:.1e}'.format(times[10]) + " s"
    Ax3.set_title(titlestr3, fontsize=FS+2)

    Cplot4 = Ax4.contourf(x, y, d4.T, 40, cmap='jet')
    cbar4 = plt.colorbar(Cplot4, ax=Ax4, pad=0.02, format='%.2e')
    cbar4.set_label('[T/s]', rotation=270, labelpad=15, size=FS)
    cbar4.ax.tick_params(labelsize=FS-2)
    Ax4.set_xlabel('X [m]', fontsize=FS)
    Ax4.set_ylabel('Y [m]', fontsize=FS, labelpad=-10)
    Ax4.tick_params(labelsize=FS-2)
    Ax4.set_xbound(np.min(x), np.max(x))
    Ax4.set_ybound(np.min(y), np.max(y))
    titlestr4 = "dBzdt = " + '{:.1e}'.format(times[20]) + " s"
    Ax4.set_title(titlestr4, fontsize=FS+2)


if __name__ == '__main__':
    run()
    plt.show()
