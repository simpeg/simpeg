"""
MT: 3D: Forward
===============

Forward model 3D MT data.

Test script to use SimPEG.NSEM platform to forward model
impedance and tipper synthetic data.
"""

import SimPEG as simpeg
from SimPEG.EM import NSEM
import numpy as np
import matplotlib.pyplot as plt

try:
    from pymatsolver import Pardiso as solver
except:
    from SimPEG import solver


def run(plotIt=True):
    """
        MT: 3D: Forward
        ===============

        Forward model 3D MT data.

    """

    # Make a mesh
    M = simpeg.Mesh.TensorMesh(
        [
            [(100, 9, -1.5), (100., 13), (100, 9, 1.5)],
            [(100, 9, -1.5), (100., 13), (100, 9, 1.5)],
            [(50, 10, -1.6), (50., 10), (50, 6, 2)]
        ], x0=['C', 'C', -14926.8217]
    )
    # Setup the model
    conds = [1,1e-2]
    sig = simpeg.Utils.ModelBuilder.defineBlock(
        M.gridCC, [-100, -100, -350], [100, 100, -150], conds
    )
    sig[M.gridCC[:, 2] > 0] = 1e-8
    sig[M.gridCC[:, 2] < -1000] = 1e-1
    sigBG = np.zeros(M.nC) + conds[1]
    sigBG[M.gridCC[:, 2] > 0] = 1e-8
    if plotIt:
        collect_obj, line_obj = M.plotSlice(np.log10(sig), grid=True, normal='X')
        color_bar = plt.colorbar(collect_obj)

    # Setup the the survey object
    # Receiver locations
    rx_x, rx_y = np.meshgrid(np.arange(-600, 601, 100), np.arange(-600, 601, 100))
    rx_loc = np.hstack((simpeg.Utils.mkvc(rx_x, 2), simpeg.Utils.mkvc(rx_y, 2), np.zeros((np.prod(rx_x.shape), 1))))

    # Make a receiver list
    rxList = []
    for rx_orientation in ['xx', 'xy', 'yx', 'yy']:
        rxList.append(NSEM.Rx.Point_impedance3D(
            locs=rx_loc, orientation=rx_orientation, component='real'))
        rxList.append(NSEM.Rx.Point_impedance3D(
            locs=rx_loc, orientation=rx_orientation, component='imag'))
    for rx_orientation in ['zx', 'zy']:
        rxList.append(NSEM.Rx.Point_tipper3D(
            locs=rx_loc, orientation=rx_orientation, component='real'))
        rxList.append(NSEM.Rx.Point_tipper3D(
            locs=rx_loc, orientation=rx_orientation, component='imag'))

    # Source list
    srcList = [
        NSEM.Src.Planewave_xy_1Dprimary(rxList=rxList, freq=freq)
        for freq in np.logspace(4, -2, 13)
    ]
    # Survey MT
    survey = NSEM.Survey(srcList=srcList)

    # Setup the simulation object
    simulation = NSEM.Simulation3D_ePrimSec(
        mesh=M, survey=survey, sigma=sig, sigmaPrimary=sigBG)
    simulation.solver = solver

    # Calculate the data
    fields = simulation.fields()
    dataVec = NSEM.Data(
        survey=simulation.survey, dobs=simulation.dpred(f=fields))

    # Add uncertainty to the data - 10% standard
    # devation and 0 floor
    dataVec.standard_deviation = np.ones_like(
        dataVec.dobs)* 0.1
    dataVec.noise_floor = np.zeros_like(dataVec.dobs)

    # Add plots
    if plotIt:
        # Plot the data
        # On and off diagonal (on left and right axis, respectively)
        fig, axes = plt.subplots(2, 1, figsize=(7, 5))
        plt.subplots_adjust(right=0.8)
        [(ax.invert_xaxis(), ax.set_xscale('log')) for ax in axes]
        ax_r, ax_p = axes
        ax_r.set_yscale('log')
        ax_r.set_ylabel('Apparent resistivity [xy-yx]')
        ax_r_on = ax_r.twinx()
        ax_r_on.set_yscale('log')
        ax_r_on.set_ylabel('Apparent resistivity [xx-yy]')
        ax_p.set_ylabel('Apperent phase')
        ax_p.set_xlabel('Frequency [Hz]')
        # Start plotting
        ax_r = dataVec.plot_app_res(
            np.array([-200, 0]),
            components=['xy', 'yx'], ax=ax_r, errorbars=True)
        ax_r_on = dataVec.plot_app_res(
            np.array([-200, 0]),
            components=['xx', 'yy'], ax=ax_r_on, errorbars=True)
        ax_p = dataVec.plot_app_phs(
            np.array([-200, 0]),
            components=['xx', 'xy', 'yx', 'yy'], ax=ax_p, errorbars=True)
        ax_p.legend(bbox_to_anchor=(1.05, 1), loc=2)



if __name__ == '__main__':

    do_plots = True
    run(do_plots)
    if do_plots:
        plt.show()


