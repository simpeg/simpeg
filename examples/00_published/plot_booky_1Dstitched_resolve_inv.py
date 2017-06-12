import numpy as np
import cPickle as pickle
from SimPEG import (
    Mesh, Maps, Utils, DataMisfit, Regularization,
    Optimization, Inversion, InvProblem, Directives
)
import SimPEG.EM as EM
import matplotlib.pyplot as plt
from pymatsolver import PardisoSolver
from scipy.constants import mu_0
import h5py
from scipy.spatial import cKDTree

"""
Heagy et al., 2017 1D RESOLVE Bookpurnong Inversion
===================================================

In this example, perform a stitched 1D inversion of the Bookpurnong RESOLVE
data. The original data can be downloaded from:
`https://storage.googleapis.com/simpeg/bookpurnong/bookpurnong.tar.gz <https://storage.googleapis.com/simpeg/bookpurnong/bookpurnong.tar.gz>`_

The forward simulation is performed on the cylindrically symmetric mesh using
:code:`SimPEG.EM.FDEM`.

Heagy, L.J., R. Cockett, S. Kang, G.K. Rosenkjaer, D.W. Oldenburg,
2017 (in review), A framework for simulation and inversion in electromagnetics.
Computers & Geosciences

The paper is available at:
https://arxiv.org/abs/1610.00804

"""


def resolve_1Dinversions(
    mesh, dobs, src_height, freqs, m0, mref, mapping,
    std=0.08, floor=1e-14, rxOffset=7.86
):
    """
    Perform a single 1D inversion for a RESOLVE sounding for Horizontal
    Coplanar Coil data (both real and imaginary).

    :param discretize.CylMesh mesh: mesh used for the forward simulation
    :param numpy.array dobs: observed data
    :param float src_height: height of the source above the ground
    :param numpy.array freqs: frequencies
    :param numpy.array m0: starting model
    :param numpy.array mref: reference model
    :param Maps.IdentityMap mapping: mapping that maps the model to electrical conductivity
    :param float std: percent error used to construct the data misfit term
    :param float floor: noise floor used to construct the data misfit term
    :param float rxOffset: offset between source and receiver.
    """

    # ------------------- Forward Simulation ------------------- #
    # set up the receivers
    bzr = EM.FDEM.Rx.Point_bSecondary(
        np.array([[rxOffset, 0., src_height]]),
        orientation='z',
        component='real'
    )

    bzi = EM.FDEM.Rx.Point_b(
        np.array([[rxOffset, 0., src_height]]),
        orientation='z',
        component='imag'
    )

    # source location
    srcLoc = np.array([0., 0., src_height])
    srcList = [
        EM.FDEM.Src.MagDipole([bzr, bzi], freq, srcLoc, orientation='Z')
        for freq in freqs
    ]

    # construct a forward simulation
    survey = EM.FDEM.Survey(srcList)
    prb = EM.FDEM.Problem3D_b(mesh, sigmaMap=mapping, Solver=PardisoSolver)
    prb.pair(survey)

    # ------------------- Inversion ------------------- #
    # data misfit term
    survey.dobs = dobs
    dmisfit = DataMisfit.l2_DataMisfit(survey)
    uncert = abs(dobs) * std + floor
    dmisfit.W = 1./uncert

    # regularization
    regMesh = Mesh.TensorMesh([mesh.hz[mapping.maps[-1].indActive]])
    reg = Regularization.Simple(regMesh)
    reg.mref = mref

    # optimization
    opt = Optimization.InexactGaussNewton(maxIter=10)

    # statement of the inverse problem
    invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)

    # Inversion directives and parameters
    target = Directives.TargetMisfit()
    beta = Directives.BetaSchedule(coolingFactor=1, coolingRate=2)
    inv = Inversion.BaseInversion(invProb, directiveList=[beta, target])

    invProb.beta = 2.   # Fix beta in the nonlinear iterations
    reg.alpha_s = 1e-3
    reg.alpha_x = 1.
    prb.counter = opt.counter = Utils.Counter()
    opt.LSshorten = 0.5
    opt.remember('xc')

    # run the inversion
    mopt = inv.run(m0)
    return mopt, invProb.dpred, survey.dobs


def run(runIt=False, plotIt=True, saveFig=False):
    # Load resolve data
    resolve = h5py.File("./downloads/booky_resolve.hdf5", "r")
    river_path = resolve["river_path"].value    # River path
    nSounding = resolve["data"].shape[0]    # the # of soundings
    # Bird height from surface
    b_height_resolve = resolve["src_elevation"].value

    cpi_inds = [0, 2, 6, 8, 10]  # Indices for HCP in-phase
    cpq_inds = [1, 3, 7, 9, 11]  # Indices for HCP quadrature
    frequency_cp = resolve["frequency_cp"].value

    cs, ncx, ncz, npad = 1., 10., 10., 20
    hx = [(cs, ncx), (cs, npad, 1.3)]
    npad = 12
    temp = np.logspace(np.log10(1.), np.log10(12.), 19)
    temp_pad = temp[-1] * 1.3 ** np.arange(npad)
    hz = np.r_[temp_pad[::-1], temp[::-1], temp, temp_pad]
    mesh = Mesh.CylMesh([hx, 1, hz], '00C')
    active = mesh.vectorCCz < 0.
    rxOffset = 7.86
    bp = -mu_0/(4*np.pi*rxOffset**3)

    if runIt:

        actMap = Maps.InjectActiveCells(
            mesh, active, np.log(1e-8), nC=mesh.nCz
        )
        mapping = Maps.ExpMap(mesh) * Maps.SurjectVertical1D(mesh) * actMap
        sig_half = 1e-1
        sig_air = 1e-8
        sigma = np.ones(mesh.nCz)*sig_air
        sigma[active] = sig_half
        m0 = np.log(1e-1) * np.ones(active.sum())
        mref = np.log(1e-1) * np.ones(active.sum())

        mopt_re = []
        dpred_re = []
        dobs_re = []

        nskip = 40
        std = np.repeat(np.r_[np.ones(3)*0.1, np.ones(2)*0.15], 2)
        floor = abs(20 * bp * 1e-6)

        for rxind in range(nSounding):
            dobs = np.c_[
                resolve["data"][rxind, :][cpi_inds].astype(float),
                resolve["data"][rxind, :][cpq_inds].astype(float)
            ].flatten() * bp * 1e-6
            src_height = b_height_resolve[rxind].astype(float)
            mopt, dpred, dobs = resolve_1Dinversions(
                mesh, dobs, src_height, frequency_cp, m0, mref, mapping,
                std=std, floor=floor
                )

            mopt_re.append(mopt)
            dpred_re.append(dpred)
            dobs_re.append(dobs)

        mopt_re = np.vstack(mopt_re)
        dpred_re = np.vstack(dpred_re)
        dobs_re = np.vstack(dobs_re)

        np.save("mopt_re_final", mopt_re)
        np.save("dobs_re_final", dobs_re)
        np.save("dpred_re_final", dpred_re)

    if plotIt:

        mopt_re = resolve["mopt"].value
        dobs_re = resolve["dobs"].value
        dpred_re = resolve["dpred"].value

        sigma = np.exp(mopt_re)
        indz = -7
        cmap = "jet"

        # dummy figure for colobar
        fig = plt.figure()
        out = plt.scatter(
            np.ones(3), np.ones(3), c=np.linspace(-2, 1, 3), cmap=cmap
        )
        plt.close(fig)

        # plot recovered stitched conductivity model
        fig = plt.figure(figsize=(5, 5))
        ax = plt.subplot(111)
        temp = sigma[:, indz]
        tree = cKDTree(zip(resolve["xy"][:, 0], resolve["xy"][:, 1]))
        d, d_inds = tree.query(
            zip(resolve["xy"][:, 0], resolve["xy"][:, 1]), k=20
        )
        w = 1. / (d+100.)**2.
        w = Utils.sdiag(1./np.sum(w, axis=1)) * (w)
        xy = resolve["xy"]
        temp = (temp.flatten()[d_inds] * w).sum(axis=1)
        Utils.plot2Ddata(
            xy, temp, ncontour=100, scale="log", dataloc=False,
            contourOpts={"cmap": cmap, "vmin": -2, "vmax": 1.}, ax=ax
        )
        ax.plot(
            resolve["xy"][:, 0], resolve["xy"][:, 1], 'k.', alpha=0.02, ms=1
        )
        ax.set_title(
            ("%.1f m below surface") % (-mesh.vectorCCz[active][indz])
        )
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        xticks = [460000, 463000]
        yticks = [6195000, 6198000, 6201000]
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([str(f) for f in xticks])
        ax.set_yticklabels([str(f) for f in yticks])
        cb = plt.colorbar(
            out, ax=ax, ticks=np.linspace(-2, 1, 4), format="$10^{%.1f}$"
        )
        cb.set_ticklabels(["0.01", "0.1", "1", "10"])
        cb.set_label("Conductivity (S/m)")
        ax.plot(river_path[:, 0], river_path[:, 1], 'k-', lw=0.5)
        plt.tight_layout()
        plt.show()

        if saveFig is True:
            fig.savefig("resolve_cond_9_9m.png", dpi=200)

        # plot observed and predicted data
        freq_ind = 0
        fig = plt.figure(figsize=(11*0.8, 6*0.8))
        title = ["Observed (In-phase 400 Hz)", "Predicted (In-phase 400 Hz)"]
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        axs = [ax1, ax2]
        temp_dobs = dobs_re[:, freq_ind].copy()
        # temp_dobs[mask_:_data] = np.nan
        ax.plot(river_path[:, 0], river_path[:, 1], 'k-', lw=0.5)
        out = Utils.plot2Ddata(
            resolve["xy"].value, temp_dobs/abs(bp)*1e6, ncontour=100,
            scale="log", dataloc=False, ax=ax1, contourOpts={"cmap": "viridis"}
        )
        vmin, vmax = out[0].get_clim()
        # cb = plt.colorbar(out[0], ticks=np.linspace(vmin, vmax, 3), ax=ax1, format="%.1e", fraction=0.046, pad=0.04)
        # cb.set_label("Bz (ppm)")
        temp_dpred = dpred_re[:, freq_ind].copy()
        # temp_dpred[mask_:_data] = np.nan
        ax.plot(river_path[:, 0], river_path[:, 1], 'k-', lw=0.5)
        Utils.plot2Ddata(
            resolve["xy"].value, temp_dpred/abs(bp)*1e6, ncontour=100,
            scale="log", dataloc=False,
            contourOpts={"vmin": vmin, "vmax": vmax, "cmap": "viridis"}, ax=ax2
        )
        cb = plt.colorbar(
            out[0], ticks=np.linspace(vmin, vmax, 3), ax=ax2,
            format="%.1e", fraction=0.046, pad=0.04
        )
        cb.set_label("Bz (ppm)")

        for i, ax in enumerate(axs):
            xticks = [460000, 463000]
            yticks = [6195000, 6198000, 6201000]
            xloc, yloc = 462100.0, 6196500.0
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.plot(xloc, yloc, 'wo')
            ax.plot(river_path[:, 0], river_path[:, 1], 'k', lw=0.5)

            ax.set_aspect("equal")
            ax.plot(
                resolve["xy"][:, 0], resolve["xy"][:, 1], 'k.', alpha=0.02,
                ms=1
            )

            if i == 1:
                ax.set_yticklabels([str(" ") for f in yticks])
            else:
                ax.set_yticklabels([str(f) for f in yticks])
                ax.set_ylabel("Northing (m)")
            ax.set_xlabel("Easting (m)")
            ax.set_title(title[i])

        plt.tight_layout()
        plt.show()

        if saveFig is True:
            fig.savefig("obspred_resolve.png", dpi=200)

if __name__ == '__main__':
    run(runIt=False, plotIt=True, saveFig=True)
