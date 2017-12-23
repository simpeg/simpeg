"""
2.5D DC-IP inversion of Dipole Dipole array with Topography
===========================================================

This is an example for 2.5D DC-IP Inversion.
For DC inversion, a resisistivity model (Ohm-m) is generated having conductive
and resistive cylinders; they are respectively located right and left sides
of the subsurface.
For IP inversion, a chargeability model (V/V) is generated having a chargeable
cyinder located at the center.
Default `survey_type` is dipole-dipole, but this can be changed.
to 'pole-dipole', 'dipole-pole', and 'pole-pole'.
By running DC and IP simulations synthetic DC and IP data are generated,
respectively. Following two-stage approach (Oldenburg et al, 1999),
first DC data is inverted to recover a resistivity model. Then by using
the obtained resistivity model, sensitivity function is formed and used for
subsequent IP inversion to recover a chargeability model.
"""

from SimPEG import DC, IP
from SimPEG import Maps, Utils
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from pylab import hist
try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver


def run(plotIt=True, survey_type="dipole-dipole"):
    np.random.seed(1)
    # Initiate I/O class for DC
    IO = DC.IO()
    # Obtain ABMN locations

    xmin, xmax = 0., 200.
    ymin, ymax = 0., 0.
    zmin, zmax = 0, 0
    endl = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
    # Generate DC survey object
    survey_dc = DC.Utils.gen_DCIPsurvey(endl, survey_type=survey_type, dim=2,
                                     a=10, b=10, n=10)
    survey_dc.getABMN_locations()
    survey_dc = IO.from_ambn_locations_to_survey(
        survey_dc.a_locations, survey_dc.b_locations,
        survey_dc.m_locations, survey_dc.n_locations,
        survey_type, data_dc_type='volt', data_ip_type='volt'
    )

    # Obtain 2D TensorMesh
    mesh, actind = IO.set_mesh()
    topo, mesh1D = DC.Utils.genTopography(mesh, -10, 0, its=100)
    actind = Utils.surface2ind_topo(mesh, np.c_[mesh1D.vectorCCx, topo])
    survey_dc.drapeTopo(mesh, actind, option="top")

    # Build conductivity and chargeability model
    blk_inds_c = Utils.ModelBuilder.getIndicesSphere(
        np.r_[60., -25.], 12.5, mesh.gridCC
    )
    blk_inds_r = Utils.ModelBuilder.getIndicesSphere(
        np.r_[140., -25.], 12.5, mesh.gridCC
    )
    blk_inds_charg = Utils.ModelBuilder.getIndicesSphere(
        np.r_[100., -25], 12.5, mesh.gridCC
    )
    layer_inds = mesh.gridCC[:, 1] > -5.
    sigma = np.ones(mesh.nC)*1./100.
    sigma[blk_inds_c] = 1./10.
    sigma[blk_inds_r] = 1./1000.
    sigma[~actind] = 1./1e8
    rho = 1./sigma
    charg = np.zeros(mesh.nC)
    charg[blk_inds_charg] = 0.1

    # Show the true conductivity model
    if plotIt:
        fig, axs = plt.subplots(2,1, figsize=(12, 6))
        temp_rho = rho.copy()
        temp_rho[~actind] = np.nan
        temp_charg = charg.copy()
        temp_charg[~actind] = np.nan

        out1 = mesh.plotImage(
            temp_rho, grid=True, ax=axs[0], gridOpts={'alpha': 0.2},
            clim=(10, 1000),
            pcolorOpts={"cmap": "viridis", "norm": colors.LogNorm()}
        )
        out2 = mesh.plotImage(
            temp_charg, grid=True, ax=axs[1], gridOpts={'alpha': 0.2},
            clim=(0, 0.1),
            pcolorOpts={"cmap": "magma"}
        )
        for i in range(2):
            axs[i].plot(
                survey_dc.electrode_locations[:, 0],
                survey_dc.electrode_locations[:, 1], 'kv'
            )
            axs[i].set_xlim(IO.grids[:, 0].min(), IO.grids[:, 0].max())
            axs[i].set_ylim(-IO.grids[:, 1].max(), IO.grids[:, 1].min())
            axs[i].set_aspect('equal')
        cb = plt.colorbar(out1[0], ax=axs[0])
        cb.set_label("Resistivity (ohm-m)")
        cb = plt.colorbar(out2[0], ax=axs[1])
        cb.set_label("Chargeability")

        plt.show()

    # Use Exponential Map: m = log(rho)
    actmap = Maps.InjectActiveCells(
        mesh, indActive=actind, valInactive=np.log(1e8)
    )
    mapping = Maps.ExpMap(mesh) * actmap

    # Generate mtrue_dc for resistivity
    mtrue_dc = np.log(rho[actind])

    # Generate 2.5D DC problem
    # "N" means potential is defined at nodes
    prb = DC.Problem2D_N(
        mesh, rhoMap=mapping, storeJ=True,
        Solver=Solver
    )
    # Pair problem with survey
    try:
        prb.pair(survey_dc)
    except:
        survey_dc.unpair()
        prb.pair(survey_dc)

    # Make synthetic DC data with 5% Gaussian noise
    dtrue_dc = survey_dc.makeSyntheticData(mtrue_dc, std=0.05, force=True)
    IO.data_dc = dtrue_dc

    # Generate mtrue_ip for chargability
    mtrue_ip = charg[actind]
    # Generate 2.5D DC problem
    # "N" means potential is defined at nodes
    prb_ip = IP.Problem2D_N(
        mesh, etaMap=actmap, storeJ=True, rho=rho,
        Solver=Solver
    )
    survey_ip = IP.from_dc_to_ip_survey(survey_dc, dim="2.5D")
    prb_ip.pair(survey_ip)
    dtrue_ip = survey_ip.makeSyntheticData(mtrue_ip, std=0.05)

    IO.data_ip = dtrue_ip

    # Show apparent resisitivty pseudo-section
    if plotIt:
        IO.plotPseudoSection(
            data_type='apparent_resistivity', scale='log',
            cmap='viridis'
        )
        plt.show()

    # Show apparent chargeability pseudo-section
    if plotIt:
        IO.plotPseudoSection(
            data_type='apparent_chargeability', scale='linear',
            cmap='magma'
        )
        plt.show()

    # Show apparent resisitivty histogram
    if plotIt:
        fig = plt.figure(figsize=(10, 4))
        ax1 = plt.subplot(121)
        out = hist(np.log10(abs(IO.voltages)), bins=20)
        ax1.set_xlabel("log10 DC voltage (V)")
        ax2 = plt.subplot(122)
        out = hist(IO.apparent_resistivity, bins=20)
        ax2.set_xlabel("Apparent Resistivity ($\Omega$m)")
        plt.tight_layout()
        plt.show()

    # Set initial model based upon histogram
    m0_dc = np.ones(actmap.nP)*np.log(100.)
    # Set uncertainty
    # floor
    eps_dc = 10**(-3.2)
    # percentage
    std_dc = 0.05

    mopt_dc, pred_dc = DC.run_inversion(
        m0_dc, survey_dc, actind, mesh, std_dc, eps_dc,
        beta0_ratio=1e0,
        use_sensitivity_weight=True
        )

    # Convert obtained inversion model to resistivity
    # rho = M(m), where M(.) is a mapping

    rho_est = mapping*mopt_dc
    rho_est[~actind] = np.nan
    rho_true = rho.copy()
    rho_true[~actind] = np.nan

    # show recovered conductivity
    if plotIt:
        vmin, vmax = rho.min(), rho.max()
        fig, ax = plt.subplots(2, 1, figsize=(20, 6))
        out1 = mesh.plotImage(
                rho_true, clim=(10, 1000),
                pcolorOpts={"cmap": "viridis", "norm": colors.LogNorm()},
                ax=ax[0]
        )
        out2 = mesh.plotImage(
            rho_est, clim=(10, 1000),
            pcolorOpts={"cmap": "viridis", "norm": colors.LogNorm()},
            ax=ax[1]
        )
        out = [out1, out2]
        for i in range(2):
            ax[i].plot(
                survey_dc.electrode_locations[:, 0],
                survey_dc.electrode_locations[:, 1], 'kv'
            )
            ax[i].set_xlim(IO.grids[:, 0].min(), IO.grids[:, 0].max())
            ax[i].set_ylim(-IO.grids[:, 1].max(), IO.grids[:, 1].min())
            cb = plt.colorbar(out[i][0], ax=ax[i])
            cb.set_label("Resistivity ($\Omega$m)")
            ax[i].set_xlabel("Northing (m)")
            ax[i].set_ylabel("Elevation (m)")
            ax[i].set_aspect('equal')
        plt.tight_layout()
        plt.show()

    # Show apparent resisitivty histogram
    if plotIt:
        fig = plt.figure(figsize=(10, 4))
        ax1 = plt.subplot(121)
        out = hist(np.log10(abs(IO.voltages_ip)), bins=20)
        ax1.set_xlabel("log10 IP voltage (V)")
        ax2 = plt.subplot(122)
        out = hist(IO.apparent_chargeability, bins=20)
        ax2.set_xlabel("Apparent Chargeability (V/V)")
        plt.tight_layout()
        plt.show()


    # Set initial model based upon histogram
    m0_ip = np.ones(actmap.nP)*1e-10
    # Set uncertainty
    # floor
    eps_ip = 10**(-4)
    # percentage
    std_ip = 0.05
    # Clean sensitivity function formed with true resistivity
    prb_ip._Jmatrix = None
    # Input obtained resistivity to form sensitivity
    prb_ip.rho = mapping*mopt_dc
    mopt_ip, _ = IP.run_inversion(
        m0_ip, survey_ip, actind, mesh, std_ip, eps_ip,
        upper=np.Inf, lower=0.,
        beta0_ratio=1e0,
        use_sensitivity_weight=True
    )

    # Convert obtained inversion model to chargeability
    # charg = M(m), where M(.) is a mapping for cells below topography

    charg_est = actmap*mopt_ip
    charg_est[~actind] = np.nan
    charg_true = charg.copy()
    charg_true[~actind] = np.nan

    # show recovered chargeability
    if plotIt:
        fig, ax = plt.subplots(2, 1, figsize=(20, 6))
        out1 = mesh.plotImage(
                charg_true, clim=(0, 0.1),
                pcolorOpts={"cmap": "magma"},
                ax=ax[0]
        )
        out2 = mesh.plotImage(
            charg_est, clim=(0, 0.1),
            pcolorOpts={"cmap": "magma"},
            ax=ax[1]
        )
        out = [out1, out2]
        for i in range(2):
            ax[i].plot(
                survey_dc.electrode_locations[:, 0],
                survey_dc.electrode_locations[:, 1], 'rv'
            )
            ax[i].set_xlim(IO.grids[:, 0].min(), IO.grids[:, 0].max())
            ax[i].set_ylim(-IO.grids[:, 1].max(), IO.grids[:, 1].min())
            cb = plt.colorbar(out[i][0], ax=ax[i])
            cb.set_label("Resistivity ($\Omega$m)")
            ax[i].set_xlabel("Northing (m)")
            ax[i].set_ylabel("Elevation (m)")
            ax[i].set_aspect('equal')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    survey_type = 'dipole-dipole'
    run(survey_type=survey_type, plotIt=True)
