import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import mu_0

from SimPEG import Mesh, Utils, Maps
from SimPEG.EM import FDEM

# Try importing PardisoSolver from pymatsolver
# otherwise, use SolverLU from SimPEG
try:
    from pymatsolver import PardisoSolver as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

# Set a nice colormap!
plt.set_cmap(plt.get_cmap('viridis'))



def run(plotIt=True):
    """
        Mesh: Plotting with defining range
        ==================================

        When using a large Mesh with the cylindrical code, it is advantageous
        to define a :code:`range_x` and :code:`range_y` when plotting with
        vectors. In this case, only the region inside of the range is
        interpolated. In particular, you often want to ignore padding cells.

    """

    # ## Model Parameters
    #
    # We define a
    # - resistive halfspace and
    # - conductive sphere
    #    - radius of 30m
    #    - center is 50m below the surface

    # electrical conductivities in S/m
    sig_halfspace = 1e-6
    sig_sphere = 1e0
    sig_air = 1e-8



    # depth to center, radius in m
    sphere_z = -50.
    sphere_radius = 30.


    # ## Survey Parameters
    #
    # - Transmitter and receiver 20m above the surface
    # - Receiver offset from transmitter by 8m horizontally
    # - 25 frequencies, logaritmically between $10$ Hz and $10^5$ Hz


    boom_height = 20.
    rx_offset = 8.
    freqs = np.r_[1e1, 1e5]

    # source and receiver location in 3D space
    src_loc = np.r_[0., 0., boom_height]
    rx_loc = np.atleast_2d(np.r_[rx_offset, 0., boom_height])



    # print the min and max skin depths to make sure mesh is fine enough and
    # extends far enough

    def skin_depth(sigma, f):
        return 500./np.sqrt(sigma * f)

    print(
        'Minimum skin depth (in sphere): {:.2e} m '.format(
            skin_depth(sig_sphere, freqs.max())
        )
    )
    print(
        'Maximum skin depth (in background): {:.2e} m '.format(
            skin_depth(sig_halfspace, freqs.min())
        )
    )


    # ## Mesh
    #
    # Here, we define a cylindrically symmetric tensor mesh.
    #
    # ### Mesh Parameters
    #
    # For the mesh, we will use a cylindrically symmetric tensor mesh. To
    # construct a tensor mesh, all that is needed is a vector of cell widths in
    # the x and z-directions. We will define a core mesh region of uniform cell
    # widths and a padding region where the cell widths expand "to infinity".

    # x-direction
    csx = 2  # core mesh cell width in the x-direction
    ncx = np.ceil(1.2*sphere_radius/csx)  # number of core x-cells (uniform mesh slightly beyond sphere radius)
    npadx = 50  # number of x padding cells

    # z-direction
    csz = 1  # core mesh cell width in the z-direction
    ncz = np.ceil(1.2*(boom_height - (sphere_z-sphere_radius))/csz) # number of core z-cells (uniform cells slightly below bottom of sphere)
    npadz = 52  # number of z padding cells

    # padding factor (expand cells to infinity)
    pf = 1.3

    # cell spacings in the x and z directions
    hx = Utils.meshTensor([(csx, ncx), (csx, npadx, pf)])
    hz = Utils.meshTensor([(csz, npadz, -pf), (csz, ncz), (csz, npadz, pf)])

    # define a SimPEG mesh
    mesh = Mesh.CylMesh([hx, 1, hz], x0 = np.r_[0.,0., -hz.sum()/2.-boom_height])


    # ### Plot the mesh
    #
    # Below, we plot the mesh. The cyl mesh is rotated around x=0. Ensure that
    # each dimension extends beyond the maximum skin depth.
    #
    # Zoom in by changing the xlim and zlim.

    # X and Z limits we want to plot to. Try
    xlim = np.r_[0., 2.5e6]
    zlim = np.r_[-2.5e6, 2.5e6]

    fig, ax = plt.subplots(1,1)
    mesh.plotGrid(ax=ax)

    ax.set_title('Simulation Mesh')
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)

    print(
        'The maximum skin depth is (in background): {:.2e} m. '
        'Does the mesh go sufficiently past that?'.format(
            skin_depth(sig_halfspace, freqs.min())
        )
    )


    # ## Put Model on Mesh
    #
    # Now that the model parameters and mesh are defined, we can define
    # electrical conductivity on the mesh.
    #
    # The electrical conductivity is defined at cell centers when using the
    # finite volume method. So here, we define a vector that contains an
    # electrical conductivity value for every cell center.

    # create a vector that has one entry for every cell center
    sigma = sig_air*np.ones(mesh.nC)  # start by defining the conductivity of the air everwhere
    sigma[mesh.gridCC[:, 2] < 0.] = sig_halfspace  # assign halfspace cells below the earth

    # indices of the sphere (where (x-x0)**2 + (z-z0)**2 <= R**2)
    sphere_ind =(
        (mesh.gridCC[:,0]**2 + (mesh.gridCC[:,2] - sphere_z)**2) <=
        sphere_radius**2
    )
    sigma[sphere_ind] = sig_sphere  # assign the conductivity of the sphere

    # Plot a cross section of the conductivity model
    fig, ax = plt.subplots(1, 1)
    cb = plt.colorbar(mesh.plotImage(np.log10(sigma), ax=ax, mirror=True)[0])

    # plot formatting and titles
    cb.set_label('$\log_{10}\sigma$', fontsize=13)
    ax.axis('equal')
    ax.set_xlim([-120., 120.])
    ax.set_ylim([-100., 30.])
    ax.set_title('Conductivity Model')


    # ## Set up the Survey
    #
    # Here, we define sources and receivers. For this example, the receivers
    # are magnetic flux recievers, and are only looking at the secondary field
    # (eg. if a bucking coil were used to cancel the primary). The source is a
    # vertical magnetic dipole with unit moment.


    # Define the receivers, we will sample the real secondary magnetic flux
    # density as well as the imaginary magnetic flux density

    bz_r = FDEM.Rx.Point_bSecondary(
        locs=rx_loc, orientation='z', component='real'
    )  # vertical real b-secondary
    bz_i = FDEM.Rx.Point_b(
        locs=rx_loc, orientation='z', component='imag'
    )  # vertical imag b (same as b-secondary)

    rxList = [bz_r, bz_i]  # list of receivers


    # Define the list of sources - one source for each frequency. The source is
    # a point dipole oriented in the z-direction

    srcList = [
        FDEM.Src.MagDipole(rxList, f, src_loc, orientation='z') for f in freqs
    ]

    print(
        'There are {nsrc} sources (same as the number of frequencies - {nfreq}). '
        'Each source has {nrx} receivers sampling the resulting b-fields'.format(
            nsrc = len(srcList),
            nfreq = len(freqs),
            nrx = len(rxList)
        )
    )


    # ## Set up Forward Simulation
    #
    # A forward simulation consists of a paired SimPEG problem and Survey.
    # For this example, we use the E-formulation of Maxwell's equations,
    # solving the second-order system for the electric field, which is defined
    # on the cell edges of the mesh. This is the `prob` variable below. The
    # `survey` takes the source list which is used to construct the RHS for the
    # problem. The source list also contains the receiver information, so the
    # `survey` knows how to sample fields and fluxes that are produced by
    # solving the `prob`.

    # define a problem - the statement of which discrete pde system we want to
    # solve
    prob = FDEM.Problem3D_e(mesh, sigmaMap=Maps.IdentityMap(mesh))
    prob.solver = Solver

    survey = FDEM.Survey(srcList)

    # tell the problem and survey about each other - so the RHS can be
    # constructed for the problem and the
    # resulting fields and fluxes can be sampled by the receiver.
    prob.pair(survey)


    # ### Solve the forward simulation
    #
    # Here, we solve the problem for the fields everywhere on the mesh.
    fields = prob.fields(sigma)


    # ### Plot the fields
    #
    # Lets look at the physics!

    # log-scale the colorbar
    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    def plotMe(field, ax):
        plt.colorbar(mesh.plotImage(
            field, vType='F', view='vec',
            range_x=[-100., 100.], range_y=[-180., 60.],
            pcolorOpts={
                    'norm': LogNorm(), 'cmap': plt.get_cmap('viridis')
                },
            streamOpts={'color': 'k'},
            ax=ax, mirror=True
        )[0], ax=ax)

    plotMe(fields[srcList[0], 'bSecondary'].real, ax[0])
    ax[0].set_title('Real B-Secondary, {}Hz'.format(freqs[0]))

    plotMe(fields[srcList[1], 'bSecondary'].real, ax[1])
    ax[1].set_title('Real B-Secondary, {}Hz'.format(freqs[1]))

    plt.tight_layout()

    if plotIt:
        plt.show()

if __name__ == '__main__':
    run(plotIt=True)


