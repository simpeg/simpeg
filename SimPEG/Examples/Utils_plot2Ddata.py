from SimPEG import EM, np, Utils


def run(plotIt = True):
    """

        Plotting 2D data
        ================

        Often measured data is in 2D, but locations are not gridded.
        Data can be vectoral, hence we want to plot direction and
        amplitude of the vector. Following example use SimPEG's
        analytic function (electric dipole) to generate data
        at 2D plane.

    """
    # Make un-gridded xyz points

    x = np.linspace(-50, 50, 30)
    x += np.random.randn(x.size)*0.1*x
    y = np.linspace(-50, 50, 30)
    y += np.random.randn(x.size)*0.1*y
    z = np.r_[50.]
    xyz = Utils.ndgrid(x, y, z)
    sig = 1.
    f = np.r_[1.]
    srcLoc = np.r_[0., 0., 0.]

    # Use analytic fuction to compute Ex, Ey, Ez
    Ex, Ey, Ez = EM.Analytics.E_from_ElectricDipoleWholeSpace(xyz, srcLoc, sig, f)

    if plotIt:
        import matplotlib.pyplot as plt
        plt.figure()
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        # Plot Real Ex (scalar)
        cont1, ax1 = Utils.plot2Ddata(xyz, Ex.real, dataloc=True,
                                      ax=ax1, contourOpts={"cmap": "viridis"})
        # Make it as (ndata,2) matrix
        E = np.c_[Ex, Ey]
        # Plot Real E (vector)
        cont2, ax2 = Utils.plot2Ddata(xyz, E.real, vec=True,
                                      ax=ax2, contourOpts={"cmap": "viridis"})
        cb1 = plt.colorbar(cont1, ax=ax1, orientation="horizontal")
        cb2 = plt.colorbar(cont2, ax=ax2, orientation="horizontal")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")

        ax1.set_aspect('equal', adjustable='box')
        ax2.set_aspect('equal', adjustable='box')
        plt.show()



if __name__ == '__main__':
    run()



