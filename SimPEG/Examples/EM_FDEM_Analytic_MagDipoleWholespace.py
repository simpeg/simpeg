import numpy as np
from SimPEG import Mesh, Maps, Utils
import SimPEG.EM as EM


def run(XYZ=None, loc=np.r_[0., 0., 0.], sig=1.0, freq=1.0, orientation='Z',
        plotIt=True):

    """
        EM: Magnetic Dipole in a Whole-Space
        ====================================

        Here we plot the magnetic flux density from a harmonic dipole in a
        wholespace.

    """

    if XYZ is None:
        # avoid putting measurement points where source is
        x = np.arange(-100.5, 100.5, step = 1.)
        y = np.r_[0]
        z = x
        XYZ = Utils.ndgrid(x, y, z)

    Bx, By, Bz = EM.Analytics.FDEM.MagneticDipoleWholeSpace(XYZ, loc, sig,
                                                            freq,
                                                            orientation=orientation)
    absB = np.sqrt(Bx*Bx.conj()+By*By.conj()+Bz*Bz.conj()).real

    if plotIt:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        bxplt = Bx.reshape(x.size, z.size)
        bzplt = Bz.reshape(x.size, z.size)
        pc = ax.pcolor(x, z, absB.reshape(x.size, z.size), norm=LogNorm())
        ax.streamplot(x, z, bxplt.real, bzplt.real, color='k', density=1)
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([z.min(), z.max()])
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        cb = plt.colorbar(pc, ax = ax)
        cb.set_label('|B| (T)')
        plt.show()

        return fig, ax

if __name__ == '__main__':
    run()
