from SimPEG import Mesh, np, PF


def run(plotIt=True):
    """
        PF: Magnetics: Analytics
        ========================

        Comparing the magnetics field in Vancouver to Seoul

    """

    xr = np.linspace(-300, 300, 41)
    yr = np.linspace(-300, 300, 41)
    X, Y = np.meshgrid(xr, yr)
    Z = np.ones((np.size(xr), np.size(yr)))*150

    # Bz component in Korea
    inckr = -8. + 3./60
    deckr = 54. + 9./60
    btotkr = 50898.6
    Bokr = PF.MagAnalytics.IDTtoxyz(inckr, deckr, btotkr)

    bx, by, bz = PF.MagAnalytics.MagSphereAnaFunA(
        X, Y, Z, 100., 0., 0., 0., 0.01, Bokr, 'secondary'
    )
    Bzkr = np.reshape(bz, (np.size(xr), np.size(yr)), order='F')

    # Bz component in Canada
    incca = 16. + 49./60
    decca = 70. + 19./60
    btotca = 54692.1
    Boca = PF.MagAnalytics.IDTtoxyz(incca, decca, btotca)

    bx, by, bz = PF.MagAnalytics.MagSphereAnaFunA(
        X, Y, Z, 100., 0., 0., 0., 0.01, Boca, 'secondary'
    )
    Bzca = np.reshape(bz, (np.size(xr), np.size(yr)), order='F')

    if plotIt:
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig = plt.figure(figsize=(14, 5))

        ax1 = plt.subplot(121)
        dat1 = plt.imshow(Bzkr, extent=[min(xr), max(xr), min(yr), max(yr)])
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        ax1.set_xlabel('East-West (m)')
        ax1.set_ylabel('South-North (m)')
        plt.colorbar(dat1, cax=cax1)
        ax1.set_title('$B_z$ field at Seoul, South Korea')

        ax2 = plt.subplot(122)
        dat2 = plt.imshow(Bzca, extent=[min(xr), max(xr), min(yr), max(yr)])
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        ax2.set_xlabel('East-West (m)')
        ax2.set_ylabel('South-North (m)')
        plt.colorbar(dat2, cax=cax2)
        ax2.set_title('$B_z$ field at Vancouver, Canada')
        plt.show()

if __name__ == '__main__':
    run()
