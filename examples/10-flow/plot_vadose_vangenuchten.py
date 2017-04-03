"""
FLOW: Vadose: van Genuchten
===========================

Shows the water retention curve and the hydraulic conductivity
function for a number of soil types.

For more information about the parameters used see:

The RETC code for quantifying the hydraulic functions of unsaturated
soils, Van Genuchten, M Th, Leij, F J, Yates, S R
"""
import matplotlib.pyplot as plt

from SimPEG import Mesh
from SimPEG.FLOW import Richards


def run(plotIt=True):
    mesh = Mesh.TensorMesh([10])
    VGparams = Richards.Empirical.VanGenuchtenParams()
    leg = []
    for p in dir(VGparams):
        if p[0] == '_':
            continue
        leg += [p]
        params = getattr(VGparams, p)
        k_fun, theta_fun = Richards.Empirical.van_genuchten(mesh, **params)
        theta_fun.plot(ax=plt.subplot(121))
        k_fun.plot(ax=plt.subplot(122))

    plt.legend(leg, loc=3)

if __name__ == '__main__':
    run()
    plt.show()
