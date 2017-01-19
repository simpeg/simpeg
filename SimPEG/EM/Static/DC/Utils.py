from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


def WennerSrcList(nElecs, aSpacing, in2D=False, plotIt=False):
    """
    Source list for a Wenner Array
    """

    import SimPEG.EM.Static.DC as DC

    elocs = np.arange(0, aSpacing*nElecs, aSpacing)
    elocs -= (nElecs*aSpacing - aSpacing)/2
    space = 1
    WENNER = np.zeros((0,), dtype=int)
    for ii in range(nElecs):
        for jj in range(nElecs):
            test = np.r_[jj,jj+space, jj+space*2, jj+space*3]
            if np.any(test >= nElecs):
                break
            WENNER = np.r_[WENNER, test]
        space += 1
    WENNER = WENNER.reshape((-1, 4))

    if plotIt:
        for i, s in enumerate('rbkg'):
            plt.plot(elocs[WENNER[:, i]], s+'.')
        plt.show()

    # Create sources and receivers
    i = 0
    if in2D:
        def getLoc(ii, abmn):
            return np.r_[elocs[WENNER[ii, abmn]], 0]
    else:
        def getLoc(ii, abmn):
            return np.r_[elocs[WENNER[ii, abmn]], 0, 0]
    srcList = []
    for i in range(WENNER.shape[0]):
        rx = DC.Rx.Dipole(getLoc(i, 1).reshape([1, -1]),
                          getLoc(i, 2).reshape([1, -1]))
        src = DC.Src.Dipole([rx], getLoc(i, 0), getLoc(i, 3))
        srcList += [src]

    return srcList
