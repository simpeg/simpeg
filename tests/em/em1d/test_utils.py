from collections import namedtuple

import pytest
import libdlf
import numpy as np
import numpy.testing as npt
from empymod.transform import get_dlf_points

from simpeg.electromagnetics.utils.em1d_utils import get_splined_dlf_points

FILTERS = [f"hankel.{filt}" for filt in libdlf.hankel.__all__] + [
    f"fourier.{filt}" for filt in libdlf.fourier.__all__
]


@pytest.mark.parametrize("filt", FILTERS)
@pytest.mark.parametrize("n_points", [1, 5, 10])
def test_splined_dlf(filt, n_points):
    f_type, f_name = filt.split(".")
    f_module = getattr(libdlf, f_type)
    filt = getattr(f_module, f_name)
    base, *vals = filt()
    if len(vals) == 2:
        v0, v1 = vals
    else:
        v0 = v1 = vals[0]
    factor = np.around([base[1] / base[0]], 15)
    filt_type = namedtuple("Filter", "base v0 v1 factor")
    filt = filt_type(base, v0, v1, factor)

    r_s = np.logspace(-1, 1, n_points)

    out1, out2 = get_splined_dlf_points(filt, r_s.min(), r_s.max())
    test1, test2 = get_dlf_points(filt, r_s, -1)

    npt.assert_allclose(out1, test1[0])
    npt.assert_allclose(out2, test2)
