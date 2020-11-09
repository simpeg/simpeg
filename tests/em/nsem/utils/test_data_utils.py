# Copyright (c) 2013 SimPEG Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the SimPEG project (https://simpeg.xyz)
import numpy as np
from SimPEG.electromagnetics.natural_source.utils.data_utils import rec_to_ndarr


def test_rec_to_ndarr():
    sample_rec_arr = np.ma.masked_array(data=[(1, 3), (2, 4)], dtype=[('a', int), ('b', float)])
    res = rec_to_ndarr(rec_arr=sample_rec_arr, data_type=float)
    exp_nd_array = np.array([[1, 3], [2, 4]], dtype=float)
    np.testing.assert_almost_equal(res, exp_nd_array)
