# Copyright (c) 2013 SimPEG Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the SimPEG project (https://simpeg.xyz)
from ....utils.code_utils import deprecate_module

deprecate_module("plotDataTypes", "plot_data_types", "0.15.0")

from .plot_data_types import *
