# Copyright (c) 2013 SimPEG Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the SimPEG project (https://simpeg.xyz)
from ...utils.code_utils import deprecate_module

deprecate_module("AnalyticUtils", "analytic_utils", "0.15.0")

from .analytic_utils import *
