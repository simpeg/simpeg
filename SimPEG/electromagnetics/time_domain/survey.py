# Copyright (c) 2013 SimPEG Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the SimPEG project (https://simpeg.xyz)
import properties
from ...survey import BaseSurvey
from .sources import BaseTDEMSrc


####################################################
# Survey
####################################################


class Survey(BaseSurvey):
    """
    Time domain electromagnetic survey
    """

    source_list = properties.List(
        "A list of sources for the survey",
        properties.Instance("A SimPEG source", BaseTDEMSrc),
        default=[],
    )

    def __init__(self, source_list=None, **kwargs):
        super(Survey, self).__init__(source_list, **kwargs)
