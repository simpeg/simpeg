from __future__ import division, print_function

import properties
import warnings

from ...Utils import Zero
from ...Survey import BaseSurvey
from .SrcTDEM import BaseTDEMSrc


###############################################################################
#                                                                             #
#                                  Survey                                     #
#                                                                             #
###############################################################################

class Survey(BaseSurvey):
    """
    Time domain electromagnetic survey
    """

    srcList = properties.List(
        "A list of sources for the survey",
        properties.Instance(
            "A frequency domain EM source",
            BaseTDEMSrc
        ),
        required=True
    )

    def __init__(self, **kwargs):
        super(Survey, self).__init__(**kwargs)

