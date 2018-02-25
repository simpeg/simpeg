from __future__ import division, print_function

import properties
import warnings

from ...Utils import Zero
from ...NewSurvey import BaseSurvey
from ..NewBase import BaseEMSrc


###############################################################################
#                                                                             #
#                                  Survey                                     #
#                                                                             #
###############################################################################

class Survey(BaseEMSrc):
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

