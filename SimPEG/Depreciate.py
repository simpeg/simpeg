from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings


def use_old_mappings():
    warnings.warn(
        'The `mapping` property has been depreciated, please use `sigmaMap`'
    )

    from SimPEG import Maps
    from SimPEG.EM.Base import BaseEMProblem

    BaseEMProblem._defaults = {
        'sigmaMap': Maps.IdentityMap()
    }
    BaseEMProblem._depreciate_maps = True
