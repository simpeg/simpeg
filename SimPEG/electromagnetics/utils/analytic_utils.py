import numpy as np
from scipy.special import ellipk, ellipe
from scipy.constants import mu_0

import properties

from discretize.base import BaseMesh
from discretize import TensorMesh

from ...utils import mkvc


def MagneticDipoleVectorPotential(
    srcLoc, obsLoc, component, moment=1.0, orientation=np.r_[0.0, 0.0, 1.0], mu=mu_0
):
    """
    This code has been deprecated after SimPEG 0.11.5. Please use geoana instead. "

    .. code::

        >> pip install geoana
        >> from geoana.electromagnetics.static import MagneticDipoleWholeSpace
    """

    raise Exception(
        "This code has been deprecated after SimPEG 0.11.5. "
        "Please use geoana instead. "
        "\n >> pip install geoana "
        "\n >> from geoana.electromagnetics.static import MagneticDipoleWholeSpace"
    )


def MagneticDipoleFields(
    srcLoc, obsLoc, component, orientation="Z", moment=1.0, mu=mu_0
):
    """
    This code has been deprecated after SimPEG 0.11.5. Please use geoana instead. "

    .. code::

        >> pip install geoana
        >> from geoana.electromagnetics.static import MagneticDipoleWholeSpace
    """

    raise Exception(
        "This code has been deprecated after SimPEG 0.11.5. "
        "Please use geoana instead. "
        "\n >> pip install geoana "
        "\n >> from geoana.electromagnetics.static import MagneticDipoleWholeSpace"
    )


def MagneticLoopVectorPotential(
    srcLoc, obsLoc, component, radius, orientation="Z", mu=mu_0
):
    """
    This code has been deprecated after SimPEG 0.11.5. Please use geoana instead. "

    .. code::

        >> pip install geoana
        >> from geoana.electromagnetics.static import MagneticDipoleWholeSpace
    """

    raise Exception(
        "This code has been deprecated after SimPEG 0.11.5. "
        "Please use geoana instead. "
        "\n >> pip install geoana "
        "\n >> from geoana.electromagnetics.static import CircularLoopWholeSpace"
    )
