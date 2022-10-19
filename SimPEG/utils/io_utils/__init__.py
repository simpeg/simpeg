from __future__ import absolute_import
from __future__ import print_function

from .io_utils_general import read_GOCAD_ts, download
from .io_utils_pf import (
    read_mag3d_ubc,
    write_mag3d_ubc,
    read_grav3d_ubc,
    write_grav3d_ubc,
    read_gg3d_ubc,
    write_gg3d_ubc,
)

from .io_utils_electromagnetics import (
    read_dcip2d_ubc,
    read_dcip3d_ubc,
    read_dcipoctree_ubc,
    read_dcip_xyz,
    write_dcip2d_ubc,
    write_dcip3d_ubc,
    write_dcipoctree_ubc,
    write_dcip_xyz,
)

# Deprecated
from .io_utils_pf import (
    readUBCmagneticsObservations,
    writeUBCmagneticsObservations,
    readUBCgravityObservations,
    writeUBCgravityObservations,
)
