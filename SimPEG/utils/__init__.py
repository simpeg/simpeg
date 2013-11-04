import matutils
import sputils
import lomutils
import interputils
import ModelBuilder
from matutils import getSubArray, mkvc, ndgrid, ind2sub, sub2ind
from sputils import spzeros, kron3, speye, sdiag
from lomutils import volTetra, faceInfo, inv2X2BlockDiagonal, inv3X3BlockDiagonal, indexCube, exampleLomGird
from interputils import interpmat