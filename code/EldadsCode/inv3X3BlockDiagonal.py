from utils import *
from sputils import *


def inv3X3BlockDiagonal(a11, a12, a13, a21, a22, a23, a31, a32, a33):

    a11 = mkvc(a11)
    a12 = mkvc(a12)
    a13 = mkvc(a13)
    a21 = mkvc(a21)
    a22 = mkvc(a22)
    a23 = mkvc(a23)
    a31 = mkvc(a31)
    a32 = mkvc(a32)
    a33 = mkvc(a33)

    detA = a31*a12*a23 - a31*a13*a22 - a21*a12*a33 + a21*a13*a32 + a11*a22*a33 - a11*a23*a32

    b11 = +(a22*a33 - a23*a32)/detA
    b12 = -(a12*a33 - a13*a32)/detA
    b13 = +(a12*a23 - a13*a22)/detA

    b21 = +(a31*a23 - a21*a33)/detA
    b22 = -(a31*a13 - a11*a33)/detA
    b23 = +(a21*a13 - a11*a23)/detA

    b31 = -(a31*a22 - a21*a32)/detA
    b32 = +(a31*a12 - a11*a32)/detA
    b33 = -(a21*a12 - a11*a22)/detA

    B = appendBottom3(
        appendRight3(sdiag(b11), sdiag(b12),  sdiag(b13)),
        appendRight3(sdiag(b21), sdiag(b22),  sdiag(b23)),
        appendRight3(sdiag(b31), sdiag(b32),  sdiag(b33)))

    return B
