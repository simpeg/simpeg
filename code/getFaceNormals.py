from numpy import *
from utils import *


def getFaceNormals(X, Y, Z):
# compute the x normals
	d1xp = diffp(X,2,3)
	d1yp = diffp(Y,2,3)
	d1zp = diffp(Z,2,3)

	d1xm = diffm(X,3,2)
	d1ym = diffm(Y,3,2)
	d1zm = diffm(Z,3,2)

	# normals 
	n1x = d1yp*d1zm - d1zp*d1ym
	n1y = d1zp*d1xm - d1xp*d1zm
	n1z = d1xp*d1ym - d1yp*d1xm
	normn1 = sqrt(n1x**2 + n1y**2 + n1z**2)
	n1x = n1x / normn1;
	n1y = n1y / normn1;
	n1z = n1z / normn1;

	area1 = normn1/2


	# compute the y normals 
	d2xp = diffp(X,1,3)
	d2yp = diffp(Y,1,3)
	d2zp = diffp(Z,1,3)

	d2xm = diffm(X,1,3)
	d2ym = diffm(Y,1,3)
	d2zm = diffm(Z,1,3)

	# normals 
	n2x = d2yp*d2zm - d2zp*d2ym
	n2y = d2zp*d2xm - d2xp*d2zm
	n2z = d2xp*d2ym - d2yp*d2xm
	normn2 = sqrt(n2x**2 + n2y**2 + n2z**2)
	n2x = n2x / normn2
	n2y = n2y / normn2
	n2z = n2z / normn2

	area2 = normn2/2

	# compute the z normals 
	d3xp = diffp(X,1,2)
	d3yp = diffp(Y,1,2)
	d3zp = diffp(Z,1,2)

	d3xm = diffm(X,2,1)
	d3ym = diffm(Y,2,1)
	d3zm = diffm(Z,2,1)

	# normals 
	n3x = d3yp*d3zm - d3zp*d3ym
	n3y = d3zp*d3xm - d3xp*d3zm
	n3z = d3xp*d3ym - d3yp*d3xm;
	normn3 = sqrt(n3x**2 + n3y**2 + n3z**2);
	n3x = n3x / normn3;
	n3y = n3y / normn3;
	n3z = n3z / normn3;

	area3 = normn3/2;

	return (n1x,n1y,n1z,n2x,n2y,n2z,n3x,n3y,n3z,area1,area2,area3)

if __name__ == '__main__':

    X, Y, Z = mgrid[0:4, 0:5, 0:6]

    t = getFaceNormals(X, Y, Z)
