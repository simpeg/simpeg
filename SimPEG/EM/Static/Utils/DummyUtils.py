from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from .. import DC, IP


def writeUBC_DCobs(fileName, DCsurvey, dim, surveyType, iptype=0):
    """

        # Moved from old DCIP branch
        # This code needs to be verified ..
        Write UBC GIF DCIP 2D or 3D observation file

        :param string fileName: including path where the file is written out
        :param Survey DCsurvey: DC survey class object
        :param string dim:  either '2D' | '3D'
        :param string surveyType:  either 'SURFACE' | 'GENERAL'
        :rtype: file
        :return: UBC2D-Data file
    """

    from SimPEG import mkvc

    assert (dim == '2D') | (dim == '3D'), "Data must be either '2D' | '3D'"
    assert (surveyType == 'SURFACE') | (surveyType == 'GENERAL') | (surveyType == 'SIMPLE'), "Data must be either 'SURFACE' | 'GENERAL' | 'SIMPLE'"

    fid = open(fileName, 'w')
    fid.write('! ' + surveyType + ' FORMAT\n')

    if iptype!=0:
        fid.write('IPTYPE=%i\n'%iptype)

    else:
        fid.write('! ' + stype + ' FORMAT\n')

    count = 0

    for ii in range(DCsurvey.nSrc):

        tx = np.c_[DCsurvey.srcList[ii].loc]

        rx = DCsurvey.srcList[ii].rxList[0].locs

        nD = DCsurvey.srcList[ii].nD

        M = rx[0]
        N = rx[1]

        # Adapt source-receiver location for dim and surveyType
        if dim == '2D':

            if surveyType == 'SIMPLE':

                # fid.writelines("%e " % ii for ii in mkvc(tx[0, :]))
                A = np.repeat(tx[0, 0], M.shape[0], axis=0)
                B = np.repeat(tx[0, 1], M.shape[0], axis=0)
                M = M[:, 0]
                N = N[:, 0]

                np.savetxt(fid, np.c_[A, B, M, N , DCsurvey.dobs[count:count+nD], DCsurvey.std[count:count+nD] ], fmt='%e', delimiter=' ', newline='\n')

            else:

                if surveyType == 'SURFACE':

                    fid.writelines("%f " % ii for ii in mkvc(tx[0, :]))
                    M = M[:, 0]
                    N = N[:, 0]

                if surveyType == 'GENERAL':

                    # Flip sign for z-elevation to depth
                    tx[2::2, :] = -tx[2::2, :]

                    fid.writelines("%e " % ii for ii in mkvc(tx[::2, :]))
                    M = M[:, 0::2]
                    N = N[:, 0::2]

                    # Flip sign for z-elevation to depth
                    M[:, 1::2] = -M[:, 1::2]
                    N[:, 1::2] = -N[:, 1::2]

                fid.write('%i\n' % nD)
                np.savetxt(fid, np.c_[ M, N , DCsurvey.dobs[count:count+nD], DCsurvey.std[count:count+nD] ], fmt='%f', delimiter=' ', newline='\n')

        if dim == '3D':

            if surveyType == 'SURFACE':

                fid.writelines("%e " % ii for ii in mkvc(tx[0:2, :]))
                M = M[:, 0:2]
                N = N[:, 0:2]

            if surveyType == 'GENERAL':

                fid.writelines("%e " % ii for ii in mkvc(tx[0:3, :]))

            fid.write('%i\n'% nD)
            np.savetxt(fid, np.c_[ M, N , DCsurvey.dobs[count:count+nD], DCsurvey.std[count:count+nD] ], fmt='%e', delimiter=' ', newline='\n')
            fid.write('\n')

        count += nD

    fid.close()


def readUBC_DC3Dobs(fileName, rtype='DC'):
    """
        Read UBC GIF IP 3D observation file and generate survey

        :param string fileName:, path to the UBC GIF 3D obs file
        :rtype: Survey
        :return: DCIPsurvey

    """
    zflag = True # Flag for z value provided

    # Load file
    if rtype == 'IP':
        obsfile = np.genfromtxt(fileName, delimiter=' \n', dtype=np.str, comments='IPTYPE')

    elif rtype == 'DC':
        obsfile = np.genfromtxt(fileName, delimiter=' \n', dtype=np.str, comments='!')

    else:
        print "rtype must be 'DC'(default) | 'IP'"

    # Pre-allocate
    srcLists = []
    Rx = []
    d = []
    wd = []

    # Countdown for number of obs/tx
    count = 0
    for ii in range(obsfile.shape[0]):

        # Skip if blank line
        if not obsfile[ii]:
            continue

        # First line or end of a transmitter block, read transmitter info
        if count==0:
            # Read the line
            temp = (np.fromstring(obsfile[ii], dtype=float, sep=' ').T)
            count = int(temp[-1])

            # Check if z value is provided, if False -> nan
            if len(temp) == 5:
                tx = np.r_[temp[0:2], np.nan, temp[2:4], np.nan]

                zflag = False # Pass on the flag to the receiver loc

            else:
                tx = temp[:-1]

            rx = []
            continue

        temp = np.fromstring(obsfile[ii], dtype=float, sep=' ') # Get the string

        # Filter out negative IP
#        if temp[-2] < 0:
#            count = count -1
#            print "Negative!"
#
#        else:

        # If the Z-location is provided, otherwise put nan
        if zflag:

            rx.append(temp[:-2])
            # Check if there is data with the location
            if len(temp) == 8:
                d.append(temp[-2])
                wd.append(temp[-1])

        else:
            rx.append(np.r_[temp[0:2], np.nan, temp[2:4], np.nan] )
            # Check if there is data with the location
            if len(temp)==6:
                d.append(temp[-2])
                wd.append(temp[-1])

        count = count -1

        # Reach the end of transmitter block, append the src, rx and continue
        if count == 0:
            rx = np.asarray(rx)
            Rx = DC.RxDipole(rx[:, :3], rx[:, 3:])
            srcLists.append( DC.SrcDipole( [Rx], tx[:3], tx[3:]) )

    # Create survey class
    survey = DC.SurveyDC(srcLists)

    survey.dobs = np.asarray(d)
    survey.std = np.asarray(wd)

    return {'DCsurvey': survey}


def readUBC_DC2Dobs(fileName):
    """
        ------- NEEDS TO BE UPDATED ------
        Read UBC GIF 2D observation file and generate arrays for tx-rx location

        :param string fileName: path to the UBC GIF 2D model file
        :rtype: (DC.Src, DC.Rx, ??, ??)
        :return: source_locs, rx_locs, ??, ??
    """

    # Load file
    obsfile = np.genfromtxt(fileName, delimiter=' \n', dtype=np.str, comments='!')

    # Check first line and figure out if 2D or 3D file format
    line = np.array(obsfile[0].split(), dtype=float)

    tx_A = []
    tx_B = []
    rx_M = []
    rx_N = []
    d = []
    wd = []

    for ii in range(obsfile.shape[0]):

        # If len == 3, then simple format where tx-rx is listed on each line
        if len(line) == 4:

            temp = np.fromstring(obsfile[ii], dtype=float, sep=' ')
            tx_A = np.hstack((tx_A, temp[0]))
            tx_B = np.hstack((tx_B, temp[1]))
            rx_M = np.hstack((rx_M, temp[2]))
            rx_N = np.hstack((rx_N, temp[3]))

    rx = np.transpose(np.array((rx_M, rx_N)))
    tx = np.transpose(np.array((tx_A, tx_B)))

    return tx, rx, d, wd


def readUBC_DC2Dpre(fileName):
    """
        Read UBC GIF DCIP 2D observation file and generate arrays for tx-rx location

        Input:
        :param string fileName: path to the UBC GIF 3D obs file
        :rtype: DC.Survey
        :return: DCsurvey

        Created on Mon March 9th, 2016 << Doug's 70th Birthday !! >>

        @author: dominiquef

    """

    obsfile = np.genfromtxt(fileName, delimiter=' \n', dtype=np.str, comments='!')

    # Pre-allocate
    srcLists = []
    Rx = []
    d = []
    zflag = True  # Flag for z value provided

    for ii in range(obsfile.shape[0]):

        if not obsfile[ii]:
            continue

        # First line is transmitter with number of receivers

        temp = (np.fromstring(obsfile[ii], dtype=float, sep=' ').T)

        # Check if z value is provided, if False -> nan
        if len(temp) == 5:
            tx = np.r_[temp[0], np.nan, np.nan, temp[1], np.nan, np.nan]
            zflag = False

        else:
            tx = np.r_[temp[0], np.nan, temp[1], temp[2], np.nan, temp[3]]

        if zflag:
            rx = np.c_[temp[4], np.nan, temp[5], temp[6], np.nan, temp[7]]

        else:
            rx = np.c_[temp[2], np.nan, np.nan, temp[3], np.nan, np.nan]
            # Check if there is data with the location

        d.append(temp[-1])

        Rx = DC.RxDipole(rx[:, :3], rx[:, 3:])
        srcLists.append( DC.SrcDipole( [Rx], tx[:3], tx[3:]) )

    # Create survey class
    survey = DC.SurveyDC(srcLists)

    survey.dobs = np.asarray(d)

    return {'DCsurvey': survey}


def readUBC_DC2DMesh(fileName):
    """
        Read UBC GIF 2DTensor mesh and generate 2D Tensor mesh in simpeg

        :param string fileName: path to the UBC GIF mesh file
        :rtype: Mesh.TensorMesh
        :return: SimPEG TensorMesh 2D object

        Created on Thu Nov 12 13:14:10 2015

        @author: dominiquef

    """

    fopen = open(fileName, 'r')

    # Read down the file and unpack dx vector
    def unpackdx(fid, nrows):
        for ii in range(nrows):

            line = fid.readline()
            var = np.array(line.split(), dtype=float)

            if ii==0:
                x0= var[0]
                xvec = np.ones(int(var[2])) * (var[1] - var[0]) / int(var[2])
                xend = var[1]

            else:
                xvec = np.hstack((xvec, np.ones(int(var[1])) * (var[0] - xend) / int(var[1])))
                xend = var[0]

        return x0, xvec

    # Start with dx block
    # First line specifies the number of rows for x-cells
    line = fopen.readline()
    nl = np.array(line.split(), dtype=float)

    [x0, dx] = unpackdx(fopen, nl)

    # Move down the file until reaching the z-block
    line = fopen.readline()
    if not line:
        line = fopen.readline()

    # End with dz block
    # First line specifies the number of rows for z-cells
    line = fopen.readline()
    nl = np.array(line.split(), dtype=float)

    [z0, dz] = unpackdx(fopen, nl)

    # Flip z0 to be the bottom of the mesh for SimPEG
    z0 = z0 - sum(dz)
    dz = dz[::-1]
    # Make the mesh using SimPEG

    from SimPEG import Mesh
    tensMsh = Mesh.TensorMesh([dx, dz], (x0, z0))
    return tensMsh


def xy_2_lineID(DCsurvey):
    """
        Read DC survey class and append line ID.
        Assumes that the locations are listed in the order
        they were collected. May need to generalize for random
        point locations, but will be more expensive

        :param numpy.array DCdict: Vectors of station location
        :rtype: numpy.array
        :return: LineID Vector of integers

        Created on Thu Feb 11, 2015

        @author: dominiquef

    """

    # Compute unit vector between two points
    nstn = DCsurvey.nSrc

    # Pre-allocate space
    lineID = np.zeros(nstn)

    linenum = 0
    indx = 0

    for ii in range(nstn):

        if ii == 0:

            A = DCsurvey.srcList[ii].loc[0]
            B = DCsurvey.srcList[ii].loc[1]

            xout = np.mean([A[0:2], B[0:2]], axis = 0)

            xy0 = A[:2]
            xym = xout

            # Deal with replicate pole location
            if np.all(xy0 == xym):

                xym[0] = xym[0] + 1e-3

            continue

        A = DCsurvey.srcList[ii].loc[0]
        B = DCsurvey.srcList[ii].loc[1]

        xin = np.mean([A[0:2], B[0:2]], axis=0)

        # Compute vector between neighbours
        vec1, r1 = r_unit(xout, xin)

        # Compute vector between current stn and mid-point
        vec2, r2 = r_unit(xym, xin)

        # Compute vector between current stn and start line
        vec3, r3 = r_unit(xy0, xin)

        # Compute vector between mid-point and start line
        vec4, r4 = r_unit(xym, xy0)

        # Compute dot product
        ang1 = np.abs(vec1.dot(vec2))
        ang2 = np.abs(vec3.dot(vec4))

        # If the angles are smaller then 45d, than next point is on a new line
        if ((ang1 < np.cos(np.pi/4.)) | (ang2 < np.cos(np.pi/4.))) & (np.all(np.r_[r1, r2, r3, r4] > 0)):

            # Re-initiate start and mid-point location
            xy0 = A[:2]
            xym = xin

            # Deal with replicate pole location
            if np.all(xy0 == xym):

                xym[0] = xym[0] + 1e-3

            linenum += 1
            indx = ii

        else:
            xym = np.mean([xy0, xin], axis = 0)

        lineID[ii] = linenum
        xout = xin

    return lineID


def r_unit(p1, p2):
    """
    r_unit(x, y) : Function computes the unit vector
    between two points with coordinates p1(x1, y1) and p2(x2, y2)

    """

    assert len(p1) == len(p2), 'locs must be the same shape.'

    dx = []
    for ii in range(len(p1)):
        dx.append((p2[ii] - p1[ii]))

    # Compute length of vector
    r = np.linalg.norm(np.asarray(dx))

    if r != 0:
        vec = dx/r

    else:
        vec = np.zeros(len(p1))

    return vec, r


def getSrc_locs(DCsurvey):

    srcMat = np.zeros((DCsurvey.nSrc, 2, 3))
    for ii in range(DCsurvey.nSrc):
        srcMat[ii, :, :] = np.asarray(DCsurvey.srcList[ii].loc)

    return srcMat
