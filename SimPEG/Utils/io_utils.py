from __future__ import print_function
import numpy as np
import time as tm
import re
import warnings
from SimPEG import Utils
from discretize import TreeMesh


def read_GOCAD_ts(tsfile):
    """

    Read GOCAD triangulated surface (*.ts) file
    INPUT:
    tsfile: Triangulated surface

    OUTPUT:
    vrts : Array of vertices in XYZ coordinates [n x 3]
    trgl : Array of index for triangles [m x 3]. The order of the vertices
            is important and describes the normal
            n = cross( (P2 - P1 ) , (P3 - P1) )

    Author: @fourndo


    .. note::

        Remove all attributes from the GoCAD surface before exporting it!

    """

    fid = open(tsfile, 'r')
    line = fid.readline()

    # Skip all the lines until the vertices
    VRTX, TRGL = [], []
    while 'END' not in line:

        while 'VRTX' not in line:
            line = fid.readline()
            if 'END\n' in line:
                return VRTX, TRGL

        vrtx = []
        # Run down all the vertices and save in array
        while np.any(['VRTX' in line, 'PVRTX' in line]):
            l_input = re.split('[\s*]', line)
            temp = np.array(l_input[2:5])
            vrtx.append(temp.astype(np.float))

            # Read next line
            line = fid.readline()

        VRTX += [np.asarray(vrtx)]

        # Skip lines to the triangles
        while 'TRGL' not in line:
            line = fid.readline()

        # Run down the list of triangles
        trgl = []

        # Run down all the vertices and save in array
        while 'TRGL' in line:
            l_input = re.split('[\s*]', line)
            temp = np.array(l_input[1:4])
            trgl.append(temp.astype(np.int))

            # Read next line
            line = fid.readline()

        TRGL += [np.asarray(trgl)]

    return VRTX, TRGL


def read_GOCAD_pl(plFile):
    """

    Read GOCAD polyline file (*.pl)
    INPUT:
    plFile: Polyline object

    OUTPUT:
    vrts : List Array of vertices in XYZ coordinates [n x 3]
    segs : List Array of index for segments [m x 3]. The order of the vertices
            is important and describes the normal
            n = cross( (P2 - P1 ) , (P3 - P1) )

    Author: @fourndo


    .. note::

        Remove all attributes from the GoCAD surface before exporting it!

    """

    fid = open(plFile, 'r')
    line = fid.readline()

    # Skip all the lines until the vertices
    VRTX, SEGS = [], []
    while 'END' not in line:

        while 'VRTX' not in line:
            line = fid.readline()

        vrtx = []
        # Run down all the vertices and save in array
        while np.any(['VRTX' in line, 'PVRTX' in line]):
            l_input = re.split('[\s*]', line)
            temp = np.array(l_input[2:5])
            vrtx.append(temp.astype(np.float))

            # Read next line
            line = fid.readline()

        VRTX += [np.asarray(vrtx)]

        # Skip lines to the triangles
        while 'SEG' not in line:
            line = fid.readline()

        # Run down the list of triangles
        segs = []

        # Run down all the vertices and save in array
        while 'SEG' in line:
            l_input = re.split('[\s*]', line)
            temp = np.array(l_input[1:3])
            segs.append(temp.astype(np.int))

            # Read next line
            line = fid.readline()

        SEGS += [np.asarray(segs)]

    return VRTX, SEGS


def surface2inds(vrtx, trgl, mesh, boundaries=True, internal=True):
    """"
    Function to read gocad polystructure file and output indexes of
    mesh with in the structure.

    """
    import vtk
    import vtk.util.numpy_support as npsup

    # Adjust the index
    trgl = trgl - 1

    # Make vtk pts
    ptsvtk = vtk.vtkPoints()
    ptsvtk.SetData(npsup.numpy_to_vtk(vrtx, deep=1))

    # Make the polygon connection
    polys = vtk.vtkCellArray()
    for face in trgl:
        poly = vtk.vtkPolygon()
        poly.GetPointIds().SetNumberOfIds(len(face))
        for nrv, vert in enumerate(face):
            poly.GetPointIds().SetId(nrv, vert)
        polys.InsertNextCell(poly)

    # Make the polydata, structure of connections and vrtx
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(ptsvtk)
    polyData.SetPolys(polys)

    # Make implicit func
    ImpDistFunc = vtk.vtkImplicitPolyDataDistance()
    ImpDistFunc.SetInput(polyData)

    # Convert the mesh
    vtkMesh = vtk.vtkRectilinearGrid()
    vtkMesh.SetDimensions(mesh.nNx, mesh.nNy, mesh.nNz)
    vtkMesh.SetXCoordinates(npsup.numpy_to_vtk(mesh.vectorNx, deep=1))
    vtkMesh.SetYCoordinates(npsup.numpy_to_vtk(mesh.vectorNy, deep=1))
    vtkMesh.SetZCoordinates(npsup.numpy_to_vtk(mesh.vectorNz, deep=1))
    # Add indexes
    vtkInd = npsup.numpy_to_vtk(np.arange(mesh.nC), deep=1)
    vtkInd.SetName('Index')
    vtkMesh.GetCellData().AddArray(vtkInd)

    extractImpDistRectGridFilt = vtk.vtkExtractGeometry()  # Object constructor
    extractImpDistRectGridFilt.SetImplicitFunction(ImpDistFunc)  #
    extractImpDistRectGridFilt.SetInputData(vtkMesh)

    if boundaries is True:
        extractImpDistRectGridFilt.ExtractBoundaryCellsOn()

    else:
        extractImpDistRectGridFilt.ExtractBoundaryCellsOff()

    if internal is True:
        extractImpDistRectGridFilt.ExtractInsideOn()

    else:
        extractImpDistRectGridFilt.ExtractInsideOff()

    print("Extracting indices from grid...")
    # Executing the pipe
    extractImpDistRectGridFilt.Update()

    # Get index inside
    insideGrid = extractImpDistRectGridFilt.GetOutput()
    insideGrid = npsup.vtk_to_numpy(insideGrid.GetCellData().GetArray('Index'))

    # Return the indexes inside
    return insideGrid


def download(
    url, folder='.', overwrite=False, verbose=True
):
    """
    Function to download all files stored in a cloud directory

    :param str url: url or list of urls for the file(s) to be downloaded ("https://...")
    :param str folder: folder to where the directory is created and files downloaded (default is the current directory)
    :param bool overwrite: overwrite if a file with the specified name already exists
    :param bool verbose: print out progress
    """

    # Download from cloud
    import urllib
    import os
    import sys

    def rename_path(downloadpath):
        splitfullpath = downloadpath.split(os.path.sep)

        # grab just the filename
        fname = splitfullpath[-1]
        fnamesplit = fname.split('.')
        newname = fnamesplit[0]

        # check if we have already re-numbered
        newnamesplit = newname.split('(')

        # add (num) to the end of the filename
        if len(newnamesplit) == 1:
            num = 1
        else:
            num = int(newnamesplit[-1][:-1])
            num += 1

        newname = '{}({}).{}'.format(newnamesplit[0], num, fnamesplit[-1])
        return os.path.sep.join(
            splitfullpath[:-1] + newnamesplit[:-1] + [newname]
        )

    # grab the correct url retriever
    if sys.version_info < (3,):
        urlretrieve = urllib.urlretrieve
    else:
        urlretrieve = urllib.request.urlretrieve

    # ensure we are working with absolute paths and home directories dealt with
    folder = os.path.abspath(os.path.expanduser(folder))

    # make the directory if it doesn't currently exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    if isinstance(url, str):
        filenames = [url.split('/')[-1]]
    elif isinstance(url, list):
        filenames = [u.split('/')[-1] for u in url]

    downloadpath = [os.path.sep.join([folder, f]) for f in filenames]

    # check if the directory already exists
    for i, download in enumerate(downloadpath):
        if os.path.exists(download):
            if overwrite is True:
                if verbose is True:
                    print("overwriting {}".format(download))
            elif overwrite is False:
                while os.path.exists is True:
                    download = rename_path(download)

                if verbose is True:
                    print(
                        "file already exists, new file is called {}".format(
                            download
                        )
                    )
                downloadpath[i] = download

    # download files
    urllist = url if isinstance(url, list) else [url]
    for u, f in zip(urllist, downloadpath):
        print("Downloading {}".format(u))
        urlretrieve(u, f)
        print("   saved to: " + f)

    print("Download completed!")
    return downloadpath if isinstance(url, list) else downloadpath[0]


def readUBCmagneticsObservations(obs_file):
    """
        Read and write UBC mag file format

        INPUT:
        :param fileName, path to the UBC obs mag file

        OUTPUT:
        :param survey
        :param M, magnetization orentiaton (MI, MD)
    """
    from SimPEG.PF import BaseMag
    fid = open(obs_file, 'r')

    # First line has the inclination,declination and amplitude of B0
    line = fid.readline()
    B = np.array(line.split(), dtype=float)

    # Second line has the magnetization orientation and a flag
    line = fid.readline()
    M = np.array(line.split(), dtype=float)

    # Third line has the number of rows
    line = fid.readline()
    ndat = int(line.strip())

    # Pre-allocate space for obsx, obsy, obsz, data, uncert
    line = fid.readline()
    temp = np.array(line.split(), dtype=float)

    d = np.zeros(ndat, dtype=float)
    wd = np.zeros(ndat, dtype=float)
    locXYZ = np.zeros((ndat, 3), dtype=float)

    ii = 0
    while ii < ndat:

        temp = np.array(line.split(), dtype=float)
        if len(temp) > 0:
            locXYZ[ii, :] = temp[:3]

            if len(temp) > 3:
                d[ii] = temp[3]

                if len(temp) == 5:
                    wd[ii] = temp[4]
            ii += 1
        line = fid.readline()

    rxLoc = BaseMag.RxObs(locXYZ)
    srcField = BaseMag.SrcField([rxLoc], param=(B[2], B[0], B[1]))
    survey = BaseMag.LinearSurvey(srcField)
    survey.dobs = d
    survey.std = wd
    return survey, M


def writeUBCmagneticsObservations(filename, survey, d):
    """
    writeUBCobs(filename,B,M,rxLoc,d,wd)

    Function writing an observation file in UBC-MAG3D format.

    INPUT
    filename    : Name of out file including directory
    survey
    flag          : dobs | dpred

    OUTPUT
    Obsfile

    Created on Dec, 27th 2015

    @author: dominiquef
    """

    B = survey.srcField.param

    rxLoc = survey.srcField.rxList[0].locs

    wd = survey.std

    data = np.c_[rxLoc, d, wd]
    head = ('%6.2f %6.2f %6.2f\n' % (B[1], B[2], B[0]) +
            '%6.2f %6.2f %6.2f\n' % (B[1], B[2], 1) +
            '%i' % len(d))
    np.savetxt(
        filename, data, fmt='%e', delimiter=' ',
        newline='\n', header=head, comments=''
    )

    print("Observation file saved to: " + filename)


def readUBCgravityObservations(obs_file):
    """
    Read UBC grav file format

    INPUT:
    :param fileName, path to the UBC obs grav file

    OUTPUT:
    :param survey

    """
    from SimPEG.PF import BaseGrav
    fid = open(obs_file, 'r')

    # First line has the number of rows
    line = fid.readline()
    ndat = int(line.split()[0])

    # Pre-allocate space for obsx, obsy, obsz, data, uncert
    line = fid.readline()

    d = np.zeros(ndat, dtype=float)
    wd = np.zeros(ndat, dtype=float)
    locXYZ = np.zeros((ndat, 3), dtype=float)

    ii = 0
    while ii < ndat:
        temp = np.array(line.split(), dtype=float)
        if len(temp) > 0:
            locXYZ[ii, :] = temp[:3]
            d[ii] = temp[3]
            wd[ii] = temp[4]
            ii += 1
        line = fid.readline()

    rxLoc = BaseGrav.RxObs(locXYZ)
    srcField = BaseGrav.SrcField([rxLoc])
    survey = BaseGrav.LinearSurvey(srcField)
    survey.dobs = -d
    survey.std = wd
    return survey


def writeUBCgravityObservations(filename, survey, d):
    """
        Write UBC grav file format

        INPUT:
        :param: fileName, path to the UBC obs grav file
        :param: survey Gravity object
        :param: data array

    """
    rxLoc = survey.srcField.rxList[0].locs

    wd = survey.std

    data = np.c_[rxLoc, -d, wd]
    head = ('%i' % len(d))
    np.savetxt(
        filename, data, fmt='%e', delimiter=' ',
        newline='\n', header=head, comments=''
    )

    print("Observation file saved to: " + filename)


def writeVectorUBC(mesh, fileName, model):
    """
        Writes a vector model associated with a SimPEG TensorMesh
        to a UBC-GIF format model file.

        :param string fileName: File to write to
        :param numpy.ndarray model: The model
    """

    if model.ndim == 1:
        # Catch the standard case that the model is (3*nC,1) instead of (nC,3)
        model = model.reshape((-1, 3), order='F')
        
    modelMatTR = np.zeros_like(model)

    if isinstance(mesh, TreeMesh):
        ubc_order = mesh._ubc_order

        for ii in range(3):
            modelMatTR[:, ii] = model[ubc_order, ii]

    else:
        for ii in range(3):
            # Reshape model to a matrix
            modelMat = mesh.r(model[:, ii], 'CC', 'CC', 'M')
            # Transpose the axes
            modelMatT = modelMat.transpose((2, 0, 1))
            # Flip z to positive down
            modelMatTR[:, ii] = Utils.mkvc(modelMatT[::-1, :, :])

    # Flip Z for UBC file format
    modelMatTR[:, 2] *= -1

    np.savetxt(fileName, modelMatTR)


def readVectorUBC(mesh, fileName):
    """
        Read a vector model associated with a SimPEG TensorMesh
        to a UBC-GIF format model file.

        :param string fileName: File to write to
        :param numpy.ndarray model: The model
    """

    # f = open(fileName, 'r')
    model = np.loadtxt(fileName)
    vModel = np.zeros((mesh.nC, 3))
    # f.close()
    if isinstance(mesh, TreeMesh):
        ubc_order = mesh._ubc_order

        for ii in range(3):
            vModel[ubc_order, ii] = model[:, ii]
    else:

        for ii in range(3):
            comp = np.reshape(model[:, ii], (mesh.nCz, mesh.nCx, mesh.nCy), order='F')
            comp = comp[::-1, :, :]
            comp = np.transpose(comp, (1, 2, 0))
            vModel[:, ii] = Utils.mkvc(comp)

    # Flip the z vector
    vModel[:, 2] *= -1
    return vModel


class readFTMGdataFile:
    """
    Patch object for holding FTMG data

    """
    #

    def __init__(self, file_path=None):
        if file_path is not None:
            # initiate variables
            self.lines = []
            self.project_name = ''
            self.date = ''
            self.id = ''
            # read the data file
            input_file = open(file_path)
            # extract lines
            cnt = 0
            cnt_line = 0
            header_ids = []
            lines = []
            for dline in input_file:
                # check for header line
                if '/' in dline:
                    if cnt == 1:
                        splt = dline.split('/')
                        self.project_name = splt[1]
                    if cnt == 2:
                        splt = dline.split('/')
                        self.date = splt[1]
                    if cnt == 3:
                        splt = dline.split('/')
                        self.id = splt[1]
                    if cnt == 17:
                        splt = dline.split('/')
                        header_ids = splt[1].split()
                    cnt += 1
                else:
                    if 'Line' in dline:
                        if cnt_line == 0:
                            splt = dline.split()
                            new_line = ftmgLine(splt[1])
                            lines.append(new_line)
                            cnt_line += 1
                        else:
                            # print('number of stations: {0} in line: {1}'.format(len(lines[0].stations), lines[0].line_id))
                            self.lines.append(lines[0])
                            splt = dline.split()
                            lines[0] = ftmgLine(splt[1])
                    else:
                        text_line = dline.split()
                        data_dict = {}
                        # check that we have the same amount of header ids
                        # as we do data columns
                        # print("hdr length: {0} & data length {1}".format(len(header_ids), len(text_line)))
                        if len(header_ids) == len(text_line):
                            for idx, ids in enumerate(header_ids):
                                data_dict[ids] = float(text_line[idx])
                                # print(ids, data_dict[ids])
                            # create a stations class
                            lines[0].addStation(ftmgStation(data_obj=data_dict))
                        else:
                            print('[ERROR] column id and data column mismatch or blank line found!!!')
            # print('number of stations: {0} in line: {1}'.format(len(lines[0].stations), lines[0].line_id))
            self.lines.append(lines[0])
        else:
            print("Please define an input file!!!!! If you ain't first your last!!!")

    def getUtmLocations(self, limits=None, ground=False):
        """
        Return a numpy array of the easting and northings

        """

        easting = []
        northing = []
        elevation = []
        if limits is None:
            for line in range(len(self.lines)):
                for stn in range(len(self.lines[line].stations)):
                    easting.append(self.lines[line].stations[stn].easting)
                    northing.append(self.lines[line].stations[stn].northing)
                    # see if ground elevation is requested
                    if ground:
                        z_ground = (self.lines[line].stations[stn].alt_gps -
                                    self.lines[line].stations[stn].radar)
                        elevation.append(z_ground)
                    else:
                        elevation.append(self.lines[line].stations[stn].alt_gps)
        else:
            for line in range(len(self.lines)):
                for stn in range(len(self.lines[line].stations)):
                    x_stn = self.lines[line].stations[stn].easting
                    y_stn = self.lines[line].stations[stn].northing
                    # check that station is within the limits
                    if limits[0] < x_stn < limits[1] and limits[2] < y_stn < limits[3]:
                        easting.append(self.lines[line].stations[stn].easting)
                        northing.append(self.lines[line].stations[stn].northing)
                        # see if ground elevation is requested
                        if ground:
                            z_ground = (self.lines[line].stations[stn].alt_gps -
                                        self.lines[line].stations[stn].radar)
                            elevation.append(z_ground)
                        else:
                            elevation.append(self.lines[line].stations[stn].alt_gps)

        return np.asarray(easting), np.asarray(northing), np.asarray(elevation)

    def getComponentData(self, component=None, limits=None, inducing_field=None):
        output_comp = []

        if component.lower() == 'tmi':
            assert inducing_field is not None, "Inducing field parameters must be provided for 'TMI' data"

        # check for data extents requested

        # check thats components are supplied
        if component is not None:
            for line in range(len(self.lines)):
                for stn in range(len(self.lines[line].stations)):

                    if limits is not None:
                        x_stn = self.lines[line].stations[stn].easting
                        y_stn = self.lines[line].stations[stn].northing

                        if not (limits[0] < x_stn < limits[1]) and (limits[2] < y_stn < limits[3]):
                            continue

                    # check if component was aquired
                    if component in self.lines[line].stations[stn].comps:

                        if component.lower() == 'tmi':
                            output_comp.append(self.lines[line].stations[stn].comps[component] - inducing_field[0])
                        else:
                            output_comp.append(self.lines[line].stations[stn].comps[component])
        # else:
        #     # check thats components are supplied
        #     if component is not None:
        #         for line in range(len(self.lines)):
        #             for stn in range(len(self.lines[line].stations)):
        #                 x_stn = self.lines[line].stations[stn].easting
        #                 y_stn = self.lines[line].stations[stn].northing
        #                 # check that station is within the limits
        #                 if limits[0] < x_stn < limits[1] and limits[2] < y_stn < limits[3]:
        #                     # check if component was aquired
        #                     if component in self.lines[line].stations[stn].comps:
        #                         output_comp.append(self.lines[line].stations[stn].comps[component])

        return output_comp

    def createFtmgSurvey(self, inducing_field=None, limits=None, force_comp=None):
        """
        Returns a SimPEF survey Object for FTMG to be fed into inversion scheme

        """
        print("TODO")
        # check that we have an inducing field
        from SimPEG.PF import BaseMag
        if inducing_field is not None:
            # check data extents are requested
            if limits is None:
                # check that we have the proper amount of info in array
                if inducing_field.shape[0] == 3:
                    # get observation data
                    x, y, z = self.getUtmLocations()
                    comps = self.getListOfAvailableComponents(format_type='simpeg', force_comp=force_comp)
                    # Create a MAGsurvey
                    rxLoc = np.c_[Utils.mkvc(x.T), Utils.mkvc(y.T), Utils.mkvc(z.T)]
                    Rx = BaseMag.RxObs(rxLoc)
                    srcField = BaseMag.SrcField([Rx], param=inducing_field)
                    survey = BaseMag.LinearSurvey(srcField, components=comps)
                    # create dobs
                    dobs_list = []
                    components = self.getListOfAvailableComponents(force_comp=force_comp)
                    for comp in components:
                        # if comp is not 'TMI':
                        print('component added: {0}'.format(comp))
                        dobs_list.append(self.getComponentData(component=comp, inducing_field=inducing_field))
                    # now sort the data to SimPEG format
                    dobs = np.vstack(dobs_list)
                    size = dobs.shape[0] * dobs.shape[1]
                    survey.dobs = np.reshape(dobs, size, order='F')

                    return survey

                else:
                    print('[ERROR] array with Inducing field parameters must be size 3!!!')
            else:
                # get locations with limits
                x, y, z = self.getUtmLocations(limits=limits)
                comps = self.getListOfAvailableComponents(format_type='simpeg', force_comp=force_comp)
                # Create a MAGsurvey
                rxLoc = np.c_[Utils.mkvc(x.T), Utils.mkvc(y.T), Utils.mkvc(z.T)]
                Rx = BaseMag.RxObs(rxLoc)
                srcField = BaseMag.SrcField([Rx], param=inducing_field)
                survey = BaseMag.LinearSurvey(srcField, components=comps)
                # create dobs
                dobs_list = []
                components = self.getListOfAvailableComponents(force_comp=force_comp)
                for comp in components:
                    # if comp is not 'TMI':
                    print('component added: {0}'.format(comp))
                    dobs_list.append(self.getComponentData(component=comp, limits=limits))
                # now sort the data to SimPEG format
                dobs = np.vstack(dobs_list)
                size = dobs.shape[0] * dobs.shape[1]
                survey.dobs = np.reshape(dobs, size, order='F')

                return survey

        else:
            print('[ERROR] Please include numpy array with Inducing field parameters!!!')

    def getListOfAvailableComponents(self, keep_tmi=False, format_type='std', force_comp=None):
        """
        Returns the components collected for given readFTMGdata

        """
        available_components = self.lines[0].stations[0].available_components
        if force_comp is not None:
            available_components = force_comp
        # list_ = []
        # if not keep_tmi:
        #     for comp in available_components:
        #             # if comp is not 'TMI':
        #             list_.append(comp)
        # else:
        list_ = available_components

        if format_type == 'simpeg':
            list_bu = list_
            list_ = []
            if 'Bxx' in list_bu:
                list_.append("dbx_dx")
            if 'Byy' in list_bu:
                list_.append("dby_dy")
            if 'Bzz' in list_bu:
                list_.append("dbz_dz")
            if 'Bxz' in list_bu:
                list_.append("dbx_dz")
            if 'Bxy' in list_bu:
                list_.append("dbx_dy")
            if 'Byz' in list_bu:
                list_.append("dby_dz")
            if 'Bz' in list_bu:
                list_.append("bz")
            if 'Bx' in list_bu:
                list_.append("bx")
            if 'By' in list_bu:
                list_.append("by")
            if 'TMI' in list_bu:
                list_.append("tmi")
        return list_


class ftmgLine:
    """
    Line class for storing the observations

    """

    def __init__(self, line_id):
        self.stations = []
        self.line_id = line_id

    def addStation(self, station):
        self.stations.append(station)


class ftmgStation:
    """
    station class for hold station observation info

    """

    def __init__(self, data_obj=None):
        self.available_components = []
        self.comps = {}
        self.easting = None
        self.northing = None
        self.lat = None
        self.lon = None
        self.alt_gps = None
        self.radar = None
        self.date = None
        self.gps_sec = None
        self.I1 = None
        self.I2 = None
        if data_obj is not None:
            # assign all availble data
            self.easting = data_obj['x']
            self.northing = data_obj['y']
            self.lat = data_obj['lat']
            self.lon = data_obj['lon']
            self.alt_gps = data_obj['alt_gps']
            self.radar = data_obj['radar']
            self.date = data_obj['date']
            self.gps_sec = data_obj['GPS_sec']
            if 'Bxx' in data_obj:
                # self.Bxx = data_obj['Bxx']
                self.comps['Bxx'] = data_obj['Bxx']
                self.available_components.append('Bxx')
            if 'Bxy' in data_obj:
                self.comps['Bxy'] = data_obj['Bxy']
                # self.Bxy = data_obj['Bxy']
                self.available_components.append('Bxy')
            if 'Bxz' in data_obj:
                self.comps['Bxz'] = data_obj['Bxz']
                # self.Bxz = data_obj['Bxz']
                self.available_components.append('Bxz')
            if 'Byy' in data_obj:
                self.comps['Byy'] = data_obj['Byy']
                # self.Byy = data_obj['Byy']
                self.available_components.append('Byy')
            if 'Byz' in data_obj:
                self.comps['Byz'] = data_obj['Byz']
                # self.Byz = data_obj['Byz']
                self.available_components.append('Byz')
            if 'Bzz' in data_obj:
                self.comps['Bzz'] = data_obj['Bzz']
                # self.Bzz = data_obj['Bzz']
                self.available_components.append('Bzz')
            if 'Bx' in data_obj:
                self.comps['Bx'] = data_obj['Bx']
                # self.Bx = data_obj['Bx']
                self.available_components.append('Bx')
            if 'By' in data_obj:
                self.comps['By'] = data_obj['By']
                # self.By = data_obj['By']
                self.available_components.append('By')
            if 'Bz' in data_obj:
                self.comps['Bz'] = data_obj['Bz']
                # self.Bz = data_obj['Bz']
                self.available_components.append('Bz')
            if 'TMI' in data_obj:
                self.comps['TMI'] = data_obj['TMI']
                # self.tmi = data_obj['TMI']
                self.available_components.append('TMI')
            self.I1 = data_obj['I1']
            self.I2 = data_obj['I2']
