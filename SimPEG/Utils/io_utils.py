from __future__ import print_function
import numpy as np
from SimPEG import Mesh
import time as tm
import re
import warnings


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

    import re
    import vtk
    import vtk.util.numpy_support as npsup

    fid = open(tsfile, 'r')
    line = fid.readline()

    # Skip all the lines until the vertices
    while re.match('TFACE', line) == None:
        line = fid.readline()

    line = fid.readline()
    vrtx = []

    # Run down all the vertices and save in array
    while re.match('VRTX', line):
        l_input = re.split('[\s*]', line)
        temp = np.array(l_input[2:5])
        vrtx.append(temp.astype(np.float))

        # Read next line
        line = fid.readline()

    vrtx = np.asarray(vrtx)

    # Skip lines to the triangles
    while re.match('TRGL', line) == None:
        line = fid.readline()

    # Run down the list of triangles
    trgl = []

    # Run down all the vertices and save in array
    while re.match('TRGL', line):
        l_input = re.split('[\s*]', line)
        temp = np.array(l_input[1:4])
        trgl.append(temp.astype(np.int))

        # Read next line
        line = fid.readline()

    trgl = np.asarray(trgl)

    return vrtx, trgl


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


def remoteDownload(
    url, remoteFiles, path='.', directory='SimPEGtemp', rm_previous=False
):
    """
    Function to download all files stored in a cloud directory

    :param str url: url of the storage bucket ("http://...")
    :param list remoteFiles: List of file names to download from the storate bucket
    :param str path: path to where the directory is created and files downloaded (default is the current directory)
    :param str directory: name of the directory to be created and have content downloaded to
    :param bool rm_previous: remove file and contents if a directory with the specified name already exists
    """

    # Download from cloud
    import urllib
    import shutil
    import os
    import sys

    def rename_path(downloadpath):
        splitfullpath = downloadpath.split(os.path.sep)
        curdir = splitfullpath[-1]
        splitdir = curdir.split('(')
        rootdir = splitdir[0]

        # add (num) to the end of the filename
        if len(splitdir) == 1:
            num = 1
        else:
            num = int(splitdir[-1][:-1])
            num += 1

        return os.path.sep.join(
            splitfullpath[:-1] + [rootdir + '({})'.format(num)]
        )

    # grab the correct url retriever
    if sys.version_info < (3,):
        urlretrieve = urllib.urlretrieve
    else:
        urlretrieve = urllib.request.urlretrieve

    # ensure we are working with absolute paths
    path = os.path.abspath(path)
    downloadpath = os.path.sep.join([path]+[directory])
    # check if the directory already exists
    while os.path.exists(downloadpath):
        if rm_previous is True:
            print("removing previous contents of {}".format(downloadpath))
            shutil.rmtree(downloadpath)
        elif rm_previous is False:
            downloadpath = rename_path(downloadpath)
            print(
                "directory already exists, new downloads will be in {}".format(
                    downloadpath
                )
            )

    # create the directory
    os.makedirs(downloadpath+os.path.sep)

    # download files
    print("Download files from {}...".format(url))
    for file in remoteFiles:
        print("   Retrieving: " + file)
        urlretrieve(url + file, os.path.sep.join([downloadpath]+[file]))

    print("Download completed!")
    return downloadpath
