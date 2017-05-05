from __future__ import print_function
import numpy as np
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
