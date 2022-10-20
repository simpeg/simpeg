from __future__ import print_function
import numpy as np


def read_GOCAD_ts(tsfile):
    r"""Read GOCAD triangulated surface (*.ts) file

    Parameters
    ----------
    tsfile : str
        Path to triangulated surface (*.ts) file

    Returns
    -------
    vrts : (n, 3) numpy.ndarray
        Vertices in XYZ coordinates
    trgl : (m, 3) numpy.ndarray of int
        Array of indexes where each row represents the indexes for a particular
        triangle. Note that the order of the vertices matter, as they defined
        normal vectors; i.e. :math:`\hat{n} = (\mathbf{p_2 - p_1}) \times (\mathbf{p_3 - p_1})`
    """

    import re

    fid = open(tsfile, "r")
    line = fid.readline()

    # Skip all the lines until the vertices
    while re.match("TFACE", line) is None:
        line = fid.readline()

    line = fid.readline()
    vrtx = []

    # Run down all the vertices and save in array
    while re.match("VRTX", line):
        l_input = re.split("[\s*]", line)
        temp = np.array(l_input[2:5])
        vrtx.append(temp.astype(np.float))

        # Read next line
        line = fid.readline()

    vrtx = np.asarray(vrtx)

    # Skip lines to the triangles
    while re.match("TRGL", line) is None:
        line = fid.readline()

    # Run down the list of triangles
    trgl = []

    # Run down all the vertices and save in array
    while re.match("TRGL", line):
        l_input = re.split("[\s*]", line)
        temp = np.array(l_input[1:4])
        trgl.append(temp.astype(np.int))

        # Read next line
        line = fid.readline()

    trgl = np.asarray(trgl)

    return vrtx, trgl


def download(url, folder=".", overwrite=False, verbose=True):
    """Download all files stored in a cloud directory.

    Parameters
    ----------
    url : str or list of str
        A single URL or a list of URLs for all cloud directories containing files
        you wish to download.
    folder : str
        Path to the directory where all files are downloaded to. This function will
        create the directory if the directory does not already exist.
    overwrite : bool
        If ``True``, the function will overwrite preexisting files with new files
        in the case they both have identical names. If ``False``, the name of the
        file being downloaded is change if it is identical to a preexisting file name.
    verbose : bool
        Print download progress
    """

    # Download from cloud
    import urllib.request
    import os
    import sys

    def rename_path(downloadpath):
        splitfullpath = downloadpath.split(os.path.sep)

        # grab just the filename
        fname = splitfullpath[-1]
        fnamesplit = fname.split(".")
        newname = fnamesplit[0]

        # check if we have already re-numbered
        newnamesplit = newname.split("(")

        # add (num) to the end of the filename
        if len(newnamesplit) == 1:
            num = 1
        else:
            num = int(newnamesplit[-1][:-1])
            num += 1

        newname = "{}({}).{}".format(newnamesplit[0], num, fnamesplit[-1])
        return os.path.sep.join(splitfullpath[:-1] + newnamesplit[:-1] + [newname])

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
        filenames = [url.split("/")[-1]]
    elif isinstance(url, list):
        filenames = [u.split("/")[-1] for u in url]

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
                    print("file already exists, new file is called {}".format(download))
                downloadpath[i] = download

    # download files
    urllist = url if isinstance(url, list) else [url]
    for u, f in zip(urllist, downloadpath):
        print("Downloading {}".format(u))
        urlretrieve(u, f)
        print("   saved to: " + f)

    print("Download completed!")
    return downloadpath if isinstance(url, list) else downloadpath[0]
