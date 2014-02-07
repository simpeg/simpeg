#!/usr/bin/python
"""
Input and output functions.
"""


import os as _os
import errno as _errno
import sys as _sys
import numpy as _np

from petsc4py import PETSc as _PETSc
import fileinput as _fl

def vecToArray(obj):
    """ Converts a PETSc vector to a numpy array, available on *all* MPI nodes.

        Args:
            obj (petsc4py.PETSc.Vec): input vector.

        Returns:
            numpy.array :
        """
    # scatter vector 'obj' to all processes
    comm = obj.getComm()
    scatter, obj0 = _PETSc.Scatter.toAll(obj)
    scatter.scatter(obj, obj0, False, _PETSc.Scatter.Mode.FORWARD)

    return _np.asarray(obj0)

    # deallocate
    comm.barrier()
    scatter.destroy()
    obj0.destroy()


def vecToArray0(obj):
    """ Converts a PETSc vector to a numpy array available on MPI node 0.

        Args:
            obj (petsc4py.PETSc.Vec): input vector.

        Returns:
            numpy.array :
        """
    # scatter vector 'obj' to process 0
    comm = obj.getComm()
    rank = comm.getRank()
    scatter, obj0 = _PETSc.Scatter.toZero(obj)
    scatter.scatter(obj, obj0, False, _PETSc.Scatter.Mode.FORWARD)

    if rank == 0:   return _np.asarray(obj0)

    # deallocate
    comm.barrier()
    scatter.destroy()
    obj0.destroy()


def arrayToVec(vecArray):
    """ Converts a (global) array to a PETSc vector over :attr:`petsc4py.PETSc.COMM_WORLD`.

        Args:
            vecArray (array or numpy.array): input vector.

        Returns:
            petsc4py.PETSc.Vec() :
        """
    vec = _PETSc.Vec().create(comm=_PETSc.COMM_WORLD)
    vec.setSizes(len(vecArray))
    vec.setUp()
    (Istart,Iend) = vec.getOwnershipRange()
    return vec.createWithArray(vecArray[Istart:Iend],
            comm=_PETSc.COMM_WORLD)
    vec.destroy()


def arrayToMat(matArray):
    """ Converts a (global) 2D array to a PETSc matrix over :attr:`petsc4py.PETSc.COMM_WORLD`.

        Args:
            matArray (array or numpy.array): input square array.

        :rtype: petsc4py.PETSc.Mat()

        .. important::
            Requires `SciPy <http://www.scipy.org>`_.

        """
    try:
        import scipy.sparse as sparse
    except:
        print '\nERROR: loading matrices from txt files requires Scipy!'
        return

    matSparse =matArray

    mat = _PETSc.Mat().createAIJ(size=matSparse.shape,comm=_PETSc.COMM_WORLD)
    (Istart,Iend) = mat.getOwnershipRange()

    ai = matSparse.indptr[Istart:Iend+1] - matSparse.indptr[Istart]
    aj = matSparse.indices[matSparse.indptr[Istart]:matSparse.indptr[Iend]]
    av = matSparse.data[matSparse.indptr[Istart]:matSparse.indptr[Iend]]

    mat.setValuesCSR(ai,aj,av)
    mat.assemble()

    return mat
    mat.destroy()

def matToSparse(mat):
    """ Converts a PETSc matrix to a (global) sparse matrix.

        Args:
            mat (petsc4py.PETSc.Mat): input PETSc matrix.

        :rtype: scipy.sparse.csr_matrix

        .. important::
            Requires `SciPy <http://www.scipy.org>`_.

        """
    import scipy.sparse as sparse

    data = mat.getValuesCSR()

    (Istart,Iend) = mat.getOwnershipRange()
    columns = mat.getSize()[0]
    sparseSubMat = sparse.csr_matrix(data[::-1],shape=(Iend-Istart,columns))

    comm = _PETSc.COMM_WORLD

    sparseSubMat = comm.tompi4py().allgather(sparseSubMat)

    return sparse.vstack(sparseSubMat)

def adjToH(adj,d=[0],amp=[0.]):
    """ Creates a 1 particle PETSc-type Hamiltonian matrix from a PETSc adjacency matrix.

        Args:
            adj (petsc4py.PETSc.Mat): input PETSc-type adjacency matrix.

            d (array of ints): an array containing *integers* indicating the nodes
                        where diagonal defects are to be placed (e.g. ``d=[0,1,4]``).

            amp (array of floats):   an array containing *floats* indicating the diagonal defect
                        amplitudes corresponding to each element in ``d`` (e.g. ``amp=[0.5,-1,4.2]``).

        Returns:
             : 1 particle Hamiltonian matrix
        :rtype: petsc4py.PETSc.Mat()

        Warning:
            * The size of ``a`` and ``d`` must be identical

                >>> amp = [0.5,-1.,4.2]
                >>> len(d) == len(amp)
                True

            * Elements of ``d`` can range from :math:`[0,N-1]` where the adjacency matrix is :math:`N\\times N`.

        """
    (Istart,Iend) = adj.getOwnershipRange()
    diagSum = []
    for i in range(Istart,Iend):
        diagSum.append(_np.sum(adj.getRow(i)[-1]))
        for j,val in enumerate(d):
            if i==val:  diagSum[i-Istart] += amp[j]

    mat = _PETSc.Mat().create(comm=_PETSc.COMM_WORLD)
    mat.setSizes(adj.getSize())
    mat.setUp()

    for i in range(Istart,Iend):
        mat.setValue(i,i,diagSum[i-Istart])

    mat.assemble()
    mat.axpy(-1,adj)

    return mat
    mat.destroy()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---------------------- Vec I/O functions ---------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def exportVec(vec,filename,filetype):
    """ Export a PETSc vector to a file.

        Args:
            vec (petsc4py.PETSc.Vec): input vector.
            filename (str): path to desired output file.
            filetype (str): the filetype of the exported vector.

                            * ``'txt'`` - a column vector in text format.
                            * ``'bin'`` - a PETSc binary vector.
        """
    if _os.path.isabs(filename):
        outDir = _os.path.dirname(filename)
    else:
        outDir = './'+_os.path.dirname(filename)

    # create output directory if it doesn't exist
    try:
        _os.mkdir(outDir)
    except OSError as exception:
        if exception.errno != _errno.EEXIST:
            raise

    if filetype == 'txt':
        # scatter prob to process 0
        comm = vec.getComm()
        rank = comm.getRank()
        scatter, vec0 = _PETSc.Scatter.toZero(vec)
        scatter.scatter(vec, vec0, False, _PETSc.Scatter.Mode.FORWARD)

        # use process 0 to write to text file
        if rank == 0:
            array0 = _np.asarray(vec0)
            with open(filename,'w') as f:
                for i in range(len(array0)):
                    f.write('{0: .12e}\n'.format(array0[i]))

        # deallocate
        comm.barrier()
        scatter.destroy()
        vec0.destroy()

    elif filetype == 'bin':
        binSave = _PETSc.Viewer().createBinary(filename, 'w')
        binSave(vec)
        binSave.destroy()

    vec.comm.barrier()

def loadVec(filename,filetype):
    """ Import a PETSc vector from a file.

        Args:
            filename (str): path to input file.
            filetype (str): the filetype.

                            * ``'txt'`` - a column vector in text format.
                            * ``'bin'`` - a PETSc binary vector.
        """
    if filetype == 'txt':
        try:
            vecArray = _np.loadtxt(filename,dtype=_PETSc.ScalarType)
            return arrayToVec(vecArray)
        except:
            print "\nERROR: input state space file " + filename\
                + " does not exist or is in an incorrect format"
            _sys.exit()

    elif filetype == 'bin':
        binLoad = _PETSc.Viewer().createBinary(filename, 'r')
        try:
            return _PETSc.Vec().load(binLoad)
        except:
            print "\nERROR: input state space file " + filename\
                + " does not exist or is in an incorrect format"
            _sys.exit()
        binLoad.destroy()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---------------------- Mat I/O functions ---------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def exportMat(mat,filename,filetype,mattype=None):
    """ Export a PETSc matrix to a file.

        Args:
            mat (petsc4py.PETSc.Mat): input matrix.
            filename (str): path to desired output file.
            filetype (str): the filetype of the exported vector.

                            * ``'txt'`` - a 2D matrix array in text format.
                            * ``'bin'`` - a PETSc binary matrix.
            mattype (str): (``None``,``'adj'``) - if set to ``adj``, only
                            integers ``0`` and ``1`` are written. Note
                            that this only applied in ``txt`` mode.
        """
    rank = _PETSc.Comm.Get_rank(_PETSc.COMM_WORLD)

    if _os.path.isabs(filename):
        outDir = _os.path.dirname(filename)
    else:
        outDir = './'+_os.path.dirname(filename)

    # create output directory if it doesn't exist
    try:
        _os.mkdir(outDir)
    except OSError as exception:
        if exception.errno != _errno.EEXIST:
            raise

    if filetype == 'txt':
        txtSave = _PETSc.Viewer().createASCII(filename, 'w',
            format=_PETSc.Viewer.Format.ASCII_DENSE, comm=_PETSc.COMM_WORLD)
        txtSave(mat)
        txtSave.destroy()

        if rank == 0:
            for line in _fl.FileInput(filename,inplace=1):
                if line[2] != 't':
                    if mattype == 'adj':
                        line = line.replace(" i","j")
                        line = line.replace(" -","-")
                        line = line.replace("+-","-")
                        line = line.replace("0000e+01+0.00000e+00j","")
                        line = line.replace(".00000e+00+0.00000e+00j","")
                        line = line.replace(".","")
                        line = line.replace(" -","\t-")
                        line = line.replace("  ","\t")
                        line = line.replace(" ","")
                        line = line.replace("\t"," ")
                        print line,
                    else:
                        line = line.replace(" i","j")
                        line = line.replace(" -","-")
                        line = line.replace("+-","-")
                        print line,

    elif filetype == 'bin':
        binSave = _PETSc.Viewer().createBinary(filename, 'w', comm=_PETSc.COMM_WORLD)
        binSave(mat)
        binSave.destroy()

    mat.comm.barrier()

def loadMat(filename,filetype,delimiter=None):
    """ Import a PETSc matrix from a file.

        Args:
            filename (str): path to input file.
            filetype (str): the filetype.

                            * ``'txt'`` - a 2D matrix array in text format.
                            * ``'bin'`` - a PETSc matrix vector.

            delimiter (str): this is passed to `numpy.genfromtxt\
                <http://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html>`_
                in the case of strange delimiters in an imported ``txt`` file.
        """
    if filetype == 'txt':
        try:
            try:
                if delimiter is None:
                    matArray = _np.genfromtxt(filename,dtype=_PETSc.ScalarType)
                else:
                    matArray = _np.genfromtxt(filename,dtype=_PETSc.ScalarType,delimiter=delimiter)
            except:
                filefix = []
                for line in _fl.FileInput(filename,inplace=0):
                    if line[2] != 't':
                        line = line.replace(" i","j")
                        line = line.replace(" -","-")
                        line = line.replace("+-","-")
                        filefix.append(line)

                matArray = _np.genfromtxt(filefix,dtype=_PETSc.ScalarType)

            return arrayToMat(matArray)
        except:
            print "\nERROR: input state space file " + filename\
                + " does not exist or is in an incorrect format"
            _sys.exit()

    elif filetype == 'bin':
        binLoad = _PETSc.Viewer().createBinary(filename, 'r')
        try:
            return _PETSc.Mat().load(binLoad)
        except:
            print "\nERROR: input state space file " + filename\
                + " does not exist or is in an incorrect format"
            _sys.exit()
        binLoad.destroy()

def exportVecToMat(vec,filename,filetype):
    """ Export a :math:`N^2` element PETSc vector as a :math:`N\\times N` matrix.

        This is useful when wanting to view the full statespace of a 2 particle
        quantum walk.

        Args:
            vec (petsc4py.PETSc.Vec): input :math:`N^2` element vector.
            filename (str): path to desired output file.
            filetype (str): the filetype of the exported vector.

                            * ``'txt'`` - an :math:`N\\times N` 2D matrix array in text format.
                            * ``'bin'`` - an :math:`N\\times N` PETSc binary matrix.
        """
    rank = _PETSc.Comm.Get_rank(_PETSc.COMM_WORLD)

    if _os.path.isabs(filename):
        outDir = _os.path.dirname(filename)
    else:
        outDir = './'+_os.path.dirname(filename)

    # create output directory if it doesn't exist
    try:
        _os.mkdir(outDir)
    except OSError as exception:
        if exception.errno != _errno.EEXIST:
            raise

    vecArray = vecToArray(vec)
    matArray = vecArray.reshape([_np.sqrt(vecArray.size),_np.sqrt(vecArray.size)])

    if filetype == 'txt':
        #if rank == 0:  _np.savetxt(filename,matArray)
        txtSave = _PETSc.Viewer().createASCII(filename, 'w',
            format=_PETSc.Viewer.Format.ASCII_DENSE, comm=_PETSc.COMM_WORLD)
        txtSave(arrayToMat(matArray))
        txtSave.destroy()

        if rank == 0:
            for line in _fl.FileInput(filename,inplace=1):
                if line[2] != 't':
                    line = line.replace(" i","j")
                    line = line.replace(" -","-")
                    line = line.replace("+-","-")
                    print line,

    elif filetype == 'bin':
        binSave = _PETSc.Viewer().createBinary(filename, 'w', comm=_PETSc.COMM_WORLD)
        binSave(arrayToMat(matArray))
        binSave.destroy()
    vec.comm.barrier()


def loadMatToVec(filename,filetype):
    """ Load a :math:`N\\times N` matrix as a :math:`N^2` element PETSc vector.

        This is useful when wanting to import the full statespace of a 2 particle
        quantum walk to use for propagation.

        Args:
            filename (str): path to the input file.
            filetype (str): the filetype

                            * ``'txt'`` - an :math:`N\\times N` 2D matrix array in text format.
                            * ``'bin'`` - **Not yet implemented! Please use a txt \
                                         format for this type of import**.
        """
    if filetype == 'txt':
        try:
            try:
                matArray = _np.loadtxt(filename,dtype=_PETSc.ScalarType)
            except:
                filefix = []
                for line in _fl.FileInput(filename,inplace=0):
                    if line[2] != 't':
                        line = line.replace(" i","j")
                        line = line.replace(" -","-")
                        line = line.replace("+-","-")
                        filefix.append(line)

                matArray = _np.loadtxt(filefix,dtype=_PETSc.ScalarType)

            vecArray = matArray.reshape(matArray.shape[0]**2)
            return arrayToVec(vecArray)
        except:
            print "\nERROR: input state space file " + filename\
                + " does not exist or is in an incorrect format"
            _sys.exit()

    elif filetype == 'bin':
        print '\nERROR: only works for txt storage!'
        _sys.exit()
