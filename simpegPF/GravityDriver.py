import re, os
from SimPEG import Mesh, np
import BaseGrav, Gravity

class GravityDriver_Inv(object):
    """docstring for GravityDriver_Inv"""

    def __init__(self, input_file=None):
        if input_file is not None:
            self.basePath = os.path.sep.join(input_file.split(os.path.sep)[:-1])
            if len(self.basePath) > 0:
                self.basePath += os.path.sep
            self.readDriverFile(input_file.split(os.path.sep)[-1])


    def readDriverFile(self, input_file):
        """
            Read input files for forward modeling GRAV data with integral form
            INPUT:
            input_file: File name containing the forward parameter

            OUTPUT:
            mshfile
            obsfile
            topofile
            start model
            ref model
            weightfile
            chi_target
            as, ax ,ay, az
            upper, lower bounds
            lp, lqx, lqy, lqz
            eps_p, eps_q
            # All files should be in the working directory, otherwise the path must
            # be specified.

        """


        fid = open(self.basePath + input_file,'r')

        # Line 1
        line = fid.readline()
        l_input  = line.split('!')
        mshfile = l_input[0].rstrip()

        # Line 2
        line = fid.readline()
        l_input  = line.split('!')
        obsfile = l_input[0].rstrip()

        # Line 3
        line = fid.readline()
        l_input = re.split('[!\s]',line)
        if l_input=='null':
            topofile = []

        else:
            topofile = l_input[0].rstrip()


        # Line 4
        line = fid.readline()
        l_input = re.split('[!\s]',line)
        if l_input[0]=='VALUE':
            mstart = float(l_input[1])

        else:
            mstart = l_input[0].rstrip()

        # Line 5
        line = fid.readline()
        l_input = re.split('[!\s]',line)
        if l_input[0]=='VALUE':
            mref = float(l_input[1])

        else:
            mref = l_input[0].rstrip()


        # Line 6
        line = fid.readline()
        l_input = re.split('[!\s]',line)
        if l_input=='DEFAULT':
            wgtfile = []

        else:
            wgtfile = l_input[0].rstrip()

        # Line 7
        line = fid.readline()
        l_input = re.split('[!\s]',line)
        chi = float(l_input[0])

        # Line 8
        line = fid.readline()
        l_input = re.split('[!\s]',line)
        val = np.array(l_input[0:4])
        alphas = val.astype(np.float)

        # Line 9
        line = fid.readline()
        l_input = re.split('[!\s]',line)
        if l_input[0]=='VALUE':
            val   = np.array(l_input[1:3])
            bounds = val.astype(np.float)

        else:
            bounds = l_input[0].rstrip()

        # Line 10
        line = fid.readline()
        l_input = re.split('[!\s]',line)
        if l_input[0]=='VALUE':
            val   = np.array(l_input[1:6])
            lpnorms = val.astype(np.float)

        else:
            lpnorms = l_input[0].rstrip()

        # Line 11
        line = fid.readline()
        l_input = re.split('[!\s]',line)
        if l_input[0]=='VALUE':
            val   = np.array(l_input[1:3])
            eps = val.astype(np.float)

        else:
            eps = l_input[0].rstrip()

        self.mshfile  = mshfile
        self.obsfile  = obsfile
        self.topofile = topofile
        self.mstart   = mstart
        self._mrefInput = mref
        self.wgtfile  = wgtfile
        self.chi      = chi
        self.alphas   = alphas
        self.bounds   = bounds
        self.lpnorms  = lpnorms
        self.eps      = eps

    @property
    def mesh(self):
        if getattr(self, '_mesh', None) is None:
            self._mesh = Mesh.TensorMesh.readUBC(self.basePath + self.mshfile)
        return self._mesh

    @property
    def survey(self):
        if getattr(self, '_survey', None) is None:
            self._survey = self.readGravityObservations(self.obsfile)
        return self._survey

    @property
    def activeCells(self):
        if getattr(self, '_activeCells', None) is None:
            if self.topofile == 'null':
                self._activeCells = np.arange(mesh.nC)
            else:
                topo = np.genfromtxt(self.basePath + self.topofile, skip_header=1)
                # Find the active cells
                self._activeCells = Gravity.getActiveTopo(self.mesh,topo,'N')
        return self._activeCells


    @property
    def nC(self):
        if getattr(self, '_nC', None) is None:
            self._nC = len(self.activeCells)
        return self._nC

    @property
    def m0(self):
        if getattr(self, '_m0', None) is None:
            if isinstance(self.mstart, float):
                self._m0 = np.ones(self.nC) * self.mstart
            else:
                self._m0 = Utils.meshutils.readUBCTensorModel(self.basePath + self.mstart,self.mesh)
                self._m0 = self._m0[self.activeCells]

        return self._m0

    @property
    def mref(self):
        if getattr(self, '_mref', None) is None:
            if isinstance(self._mrefInput, float):
                self._mref = np.ones(self.nC) * self._mrefInput
            else:
                self._mref = Utils.meshutils.readUBCTensorModel(self.basePath + self._mrefInput, self.mesh)
                self._mref = self._mref[self.activeCells]
        return self._mref



    def readGravityObservations(self, obs_file):
        """
        Read UBC grav file format

        INPUT:
        :param fileName, path to the UBC obs grav file

        OUTPUT:
        :param survey

        """

        fid = open(obs_file,'r')

        # First line has the number of rows
        line = fid.readline()
        ndat = np.array(line.split(),dtype=int)

        # Pre-allocate space for obsx, obsy, obsz, data, uncert
        line = fid.readline()
        temp = np.array(line.split(),dtype=float)

        d  = np.zeros(ndat, dtype=float)
        wd = np.zeros(ndat, dtype=float)
        locXYZ = np.zeros( (ndat,3), dtype=float)

        for ii in range(ndat):

            temp = np.array(line.split(),dtype=float)
            locXYZ[ii,:] = temp[:3]
            d[ii] = temp[3]
            wd[ii] = temp[4]
            line = fid.readline()

        rxLoc = BaseGrav.RxObs(locXYZ)
        srcField = BaseGrav.SrcField([rxLoc])
        survey = BaseGrav.LinearSurvey(srcField)
        survey.dobs =  d
        survey.std =  wd
        return survey
