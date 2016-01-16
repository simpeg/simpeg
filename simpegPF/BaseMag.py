from SimPEG import Maps, Survey, Utils, np, sp
from scipy.constants import mu_0
import re

class BaseMagSurvey(Survey.BaseSurvey):
    """Base Magnetics Survey"""

    rxLoc = None #: receiver locations
    rxType = None #: receiver type

    def __init__(self, **kwargs):
        Survey.BaseSurvey.__init__(self, **kwargs)


    def setBackgroundField(self, Inc, Dec, Btot):

        Bx = Btot*np.cos(Inc/180.*np.pi)*np.sin(Dec/180.*np.pi)
        By = Btot*np.cos(Inc/180.*np.pi)*np.cos(Dec/180.*np.pi)
        Bz = -Btot*np.sin(Inc/180.*np.pi)

        self.B0 = np.r_[Bx,By,Bz]

    @property
    def Qfx(self):
        if getattr(self, '_Qfx', None) is None:
            self._Qfx = self.prob.mesh.getInterpolationMat(self.rxLoc,'Fx')
        return self._Qfx

    @property
    def Qfy(self):
        if getattr(self, '_Qfy', None) is None:
            self._Qfy = self.prob.mesh.getInterpolationMat(self.rxLoc,'Fy')
        return self._Qfy

    @property
    def Qfz(self):
        if getattr(self, '_Qfz', None) is None:
            self._Qfz = self.prob.mesh.getInterpolationMat(self.rxLoc,'Fz')
        return self._Qfz

    def projectFields(self, u):
        """
            This function projects the fields onto the data space.

            Especially, here for we use total magnetic intensity (TMI) data,
            which is common in practice.

            First we project our B on to data location

            .. math::

                \mathbf{B}_{rec} = \mathbf{P} \mathbf{B}

            then we take the dot product between B and b_0

            .. math ::

                \\text{TMI} = \\vec{B}_s \cdot \hat{B}_0

        """
        #TODO: There can be some different tyes of data like |B| or B

        bfx = self.Qfx*u['B']
        bfy = self.Qfy*u['B']
        bfz = self.Qfz*u['B']

        # Generate unit vector
        B0 = self.prob.survey.B0
        Bot = np.sqrt(B0[0]**2+B0[1]**2+B0[2]**2)
        box = B0[0]/Bot
        boy = B0[1]/Bot
        boz = B0[2]/Bot

        # return bfx*box + bfx*boy + bfx*boz
        return bfx*box + bfy*boy + bfz*boz


    @Utils.count
    def projectFieldsDeriv(self, B):
        """
            This function projects the fields onto the data space.

            .. math::

                \\frac{\partial d_\\text{pred}}{\partial \mathbf{B}} = \mathbf{P}

            Especially, this function is for TMI data type

        """
        # Generate unit vector
        B0 = self.prob.survey.B0
        Bot = np.sqrt(B0[0]**2+B0[1]**2+B0[2]**2)
        box = B0[0]/Bot
        boy = B0[1]/Bot
        boz = B0[2]/Bot

        return self.Qfx*box+self.Qfy*boy+self.Qfz*boz


    def projectFieldsAsVector(self, B):

        bfx = self.Qfx*B
        bfy = self.Qfy*B
        bfz = self.Qfz*B

        return np.r_[bfx, bfy, bfz]

class MagSurveyBx(object):
    """docstring for MagSurveyBx"""
    def __init__(self, **kwargs):
        Survey.BaseData.__init__(self, **kwargs)

    def projectFields(self, B):
        bfx = self.Qfx*B
        return bfx


class BaseMagMap(Maps.IdentityMap):
    """BaseMagMap"""

    def __init__(self, mesh, **kwargs):
        Maps.IdentityMap.__init__(self, mesh)

    def _transform(self, m):

        return mu_0*(1 + m)

    def deriv(self, m):

        return mu_0*sp.identity(self.nP)

class WeightMap(Maps.IdentityMap):
    """Weighted Map for distributed parameters"""

    def __init__(self, mesh, weight, **kwargs):
        Maps.IdentityMap.__init__(self, mesh)
        self.mesh = mesh
        self.weight = weight

    def _transform(self, m):        
        return m*self.weight

    def deriv(self, m):
        return Utils.sdiag(self.weight)    


    

def readUBCmagObs(obs_file):
    
    """
    Read and write UBC mag file format
    
    INPUT:
    :param fileName, path to the UBC obs mag file
    
    OUTPUT:
    :param dobs, observation in (x y z [data] [wd])
    :param B, primary field information (BI, BD, B0)
    :param M, magnetization orentiaton (MI, MD)
    
    """

    fid = open(obs_file,'r') 

    # First line has the inclination,declination and amplitude of B0
    line = fid.readline()
    B = np.array(line.split(),dtype=float)

    # Second line has the magnetization orientation and a flag 
    line = fid.readline()
    M = np.array(line.split(),dtype=float)

    # Third line has the number of rows
    line = fid.readline()
    ndat = np.array(line.split(),dtype=int)

    # Pre-allocate space for obsx, obsy, obsz, data, uncert
    line = fid.readline()
    temp = np.array(line.split(),dtype=float) 
        
    dobs = np.zeros((ndat,len(temp)), dtype=float)
    
    
    for ii in range(ndat):
        
        dobs[ii,:] = np.array(line.split(),dtype=float) 
        line = fid.readline()
        
    return B, M, dobs

def read_MAGfwr_inp(input_file):

    """Read input files for forward modeling MAG data with integral form
    INPUT:
    input_file: File name containing the forward parameter
    
    OUTPUT:
    mshfile
    obsfile
    modfile
    magfile
    topofile
    # All files should be in the working directory, otherwise the path must
    # be specified.

    Created on Jul 17, 2013
    
    @author: dominiquef
    """

    
    fid = open(input_file,'r')
    
    line = fid.readline()
    l_input  = line.split('!')
    mshfile = l_input[0].rstrip()
    
    line = fid.readline()
    l_input  = line.split('!')
    obsfile = l_input[0].rstrip()
    
    line = fid.readline()
    l_input = line.split('!') 
    modfile = l_input[0].rstrip()
    
    line = fid.readline()
    l_input = line.split('!') 
    if l_input=='null':
        magfile = []
        
    else:
        magfile = l_input[0].rstrip()
        
        
    line = fid.readline()
    l_input = line.split('!') 
    if l_input=='null':
        topofile = []
        
    else:
        topofile = l_input[0].rstrip()
      
    return mshfile, obsfile, modfile, magfile, topofile

def read_MAGinv_inp(input_file):
    """Read input files for forward modeling MAG data with integral form
    INPUT:
    input_file: File name containing the forward parameter
    
    OUTPUT:
    mshfile
    obsfile
    topofile
    start model
    ref model
    mag model
    weightfile
    chi_target
    as, ax ,ay, az
    upper, lower bounds
    lp, lqx, lqy, lqz

    # All files should be in the working directory, otherwise the path must
    # be specified.

    Created on Dec 21th, 2015
    
    @author: dominiquef
    """

    
    fid = open(input_file,'r')
    
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
        magfile = []
        
    else:
        magfile = l_input[0].rstrip()

    # Line 7
    line = fid.readline()
    l_input = re.split('[!\s]',line) 
    if l_input=='DEFAULT':
        wgtfile = []
        
    else:
        wgtfile = l_input[0].rstrip()

    # Line 8
    line = fid.readline()
    l_input = re.split('[!\s]',line) 
    chi = float(l_input[0])

    # Line 9
    line = fid.readline()
    l_input = re.split('[!\s]',line) 
    val = np.array(l_input[0:4])
    alphas = val.astype(np.float)

    # Line 10
    line = fid.readline()
    l_input = re.split('[!\s]',line) 
    if l_input[0]=='VALUE':
        val   = np.array(l_input[1:3])
        bounds = val.astype(np.float)
        
    else:
        bounds = l_input[0].rstrip()

    # Line 11
    line = fid.readline()
    l_input = re.split('[!\s]',line) 
    if l_input[0]=='VALUE':
        val   = np.array(l_input[1:6])
        lpnorms = val.astype(np.float)
        
    else:
        lpnorms = l_input[0].rstrip()

    return mshfile, obsfile, topofile, mstart, mref, magfile, wgtfile, chi, alphas, bounds, lpnorms

def read_GOCAD_ts(tsfile):
    """Read GOCAD triangulated surface (*.ts) file
    INPUT:
    tsfile: Triangulated surface
    
    OUTPUT:
    vrts : Array of vertices in XYZ coordinates [n x 3]
    trgl : Array of index for triangles [m x 3]. The order of the vertices 
            is important and describes the normal
            n = cross( (P2 - P1 ) , (P3 - P1) )


    Created on Jan 13th, 2016
    
    Author: @fourndo
    """

    
    fid = open(tsfile,'r')
    line = fid.readline()
    
    # Skip all the lines until the vertices
    while re.match('TFACE',line)==None:
        line = fid.readline()
    
    line = fid.readline()
    vrtx = []
    
    # Run down all the vertices and save in array
    while re.match('VRTX',line):
        l_input  = re.split('[\s*]',line)
        temp = np.array(l_input[2:5])
        vrtx.append(temp.astype(np.float))
        
        # Read next line
        line = fid.readline()
    
    vrtx = np.asarray(vrtx)
    
    # Run down the list of triangles
    trgl = []
    
    # Run down all the vertices and save in array
    while re.match('TRGL',line):
        l_input  = re.split('[\s*]',line)
        temp = np.array(l_input[1:4])
        trgl.append(temp.astype(np.int))
        
        # Read next line
        line = fid.readline()
     
    trgl = np.asarray(trgl)
    
    return vrtx, trgl
    
def gocad2vtk(gcFile,mesh):
    """"
    Function to read gocad polystructure file and output indexes of mesh with in the structure.
    
    """
    import vtk, vtk.util.numpy_support as npsup
    
    vrtx, trgl = read_GOCAD_ts(gcFile)
    # Adjust the index
    trgl = trgl - 1
    
    # Make vtk pts
    ptsvtk = vtk.vtkPoints()
    ptsvtk.SetData(npsup.numpy_to_vtk(vrtx,deep=1))
    
    # Make the polygon connection
    polys = vtk.vtkCellArray()
    for face in trgl:
        poly = vtk.vtkPolygon()
        poly.GetPointIds().SetNumberOfIds(len(face))
        for nrv, vert in enumerate(face):
            poly.GetPointIds().SetId(nrv,vert)
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
    vtkMesh.SetDimensions(mesh.nNx,mesh.nNy,mesh.nNz)
    vtkMesh.SetXCoordinates(npsup.numpy_to_vtk(mesh.vectorNx,deep=1))    
    vtkMesh.SetYCoordinates(npsup.numpy_to_vtk(mesh.vectorNy,deep=1))    
    vtkMesh.SetZCoordinates(npsup.numpy_to_vtk(mesh.vectorNz,deep=1)) 
    # Add indexes
    vtkInd = npsup.numpy_to_vtk(np.arange(mesh.nC),deep=1)
    vtkInd.SetName('Index')
    vtkMesh.GetCellData().AddArray(vtkInd)
    
    extractImpDistRectGridFilt = vtk.vtkExtractGeometry() # Object constructor
    extractImpDistRectGridFilt.SetImplicitFunction(ImpDistFunc) #
    extractImpDistRectGridFilt.SetInputData(vtkMesh)
    extractImpDistRectGridFilt.ExtractBoundaryCellsOn()
    extractImpDistRectGridFilt.ExtractInsideOn()
    
    # Executing the pipe
    extractImpDistRectGridFilt.Update()
    
    insideGrid = extractImpDistRectGridFilt.GetOutput()
    # Return the indexes inside
    return npsup.vtk_to_numpy(insideGrid.GetCellData().GetArray('Index'))