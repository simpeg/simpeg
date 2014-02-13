import numpy as np

class Sources(object):
    """
        Class creates base sources
    """
    def __init__(self):
        raise Exception('Sources is a base class that requires a mesh. Inherit to your favorite Mesh class.')


    def magneticDipoleSource(self,dipoleLoc,dimDip=3):
        """
        Calculates the response from magnetic dipole source(s).

        Inputs:
            mesh      - mesh object
            dipoleLoc - an array of [n x 3] dipole locations (xyz)
            dimDip    - dipole dimension (e.g., x = 1,y = 2,z = 3)

        Outputs:
            b - the magnetic field on the face grid
        """

        def getMagneticDipolePotential(loc,m,grid,dimAxis):
            # Get the potential of the dipole
            # Returns A(# faces x # transmitters)

            nL = grid.shape[0]
            nT = loc.shape[0]
            fullLoc = np.ones((nL,3))
            A = np.zeros((nL,nT))
            for ii in range(nT):
                fullM = np.ones((nL,3))*m[ii,:]
                br = grid - fullLoc*loc[ii,:]
                cp = np.cross(fullM,br)
                r = np.sqrt(br[:,0]**2 + br[:,1]**2 + br[:,2]**2)
                A[:,ii] = ((1e-6)*cp[:,dimAxis-1])/r**3
            return A

        dipoleLoc = np.atleast_2d(dipoleLoc)
        dimDip = np.atleast_2d(dimDip)
        m = np.zeros((dipoleLoc.shape[0],3))
        if (dimDip.shape[0] == 1):
            try:
                m[:,dimDip-1] = 1
            except IndexError:
                print "magneticSource:Error: Dipole dimension should be 1 (x), 2 (y), or 3 (z)."
                raise
        elif (dimDip.shape[0] == dipoleLoc.shape[0]):
            try:
                for jj in range(dipoleLoc.shape[0]):
                    m[jj,dimDip[jj] - 1] = 1

            except IndexError:
                print "magneticSource:Error: Dipole dimension should be 1 (x), 2 (y), or 3 (z)."
                raise
        else:
            print "magneticSource:Error: Dipole direction should also be vector of same length as dipole locations."
            raise

        # Get magnetic potential at each set of orthogonal faces:
        Ax = getMagneticDipolePotential(dipoleLoc,m,self.gridEx,1)
        Ay = getMagneticDipolePotential(dipoleLoc,m,self.gridEy,2)
        Az = getMagneticDipolePotential(dipoleLoc,m,self.gridEz,3)

        # Combine potential
        A = np.concatenate((Ax, Ay, Az),axis=0)

        # B = curl A
        CURL = self.edgeCurl
        self._src = CURL*A

        return self._src

    def simpleLoopSource(self,loc,Lx,Ly=None):
        """
        Returns unit values for a simple loop source(s)

        Inputs:
            mesh - mesh object
            loc  - an array of [n x 3] loop centre location (xyz)
            Lx   - length of loop in x-direction (meters)
            Ly   - length of loop in y-direction (default is Lx)

        Outputs:
            e - Unit values of current on the edge grid
        """

        # Check for default value of Ly:
        Lx = np.atleast_2d(Lx)
        if Ly is None:
            Ly = Lx
        else:
            Ly = np.atleast_2d(Ly)

        # Number of loops
        loc = np.atleast_2d(loc)
        nL = loc.shape[0]
        normLoc = np.zeros((nL,3))
        sub = np.zeros((nL,3),'i')

        # Check number of values for loop sizes:
        if (Lx.shape[0] != Ly.shape[0]):
            print "simpleLoopSource:Error: Lx and Ly differ in lengths"
            raise

        if (Lx.shape[0] == 1):
            Lx = np.ones((nL,1))*Lx
            Ly = np.ones((nL,1))*Ly

        # widths of cells:
        hx = self.x0[0] + np.cumsum(self.hx)
        hy = self.x0[1] + np.cumsum(self.hy)
        hz = self.x0[2] + np.cumsum(self.hz)
        # Normalize the location to the grid:
        for ii in range(nL):
        # x
            diff = abs(loc[ii,0] - hx)
            iInd = np.argmin(diff)
            normLoc[ii,0] = iInd + diff[iInd]/hx[iInd]
        # y
            diff = abs(loc[ii,1] - hy)
            jInd = np.argmin(diff)
            normLoc[ii,1] = jInd + diff[jInd]/hy[jInd]
        # z
            diff = abs(loc[ii,2] - hz)
            kInd = np.argmin(diff)
            normLoc[ii,2] = kInd - diff[kInd]/hz[kInd]
            sub[ii,0] = iInd
            sub[ii,1] = jInd
            sub[ii,2] = kInd

        # Get node and edge grids
        self._src = np.zeros((self.edge.shape[0],nL))
        # Edge grid matrix
        [Ex,Ey,Ez] = self.r(self.edge,'E','E','M')
        # No topo for now, but this could be changed
        EzSource = np.zeros((Ez.shape))
        ezF = self.r(EzSource,'Ez','Ez','V')
        # Loops for each loc
        for ii in range(nL):
            ExSource = np.zeros((Ex.shape))
            EySource = np.zeros((Ey.shape))
        # Find the node closest to loc (round)
            nodeLoc = np.around(normLoc[ii,:])
        # Put at bottom of cell
            nodeLoc[2] = np.floor(normLoc[ii,2])

        # Fill edges in counter-clockwise format
        #    First x:
            i = sub[ii,0]
            j = sub[ii,1]
            k = sub[ii,2]
            eDist1 = 0.0; eDist2 = 0.0
            end1 = False; end2 = False
            i1 = i-1; i2 = i+1
            #### GET I1,I2,J1,J2 FIRST AND THEN FILL EX AND EY. NEED THEM FOR BOTH
            while True:
                try:
                    eDist1 = eDist1 + Ex[i1,j,k]
                    eDist2 = eDist2 + Ex[i2,j,k]
                except IndexError:
                    print "simpleLoopSource:Error: Loop goes off of x-edge of mesh."
                    raise

                if (np.around(abs(eDist1 - Lx[ii,0]/2)/Ex[i1,j,k]) < 1 and end1 == False):
                    end1 = True
                else:
                    i1 = i1 - 1

                if (np.around(abs(eDist2 - Lx[ii,0]/2)/Ex[i2,j,k]) < 1 and end2 == False):
                    end2 = True
                else:
                    i2 = i2 + 1

                if (end1 == True and end2 == True):
                    break

            eDist1 = 0.0; eDist2 = 0.0
            end1 = False; end2 = False
            j1 = j-1; j2 = j+1
            while True:
                try:
                    eDist1 = eDist1 + Ey[i1,j1,k]
                    eDist2 = eDist2 + Ey[i2,j2,k]
                except IndexError:
                    print "simpleLoopSource:Error: Loop goes off of y-edge of mesh."
                    raise

                if (np.around(abs(eDist1 - Ly[ii,0]/2)/Ey[i1,j1,k]) < 1 and end1 == False):
                    end1 = True
                else:
                    j1 = j1 - 1


                if (np.around(abs(eDist2 - Ly[ii,0]/2)/Ey[i2,j2,k]) < 1 and end2 == False):
                    end2 = True
                else:
                    j2 = j2 + 1

                if (end1 == True and end2 == True):
                    break

            # Fill values counter-clockwise:
            for jj in range(i1,i2):
                ExSource[jj,j1,k] =  1.0
                ExSource[jj,j2,k] = -1.0

            for jj in range(j1,j2):
                EySource[i1,jj,k] = -1.0
                EySource[i2,jj,k] =  1.0

            exF = self.r(ExSource,'Ex','Ex','V')
            eyF = self.r(ExSource,'Ey','Ey','V')
            # Set final e:
            self._src[:,ii] = np.concatenate((exF, eyF, ezF),axis=0)

        #Finished: return e
        return self._src
