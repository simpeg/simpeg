import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from SimPEG.Utils import mkvc, animate


class TensorView(object):
    """
    Provides viewing functions for TensorMesh

    This class is inherited by TensorMesh
    """
    def __init__(self):
        pass

    def plotImage(self, I, imageType='CC', figNum=1,ax=None,direction='z',numbering=True,annotationColor='w',showIt=False,clim=None):
        """
        Mesh.plotImage(I)

        Plots scalar fields on the given mesh.

        Input:

        :param numpy.array I: scalar field

        Optional Input:

        :param str imageType: type of image ('CC','N','F','Fx','Fy','Fz','E','Ex','Ey','Ez') or combinations, e.g. ExEy or FxFz
        :param int figNum: number of figure to plot to
        :param matplotlib.axes.Axes ax: axis to plot to
        :param str direction: slice dimensions, 3D only ('x', 'y', 'z')
        :param bool numbering: show numbering of slices, 3D only
        :param str annotationColor: color of annotation, e.g. 'w', 'k', 'b'
        :param bool showIt: call plt.show()

        .. plot::
            :include-source:

            from SimPEG import Mesh, np
            M = Mesh.TensorMesh([20, 20])
            I = np.sin(M.gridCC[:,0]*2*np.pi)*np.sin(M.gridCC[:,1]*2*np.pi)
            M.plotImage(I, showIt=True)

        .. plot::
            :include-source:

            from SimPEG import Mesh, np
            M = Mesh.TensorMesh([20,20,20])
            I = np.sin(M.gridCC[:,0]*2*np.pi)*np.sin(M.gridCC[:,1]*2*np.pi)*np.sin(M.gridCC[:,2]*2*np.pi)
            M.plotImage(I, annotationColor='k', showIt=True)

        """
        assert type(I) == np.ndarray, "I must be a numpy array"
        assert type(numbering) == bool, "numbering must be a bool"
        assert direction in ["x", "y","z"], "direction must be either x,y, or z"


        if imageType == 'CC':
            assert I.size == self.nC, "Incorrect dimensions for CC."
        elif imageType == 'N':
            assert I.size == self.nN, "Incorrect dimensions for N."
        elif imageType == 'Fx':
            if I.size != np.prod(self.vnFx): I, fy, fz = self.r(I,'F','F','M')
        elif imageType == 'Fy':
            if I.size != np.prod(self.vnFy): fx, I, fz = self.r(I,'F','F','M')
        elif imageType == 'Fz':
            if I.size != np.prod(self.vnFz): fx, fy, I = self.r(I,'F','F','M')
        elif imageType == 'Ex':
            if I.size != np.prod(self.vnEx): I, ey, ez = self.r(I,'E','E','M')
        elif imageType == 'Ey':
            if I.size != np.prod(self.vnEy): ex, I, ez = self.r(I,'E','E','M')
        elif imageType == 'Ez':
            if I.size != np.prod(self.vnEz): ex, ey, I = self.r(I,'E','E','M')
        elif imageType[0] == 'E':
            plotAll = len(imageType) == 1
            options = {"direction":direction,"numbering":numbering,"annotationColor":annotationColor,"showIt":showIt}
            fig = plt.figure(figNum)
            # Determine the subplot number: 131, 121
            numPlots = 130 if plotAll else len(imageType)/2*10+100
            pltNum = 1
            ex, ey, ez = self.r(I,'E','E','M')
            if plotAll or 'Ex' in imageType:
                ax_x = plt.subplot(numPlots+pltNum)
                self.plotImage(ex, imageType='Ex', ax=ax_x, **options)
                pltNum +=1
            if plotAll or 'Ey' in imageType:
                ax_y = plt.subplot(numPlots+pltNum)
                self.plotImage(ey, imageType='Ey', ax=ax_y, **options)
                pltNum +=1
            if plotAll or 'Ez' in imageType:
                ax_z = plt.subplot(numPlots+pltNum)
                self.plotImage(ez, imageType='Ez', ax=ax_z, **options)
                pltNum +=1
            return
        elif imageType[0] == 'F':
            plotAll = len(imageType) == 1
            options = {"direction":direction,"numbering":numbering,"annotationColor":annotationColor,"showIt":showIt}
            fig = plt.figure(figNum)
            # Determine the subplot number: 131, 121
            numPlots = 130 if plotAll else len(imageType)/2*10+100
            pltNum = 1
            fxyz = self.r(I,'F','F','M')
            if plotAll or 'Fx' in imageType:
                ax_x = plt.subplot(numPlots+pltNum)
                self.plotImage(fxyz[0], imageType='Fx', ax=ax_x, **options)
                pltNum +=1
            if plotAll or 'Fy' in imageType:
                ax_y = plt.subplot(numPlots+pltNum)
                self.plotImage(fxyz[1], imageType='Fy', ax=ax_y, **options)
                pltNum +=1
            if plotAll or 'Fz' in imageType:
                ax_z = plt.subplot(numPlots+pltNum)
                self.plotImage(fxyz[2], imageType='Fz', ax=ax_z, **options)
                pltNum +=1
            return
        else:
            raise Exception("imageType must be 'CC', 'N','Fx','Fy','Fz','Ex','Ey','Ez'")


        if ax is None:
            fig = plt.figure(figNum)
            fig.clf()
            ax = plt.subplot(111)
        else:
            assert isinstance(ax,matplotlib.axes.Axes), "ax must be an Axes!"
            fig = ax.figure

        if self.dim == 1:
            if imageType == 'CC':
                ph = ax.plot(self.vectorCCx, I, '-ro')
            elif imageType == 'N':
                ph = ax.plot(self.vectorNx, I, '-bs')
            ax.set_xlabel("x")
            ax.axis('tight')
        elif self.dim == 2:
            if imageType == 'CC':
                C = I[:].reshape(self.vnC, order='F')
            elif imageType == 'N':
                C = I[:].reshape(self.vnN, order='F')
                C = 0.25*(C[:-1, :-1] + C[1:, :-1] + C[:-1, 1:] + C[1:, 1:])
            elif imageType == 'Fx':
                C = I[:].reshape(self.vnFx, order='F')
                C = 0.5*(C[:-1, :] + C[1:, :] )
            elif imageType == 'Fy':
                C = I[:].reshape(self.vnFy, order='F')
                C = 0.5*(C[:, :-1] + C[:, 1:] )
            elif imageType == 'Ex':
                C = I[:].reshape(self.vnEx, order='F')
                C = 0.5*(C[:,:-1] + C[:,1:] )
            elif imageType == 'Ey':
                C = I[:].reshape(self.vnEy, order='F')
                C = 0.5*(C[:-1,:] + C[1:,:] )

            if clim is None:
                clim = [C.min(),C.max()]
            ph = ax.pcolormesh(self.vectorNx, self.vectorNy, C.T, vmin=clim[0], vmax=clim[1])
            ax.axis('tight')
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        elif self.dim == 3:
            if direction == 'z':

                # get copy of image and average to cell-centres is necessary
                if imageType == 'CC':
                    Ic = I[:].reshape(self.vnC, order='F')
                elif imageType == 'N':
                    Ic = I[:].reshape(self.vnN, order='F')
                    Ic = .125*(Ic[:-1,:-1,:-1]+Ic[1:,:-1,:-1] + Ic[:-1,1:,:-1]+ Ic[1:,1:,:-1]+ Ic[:-1,:-1,1:]+Ic[1:,:-1,1:] + Ic[:-1,1:,1:]+ Ic[1:,1:,1:] )
                elif imageType == 'Fx':
                    Ic = I[:].reshape(self.vnFx, order='F')
                    Ic = .5*(Ic[:-1,:,:]+Ic[1:,:,:])
                elif imageType == 'Fy':
                    Ic = I[:].reshape(self.vnFy, order='F')
                    Ic = .5*(Ic[:,:-1,:]+Ic[:,1:,:])
                elif imageType == 'Fz':
                    Ic = I[:].reshape(self.vnFz, order='F')
                    Ic = .5*(Ic[:,:,:-1]+Ic[:,:,1:])
                elif imageType == 'Ex':
                    Ic = I[:].reshape(self.vnEx, order='F')
                    Ic = .25*(Ic[:,:-1,:-1]+Ic[:,1:,:-1]+Ic[:,:-1,1:]+Ic[:,1:,:1])
                elif imageType == 'Ey':
                    Ic = I[:].reshape(self.vnEy, order='F')
                    Ic = .25*(Ic[:-1,:,:-1]+Ic[1:,:,:-1]+Ic[:-1,:,1:]+Ic[1:,:,:1])
                elif imageType == 'Ez':
                    Ic = I[:].reshape(self.vnEz, order='F')
                    Ic = .25*(Ic[:-1,:-1,:]+Ic[1:,:-1,:]+Ic[:-1,1:,:]+Ic[1:,:1,:])

                # determine number oE slices in x and y dimension
                nX = np.ceil(np.sqrt(self.nCz))
                nY = np.ceil(self.nCz/nX)

                #  allocate space for montage
                nCx = self.nCx
                nCy = self.nCy

                C = np.zeros((nX*nCx,nY*nCy))

                for iy in range(int(nY)):
                    for ix in range(int(nX)):
                        iz = ix + iy*nX
                        if iz < self.nCz:
                            C[ix*nCx:(ix+1)*nCx, iy*nCy:(iy+1)*nCy] = Ic[:, :, iz]
                        else:
                            C[ix*nCx:(ix+1)*nCx, iy*nCy:(iy+1)*nCy] = np.nan

                C = np.ma.masked_where(np.isnan(C), C)
                xx = np.r_[0, np.cumsum(np.kron(np.ones((nX, 1)), self.hx).ravel())]
                yy = np.r_[0, np.cumsum(np.kron(np.ones((nY, 1)), self.hy).ravel())]
                # Plot the mesh

                if clim is None:
                    clim = [C.min(),C.max()]
                ph = ax.pcolormesh(xx, yy, C.T, vmin=clim[0], vmax=clim[1])
                # Plot the lines
                gx =  np.arange(nX+1)*(self.vectorNx[-1]-self.x0[0])
                gy =  np.arange(nY+1)*(self.vectorNy[-1]-self.x0[1])
                # Repeat and seperate with NaN
                gxX = np.c_[gx, gx, gx+np.nan].ravel()
                gxY = np.kron(np.ones((nX+1, 1)), np.array([0, sum(self.hy)*nY, np.nan])).ravel()
                gyX = np.kron(np.ones((nY+1, 1)), np.array([0, sum(self.hx)*nX, np.nan])).ravel()
                gyY = np.c_[gy, gy, gy+np.nan].ravel()
                ax.plot(gxX, gxY, annotationColor+'-', linewidth=2)
                ax.plot(gyX, gyY, annotationColor+'-', linewidth=2)
                ax.axis('tight')

                if numbering:
                    pad = np.sum(self.hx)*0.04
                    for iy in range(int(nY)):
                        for ix in range(int(nX)):
                            iz = ix + iy*nX
                            if iz < self.nCz:
                                ax.text((ix+1)*(self.vectorNx[-1]-self.x0[0])-pad,(iy)*(self.vectorNy[-1]-self.x0[1])+pad,
                                         '#%i'%iz,color=annotationColor,verticalalignment='bottom',horizontalalignment='right',size='x-large')

        ax.set_title(imageType)
        if showIt: plt.show()
        return ph

    def plotSlice(self, v, vType='CC',
                  normal='Z', ind=None, grid=False, view='real',
                  ax=None, clim=None, showIt=False,
                  pcolorOpts={},
                  streamOpts={'color':'k'},
                  gridOpts={'color':'k'}
                  ):

        """
        Plots a slice of a 3D mesh.

        .. plot::

            from SimPEG import *
            mT = Utils.meshTensors(((2,5),(4,2),(2,5)),((2,2),(6,2),(2,2)),((2,2),(6,2),(2,2)))
            M = Mesh.TensorMesh(mT)
            q = np.zeros(M.vnC)
            q[[4,4],[4,4],[2,6]]=[-1,1]
            q = Utils.mkvc(q)
            A = M.faceDiv*M.cellGrad
            b = Solver(A).solve(q)
            M.plotSlice(M.cellGrad*b, 'F', view='vec', grid=True, showIt=True, pcolorOpts={'alpha':0.8})

        """
        viewOpts = ['real','imag','abs','vec']
        normalOpts = ['X', 'Y', 'Z']
        vTypeOpts = ['CC', 'CCv','F','E']

        # Some user error checking
        assert vType in vTypeOpts, "vType must be in ['%s']" % "','".join(vTypeOpts)
        assert self.dim == 3, 'Must be a 3D mesh.'
        assert view in viewOpts, "view must be in ['%s']" % "','".join(viewOpts)
        assert normal in normalOpts, "normal must be in ['%s']" % "','".join(normalOpts)
        assert type(grid) is bool, 'grid must be a boolean'

        szSliceDim = getattr(self, 'nC'+normal.lower()) #: Size of the sliced dimension
        if ind is None: ind = int(szSliceDim/2)
        assert type(ind) in [int, long], 'ind must be an integer'

        if ax is None:
            fig = plt.figure(1)
            fig.clf()
            ax = plt.subplot(111)
        else:
            assert isinstance(ax, matplotlib.axes.Axes), "ax must be an matplotlib.axes.Axes"
            fig = ax.figure

        # The slicing and plotting code!!

        def getIndSlice(v):
            if   normal == 'X': v = v[ind,:,:]
            elif normal == 'Y': v = v[:,ind,:]
            elif normal == 'Z': v = v[:,:,ind]
            return v

        def doSlice(v):
            if vType == 'CC':
                return getIndSlice(self.r(v,'CC','CC','M'))
            elif vType == 'CCv':
                v = self.r(v.reshape((self.nC,3),order='F'),'CC','CC','M')
                assert view == 'vec', 'Other types for CCv not yet supported'
            else:
                # Now just deal with 'F' and 'E'
                aveOp = 'ave' + vType + ('2CCV' if view == 'vec' else '2CC')
                v = getattr(self,aveOp)*v # average to cell centers (might be a vector)
                v = self.r(v.reshape((self.nC,3),order='F'),'CC','CC','M')
            if view == 'vec':
                outSlice = []
                if 'X' not in normal: outSlice.append(getIndSlice(v[0]))
                if 'Y' not in normal: outSlice.append(getIndSlice(v[1]))
                if 'Z' not in normal: outSlice.append(getIndSlice(v[2]))
                return outSlice
            else:
                return getIndSlice(self.r(v,'CC','CC','M'))

        h2d = []
        if 'X' not in normal: h2d.append(self.hx)
        if 'Y' not in normal: h2d.append(self.hy)
        if 'Z' not in normal: h2d.append(self.hz)
        tM = self.__class__(h2d) #: Temp Mesh

        out = ()
        if view in ['real','imag','abs']:
            v = getattr(np,view)(v) # e.g. np.real(v)
            v = doSlice(v)
            if clim is None:
                clim = [v.min(),v.max()]
            out += (ax.pcolormesh(tM.vectorNx, tM.vectorNy, v.T, vmin=clim[0], vmax=clim[1], **pcolorOpts),)
        elif view in ['vec']:
            U, V = doSlice(v)
            if clim is None:
                uv = np.r_[mkvc(U), mkvc(V)]
                uv = np.sqrt(uv**2)
                clim = [uv.min(),uv.max()]

            # Matplotlib seems to not support irregular
            # spaced vectors at the moment. So we will
            # Interpolate down to a regular mesh at the
            # smallest mesh size in this 2D slice.
            nxi = int(tM.hx.sum()/tM.hx.min())
            nyi = int(tM.hy.sum()/tM.hy.min())
            tMi = self.__class__([np.ones(nxi)*tM.hx.sum()/nxi,
                                  np.ones(nyi)*tM.hy.sum()/nyi])
            P = tM.getInterpolationMat(tMi.gridCC,'CC',zerosOutside=True)
            Ui = P*mkvc(U)
            Vi = P*mkvc(V)
            Ui = tMi.r(Ui, 'CC', 'CC', 'M')
            Vi = tMi.r(Vi, 'CC', 'CC', 'M')
            # End Interpolation

            out += (ax.pcolormesh(tM.vectorNx, tM.vectorNy, np.sqrt(U**2+V**2).T, vmin=clim[0], vmax=clim[1], **pcolorOpts),)
            out += (ax.streamplot(tMi.vectorCCx, tMi.vectorCCy, Ui.T, Vi.T, **streamOpts),)

        if grid:
            xXGrid = np.c_[tM.vectorNx,tM.vectorNx,np.nan*np.ones(tM.nNx)].flatten()
            xYGrid = np.c_[tM.vectorNy[0]*np.ones(tM.nNx),tM.vectorNy[-1]*np.ones(tM.nNx),np.nan*np.ones(tM.nNx)].flatten()
            yXGrid = np.c_[tM.vectorNx[0]*np.ones(tM.nNy),tM.vectorNx[-1]*np.ones(tM.nNy),np.nan*np.ones(tM.nNy)].flatten()
            yYGrid = np.c_[tM.vectorNy,tM.vectorNy,np.nan*np.ones(tM.nNy)].flatten()
            out += (ax.plot(np.r_[xXGrid,yXGrid],np.r_[xYGrid,yYGrid],**gridOpts)[0],)

        ax.set_xlabel('y' if normal == 'X' else 'x')
        ax.set_ylabel('y' if normal == 'Z' else 'z')
        ax.set_title('Slice %d' % ind)
        ax.set_xlim(*tM.vectorNx[[0,-1]])
        ax.set_ylim(*tM.vectorNy[[0,-1]])

        if showIt: plt.show()
        return out

    def plotGrid(self, nodes=False, faces=False, centers=False, edges=False, lines=True, showIt=False):
        """Plot the nodal, cell-centered and staggered grids for 1,2 and 3 dimensions.

        :param bool nodes: plot nodes
        :param bool faces: plot faces
        :param bool centers: plot centers
        :param bool edges: plot edges
        :param bool lines: plot lines connecting nodes
        :param bool showIt: call plt.show()

        .. plot::
           :include-source:

           from SimPEG import Mesh, np
           h1 = np.linspace(.1,.5,3)
           h2 = np.linspace(.1,.5,5)
           mesh = Mesh.TensorMesh([h1, h2])
           mesh.plotGrid(nodes=True, faces=True, centers=True, lines=True, showIt=True)

        .. plot::
           :include-source:

           from SimPEG import Mesh, np
           h1 = np.linspace(.1,.5,3)
           h2 = np.linspace(.1,.5,5)
           h3 = np.linspace(.1,.5,3)
           mesh = Mesh.TensorMesh([h1,h2,h3])
           mesh.plotGrid(nodes=True, faces=True, centers=True, lines=True, showIt=True)

        """
        if self.dim == 1:
            fig = plt.figure(1)
            fig.clf()
            ax = plt.subplot(111)
            xn = self.gridN
            xc = self.gridCC
            ax.hold(True)
            ax.plot(xn, np.ones(np.shape(xn)), 'bs')
            ax.plot(xc, np.ones(np.shape(xc)), 'ro')
            ax.plot(xn, np.ones(np.shape(xn)), 'k--')
            ax.grid(True)
            ax.hold(False)
            ax.set_xlabel('x1')
            if showIt: plt.show()
        elif self.dim == 2:
            fig = plt.figure(2)
            fig.clf()
            ax = plt.subplot(111)
            xn = self.gridN
            xc = self.gridCC
            xs1 = self.gridFx
            xs2 = self.gridFy

            ax.hold(True)
            if nodes: ax.plot(xn[:, 0], xn[:, 1], 'bs')
            if centers: ax.plot(xc[:, 0], xc[:, 1], 'ro')
            if faces:
                ax.plot(xs1[:, 0], xs1[:, 1], 'g>')
                ax.plot(xs2[:, 0], xs2[:, 1], 'g^')
            if edges:
                ax.plot(self.gridEx[:, 0], self.gridEx[:, 1], 'c>')
                ax.plot(self.gridEy[:, 0], self.gridEy[:, 1], 'c^')

            # Plot the grid lines
            if lines:
                NN = self.r(self.gridN, 'N', 'N', 'M')
                X1 = np.c_[mkvc(NN[0][0, :]), mkvc(NN[0][self.nCx, :]), mkvc(NN[0][0, :])*np.nan].flatten()
                Y1 = np.c_[mkvc(NN[1][0, :]), mkvc(NN[1][self.nCx, :]), mkvc(NN[1][0, :])*np.nan].flatten()
                X2 = np.c_[mkvc(NN[0][:, 0]), mkvc(NN[0][:, self.nCy]), mkvc(NN[0][:, 0])*np.nan].flatten()
                Y2 = np.c_[mkvc(NN[1][:, 0]), mkvc(NN[1][:, self.nCy]), mkvc(NN[1][:, 0])*np.nan].flatten()
                X = np.r_[X1, X2]
                Y = np.r_[Y1, Y2]
                plt.plot(X, Y)

            ax.grid(True)
            ax.hold(False)
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            if showIt: plt.show()
        elif self.dim == 3:
            fig = plt.figure(3)
            fig.clf()
            ax = fig.add_subplot(111, projection='3d')
            xn = self.gridN
            xc = self.gridCC
            xfs1 = self.gridFx
            xfs2 = self.gridFy
            xfs3 = self.gridFz

            xes1 = self.gridEx
            xes2 = self.gridEy
            xes3 = self.gridEz

            ax.hold(True)
            if nodes: ax.plot(xn[:, 0], xn[:, 1], 'bs', zs=xn[:, 2])
            if centers: ax.plot(xc[:, 0], xc[:, 1], 'ro', zs=xc[:, 2])
            if faces:
                ax.plot(xfs1[:, 0], xfs1[:, 1], 'g>', zs=xfs1[:, 2])
                ax.plot(xfs2[:, 0], xfs2[:, 1], 'g<', zs=xfs2[:, 2])
                ax.plot(xfs3[:, 0], xfs3[:, 1], 'g^', zs=xfs3[:, 2])
            if edges:
                ax.plot(xes1[:, 0], xes1[:, 1], 'k>', zs=xes1[:, 2])
                ax.plot(xes2[:, 0], xes2[:, 1], 'k<', zs=xes2[:, 2])
                ax.plot(xes3[:, 0], xes3[:, 1], 'k^', zs=xes3[:, 2])

            # Plot the grid lines
            if lines:
                NN = self.r(self.gridN, 'N', 'N', 'M')
                X1 = np.c_[mkvc(NN[0][0, :, :]), mkvc(NN[0][self.nCx, :, :]), mkvc(NN[0][0, :, :])*np.nan].flatten()
                Y1 = np.c_[mkvc(NN[1][0, :, :]), mkvc(NN[1][self.nCx, :, :]), mkvc(NN[1][0, :, :])*np.nan].flatten()
                Z1 = np.c_[mkvc(NN[2][0, :, :]), mkvc(NN[2][self.nCx, :, :]), mkvc(NN[2][0, :, :])*np.nan].flatten()
                X2 = np.c_[mkvc(NN[0][:, 0, :]), mkvc(NN[0][:, self.nCy, :]), mkvc(NN[0][:, 0, :])*np.nan].flatten()
                Y2 = np.c_[mkvc(NN[1][:, 0, :]), mkvc(NN[1][:, self.nCy, :]), mkvc(NN[1][:, 0, :])*np.nan].flatten()
                Z2 = np.c_[mkvc(NN[2][:, 0, :]), mkvc(NN[2][:, self.nCy, :]), mkvc(NN[2][:, 0, :])*np.nan].flatten()
                X3 = np.c_[mkvc(NN[0][:, :, 0]), mkvc(NN[0][:, :, self.nCz]), mkvc(NN[0][:, :, 0])*np.nan].flatten()
                Y3 = np.c_[mkvc(NN[1][:, :, 0]), mkvc(NN[1][:, :, self.nCz]), mkvc(NN[1][:, :, 0])*np.nan].flatten()
                Z3 = np.c_[mkvc(NN[2][:, :, 0]), mkvc(NN[2][:, :, self.nCz]), mkvc(NN[2][:, :, 0])*np.nan].flatten()
                X = np.r_[X1, X2, X3]
                Y = np.r_[Y1, Y2, Y3]
                Z = np.r_[Z1, Z2, Z3]
                plt.plot(X, Y, 'b-', zs=Z)

            ax.grid(True)
            ax.hold(False)
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_zlabel('x3')
            if showIt: plt.show()

    def slicer(mesh, var, imageType='CC', normal='z', index=0, ax=None, clim=None):
        assert normal in 'xyz', 'normal must be x, y, or z'
        if ax is None: ax = plt.subplot(111)
        I = mesh.r(var,'CC','CC','M')
        axes = [p for p in 'xyz' if p not in normal.lower()]
        if normal is 'x': I = I[index,:,:]
        if normal is 'y': I = I[:,index,:]
        if normal is 'z': I = I[:,:,index]
        if clim is None: clim = [I.min(),I.max()]
        p = ax.pcolormesh(getattr(mesh,'vectorN'+axes[0]),getattr(mesh,'vectorN'+axes[1]),I.T,vmin=clim[0],vmax=clim[1])
        ax.axis('tight')
        ax.set_xlabel(axes[0])
        ax.set_ylabel(axes[1])
        return p

    def videoSlicer(mesh,var,imageType='CC',normal='z',figsize=(10,8)):
        assert mesh.dim > 2, 'This is for 3D meshes only.'
        # First set up the figure, the axis, and the plot element we want to animate
        fig = plt.figure(figsize=figsize)
        ax = plt.axes()
        clim = [var.min(),var.max()]
        plt.colorbar(mesh.slicer(var, imageType=imageType, normal=normal, index=0, ax=ax, clim=clim))
        tlt = plt.title(normal)

        def animateFrame(i):
            mesh.slicer(var, imageType=imageType, normal=normal, index=i, ax=ax, clim=clim)
            tlt.set_text(normal.upper()+('-Slice: %d, %4.4f' % (i,getattr(mesh,'vectorCC'+normal)[i])))

        return animate(fig, animateFrame, frames=mesh.vnC['xyz'.index(normal)])

    def video(mesh, var, function, figsize=(10, 8), colorbar=True, skip=1):
        """
        Call a function for a list of models to create a video.

        ::

            def function(var, ax, clim, tlt, i):
                tlt.set_text('%d'%i)
                return mesh.plotImage(var, imageType='CC', ax=ax, clim=clim)

            mesh.video([model1, model2, ..., modeln],function)
        """
        # First set up the figure, the axis, and the plot element we want to animate
        fig = plt.figure(figsize=figsize)
        ax = plt.axes()
        VAR = np.concatenate(var)
        clim = [VAR.min(),VAR.max()]
        tlt = plt.title('')
        if colorbar:
            plt.colorbar(function(var[0],ax,clim,tlt,0))

        frames = np.arange(0,len(var),skip)
        def animateFrame(j):
            i = frames[j]
            function(var[i],ax,clim,tlt,i)

        return animate(fig, animateFrame, frames=len(frames))


if __name__ == '__main__':
    from SimPEG import *
    mT = Utils.meshTensors(((2,5),(4,2),(2,5)),((2,2),(6,2),(2,2)),((2,2),(6,2),(2,2)))
    M = Mesh.TensorMesh(mT)
    q = np.zeros(M.vnC)
    q[[4,4],[4,4],[2,6]]=[-1,1]
    q = Utils.mkvc(q)
    A = M.faceDiv*M.cellGrad
    b = Solver(A).solve(q)
    M.plotSlice(M.cellGrad*b, 'F', view='vec', grid=True, showIt=True, pcolorOpts={'alpha':0.8})
