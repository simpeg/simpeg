import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from SimPEG.utils import mkvc, animate


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

        .. plot:: examples/mesh/plot_image_2D.py
           :include-source:

        .. plot:: examples/mesh/plot_image_3D.py
           :include-source:
        """
        assert type(I) == np.ndarray, "I must be a numpy array"
        assert type(numbering) == bool, "numbering must be a bool"
        assert direction in ["x", "y","z"], "direction must be either x,y, or z"


        if imageType == 'CC':
            assert I.size == self.nC, "Incorrect dimensions for CC."
        elif imageType == 'N':
            assert I.size == self.nN, "Incorrect dimensions for N."
        elif imageType == 'Fx':
            if I.size != np.prod(self.nFx): I, fy, fz = self.r(I,'F','F','M')
        elif imageType == 'Fy':
            if I.size != np.prod(self.nFy): fx, I, fz = self.r(I,'F','F','M')
        elif imageType == 'Fz':
            if I.size != np.prod(self.nFz): fx, fy, I = self.r(I,'F','F','M')
        elif imageType == 'Ex':
            if I.size != np.prod(self.nEx): I, ey, ez = self.r(I,'E','E','M')
        elif imageType == 'Ey':
            if I.size != np.prod(self.nEy): ex, I, ez = self.r(I,'E','E','M')
        elif imageType == 'Ez':
            if I.size != np.prod(self.nEz): ex, ey, I = self.r(I,'E','E','M')
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
                C = I[:].reshape(self.n, order='F')
            elif imageType == 'N':
                C = I[:].reshape(self.n+1, order='F')
                C = 0.25*(C[:-1, :-1] + C[1:, :-1] + C[:-1, 1:] + C[1:, 1:])
            elif imageType == 'Fx':
                C = I[:].reshape(self.nFx, order='F')
                C = 0.5*(C[:-1, :] + C[1:, :] )
            elif imageType == 'Fy':
                C = I[:].reshape(self.nFy, order='F')
                C = 0.5*(C[:, :-1] + C[:, 1:] )
            elif imageType == 'Ex':
                C = I[:].reshape(self.nEx, order='F')
                C = 0.5*(C[:,:-1] + C[:,1:] )
            elif imageType == 'Ey':
                C = I[:].reshape(self.nEy, order='F')
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
                    Ic = I[:].reshape(self.n, order='F')
                elif imageType == 'N':
                    Ic = I[:].reshape(self.n+1, order='F')
                    Ic = .125*(Ic[:-1,:-1,:-1]+Ic[1:,:-1,:-1] + Ic[:-1,1:,:-1]+ Ic[1:,1:,:-1]+ Ic[:-1,:-1,1:]+Ic[1:,:-1,1:] + Ic[:-1,1:,1:]+ Ic[1:,1:,1:] )
                elif imageType == 'Fx':
                    Ic = I[:].reshape(self.nFx, order='F')
                    Ic = .5*(Ic[:-1,:,:]+Ic[1:,:,:])
                elif imageType == 'Fy':
                    Ic = I[:].reshape(self.nFy, order='F')
                    Ic = .5*(Ic[:,:-1,:]+Ic[:,1:,:])
                elif imageType == 'Fz':
                    Ic = I[:].reshape(self.nFz, order='F')
                    Ic = .5*(Ic[:,:,:-1]+Ic[:,:,1:])
                elif imageType == 'Ex':
                    Ic = I[:].reshape(self.nEx, order='F')
                    Ic = .25*(Ic[:,:-1,:-1]+Ic[:,1:,:-1]+Ic[:,:-1,1:]+Ic[:,1:,:1])
                elif imageType == 'Ey':
                    Ic = I[:].reshape(self.nEy, order='F')
                    Ic = .25*(Ic[:-1,:,:-1]+Ic[1:,:,:-1]+Ic[:-1,:,1:]+Ic[1:,:,:1])
                elif imageType == 'Ez':
                    Ic = I[:].reshape(self.nEz, order='F')
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

    def plotGrid(self, nodes=False, faces=False, centers=False, edges=False, lines=True, showIt=False):
        """Plot the nodal, cell-centered and staggered grids for 1,2 and 3 dimensions.

        :param bool nodes: plot nodes
        :param bool faces: plot faces
        :param bool centers: plot centers
        :param bool edges: plot edges
        :param bool lines: plot lines connecting nodes
        :param bool showIt: call plt.show()

        .. plot:: examples/mesh/plot_grid_2D.py
           :include-source:

        .. plot:: examples/mesh/plot_grid_3D.py
           :include-source:
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

        return animate(fig, animateFrame, frames=mesh.nCv['xyz'.index(normal)])

    def video(mesh,var,function,figsize=(10,8)):
        """
        Call a function for a list of models to create a video.

        ::

            def function(var, ax, clim, tlt, i):
                tlt.set_text('%%d'%%i)
                return mesh.plotImage(var, imageType='CC', ax=ax, clim=clim)

            mesh.video([model1, model2, ..., modeln],function)
        """
        # First set up the figure, the axis, and the plot element we want to animate
        fig = plt.figure(figsize=figsize)
        ax = plt.axes()
        VAR = np.concatenate(var)
        clim = [VAR.min(),VAR.max()]
        tlt = plt.title('')
        plt.colorbar(function(var[0],ax,clim,tlt,0))

        def animateFrame(i):
            function(var[i],ax,clim,tlt,i)

        return animate(fig, animateFrame, frames=len(var))


