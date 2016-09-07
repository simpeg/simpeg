import SimPEG
from SimPEG.EM.Utils.EMUtils import mu_0

class Point_NSEM(SimPEG.Survey.BaseRx):
    """
    Natural source receiver base class.

    Assumes that the data locations are xyz coordinates.

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    :param string component: real or imaginary component 'real' or 'imag'
    """

    def __init__(self, locs, orientation=None, component=None):
        assert(orientation in ['xx','xy', 'yx', 'yy']), "Orientation {0!s} not known. Orientation must be in 'x', 'y', 'z'. Arbitrary orientations have not yet been implemented.".format(orientation)
        assert(component in ['real', 'imag']), "'component' must be 'real' or 'imag', not {0!s}".format(component)

        self.orientation = orientation
        self.component = component


        SimPEG.Survey.BaseRx.__init__(self, locs, rxType=None) #TODO: remove rxType from baseRx

    # Set a mesh property
    @property
    def mesh(self):
        return self._mesh
    @mesh.setter
    def mesh(self,value):
        if value is getattr(self,'_mesh',None):
            pass
        else:
            self._mesh = value
    @property
    def src(self):
        return self._src
    @src.setter
    def src(self, value):
        self._src = value
    @property
    def f(self):
        return self._f
    @f.setter
    def f(self, value):
        self._f = value

    def locs_e(self):
        if self.locs.ndim == 3:
            loc = self.locs[:, :, 0]
        else:
            loc = self.locs
        return loc

    def locs_b(self):
        if self.locs.ndim == 3:
            loc = self.locs[:, :, 1]
        else:
            loc = self.locs
        return loc

    @property
    def Pex(self):
        if getattr(self,'_Pex',None) is None:
            self._Pex = self._mesh.getInterpolationMat(self.locs_e(), 'Ex')
        return self._Pex

    @property
    def Pey(self):
        if getattr(self,'_Pey',None) is None:
            self._Pey = self._mesh.getInterpolationMat(self.locs_e(), 'Ey')
        return self._Pey

    @property
    def Pbx(self):
        if getattr(self,'_Pbx',None) is None:
            self._Pbx = self._mesh.getInterpolationMat(self.locs_b(), 'Fx')
        return self._Pbx

    @property
    def Pby(self):
        if getattr(self,'_Pby',None) is None:
            self._Pby = self._mesh.getInterpolationMat(self.locs_b(), 'Fy')
        return self._Pby

    # Utility for convienece
    def sDiag(self, t):
        return SimPEG.Utils.sdiag(SimPEG.mkvc(t,2))

    # Get the components of the fields
    # px: x-polaration and py: y-polaration.
    @property
    def ex_px(self):
        return self.Pex*self.f[self.src, 'e_px']
    @property
    def ey_px(self):
        return self.Pey*self.f[self.src, 'e_px']
    @property
    def ex_py(self):
        return self.Pex*self.f[self.src, 'e_py']
    @property
    def ey_py(self):
        return self.Pey*self.f[self.src, 'e_py']
    @property
    def hx_px(self):
        return self.Pbx*self.f[self.src, 'b_px']/mu_0
    @property
    def hy_px(self):
        return self.Pby*self.f[self.src, 'b_px']/mu_0
    @property
    def hx_py(self):
        return self.Pbx*self.f[self.src, 'b_py']/mu_0
    @property
    def hy_py(self):
        return self.Pby*self.f[self.src, 'b_py']/mu_0
    # Get the derivatives
    def ex_px_u(self, vec):
        return self.Pex*self.f._e_pxDeriv_u(self.src,vec)
    def ey_px_u(self, vec):
        return self.Pey*self.f._e_pxDeriv_u(self.src,vec)
    def ex_py_u(self, vec):
        return self.Pex*self.f._e_pyDeriv_u(self.src,vec)
    def ey_py_u(self, vec):
        return self.Pey*self.f._e_pyDeriv_u(self.src,vec)
    def hx_px_u(self, vec):
        return self.Pbx*self.f._b_pxDeriv_u(self.src,vec)/mu_0
    def hy_px_u(self, vec):
        return self.Pby*self.f._b_pxDeriv_u(self.src,vec)/mu_0
    def hx_py_u(self, vec):
        return self.Pbx*self.f._b_pyDeriv_u(self.src,vec)/mu_0
    def hy_py_u(self, vec):
        return self.Pby*self.f._b_pyDeriv_u(self.src,vec)/mu_0
    # Define the components of the derivative
    @property
    def Hd(self):
        return self.sDiag(1./(
            self.sDiag(self.hx_px)*self.hy_py -
            self.sDiag(self.hx_py)*self.hy_px
        ))

    def Hd_uV(self, v):
        return (self.sDiag(self.hy_py)*self.hx_px_u(v) +
            self.sDiag(self.hx_px)*self.hy_py_u(v) -
            self.sDiag(self.hx_py)*self.hy_px_u(v) -
            self.sDiag(self.hy_px)*self.hx_py_u(v)
        )

    # Adjoint
    @property
    def aex_px(self):
        return SimPEG.mkvc(SimPEG.mkvc(self.f[self.src,'e_px'],2).T*self.Pex.T)
    @property
    def aey_px(self):
        return SimPEG.mkvc(SimPEG.mkvc(self.f[self.src,'e_px'],2).T*self.Pey.T)
    @property
    def aex_py(self):
        return SimPEG.mkvc(SimPEG.mkvc(self.f[self.src,'e_py'],2).T*self.Pex.T)
    @property
    def aey_py(self):
        return SimPEG.mkvc(SimPEG.mkvc(self.f[self.src,'e_py'],2).T*self.Pey.T)
    @property
    def ahx_px(self):
        return SimPEG.mkvc(SimPEG.mkvc(self.f[self.src,'b_px'],2).T/mu_0*self.Pbx.T)
    @property
    def ahy_px(self):
        return SimPEG.mkvc(SimPEG.mkvc(self.f[self.src,'b_px'],2).T/mu_0*self.Pby.T)
    @property
    def ahx_py(self):
        return SimPEG.mkvc(SimPEG.mkvc(self.f[self.src,'b_py'],2).T/mu_0*self.Pbx.T)
    @property
    def ahy_py(self):
        return SimPEG.mkvc(SimPEG.mkvc(self.f[self.src,'b_py'],2).T/mu_0*self.Pby.T)

    def aex_px_u(self, vec):
        return self.f._e_pxDeriv_u(self.src,self.Pex.T*vec,adjoint=True)
    def aey_px_u(self, vec):
        return self.f._e_pxDeriv_u(self.src,self.Pey.T*vec,adjoint=True)
    def aex_py_u(self, vec):
        return self.f._e_pyDeriv_u(self.src,self.Pex.T*vec,adjoint=True)
    def aey_py_u(self, vec):
        return self.f._e_pyDeriv_u(self.src,self.Pey.T*vec,adjoint=True)
    def ahx_px_u(self, vec):
        return self.f._b_pxDeriv_u(self.src,self.Pbx.T*vec,adjoint=True)/mu_0
    def ahy_px_u(self, vec):
        return self.f._b_pxDeriv_u(self.src,self.Pby.T*vec,adjoint=True)/mu_0
    def ahx_py_u(self, vec):
        return self.f._b_pyDeriv_u(self.src,self.Pbx.T*vec,adjoint=True)/mu_0
    def ahy_py_u(self, vec):
        return self.f._b_pyDeriv_u(self.src,self.Pby.T*vec,adjoint=True)/mu_0
    # Define the components of the derivative
    @property
    def aHd(self):
        return self.sDiag(1./(
            self.sDiag(self.ahx_px)*self.ahy_py -
            self.sDiag(self.ahx_py)*self.ahy_px
        ))
    def aHd_uV(self, x):
        return (self.ahx_px_u(self.sDiag(self.ahy_py)*x) +
            self.ahx_px_u(self.sDiag(self.ahy_py)*x) -
            self.ahy_px_u(self.sDiag(self.ahx_py)*x) -
            self.ahx_py_u(self.sDiag(self.ahy_px)*x)
        )

    def eval(self):
        """
        Function to evaluate datum for this receiver
        """
        raise NotImplementedError('SimPEG.EM.NSEM receiver has to have an eval method')

    def evalDeriv(self):
        """
        Function to evaluate datum for this receiver
        """
        raise NotImplementedError('SimPEG.EM.NSEM receiver has to have an evalDeriv method')

class Point_impedance3D(Point_NSEM):
    """
    Natural source 3D impedance receiver base class

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    :param string component: real or imaginary component 'real' or 'imag'
    """

    def __init__(self, locs, orientation=None, component=None):
        assert(orientation in ['xx','xy', 'yx', 'yy']), "Orientation {0!s} not known. Orientation must be in 'x', 'y', 'z'. Arbitrary orientations have not yet been implemented.".format(orientation)
        assert(component in ['real', 'imag']), "'component' must be 'real' or 'imag', not {0!s}".format(component)

        self.orientation = orientation
        self.component = component

        SimPEG.Survey.BaseRx.__init__(self, locs, rxType=None) #TODO: remove rxType from baseRx



    def eval(self, src, mesh, f):
        '''
        Project the fields to natural source data.

            :param SrcNSEM src: The source of the fields to project
            :param SimPEG.Mesh mesh:
            :param FieldsNSEM f: Natural source fields object to project
        '''
        # NOTE: Maybe set this as a property
        self.src = src
        self.mesh = mesh
        self.f = f

        if 'xx' in self.orientation:
            Zij = ( self.ex_px * self.hy_py - self.ex_py * self.hy_px)
        elif 'xy' in self.orientation:
            Zij = (-self.ex_px * self.hx_py + self.ex_py * self.hx_px)
        elif 'yx' in self.orientation:
            Zij = ( self.ey_px * self.hy_py - self.ey_py * self.hy_px)
        elif 'yy' in self.orientation:
            Zij = (-self.ey_px * self.hx_py + self.ey_py * self.hx_px)
        # Calculate the complex value
        f_part_complex = self.Hd * Zij

        return getattr(f_part_complex, self.component)

    def evalDeriv(self, src, mesh, f, v, adjoint=False):
        """
        The derivative of the projection wrt u

        :param NSEMsrc src: NSEM source
        :param TensorMesh mesh: Mesh defining the topology of the problem
        :param NSEMfields f: NSEM fields object of the source
        :param numpy.ndarray v: Random vector of size
        """
        self.src = src
        self.mesh = mesh
        self.f = f

        if adjoint:
            if 'xx' in self.orientation:
                Zij = self.sDiag(self.aHd * (
                    self.sDiag(self.ahy_py)*self.aex_px -
                    self.sDiag(self.ahy_px)*self.aex_py
                ))
                def ZijN_uV(x):
                    return (self.aex_px_u(self.sDiag(self.ahy_py)*x) +
                        self.ahy_py_u(self.sDiag(self.aex_px)*x) -
                        self.ahy_px_u(self.sDiag(self.aex_py)*x) -
                        self.aex_py_u(self.sDiag(self.ahy_px)*x)
                    )
            elif 'xy' in self.orientation:
                Zij = self.sDiag(self.aHd * (
                    -self.sDiag(self.ahx_py) * self.aex_px +
                    self.sDiag(self.ahx_px) * self.aex_py
                ))
                def ZijN_uV(x):
                    return (-self.aex_px_u(self.sDiag(self.ahx_py)*x) -
                        self.ahx_py_u(self.sDiag(self.aex_px)*x) +
                        self.ahx_px_u(self.sDiag(self.aex_py)*x) +
                        self.aex_py_u(self.sDiag(self.ahx_px)*x)
                    )
            elif 'yx' in self.orientation:
                Zij = self.sDiag(self.aHd*(
                    self.sDiag(self.ahy_py)*self.aey_px -
                    self.sDiag(self.ahy_px)*self.aey_py
                ))
                def ZijN_uV(x):
                    return (self.aey_px_u(self.sDiag(self.ahy_py)*x) +
                        self.ahy_py_u(self.sDiag(self.aey_px)*x) -
                        self.ahy_px_u(self.sDiag(self.aey_py)*x) -
                        self.aey_py_u(self.sDiag(self.ahy_px)*x)
                    )
            elif 'yy' in self.orientation:
                Zij = self.sDiag(self.aHd*(
                    -self.sDiag(self.ahx_py)*self.aey_px +
                    self.sDiag(self.ahx_px)*self.aey_py))
                def ZijN_uV(x):
                    return (-self.aey_px_u(self.sDiag(self.ahx_py)*x) -
                        self.ahx_py_u(self.sDiag(self.aey_px)*x) +
                        self.ahx_px_u(self.sDiag(self.aey_py)*x) +
                        self.aey_py_u(self.sDiag(self.ahx_px)*x)
                    )

            # Calculate the complex derivative
            PDeriv_real = ZijN_uV(self.aHd*v) - self.aHd_uV(Zij.T*self.aHd*v)#
            # NOTE: Need to reshape the output to go from 2*nU array to a (nU,2) matrix for each polarization
            # PDeriv_real = np.hstack((SimPEG.mkvc(PDeriv_real[:len(PDeriv_real)/2],2),SimPEG.mkvc(PDeriv_real[len(PDeriv_real)/2::],2)))
            PDeriv_real = PDeriv_real.reshape((2,self.mesh.nE)).T
            # Extract the data
            if self.component == 'imag':
                Pv = 1j*PDeriv_real
            elif self.component == 'real':
                Pv = PDeriv_real.astype(complex)
        else:
            if 'xx' in self.orientation:
                ZijN_uV = (
                    self.sDiag(self.hy_py) * self.ex_px_u(v) +
                    self.sDiag(self.ex_px) * self.hy_py_u(v) -
                    self.sDiag(self.ex_py) * self.hy_px_u(v) -
                    self.sDiag(self.hy_px) * self.ex_py_u(v)
                )
            elif 'xy' in self.orientation:
                ZijN_uV = (
                    -self.sDiag(self.hx_py) * self.ex_px_u(v) -
                    self.sDiag(self.ex_px) * self.hx_py_u(v) +
                    self.sDiag(self.ex_py) * self.hx_px_u(v) +
                    self.sDiag(self.hx_px) * self.ex_py_u(v)
                )
            elif 'yx' in self.orientation:
                ZijN_uV = (
                    self.sDiag(self.hy_py) * self.ey_px_u(v) +
                    self.sDiag(self.ey_px) * self.hy_py_u(v) -
                    self.sDiag(self.ey_py) * self.hy_px_u(v) -
                    self.sDiag(self.hy_px) * self.ey_py_u(v)
                )
            elif 'yy' in self.orientation:
                ZijN_uV = (
                    -self.sDiag(self.hx_py) * self.ey_px_u(v) -
                    self.sDiag(self.ey_px) * self.hx_py_u(v) +
                    self.sDiag(self.ey_py) * self.hx_px_u(v) +
                    self.sDiag(self.hx_px) * self.ey_py_u(v)
                )

            Zij = self.eval(src, self.mesh, self.f)
            # Calculate the complex derivative
            PDeriv_complex = self.Hd * (ZijN_uV - self.sDiag(Zij) * self.Hd_uV(v) )
            Pv = SimPEG.np.array(getattr(PDeriv_complex, self.component))

        return Pv