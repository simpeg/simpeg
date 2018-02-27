""" Module RxNSEM.py

Receivers for the NSEM problem

"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from scipy.constants import mu_0

import SimPEG
import numpy as np
from SimPEG import mkvc


class BaseRxNSEM_Point(SimPEG.Survey.BaseRx):
    """
    Natural source receiver base class.

    Assumes that the data locations are xyz coordinates.

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    :param string component: real or imaginary component 'real' or 'imag'
    """

    def __init__(self, locs, orientation=None, component=None):
        assert(orientation in ['xx', 'xy', 'yx', 'yy', 'zx', 'zy']), "Orientation {0!s} not known. Orientation must be in 'x', 'y', 'z'. Arbitrary orientations have not yet been implemented.".format(orientation)
        assert(component in ['real', 'imag']), "'component' must be 'real' or 'imag', not {0!s}".format(component)

        self.orientation = orientation
        self.component = component

        SimPEG.Survey.BaseRx.__init__(self, locs, rxType=None) # TODO: remove rxType from baseRx

    # Set a mesh property
    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        if value is getattr(self, '_mesh', None):
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

    def _locs_e(self):
        if self.locs.ndim == 3:
            loc = self.locs[:, :, 0]
        else:
            loc = self.locs
        return loc

    def _locs_b(self):
        if self.locs.ndim == 3:
            loc = self.locs[:, :, 1]
        else:
            loc = self.locs
        return loc

    # Location projection
    @property
    def Pex(self):
        if getattr(self, '_Pex', None) is None:
            self._Pex = self._mesh.getInterpolationMat(self._locs_e(), 'Ex')
        return self._Pex

    @property
    def Pey(self):
        if getattr(self, '_Pey', None) is None:
            self._Pey = self._mesh.getInterpolationMat(self._locs_e(), 'Ey')
        return self._Pey

    @property
    def Pbx(self):
        if getattr(self, '_Pbx', None) is None:
            self._Pbx = self._mesh.getInterpolationMat(self._locs_b(), 'Fx')
        return self._Pbx

    @property
    def Pby(self):
        if getattr(self, '_Pby', None) is None:
            self._Pby = self._mesh.getInterpolationMat(self._locs_b(), 'Fy')
        return self._Pby

    @property
    def Pbz(self):
        if getattr(self, '_Pbz', None) is None:
            self._Pbz = self._mesh.getInterpolationMat(self._locs_e(), 'Fz')
        return self._Pbz

    # Utility for convienece
    def _sDiag(self, t):
        return SimPEG.Utils.sdiag(mkvc(t,2))

    # Get the components of the fields
    # px: x-polaration and py: y-polaration.
    @property
    def _ex_px(self):
        return self.Pex*self.f[self.src, 'e_px']

    @property
    def _ey_px(self):
        return self.Pey*self.f[self.src, 'e_px']

    @property
    def _ex_py(self):
        return self.Pex*self.f[self.src, 'e_py']

    @property
    def _ey_py(self):
        return self.Pey*self.f[self.src, 'e_py']

    @property
    def _hx_px(self):
        return self.Pbx*self.f[self.src, 'b_px']/mu_0

    @property
    def _hy_px(self):
        return self.Pby*self.f[self.src, 'b_px']/mu_0

    @property
    def _hz_px(self):
        return self.Pbz*self.f[self.src, 'b_px']/mu_0

    @property
    def _hx_py(self):
        return self.Pbx*self.f[self.src, 'b_py']/mu_0

    @property
    def _hy_py(self):
        return self.Pby*self.f[self.src, 'b_py']/mu_0

    @property
    def _hz_py(self):
        return self.Pbz*self.f[self.src, 'b_py']/mu_0
    # Get the derivatives

    def _ex_px_u(self, vec):
        return self.Pex*self.f._e_pxDeriv_u(self.src, vec)

    def _ey_px_u(self, vec):
        return self.Pey*self.f._e_pxDeriv_u(self.src, vec)

    def _ex_py_u(self, vec):
        return self.Pex*self.f._e_pyDeriv_u(self.src, vec)

    def _ey_py_u(self, vec):
        return self.Pey*self.f._e_pyDeriv_u(self.src, vec)

    def _hx_px_u(self, vec):
        return self.Pbx*self.f._b_pxDeriv_u(self.src, vec)/mu_0

    def _hy_px_u(self, vec):
        return self.Pby*self.f._b_pxDeriv_u(self.src, vec)/mu_0

    def _hz_px_u(self, vec):
        return self.Pbz*self.f._b_pxDeriv_u(self.src, vec)/mu_0

    def _hx_py_u(self, vec):
        return self.Pbx*self.f._b_pyDeriv_u(self.src, vec)/mu_0

    def _hy_py_u(self, vec):
        return self.Pby*self.f._b_pyDeriv_u(self.src, vec)/mu_0

    def _hz_py_u(self, vec):
        return self.Pbz*self.f._b_pyDeriv_u(self.src, vec)/mu_0
    # Define the components of the derivative

    @property
    def _Hd(self):
        return self._sDiag(1./(
            self._sDiag(self._hx_px)*self._hy_py -
            self._sDiag(self._hx_py)*self._hy_px
        ))

    def _Hd_uV(self, v):
        return (
            self._sDiag(self._hy_py)*self._hx_px_u(v) +
            self._sDiag(self._hx_px)*self._hy_py_u(v) -
            self._sDiag(self._hx_py)*self._hy_px_u(v) -
            self._sDiag(self._hy_px)*self._hx_py_u(v)
        )

    # Adjoint
    @property
    def _aex_px(self):
        return mkvc(mkvc(self.f[self.src, 'e_px'], 2).T*self.Pex.T)

    @property
    def _aey_px(self):
        return mkvc(mkvc(self.f[self.src, 'e_px'], 2).T*self.Pey.T)

    @property
    def _aex_py(self):
        return mkvc(mkvc(self.f[self.src, 'e_py'], 2).T*self.Pex.T)

    @property
    def _aey_py(self):
        return mkvc(mkvc(self.f[self.src, 'e_py'], 2).T*self.Pey.T)

    @property
    def _ahx_px(self):
        return mkvc(mkvc(self.f[self.src, 'b_px'], 2).T/mu_0*self.Pbx.T)

    @property
    def _ahy_px(self):
        return mkvc(mkvc(self.f[self.src, 'b_px'], 2).T/mu_0*self.Pby.T)

    @property
    def _ahz_px(self):
        return mkvc(mkvc(self.f[self.src, 'b_px'], 2).T/mu_0*self.Pbz.T)

    @property
    def _ahx_py(self):
        return mkvc(mkvc(self.f[self.src, 'b_py'], 2).T/mu_0*self.Pbx.T)

    @property
    def _ahy_py(self):
        return mkvc(mkvc(self.f[self.src, 'b_py'], 2).T/mu_0*self.Pby.T)

    @property
    def _ahz_py(self):
        return mkvc(mkvc(self.f[self.src, 'b_py'], 2).T/mu_0*self.Pbz.T)

    # NOTE: need to add a .T at the end for the output to be (nU,)
    def _aex_px_u(self, vec):
        """
        """
        # vec is (nD,) and returns a (nU,)
        return self.f._e_pxDeriv_u(self.src, self.Pex.T*mkvc(vec,), adjoint=True)

    def _aey_px_u(self, vec):
        """
        """
        # vec is (nD,) and returns a (nU,)
        return self.f._e_pxDeriv_u(self.src, self.Pey.T*mkvc(vec,), adjoint=True)

    def _aex_py_u(self, vec):
        """
        """
        # vec is (nD,) and returns a (nU,)
        return self.f._e_pyDeriv_u(self.src, self.Pex.T*mkvc(vec,), adjoint=True)

    def _aey_py_u(self, vec):
        """
        """
        # vec is (nD,) and returns a (nU,)
        return self.f._e_pyDeriv_u(self.src, self.Pey.T*mkvc(vec,), adjoint=True)

    def _ahx_px_u(self, vec):
        """
        """
        # vec is (nD,) and returns a (nU,)
        return self.f._b_pxDeriv_u(self.src, self.Pbx.T*mkvc(vec,), adjoint=True)/mu_0

    def _ahy_px_u(self, vec):
        """
        """
        # vec is (nD,) and returns a (nU,)
        return self.f._b_pxDeriv_u(self.src, self.Pby.T*mkvc(vec,), adjoint=True)/mu_0

    def _ahz_px_u(self, vec):
        """
        """
        # vec is (nD,) and returns a (nU,)
        return self.f._b_pxDeriv_u(self.src, self.Pbz.T*mkvc(vec,), adjoint=True)/mu_0

    def _ahx_py_u(self, vec):
        """
        """
        # vec is (nD,) and returns a (nU,)
        return self.f._b_pyDeriv_u(self.src, self.Pbx.T*mkvc(vec,), adjoint=True)/mu_0

    def _ahy_py_u(self, vec):
        """
        """
        # vec is (nD,) and returns a (nU,)
        return self.f._b_pyDeriv_u(self.src, self.Pby.T*mkvc(vec,), adjoint=True)/mu_0

    def _ahz_py_u(self, vec):
        """
        """
        # vec is (nD,) and returns a (nU,)
        return self.f._b_pyDeriv_u(self.src, self.Pbz.T*mkvc(vec,), adjoint=True)/mu_0

    # Define the components of the derivative
    @property
    def _aHd(self):
        return self._sDiag(1./(
            self._sDiag(self._ahx_px)*self._ahy_py -
            self._sDiag(self._ahx_py)*self._ahy_px
        ))

    def _aHd_uV(self, x):
        return (
            self._ahx_px_u(self._sDiag(self._ahy_py)*x) +
            self._ahx_px_u(self._sDiag(self._ahy_py)*x) -
            self._ahy_px_u(self._sDiag(self._ahx_py)*x) -
            self._ahx_py_u(self._sDiag(self._ahy_px)*x)
        )

    def eval(self, src, mesh, f, return_complex=False):
        """
        Function to evaluate datum for this receiver
        """
        raise NotImplementedError('SimPEG.EM.NSEM receiver has to have an eval method')

    def evalDeriv(self, src, mesh, f, v, adjoint=False):
        """
        Function to evaluate datum for this receiver
        """
        raise NotImplementedError('SimPEG.EM.NSEM receiver has to have an evalDeriv method')


class Point_impedance1D(SimPEG.Survey.BaseRx):
    """
    Natural source 1D impedance receiver class

    :param string component: real or imaginary component 'real' or 'imag'
    """

    orientation = 'yx'

    def __init__(self, locs, component=None):
        assert(component in ['real', 'imag']), "'component' must be 'real' or 'imag', not {0!s}".format(component)

        self.component = component
        SimPEG.Survey.BaseRx.__init__(self, locs, rxType=None)

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        if value is getattr(self, '_mesh', None):
            pass
        else:
            self._mesh = value

    # Utility for convienece
    def _sDiag(self, t):
        return SimPEG.Utils.sdiag(mkvc(t, 2))

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

    @property
    def Pex(self):
        if getattr(self, '_Pex', None) is None:
            self._Pex = self._mesh.getInterpolationMat(self.locs[:, -1], 'Fx')
        return self._Pex

    @property
    def Pbx(self):
        if getattr(self, '_Pbx', None) is None:
            self._Pbx = self._mesh.getInterpolationMat(self.locs[:, -1], 'Ex')
        return self._Pbx

    @property
    def _ex(self):
        return self.Pex * mkvc(self.f[self.src, 'e_1d'], 2)

    @property
    def _hx(self):
        return self.Pbx * mkvc(self.f[self.src, 'b_1d'], 2) / mu_0

    def _ex_u(self, v):
        return self.Pex * self.f._eDeriv_u(self.src, v)

    def _hx_u(self, v):
        return self.Pbx * self.f._bDeriv_u(self.src, v) / mu_0

    def _aex_u(self, v):
        return self.f._eDeriv_u(self.src, self.Pex.T * v, adjoint=True)

    def _ahx_u(self, v):
        return self.f._bDeriv_u(self.src, self.Pbx.T * v, adjoint=True) / mu_0

    @property
    def _Hd(self):
        return self._sDiag(1./self._hx)

    def eval(self, src, mesh, f, return_complex=False):
        '''
        Project the fields to natural source data.

        :param SimPEG.EM.NSEM.SrcNSEM src: NSEM source
        :param discretize.TensorMesh mesh: Mesh defining the topology of the problem
        :param SimPEG.EM.NSEM.FieldsNSEM f: NSEM fields object of the source
        :param bool (optional) return_complex: Flag for return the complex evaluation
        :rtype: numpy.array
        :return: Evaluated data for the receiver
        '''
        # NOTE: Maybe set this as a property
        self.src = src
        self.mesh = mesh
        self.f = f

        rx_eval_complex = -self._Hd * self._ex
        # Return the full impedance
        if return_complex:
            return rx_eval_complex
        return getattr(rx_eval_complex, self.component)

    def evalDeriv(self, src, mesh, f, v, adjoint=False):
        """method evalDeriv

        The derivative of the projection wrt u

        :param SimPEG.EM.NSEM.SrcNSEM src: NSEM source
        :param discretize.TensorMesh mesh: Mesh defining the topology of the problem
        :param SimPEG.EM.NSEM.FieldsNSEM f: NSEM fields object of the source
        :param numpy.ndarray v: vector of size (nU,) (adjoint=False) and size (nD,) (adjoint=True)
        :rtype: numpy.array
        :return: Calculated derivative (nD,) (adjoint=False) and (nP,2) (adjoint=True) for both polarizations
        """
        self.src = src
        self.mesh = mesh
        self.f = f

        if adjoint:
            Z1d = self.eval(src, mesh, f, True)
            def aZ_N_uV(x):
                return -self._aex_u(x)
            def aZ_D_uV(x):
                return self._ahx_u(x)
            rx_deriv = aZ_N_uV(self._Hd.T * v) - aZ_D_uV(self._sDiag(Z1d).T * self._Hd.T * v)
            if self.component == 'imag':
                rx_deriv_component = 1j*rx_deriv
            elif self.component == 'real':
                rx_deriv_component = rx_deriv.astype(complex)
        else:
            Z1d = self.eval(src, mesh, f, True)
            Z_N_uV = -self._ex_u(v)
            Z_D_uV = self._hx_u(v)
            # Evaluate
            rx_deriv = self._Hd * (Z_N_uV - self._sDiag(Z1d) * Z_D_uV)
            rx_deriv_component = np.array(getattr(rx_deriv, self.component))
        return rx_deriv_component


class Point_impedance3D(BaseRxNSEM_Point):
    """
    Natural source 3D impedance receiver class

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string orientation: receiver orientation 'xx', 'xy', 'yx' or 'yy'
    :param string component: real or imaginary component 'real' or 'imag'
    """

    def __init__(self, locs, orientation=None, component=None):

        BaseRxNSEM_Point.__init__(self, locs, orientation=orientation, component=component)

    def eval(self, src, mesh, f, return_complex=False):
        '''
        Project the fields to natural source data.

        :param SrcNSEM src: The source of the fields to project
        :param discretize.TensorMesh mesh: topological mesh corresponding to the fields
        :param FieldsNSEM f: Natural source fields object to project
        :rtype: numpy.array
        :return: component of the impedance evaluation
        '''
        # NOTE: Maybe set this as a property
        self.src = src
        self.mesh = mesh
        self.f = f

        if 'xx' in self.orientation:
            Zij = ( self._ex_px * self._hy_py - self._ex_py * self._hy_px)
        elif 'xy' in self.orientation:
            Zij = (-self._ex_px * self._hx_py + self._ex_py * self._hx_px)
        elif 'yx' in self.orientation:
            Zij = ( self._ey_px * self._hy_py - self._ey_py * self._hy_px)
        elif 'yy' in self.orientation:
            Zij = (-self._ey_px * self._hx_py + self._ey_py * self._hx_px)
        # Calculate the complex value
        rx_eval_complex = self._Hd * Zij

        # Return the full impedance
        if return_complex:
            return rx_eval_complex
        return getattr(rx_eval_complex, self.component)

    def evalDeriv(self, src, mesh, f, v, adjoint=False):
        """
        The derivative of the projection wrt u

        :param SimPEG.EM.NSEM.SrcNSEM src: NSEM source
        :param discretize.TensorMesh mesh: Mesh defining the topology of the problem
        :param SimPEG.EM.NSEM.FieldsNSEM f: NSEM fields object of the source
        :param numpy.ndarray v: vector of size (nU,) (adjoint=False) and size (nD,) (adjoint=True)
        :rtype: numpy.array
        :return: Calculated derivative (nD,) (adjoint=False) and (nP,2) (adjoint=True) for both polarizations
        """
        self.src = src
        self.mesh = mesh
        self.f = f

        if adjoint:
            if 'xx' in self.orientation:
                Zij = self._sDiag(self._aHd * (
                    self._sDiag(self._ahy_py)*self._aex_px -
                    self._sDiag(self._ahy_px)*self._aex_py
                ))

                def ZijN_uV(x):
                    return (
                        self._aex_px_u(self._sDiag(self._ahy_py) * x) +
                        self._ahy_py_u(self._sDiag(self._aex_px) * x) -
                        self._ahy_px_u(self._sDiag(self._aex_py) * x) -
                        self._aex_py_u(self._sDiag(self._ahy_px) * x)
                    )
            elif 'xy' in self.orientation:
                Zij = self._sDiag(self._aHd * (
                    -self._sDiag(self._ahx_py) * self._aex_px +
                    self._sDiag(self._ahx_px) * self._aex_py
                ))

                def ZijN_uV(x):
                    return (
                        -self._aex_px_u(self._sDiag(self._ahx_py) * x) -
                        self._ahx_py_u(self._sDiag(self._aex_px) * x) +
                        self._ahx_px_u(self._sDiag(self._aex_py) * x) +
                        self._aex_py_u(self._sDiag(self._ahx_px) * x)
                    )
            elif 'yx' in self.orientation:
                Zij = self._sDiag(self._aHd * (
                    self._sDiag(self._ahy_py) * self._aey_px -
                    self._sDiag(self._ahy_px) * self._aey_py
                ))

                def ZijN_uV(x):
                    return (
                        self._aey_px_u(self._sDiag(self._ahy_py) * x) +
                        self._ahy_py_u(self._sDiag(self._aey_px) * x) -
                        self._ahy_px_u(self._sDiag(self._aey_py) * x) -
                        self._aey_py_u(self._sDiag(self._ahy_px) * x)
                    )
            elif 'yy' in self.orientation:
                Zij = self._sDiag(self._aHd * (
                    -self._sDiag(self._ahx_py) * self._aey_px +
                    self._sDiag(self._ahx_px) * self._aey_py))

                def ZijN_uV(x):
                    return (
                        -self._aey_px_u(self._sDiag(self._ahx_py) * x) -
                        self._ahx_py_u(self._sDiag(self._aey_px) * x) +
                        self._ahx_px_u(self._sDiag(self._aey_py) * x) +
                        self._aey_py_u(self._sDiag(self._ahx_px) * x)
                    )

            # Calculate the complex derivative
            rx_deriv_real = ZijN_uV(self._aHd * v) - self._aHd_uV(Zij.T * self._aHd * v)
            # NOTE: Need to reshape the output to go from 2*nU array to a (nU,2) matrix for each polarization
            # rx_deriv_real = np.hstack((mkvc(rx_deriv_real[:len(rx_deriv_real)/2],2),mkvc(rx_deriv_real[len(rx_deriv_real)/2::],2)))
            rx_deriv_real = rx_deriv_real.reshape((2, self.mesh.nE)).T
            # Extract the data
            if self.component == 'imag':
                rx_deriv_component = 1j * rx_deriv_real
            elif self.component == 'real':
                rx_deriv_component = rx_deriv_real.astype(complex)
        else:
            if 'xx' in self.orientation:
                ZijN_uV = (
                    self._sDiag(self._hy_py) * self._ex_px_u(v) +
                    self._sDiag(self._ex_px) * self._hy_py_u(v) -
                    self._sDiag(self._ex_py) * self._hy_px_u(v) -
                    self._sDiag(self._hy_px) * self._ex_py_u(v)
                )
            elif 'xy' in self.orientation:
                ZijN_uV = (
                    -self._sDiag(self._hx_py) * self._ex_px_u(v) -
                    self._sDiag(self._ex_px) * self._hx_py_u(v) +
                    self._sDiag(self._ex_py) * self._hx_px_u(v) +
                    self._sDiag(self._hx_px) * self._ex_py_u(v)
                )
            elif 'yx' in self.orientation:
                ZijN_uV = (
                    self._sDiag(self._hy_py) * self._ey_px_u(v) +
                    self._sDiag(self._ey_px) * self._hy_py_u(v) -
                    self._sDiag(self._ey_py) * self._hy_px_u(v) -
                    self._sDiag(self._hy_px) * self._ey_py_u(v)
                )
            elif 'yy' in self.orientation:
                ZijN_uV = (
                    -self._sDiag(self._hx_py) * self._ey_px_u(v) -
                    self._sDiag(self._ey_px) * self._hx_py_u(v) +
                    self._sDiag(self._ey_py) * self._hx_px_u(v) +
                    self._sDiag(self._hx_px) * self._ey_py_u(v)
                )

            Zij = self.eval(src, self.mesh, self.f, True)
            # Calculate the complex derivative
            rx_deriv_real = self._Hd * (ZijN_uV - self._sDiag(Zij) * self._Hd_uV(v))
            rx_deriv_component = np.array(getattr(rx_deriv_real, self.component))

        return rx_deriv_component


class Point_tipper3D(BaseRxNSEM_Point):
    """
    Natural source 3D tipper receiver base class

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    :param string component: real or imaginary component 'real' or 'imag'
    """

    def __init__(self, locs, orientation=None, component=None):

        BaseRxNSEM_Point.__init__(
            self, locs, orientation=orientation, component=component
        )

    def eval(self, src, mesh, f, return_complex=False):
        '''
        Project the fields to natural source data.

        :param SrcNSEM src: The source of the fields to project
        :param discretize.TensorMesh mesh: Mesh defining the topology of the problem
        :param FieldsNSEM f: Natural source fields object to project
        :rtype: numpy.array
        :return: Evaluated component of the impedance data
        '''
        # NOTE: Maybe set this as a property
        self.src = src
        self.mesh = mesh
        self.f = f

        if 'zx' in self.orientation:
            Tij = (- self._hy_px * self._hz_py + self._hy_py * self._hz_px)
        if 'zy' in self.orientation:
            Tij = (self._hx_px * self._hz_py - self._hx_py * self._hz_px)
        rx_eval_complex = self._Hd * Tij

        # Return the full impedance
        if return_complex:
            return rx_eval_complex
        return getattr(rx_eval_complex, self.component)

    def evalDeriv(self, src, mesh, f, v, adjoint=False):
        """
        The derivative of the projection wrt u

        :param SimPEG.EM.NSEM.SrcNSEM src: NSEM source
        :param discretize.TensorMesh mesh: Mesh defining the topology of the problem
        :param SimPEG.EM.NSEM.FieldsNSEM f: NSEM fields object of the source
        :param numpy.ndarray v: Random vector of size
        :rtype: numpy.array
        :return: Calculated derivative (nD,) (adjoint=False) and (nP,2) (adjoint=True)
            for both polarizations
        """
        self.src = src
        self.mesh = mesh
        self.f = f

        if adjoint:
            if 'zx' in self.orientation:
                Tij = self._sDiag(self._aHd * (
                    -self._sDiag(self._ahz_py) * self._ahy_px +
                    self._sDiag(self._ahz_px) * self._ahy_py)
                )

                def TijN_uV(x):
                    return (
                        -self._ahz_py_u(self._sDiag(self._ahy_px) * x) -
                        self._ahy_px_u(self._sDiag(self._ahz_py) * x) +
                        self._ahy_py_u(self._sDiag(self._ahz_px) * x) +
                        self._ahz_px_u(self._sDiag(self._ahy_py) * x)
                    )
            elif 'zy' in self.orientation:
                Tij = self._sDiag(self._aHd * (
                    self._sDiag(self._ahz_py) * self._ahx_px -
                    self._sDiag(self._ahz_px) * self._ahx_py)
                )

                def TijN_uV(x):
                    return (
                        self._ahx_px_u(self._sDiag(self._ahz_py) * x) +
                        self._ahz_py_u(self._sDiag(self._ahx_px) * x) -
                        self._ahx_py_u(self._sDiag(self._ahz_px) * x) -
                        self._ahz_px_u(self._sDiag(self._ahx_py) * x)
                    )

            # Calculate the complex derivative
            rx_deriv_real = (
                TijN_uV(self._aHd * v) -
                self._aHd_uV(Tij.T * self._aHd * v)
            )
            # NOTE: Need to reshape the output to go from 2*nU array to a (nU,2) matrix for each polarization
            # rx_deriv_real = np.hstack((mkvc(rx_deriv_real[:len(rx_deriv_real)/2],2),mkvc(rx_deriv_real[len(rx_deriv_real)/2::],2)))
            rx_deriv_real = rx_deriv_real.reshape((2, self.mesh.nE)).T
            # Extract the data
            if self.component == 'imag':
                rx_deriv_component = 1j * rx_deriv_real
            elif self.component == 'real':
                rx_deriv_component = rx_deriv_real.astype(complex)
        else:
            if 'zx' in self.orientation:
                TijN_uV = (
                    -self._sDiag(self._hy_px) * self._hz_py_u(v) -
                    self._sDiag(self._hz_py) * self._hy_px_u(v) +
                    self._sDiag(self._hy_py) * self._hz_px_u(v) +
                    self._sDiag(self._hz_px) * self._hy_py_u(v)
                )
            elif 'zy' in self.orientation:
                TijN_uV = (
                    self._sDiag(self._hz_py) * self._hx_px_u(v) +
                    self._sDiag(self._hx_px) * self._hz_py_u(v) -
                    self._sDiag(self._hx_py) * self._hz_px_u(v) -
                    self._sDiag(self._hz_px) * self._hx_py_u(v)
                )
            Tij = self.eval(src, mesh, f, True)
            # Calculate the complex derivative
            rx_deriv_complex = (
                self._Hd * (TijN_uV - self._sDiag(Tij) * self._Hd_uV(v))
            )
            rx_deriv_component = np.array(
                getattr(rx_deriv_complex, self.component)
            )

        return rx_deriv_component
