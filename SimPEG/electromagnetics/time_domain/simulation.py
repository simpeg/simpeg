import numpy as np
import scipy.sparse as sp
from scipy.constants import mu_0
import time
import properties

from ...data import Data
from ...simulation import BaseTimeSimulation
from ...utils import mkvc, sdiag, speye, Zero
from ..base import BaseEMSimulation
from .survey import Survey
from .fields import (
    Fields3D_b, Fields3D_e, Fields3D_h, Fields3D_j,
    Fields_Derivs_eb, Fields_Derivs_hj
)



class BaseTDEMSimulation(BaseTimeSimulation, BaseEMSimulation):
    """
    We start with the first order form of Maxwell's equations, eliminate and
    solve the second order form. For the time discretization, we use backward
    Euler.
    """
    clean_on_model_update = ['_Adcinv']  #: clear DC matrix factors on any model updates
    dt_threshold = 1e-8

    survey = properties.Instance(
        "a survey object", Survey, required=True
    )

    # def fields_nostore(self, m):
    #     """
    #     Solve the forward problem without storing fields

    #     :param numpy.ndarray m: inversion model (nP,)
    #     :rtype: numpy.ndarray
    #     :return numpy.ndarray: numpy.ndarray (nD,)

    #     """

    def fields(self, m):
        """
        Solve the forward problem for the fields.

        :param numpy.ndarray m: inversion model (nP,)
        :rtype: SimPEG.EM.TDEM.FieldsTDEM
        :return f: fields object
        """

        self.model = m
        f = self.fieldsPair(self)
        solution_field = self._fieldType + 'Solution'

        # set initial fields and initialize Ainv
        f[:, solution_field, 0] = self.getInitialFields()
        Ainv = None

        for time_ind, dt in enumerate(self.timeSteps):
            # keep factors if dt is the same as previous step b/c A will be the
            # same
            if Ainv is not None and (
                abs(dt-self.timeSteps[time_ind - 1]) > self.dt_threshold
            ):
                Ainv.clean()
                Ainv = None

            if Ainv is None:
                A = self.getAdiag(time_ind)
                if self.verbose:
                    print('Factoring...   (dt = {:e})'.format(dt))
                Ainv = self.Solver(A, **self.solver_opts)
                if self.verbose:
                    print('Done')

            # get rhs
            rhs = self.getRHS(time_ind)  # this should include the initial fields if time_ind == 0

            # getRHS handles the inclusion of the initial fields
            if time_ind > 0:
                Asubdiag = self.getAsubdiag(time_ind)
                rhs = rhs - Asubdiag * f[:, solution_field, time_ind]

            f[:, solution_field, time_ind+1] = Ainv * rhs

        Ainv.clean()
        return f

    def Jvec(self, m, v, f=None):

        if f is None:
            f = self.fields(m)

        self.model = m
        solution_field = self._fieldType + 'Solution'

        dun_dm_v = np.hstack([
            mkvc(self.getInitialFieldsDeriv(src, v, f=f), 2)
            for src in self.survey.srcList
        ])
        # can over-write this at each timestep
        # store the field derivs we need to project to calc full deriv
        df_dm_v = self.Fields_Derivs(self)

        Ainv = None

        for time_ind, dt in enumerate(self.timeSteps):
            if Ainv is not None and (
                abs(dt-self.timeSteps[time_ind - 1]) > self.dt_threshold
            ):
                Ainv.clean()
                Ainv = None

            if Ainv is None:
                A = self.getAdiag(time_ind)
                if self.verbose:
                    print('Factoring...   (dt = {:e})'.format(dt))
                Ainv = self.Solver(A, **self.solver_opts)
                if self.verbose:
                    print('Done')

            if time_ind > 0:
                Asubdiag = self.getAsubdiag(time_ind)

            for src_ind, src in enumerate(self.survey.source_list):
                for field_type in set([rx.projField for rx in src.receiver_list]):
                    deriv_name = '{}Deriv'.format(field_type)
                    df_dm_v[src, deriv_name, time_ind] = getattr(f, "_{}".format(deriv_name))(
                        time_ind, src, dun_dm_v[:, src_ind], v
                    )

                un_src = f[src, solution_field, time_ind+1]

                # cell centered on time mesh
                dA_dm_v = self.getAdiagDeriv(time_ind, un_src, v)
                # on nodes of time mesh
                dRHS_dm_v = self.getRHSDeriv(time_ind, src, v, f=f)

                JRHS = dRHS_dm_v - dA_dm_v

                if time_ind > 0:
                    dAsubdiag_dm_v = self.getAsubdiagDeriv(
                        time_ind, f[src, solution_field, time_ind], v
                    )

                    JRHS = JRHS - dAsubdiag_dm_v - Asubdiag * dun_dm_v[:, src_ind]

                # step in time and overwrite
                if time_ind != len(self.timeSteps+1):
                    dun_dm_v[:, src_ind] = Ainv * JRHS

        Jv = []
        for src in self.survey.srcList:
            for rx in src.receiver_list:
                Jv.append(
                    rx.evalDeriv(src, self.mesh, self.timeMesh, f,   mkvc(
                            df_dm_v[src, '%sDeriv' % rx.projField, :]
                        )
                    )
                )
        Ainv.clean()
        # del df_dm_v, dun_dm_v, Asubdiag
        # return mkvc(Jv)
        return np.hstack(Jv)

    def Jtvec(self, m, v, f=None):

        """
        Jvec computes the adjoint of the sensitivity times a vector

        .. math::
            \mathbf{J}^\\top \mathbf{v} =  \left(
            \\frac{d\mathbf{u}}{d\mathbf{m}} ^ \\top
            \\frac{d\mathbf{F}}{d\mathbf{u}} ^ \\top  +
            \\frac{\partial\mathbf{F}}{\partial\mathbf{m}} ^ \\top \\right)
            \\frac{d\mathbf{P}}{d\mathbf{F}} ^ \\top \mathbf{v}

        where

        .. math::
            \\frac{d\mathbf{u}}{d\mathbf{m}} ^\\top \mathbf{A}^\\top  +
            \\frac{d\mathbf{A}(\mathbf{u})}{d\mathbf{m}} ^ \\top =
            \\frac{d \mathbf{RHS}}{d \mathbf{m}} ^ \\top
        """

        if f is None:
            f = self.fields(m)

        self.model = m
        solution_field = self._fieldType + 'Solution'
        solution_field_deriv = '{}Deriv'.format(self._fieldType)

        # Ensure v is a data object.
        if not isinstance(v, Data):
            v = Data(self.survey, v)

        df_duT_v = self.Fields_Derivs(self)

        # same size as fields at a single timestep
        ATinv_df_duT_v = np.zeros(
            (
                len(self.survey.srcList),
                len(f[self.survey.srcList[0], solution_field, 0])
            ),
            dtype=float
        )
        JTv = np.zeros(m.shape, dtype=float)

        # Loop over sources and receivers to create a fields object:
        # PT_v, df_duT_v, df_dmT_v
        # initialize storage for PT_v (don't need to preserve over sources)
        PT_v = self.Fields_Derivs(self)
        for src in self.survey.srcList:
            # Looping over initializing field class is appending memory!
            # PT_v = Fields_Derivs(self.mesh) # initialize storage
            # #for PT_v (don't need to preserve over sources)
            # initialize size
            df_duT_v[src, '{}Deriv'.format(self._fieldType), :] = (
                np.zeros_like(f[src, self._fieldType, :])
            )

            for rx in src.receiver_list:
                rx_field_deriv = '{}Deriv'.format(rx.projField)
                PT_v[src, rx_field_deriv, :] = rx.evalDeriv(
                    src, self.mesh, self.timeMesh, f, mkvc(v[src, rx]),
                    adjoint=True
                ) # this is +=

                df_duT_fun = getattr(f, '_{}'.format(rx_field_deriv))

                for time_ind in range(self.nT + 1):
                    PT_vi = mkvc(PT_v[src, rx_field_deriv, time_ind])
                    deriv_u, deriv_m = df_duT_fun(
                        time_ind, src, None, PT_vi, adjoint=True
                    )

                    df_duT_v[src, solution_field_deriv, time_ind] = (
                        df_duT_v[src, solution_field_deriv, time_ind] +
                        mkvc(deriv_u, 2)
                    )
                    JTv = deriv_m + JTv

        # del PT_v # no longer need this (captured in df_duT_v, deriv_m)

        ATinv = None

        # Do the back-solve through time
        # if the previous timestep is the same: no need to refactor the matrix
        # for tInd, dt in zip(range(self.nT), self.timeSteps):

        for time_ind in reversed(range(self.nT)):
            # tInd = tIndP - 1
            # if time_ind > -1:
            if ATinv is not None and (
                self.timeSteps[time_ind] != self.timeSteps[time_ind+1]
            ):
                ATinv.clean()
                ATinv = None

            # refactor if we need to
            if ATinv is None:  # and tInd > -1:
                A = self.getAdiag(time_ind)
                ATinv = self.Solver(A.T, **self.solver_opts)

            # RHS will deal with the initial condition case
            if time_ind < self.nT - 1:
                Asubdiag = self.getAsubdiag(time_ind+1)

            for isrc, src in enumerate(self.survey.srcList):

                # solve against df_duT_v
                # last timestep (first to be solved)
                # if time_ind >= self.nT-1:
                rhs = mkvc(df_duT_v[src, solution_field_deriv, time_ind+1])

                if time_ind < self.nT - 1:
                    rhs = rhs - Asubdiag.T * mkvc(ATinv_df_duT_v[isrc, :])

                # if time_ind > -1:
                ATinv_df_duT_v[isrc, :] = ATinv * rhs
                # cell centered on time mesh
                un_src = f[src, solution_field, time_ind+1]
                dAT_dm_v = self.getAdiagDeriv(
                    time_ind, un_src, ATinv_df_duT_v[isrc, :], adjoint=True
                )
                dAsubdiagT_dm_v = self.getAsubdiagDeriv(
                    time_ind, f[src, solution_field, time_ind],
                    ATinv_df_duT_v[isrc, :], adjoint=True
                )
                # else:
                #     ATinv_df_duT_v[isrc, :] = rhs
                #     dAT_dm_v = Zero()
                #     dAsubdiag_dm_v = Zero()

                dRHST_dm_v = self.getRHSDeriv(
                    time_ind, src, ATinv_df_duT_v[isrc, :], adjoint=True, f=f
                )  # on nodes of time mesh

                if time_ind == 0:
                    # PT_vi = mkvc(PT_v[src, solution_field_deriv, time_ind])
                    deriv_mi = self.getInitialFieldsDeriv(
                        src, mkvc(df_duT_v[
                            src, '{}Deriv'.format(self._fieldType), time_ind+1
                        ]
                        ), f=f, adjoint=True
                    )
                    print(np.linalg.norm(deriv_mi))
                    JTv = JTv + deriv_mi

                JTv = JTv +  mkvc(-dAT_dm_v - dAsubdiagT_dm_v + dRHST_dm_v)

        if ATinv is not None:
            ATinv.clean()

        return mkvc(JTv)

    def getInitialFields(self):
        """
        Ask the sources for initial fields
        """

        Srcs = self.survey.srcList

        if self._fieldType in ['b', 'j']:
            ifields = np.zeros((self.mesh.nF, len(Srcs)))
        elif self._fieldType in ['e', 'h']:
            ifields = np.zeros((self.mesh.nE, len(Srcs)))

        if self.verbose:
            print("Calculating Initial fields")

        for i, src in enumerate(Srcs):
            ifields[:, i] = (
                ifields[:, i] + getattr(
                    src, '{}Initial'.format(self._fieldType), None
                )(self)
            )

        return ifields

    def getInitialFieldsDeriv(self, src, v, adjoint=False, f=None):

        ifieldsDeriv = getattr(
            src, '{}InitialDeriv'.format(self._fieldType), None
        )(self, v, adjoint, f)

        # take care of any utils.zero cases
        if adjoint is False:
            if self._fieldType in ['b', 'j']:
                ifieldsDeriv += np.zeros(self.mesh.nF)
            elif self._fieldType in ['e', 'h']:
                ifieldsDeriv += np.zeros(self.mesh.nE)

        elif adjoint is True:
            ifieldsDeriv += np.zeros_like(self.model)

        return ifieldsDeriv

    # Store matrix factors if we need to solve the DC problem to get the
    # initial condition
    @property
    def Adcinv(self):
        if not hasattr(self, 'getAdc'):
            raise NotImplementedError(
                "Support for galvanic sources has not been implemented for "
                "{}-formulation".format(self._fieldType)
            )
        if getattr(self, '_Adcinv', None) is None:
            if self.verbose:
                print("Factoring the system matrix for the DC problem")
            Adc = self.getAdc()
            self._Adcinv = self.Solver(Adc)
        return self._Adcinv


###############################################################################
#                                                                             #
#                                E-B Formulation                              #
#                                                                             #
###############################################################################

# ------------------------------- Problem3D_b ------------------------------- #

class Problem3D_b(BaseTDEMSimulation):
    """
    Starting from the quasi-static E-B formulation of Maxwell's equations
    (semi-discretized)

    .. math::

        \mathbf{C} \mathbf{e} + \\frac{\partial \mathbf{b}}{\partial t} =
        \mathbf{s_m} \\\\
        \mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f} \mathbf{b} -
        \mathbf{M_{\sigma}^e} \mathbf{e} = \mathbf{s_e}


    where :math:`\mathbf{s_e}` is an integrated quantity, we eliminate
    :math:`\mathbf{e}` using

    .. math::

        \mathbf{e} = \mathbf{M_{\sigma}^e}^{-1} \mathbf{C}^{\\top}
        \mathbf{M_{\mu^{-1}}^f} \mathbf{b} -
        \mathbf{M_{\sigma}^e}^{-1} \mathbf{s_e}


    to obtain a second order semi-discretized system in :math:`\mathbf{b}`

    .. math::

        \mathbf{C} \mathbf{M_{\sigma}^e}^{-1} \mathbf{C}^{\\top}
        \mathbf{M_{\mu^{-1}}^f} \mathbf{b}  +
        \\frac{\partial \mathbf{b}}{\partial t} =
        \mathbf{C} \mathbf{M_{\sigma}^e}^{-1} \mathbf{s_e} + \mathbf{s_m}


    and moving everything except the time derivative to the rhs gives

    .. math::
        \\frac{\partial \mathbf{b}}{\partial t} =
        -\mathbf{C} \mathbf{M_{\sigma}^e}^{-1} \mathbf{C}^{\\top}
        \mathbf{M_{\mu^{-1}}^f} \mathbf{b} +
        \mathbf{C} \mathbf{M_{\sigma}^e}^{-1} \mathbf{s_e} + \mathbf{s_m}

    For the time discretization, we use backward euler. To solve for the
    :math:`n+1` th time step, we have

    .. math::

        \\frac{\mathbf{b}^{n+1} - \mathbf{b}^{n}}{\mathbf{dt}} =
        -\mathbf{C} \mathbf{M_{\sigma}^e}^{-1} \mathbf{C}^{\\top}
        \mathbf{M_{\mu^{-1}}^f} \mathbf{b}^{n+1} +
        \mathbf{C} \mathbf{M_{\sigma}^e}^{-1} \mathbf{s_e}^{n+1} +
        \mathbf{s_m}^{n+1}


    re-arranging to put :math:`\mathbf{b}^{n+1}` on the left hand side gives

    .. math::

        (\mathbf{I} + \mathbf{dt} \mathbf{C} \mathbf{M_{\sigma}^e}^{-1}
         \mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f}) \mathbf{b}^{n+1} =
         \mathbf{b}^{n} + \mathbf{dt}(\mathbf{C} \mathbf{M_{\sigma}^e}^{-1}
         \mathbf{s_e}^{n+1} + \mathbf{s_m}^{n+1})

    """

    _fieldType = 'b'
    _formulation = 'EB'
    fieldsPair = Fields3D_b  #: A SimPEG.EM.TDEM.Fields3D_b object
    Fields_Derivs = Fields_Derivs_eb

    def getAdiag(self, tInd):
        """
        System matrix at a given time index

        .. math::
            (\mathbf{I} + \mathbf{dt} \mathbf{C} \mathbf{M_{\sigma}^e}^{-1}
            \mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f})

        """
        assert tInd >= 0 and tInd < self.nT

        dt = self.timeSteps[tInd]
        C = self.mesh.edgeCurl
        MeSigmaI = self.MeSigmaI
        MfMui = self.MfMui
        I = speye(self.mesh.nF)

        A = 1./dt * I + (C * (MeSigmaI * (C.T * MfMui)))

        if self._makeASymmetric is True:
            return MfMui.T * A
        return A

    def getAdiagDeriv(self, tInd, u, v, adjoint=False):
        """
        Derivative of ADiag
        """
        C = self.mesh.edgeCurl

        # def MeSigmaIDeriv(x):
        #     return self.MeSigmaIDeriv(x)

        MfMui = self.MfMui

        if adjoint:
            if self._makeASymmetric is True:
                v = MfMui * v
            return self.MeSigmaIDeriv(C.T * (MfMui * u), C.T * v, adjoint)

        ADeriv = (C * (self.MeSigmaIDeriv(C.T * (MfMui * u), v, adjoint)))

        if self._makeASymmetric is True:
            return MfMui.T * ADeriv
        return ADeriv

    def getAsubdiag(self, tInd):
        """
        Matrix below the diagonal
        """

        dt = self.timeSteps[tInd]
        MfMui = self.MfMui
        Asubdiag = - 1./dt * sp.eye(self.mesh.nF)

        if self._makeASymmetric is True:
            return MfMui.T * Asubdiag

        return Asubdiag

    def getAsubdiagDeriv(self, tInd, u, v, adjoint=False):
        return Zero() * v

    def getRHS(self, tInd):
        """
        Assemble the RHS
        """
        source_list = self.survey.source_list
        rhs = np.zeros((self.mesh.nF, len(source_list)))
        C = self.mesh.edgeCurl
        MeSigmaI = self.MeSigmaI

        for i, src in enumerate(self.survey.source_list):
            s_en = src.s_e(self, tInd+1)
            s_mn = src.s_m(self, tInd+1)

            rhs[:, i] = (
                rhs[:, i] + (C * (MeSigmaI * s_en) + s_mn)
            )

            # handle the initial condition
            if tInd == 0 and src.waveform.hasInitialFields:
                if src.srcType == "galvanic":
                    einitial = src.eInitial(self)
                    rhs[:, i] = rhs[:, i] - C * einitial

                elif src.srcType == "inductive":
                    dt = self.timeSteps[tInd]
                    binitial = src.bInitial(self)
                    rhs[:, i] = rhs[:, i] + 1/dt * binitial

        if self._makeASymmetric is True:
            MfMui = self.MfMui
            return MfMui.T * rhs
        return rhs

    def getRHSDeriv(self, tInd, src, v, adjoint=False, f=None):
        # todo: we should not be passing the fields here - the source should
        # store the intial fields and clear them if dependent on model update
        """
        Derivative of the RHS
        """

        C = self.mesh.edgeCurl
        MeSigmaI = self.MeSigmaI

        MfMui = self.MfMui

        s_e = src.s_e(self, self.times[tInd+1])
        s_mDeriv, s_eDeriv = src.evalDeriv(
            self, self.times[tInd], adjoint=adjoint
        )

        if adjoint:
            if self._makeASymmetric is True:
                v = self.MfMui * v
            if isinstance(s_e,  Zero):
                MeSigmaIDerivT_v = Zero()
            else:
                MeSigmaIDerivT_v = self.MeSigmaIDeriv(s_e, C.T * v, adjoint)

            RHSDeriv = (
                MeSigmaIDerivT_v + s_eDeriv(MeSigmaI.T * (C.T * v)) +
                s_mDeriv(v)
            )

            # handle the initial condition
            if tInd == 0 and src.waveform.hasInitialFields:
                if src.srcType == "galvanic":
                    einitial_deriv = src.eInitialDeriv(self, C.T * v, f=f, adjoint=True)
                    RHSDeriv = RHSDeriv - einitial_deriv
                elif src.srcType == "inductive":
                    dt = self.timeSteps[tInd]
                    binitial_deriv = src.bInitialDeriv(self, v, f=f, adjoint=True)
                    RHSDeriv = RHSDeriv + 1/dt * binitial_deriv
            return RHSDeriv

        if isinstance(s_e,  Zero):
            MeSigmaIDeriv_v =  Zero()
        else:
            MeSigmaIDeriv_v = self.MeSigmaIDeriv(s_e, v, adjoint)

        RHSDeriv = (
            C * MeSigmaIDeriv_v + C * MeSigmaI * s_eDeriv(v) + s_mDeriv(v)
        )

        # handle the initial condition
        if tInd == 0 and src.waveform.hasInitialFields:
            if src.srcType == "galvanic":
                einitial_deriv = src.eInitialDeriv(self, v, f=f)
                RHSDeriv = RHSDeriv - C * einitial_deriv
            elif src.srcType == "inductive":
                dt = self.timeSteps[tInd]
                binitial_deriv = src.bInitialDeriv(self, v, f=f)
                RHSDeriv = RHSDeriv + 1/dt * binitial_deriv


        if self._makeASymmetric is True:
            return self.MfMui.T * RHSDeriv
        return RHSDeriv


# ------------------------------- Problem3D_e ------------------------------- #
class Problem3D_e(BaseTDEMSimulation):
    """
        Solve the EB-formulation of Maxwell's equations for the electric field, e.

        Starting with

        .. math::

            \\nabla \\times \\mathbf{e} + \\frac{\\partial \\mathbf{b}}{\\partial t} = \\mathbf{s_m} \\
            \\nabla \\times \mu^{-1} \\mathbf{b} - \sigma \\mathbf{e} = \\mathbf{s_e}


        we eliminate :math:`\\frac{\\partial b}{\\partial t}` using

        .. math::

            \\frac{\\partial \\mathbf{b}}{\\partial t} = - \\nabla \\times \\mathbf{e} + \\mathbf{s_m}


        taking the time-derivative of Ampere's law, we see

        .. math::

            \\frac{\\partial}{\\partial t}\left( \\nabla \\times \mu^{-1} \\mathbf{b} - \\sigma \\mathbf{e} \\right) = \\frac{\\partial \\mathbf{s_e}}{\\partial t} \\
            \\nabla \\times \\mu^{-1} \\frac{\\partial \\mathbf{b}}{\\partial t} - \\sigma \\frac{\\partial\\mathbf{e}}{\\partial t} = \\frac{\\partial \\mathbf{s_e}}{\\partial t}


        which gives us

        .. math::

            \\nabla \\times \\mu^{-1} \\nabla \\times \\mathbf{e} + \\sigma \\frac{\\partial\\mathbf{e}}{\\partial t} = \\nabla \\times \\mu^{-1} \\mathbf{s_m} + \\frac{\\partial \\mathbf{s_e}}{\\partial t}


    """

    _fieldType = 'e'
    _formulation = 'EB'
    fieldsPair = Fields3D_e  #: A Fields3D_e
    Fields_Derivs = Fields_Derivs_eb

    # @profile
    # def Jtvec(self, m, v, f=None):

    #     """
    #         Jvec computes the adjoint of the sensitivity times a vector
    #     """

    #     if f is None:
    #         f = self.fields(m)

    #     self.model = m
    #     ftype = self._fieldType + 'Solution'  # the thing we solved for

    #     # Ensure v is a data object.
    #     if not isinstance(v, Data):
    #         v = Data(self.survey, v)

    #     df_duT_v = self.Fields_Derivs(self)

    #     # same size as fields at a single timestep
    #     ATinv_df_duT_v = np.zeros(
    #         (
    #             len(self.survey.srcList),
    #             len(f[self.survey.srcList[0], ftype, 0])
    #         ),
    #         dtype=float
    #     )
    #     JTv = np.zeros(m.shape, dtype=float)

    #     # Loop over sources and receivers to create a fields object:
    #     # PT_v, df_duT_v, df_dmT_v
    #     # initialize storage for PT_v (don't need to preserve over sources)
    #     PT_v = self.Fields_Derivs(self)
    #     for src in self.survey.srcList:
    #         # Looping over initializing field class is appending memory!
    #         # PT_v = Fields_Derivs(self.mesh) # initialize storage
    #         # #for PT_v (don't need to preserve over sources)
    #         # initialize size
    #         df_duT_v[src, '{}Deriv'.format(self._fieldType), :] = (
    #             np.zeros_like(f[src, self._fieldType, :])
    #         )

    #         for rx in src.rxList:
    #             PT_v[src, '{}Deriv'.format(rx.projField), :] = rx.evalDeriv(
    #                 src, self.mesh, self.timeMesh, f, mkvc(v[src, rx]),
    #                 adjoint=True
    #             )
    #             # this is +=

    #             # PT_v = np.reshape(curPT_v,(len(curPT_v)/self.timeMesh.nN,
    #             # self.timeMesh.nN), order='F')
    #             df_duTFun = getattr(f, '_{}Deriv'.format(rx.projField), None)

    #             for tInd in range(self.nT+1):
    #                 cur = df_duTFun(
    #                     tInd, src, None, mkvc(
    #                         PT_v[src, '{}Deriv'.format(rx.projField), tInd]
    #                     ),
    #                     adjoint=True
    #                 )

    #                 df_duT_v[src, '{}Deriv'.format(self._fieldType), tInd] = (
    #                     df_duT_v[src, '{}Deriv'.format(self._fieldType), tInd]
    #                     + mkvc(cur[0], 2)
    #                     )
    #                 JTv = cur[1] + JTv

    #     # no longer need this
    #     del PT_v

    #     AdiagTinv = None

    #     # Do the back-solve through time
    #     # if the previous timestep is the same: no need to refactor the matrix
    #     # for tInd, dt in zip(range(self.nT), self.timeSteps):

    #     for tInd in reversed(range(self.nT)):
    #         # tInd = tIndP - 1
    #         if AdiagTinv is not None and (
    #             tInd <= self.nT and
    #             self.timeSteps[tInd] != self.timeSteps[tInd+1]
    #         ):
    #             AdiagTinv.clean()
    #             AdiagTinv = None

    #         # refactor if we need to
    #         if AdiagTinv is None:  # and tInd > -1:
    #             Adiag = self.getAdiag(tInd)
    #             AdiagTinv = self.Solver(Adiag.T, **self.solver_opts)

    #         if tInd < self.nT - 1:
    #             Asubdiag = self.getAsubdiag(tInd+1)

    #         for isrc, src in enumerate(self.survey.srcList):

    #             # solve against df_duT_v
    #             if tInd >= self.nT-1:
    #                 # last timestep (first to be solved)
    #                 ATinv_df_duT_v[isrc, :] = AdiagTinv * df_duT_v[
    #                     src, '{}Deriv'.format(self._fieldType), tInd+1]
    #             elif tInd > -1:
    #                 ATinv_df_duT_v[isrc, :] = AdiagTinv * (
    #                     mkvc(df_duT_v[
    #                         src, '{}Deriv'.format(self._fieldType), tInd+1
    #                     ]
    #                     ) - Asubdiag.T * mkvc(ATinv_df_duT_v[isrc, :]))

    #             dAsubdiagT_dm_v = self.getAsubdiagDeriv(
    #                 tInd, f[src, ftype, tInd], ATinv_df_duT_v[isrc, :],
    #                 adjoint=True)

    #             dRHST_dm_v = self.getRHSDeriv(
    #                     tInd+1, src, ATinv_df_duT_v[isrc, :], adjoint=True
    #                     )  # on nodes of time mesh

    #             un_src = f[src, ftype, tInd+1]
    #             # cell centered on time mesh
    #             dAT_dm_v = self.getAdiagDeriv(
    #                 tInd, un_src, ATinv_df_duT_v[isrc, :], adjoint=True
    #             )

    #             JTv = JTv +  mkvc(
    #                 -dAT_dm_v - dAsubdiagT_dm_v + dRHST_dm_v
    #             )

    #     # Treating initial condition when a galvanic source is included
    #     tInd = -1
    #     Grad = self.mesh.nodalGrad

    #     for isrc, src in enumerate(self.survey.srcList):
    #         if src.srcType == "galvanic":

    #             ATinv_df_duT_v[isrc, :] = Grad*(self.Adcinv*(Grad.T*(
    #                 mkvc(df_duT_v[
    #                     src, '{}Deriv'.format(self._fieldType), tInd+1
    #                 ]
    #                 ) - Asubdiag.T * mkvc(ATinv_df_duT_v[isrc, :]))
    #             ))

    #             dRHST_dm_v = self.getRHSDeriv(
    #                     tInd+1, src, ATinv_df_duT_v[isrc, :], adjoint=True
    #                     )  # on nodes of time mesh

    #             un_src = f[src, ftype, tInd+1]
    #             # cell centered on time mesh
    #             dAT_dm_v = (
    #                 self.MeSigmaDeriv(
    #                     un_src, ATinv_df_duT_v[isrc, :], adjoint=True
    #                 )
    #             )

    #             JTv = JTv +  mkvc(
    #                 -dAT_dm_v + dRHST_dm_v
    #             )

    #     # del df_duT_v, ATinv_df_duT_v, A, Asubdiag
    #     if AdiagTinv is not None:
    #         AdiagTinv.clean()

    #     return mkvc(JTv).astype(float)

    def getAdiag(self, tInd):
        """
        Diagonal of the system matrix at a given time index
        """
        assert tInd >= 0 and tInd < self.nT

        dt = self.timeSteps[tInd]
        C = self.mesh.edgeCurl
        MfMui = self.MfMui
        MeSigma = self.MeSigma

        return C.T * (MfMui * C) + 1./dt * MeSigma

    def getAdiagDeriv(self, tInd, u, v, adjoint=False):
        """
        Deriv of ADiag with respect to electrical conductivity
        """
        assert tInd >= 0 and tInd < self.nT

        dt = self.timeSteps[tInd]
        return 1./dt * self.MeSigmaDeriv(u, v, adjoint)

    def getAsubdiag(self, tInd):
        """
        Matrix below the diagonal
        """
        assert tInd >= 0 and tInd < self.nT

        dt = self.timeSteps[tInd]
        return - 1./dt * self.MeSigma

    def getAsubdiagDeriv(self, tInd, u, v, adjoint=False):
        """
        Derivative of the matrix below the diagonal with respect to electrical
        conductivity
        """
        dt = self.timeSteps[tInd]

        if adjoint:
            return - 1./dt * self.MeSigmaDeriv(u, v, adjoint)

        return - 1./dt * self.MeSigmaDeriv(u, v, adjoint)

    def getRHS(self, tInd):
        """
        right hand side
        """
        # Omit this: Note input was tInd+1
        # if tInd == len(self.timeSteps):
        #     tInd = tInd - 1

        # dt = self.timeSteps[tInd-1]
        # s_m, s_e = self.getSourceTerm(tInd)
        # _, s_en1 = self.getSourceTerm(tInd-1)

        # return (
        #     -1./dt * (s_e - s_en1) + self.mesh.edgeCurl.T * self.MfMui * s_m
        # )

        dt = self.timeSteps[tInd]
        source_list = self.survey.source_list
        rhs = np.zeros((self.mesh.nE, len(source_list)))

        for i, src in enumerate(self.survey.source_list):
            s_en = src.s_e(self, tInd+1)
            s_mn = src.s_m(self, tInd+1)

            rhs[:, i] = (
                rhs[:, i] + (-1/dt * s_en) +
                self.mesh.edgeCurl.T * self.MfMui * s_mn
            )

            if tInd == 0 and src.waveform.hasInitialFields:
                if src.srcType == "galvanic":
                    einitial = src.eInitial(self)
                    rhs[:, i] = rhs[:, i] + 1/dt * self.MeSigma * einitial

                elif src.srcType == "inductive":
                    binitial = src.bInitial(self)
                    rhs[:, i] = rhs[:, i] + 1/dt * self.mesh.edgeCurl.T * (self.MfMui * binitial)

            else:
                s_en1 = src.s_e(self, tInd)
                rhs[:, i] = rhs[:, i] + 1/dt * s_en1

        return rhs

        # rhs = self.mesh.edgeCurl.T * self.MfMui * sm - 1./dt * s_e # (-1/dt) * (s_e - s_e_n1): neg-value comes through brackets

        # # handle the case where we have initial fields
        # if tInd == 0:

        # else:
        #     _, s_e_n1 = self.getSourceTerm(tInd)
        #     rhs = rhs + 1/dt * s_e_n1


    def getRHSDeriv(self, tInd, src, v, adjoint=False, f=None):
        # TODO: this should not take the fields!

        dt = self.timeSteps[tInd]

        if tInd == 0 and src.waveform.hasInitialFields:
            if src.srcType == "galvanic":
                einitial = src.eInitial(self)

                if adjoint:
                    einitial_deriv = src.eInitialDeriv(
                        self, self.MeSigma.T * v, adjoint=True, f=f
                    )
                    return 1/dt * (
                        einitial_deriv +
                        self.MeSigmaDeriv(einitial, v=v, adjoint=True)
                    )

                # not adjoint
                einitial_deriv = src.eInitialDeriv(self, v, f=f)
                return 1/dt * (
                    self.MeSigma * einitial_deriv +
                    self.MeSigmaDeriv(einitial, v=v)
                )

            elif src.srcType == "inductive":
                if adjoint:
                    binitial_deriv = src.bInitialDeriv(
                        self, self.MfMui.T * (self.mesh.edgeCurl * v)
                    )
                    return 1/dt * binitial_deriv

                # not adjoint
                binitial_deriv = src.bInitialDeriv(self, v, f=f)
                return (
                    1/dt * self.mesh.edgeCurl.T * (self.MfMui * binitial_deriv)
                )
        # right now, we are assuming that s_e, s_m do not depend on the model.
        return Zero()

    def getAdc(self):
        MeSigma = self.MeSigma
        Grad = self.mesh.nodalGrad
        Adc = Grad.T * MeSigma * Grad
        # Handling Null space of A
        Adc[0, 0] = Adc[0, 0] + 1.
        return Adc

    def getAdcDeriv(self, u, v, adjoint=False):
        Grad = self.mesh.nodalGrad
        if not adjoint:
            return Grad.T*self.MeSigmaDeriv(-u, v, adjoint)
        elif adjoint:
            return self.MeSigmaDeriv(-u, Grad*v, adjoint)
        return Adc

    # def clean(self):
    #     """
    #     Clean factors
    #     """
    #     if self.Adcinv is not None:
    #         self.Adcinv.clean()


###############################################################################
#                                                                             #
#                                H-J Formulation                              #
#                                                                             #
###############################################################################

# ------------------------------- Problem3D_h ------------------------------- #

class Problem3D_h(BaseTDEMSimulation):
    """
    Solve the H-J formulation of Maxwell's equations for the magnetic field h.

    We start with Maxwell's equations in terms of the magnetic field and
    current density

    .. math::

        \\nabla \\times \\rho \\mathbf{j} + \\mu \\frac{\\partial h}{\\partial t} = \\mathbf{s_m} \\
        \\nabla \\times \\mathbf{h} - \\mathbf{j} = \\mathbf{s_e}


    and eliminate :math:`\\mathbf{j}` using

    .. math::

        \\mathbf{j} = \\nabla \\times \\mathbf{h} - \\mathbf{s_e}


    giving

    .. math::

        \\nabla \\times \\rho \\nabla \\times \\mathbf{h} + \\mu \\frac{\\partial h}{\\partial t} =  \\nabla \\times \\rho \\mathbf{s_e} + \\mathbf{s_m}


    """

    _fieldType = 'h'
    _formulation = 'HJ'
    fieldsPair = Fields3D_h  #: Fields object pair
    Fields_Derivs = Fields_Derivs_hj

    def getAdiag(self, tInd):
        """
        System matrix at a given time index

        """
        assert tInd >= 0 and tInd < self.nT

        dt = self.timeSteps[tInd]
        C = self.mesh.edgeCurl
        MfRho = self.MfRho
        MeMu = self.MeMu

        return C.T * ( MfRho * C ) + 1./dt * MeMu

    def getAdiagDeriv(self, tInd, u, v, adjoint=False):
        assert tInd >= 0 and tInd < self.nT

        dt = self.timeSteps[tInd]
        C = self.mesh.edgeCurl

        if adjoint:
            return  self.MfRhoDeriv(C * u, C * v, adjoint)

        return C.T * self.MfRhoDeriv(C * u, v, adjoint)

    def getAsubdiag(self, tInd):
        assert tInd >= 0 and tInd < self.nT

        dt = self.timeSteps[tInd]

        return - 1./dt * self.MeMu

    def getAsubdiagDeriv(self, tInd, u, v, adjoint=False):
        return Zero()

    def getRHS(self, tInd):

        C = self.mesh.edgeCurl
        MfRho = self.MfRho
        s_m, s_e = self.getSourceTerm(tInd)

        return C.T * (MfRho * s_e) + s_m

    def getRHSDeriv(self, tInd, src, v, adjoint=False):
        C = self.mesh.edgeCurl
        s_m, s_e = src.eval(self, self.times[tInd])

        if adjoint is True:
            return self.MfRhoDeriv(s_e, C * v, adjoint)
        # assumes no source derivs
        return C.T * self.MfRhoDeriv(s_e, v, adjoint)

    def getRHSDeriv(self, tInd, src, v, adjoint=False):
        return Zero()  # assumes no derivs on sources

    def getAdc(self):
        D = sdiag(self.mesh.vol) * self.mesh.faceDiv
        G = D.T
        MfRhoI = self.MfRhoI
        return D * MfRhoI * G

    def getAdcDeriv(self, u, v, adjoint=False):
        D = sdiag(self.mesh.vol) * self.mesh.faceDiv
        G = D.T

        if adjoint:
            # This is the same as
            #      self.MfRhoIDeriv(G * u, D.T * v, adjoint=True)
            return self.MfRhoIDeriv(G * u, G * v, adjoint=True)
        return D * self.MfRhoIDeriv(G * u, v)

# ------------------------------- Problem3D_j ------------------------------- #

class Problem3D_j(BaseTDEMSimulation):

    """
    Solve the H-J formulation for current density

    In this case, we eliminate :math:`\partial \mathbf{h} / \partial t` and
    solve for :math:`\mathbf{j}`

    """

    _fieldType = 'j'
    _formulation = 'HJ'
    fieldsPair = Fields3D_j  #: Fields object pair
    Fields_Derivs = Fields_Derivs_hj

    def getAdiag(self, tInd):
        """
        System matrix at a given time index

        """
        assert tInd >= 0 and tInd < self.nT

        dt = self.timeSteps[tInd]
        C = self.mesh.edgeCurl
        MfRho = self.MfRho
        MeMuI = self.MeMuI
        eye = sp.eye(self.mesh.nF)

        A = C * (MeMuI * (C.T * MfRho)) + 1./dt * eye

        if self._makeASymmetric:
            return MfRho.T * A

        return A

    def getAdiagDeriv(self, tInd, u, v, adjoint=False):
        assert tInd >= 0 and tInd < self.nT

        dt = self.timeSteps[tInd]
        C = self.mesh.edgeCurl
        MfRho = self.MfRho
        MeMuI = self.MeMuI

        if adjoint:
            if self._makeASymmetric:
                v = MfRho * v
            return self.MfRhoDeriv(u, C * (MeMuI.T * (C.T * v)), adjoint)

        ADeriv = C * (MeMuI * (C.T * self.MfRhoDeriv(u, v, adjoint)))
        if self._makeASymmetric:
            return MfRho.T * ADeriv
        return ADeriv

    def getAsubdiag(self, tInd):
        assert tInd >= 0 and tInd < self.nT
        eye = sp.eye(self.mesh.nF)

        dt = self.timeSteps[tInd]

        if self._makeASymmetric:
            return -1./dt * self.MfRho.T
        return -1./dt * eye

    def getAsubdiagDeriv(self, tInd, u, v, adjoint=False):
        return Zero()

    def getRHS(self, tInd):

        if tInd == len(self.timeSteps):
            tInd = tInd - 1

        C = self.mesh.edgeCurl
        MeMuI = self.MeMuI
        dt = self.timeSteps[tInd]
        s_m, s_e = self.getSourceTerm(tInd)
        _, s_en1 = self.getSourceTerm(tInd-1)

        rhs = -1./dt * (s_e - s_en1) + C * MeMuI * s_m
        if self._makeASymmetric:
            return self.MfRho.T * rhs
        return rhs

    def getRHSDeriv(self, tInd, src, v, adjoint=False):
        return Zero()  # assumes no derivs on sources

    def getAdc(self):
        D = sdiag(self.mesh.vol) * self.mesh.faceDiv
        G = D.T
        MfRhoI = self.MfRhoI
        return D * MfRhoI * G

    def getAdcDeriv(self, u, v, adjoint=False):
        D = sdiag(self.mesh.vol) * self.mesh.faceDiv
        G = D.T

        if adjoint:
            # This is the same as
            #      self.MfRhoIDeriv(G * u, D.T * v, adjoint=True)
            return self.MfRhoIDeriv(G * u, G * v, adjoint=True)
        return D * self.MfRhoIDeriv(G * u, v)


