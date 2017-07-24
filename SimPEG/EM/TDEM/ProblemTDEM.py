from __future__ import division, print_function
import scipy.sparse as sp
import numpy as np
from SimPEG import Problem, Utils, Solver as SimpegSolver
from SimPEG.EM.Base import BaseEMProblem
from SimPEG.EM.TDEM.SurveyTDEM import Survey as SurveyTDEM
from SimPEG.EM.TDEM.FieldsTDEM import (
    FieldsTDEM, Fields3D_b, Fields3D_e, Fields3D_h, Fields3D_j, Fields_Derivs
)
from scipy.constants import mu_0
import time


class BaseTDEMProblem(Problem.BaseTimeProblem, BaseEMProblem):
    """
    We start with the first order form of Maxwell's equations, eliminate and
    solve the second order form. For the time discretization, we use backward
    Euler.
    """
    surveyPair = SurveyTDEM  #: A SimPEG.EM.TDEM.SurveyTDEM Class
    fieldsPair = FieldsTDEM  #: A SimPEG.EM.TDEM.FieldsTDEM Class

    def __init__(self, mesh, **kwargs):
        BaseEMProblem.__init__(self, mesh, **kwargs)

    # def fields_nostore(self, m):
    #     """
    #     Solve the forward problem without storing fields

    #     :param numpy.array m: inversion model (nP,)
    #     :rtype: numpy.array
    #     :return numpy.array: numpy.array (nD,)

    #     """

    def fields(self, m):
        """
        Solve the forward problem for the fields.

        :param numpy.array m: inversion model (nP,)
        :rtype: SimPEG.EM.TDEM.FieldsTDEM
        :return F: fields object
        """

        tic = time.time()
        self.model = m

        F = self.fieldsPair(self.mesh, self.survey)

        # set initial fields
        F[:, self._fieldType+'Solution', 0] = self.getInitialFields()

        # timestep to solve forward
        if self.verbose:
            print('{}\nCalculating fields(m)\n{}'.format('*'*50, '*'*50))
        Ainv = None
        for tInd, dt in enumerate(self.timeSteps):
            # keep factors if dt is the same as previous step b/c A will be the
            # same
            if Ainv is not None and (
                tInd > 0 and dt != self.timeSteps[tInd - 1]
            ):
                Ainv.clean()
                Ainv = None

            if Ainv is None:
                A = self.getAdiag(tInd)
                if self.verbose:
                    print('Factoring...   (dt = {:e})'.format(dt))
                Ainv = self.Solver(A, **self.solverOpts)
                if self.verbose:
                    print('Done')

            rhs = self.getRHS(tInd+1)  # this is on the nodes of the time mesh
            Asubdiag = self.getAsubdiag(tInd)

            if self.verbose:
                print('    Solving...   (tInd = {:d})'.format(tInd+1))
            # taking a step
            sol = Ainv * (rhs - Asubdiag * F[:, (self._fieldType + 'Solution'),
                                             tInd])

            if self.verbose:
                print('    Done...')

            if sol.ndim == 1:
                sol.shape = (sol.size, 1)
            F[:, self._fieldType+'Solution', tInd+1] = sol
        if self.verbose:
            print('{}\nDone calculating fields(m)\n{}'.format('*'*50, '*'*50))
        Ainv.clean()
        return F

    def Jvec(self, m, v, f=None):
        """
        Jvec computes the sensitivity times a vector

        .. math::
            \mathbf{J} \mathbf{v} = \\frac{d\mathbf{P}}{d\mathbf{F}} \left(
            \\frac{d\mathbf{F}}{d\mathbf{u}} \\frac{d\mathbf{u}}{d\mathbf{m}} +
            \\frac{\partial\mathbf{F}}{\partial\mathbf{m}} \\right) \mathbf{v}

        where

        .. math::
            \mathbf{A} \\frac{d\mathbf{u}}{d\mathbf{m}} +
            \\frac{\partial \mathbf{A}(\mathbf{u}, \mathbf{m})}
            {\partial\mathbf{m}} =
            \\frac{d \mathbf{RHS}}{d \mathbf{m}}
        """

        if f is None:
            f = self.fields(m)

        ftype = self._fieldType + 'Solution'  # the thing we solved for
        self.model = m

        # mat to store previous time-step's solution deriv times a vector for
        # each source
        # size: nu x nSrc

        # this is a bit silly

        # if self._fieldType is 'b' or self._fieldType is 'j':
        #     ifields = np.zeros((self.mesh.nF, len(Srcs)))
        # elif self._fieldType is 'e' or self._fieldType is 'h':
        #     ifields = np.zeros((self.mesh.nE, len(Srcs)))

        # for i, src in enumerate(self.survey.srcList):
        dun_dm_v = np.hstack([
            Utils.mkvc(
                self.getInitialFieldsDeriv(src, v, f=f), 2
            )
            for src in self.survey.srcList
        ])
        # can over-write this at each timestep
        # store the field derivs we need to project to calc full deriv
        df_dm_v = Fields_Derivs(self.mesh, self.survey)

        Adiaginv = None

        for tInd, dt in zip(range(self.nT), self.timeSteps):
            # keep factors if dt is the same as previous step b/c A will be the
            # same
            if Adiaginv is not None and (tInd > 0 and dt !=
                                         self.timeSteps[tInd - 1]):
                Adiaginv.clean()
                Adiaginv = None

            if Adiaginv is None:
                A = self.getAdiag(tInd)
                Adiaginv = self.Solver(A, **self.solverOpts)

            Asubdiag = self.getAsubdiag(tInd)

            for i, src in enumerate(self.survey.srcList):

                # here, we are lagging by a timestep, so filling in as we go
                for projField in set([rx.projField for rx in src.rxList]):
                    df_dmFun = getattr(f, '_%sDeriv' % projField, None)
                    # df_dm_v is dense, but we only need the times at
                    # (rx.P.T * ones > 0)
                    # This should be called rx.footprint

                    df_dm_v[src, '{}Deriv'.format(projField), tInd] = df_dmFun(
                        tInd, src, dun_dm_v[:, i], v
                        )

                un_src = f[src, ftype, tInd+1]

                # cell centered on time mesh
                dA_dm_v = self.getAdiagDeriv(tInd, un_src, v)
                # on nodes of time mesh
                dRHS_dm_v = self.getRHSDeriv(tInd+1, src, v)

                dAsubdiag_dm_v = self.getAsubdiagDeriv(
                    tInd, f[src, ftype, tInd], v
                )

                JRHS = dRHS_dm_v - dAsubdiag_dm_v - dA_dm_v

                # step in time and overwrite
                if tInd != len(self.timeSteps+1):
                    dun_dm_v[:, i] = Adiaginv * (
                        JRHS - Asubdiag * dun_dm_v[:, i]
                    )

        Jv = []
        for src in self.survey.srcList:
            for rx in src.rxList:
                Jv.append(
                    rx.evalDeriv(src, self.mesh, self.timeMesh, f, Utils.mkvc(
                            df_dm_v[src, '%sDeriv' % rx.projField, :]
                        )
                    )
                )
        Adiaginv.clean()
        # del df_dm_v, dun_dm_v, Asubdiag
        # return Utils.mkvc(Jv)
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
        ftype = self._fieldType + 'Solution'  # the thing we solved for

        # Ensure v is a data object.
        if not isinstance(v, self.dataPair):
            v = self.dataPair(self.survey, v)

        df_duT_v = Fields_Derivs(self.mesh, self.survey)

        # same size as fields at a single timestep
        ATinv_df_duT_v = np.zeros(
            (
                len(self.survey.srcList),
                len(f[self.survey.srcList[0], ftype, 0])
            ),
            dtype=float
        )
        JTv = np.zeros(m.shape, dtype=float)

        # Loop over sources and receivers to create a fields object:
        # PT_v, df_duT_v, df_dmT_v
        # initialize storage for PT_v (don't need to preserve over sources)
        PT_v = Fields_Derivs(self.mesh, self.survey)
        for src in self.survey.srcList:
            # Looping over initializing field class is appending memory!
            # PT_v = Fields_Derivs(self.mesh, self.survey) # initialize storage
            # #for PT_v (don't need to preserve over sources)
            # initialize size
            df_duT_v[src, '{}Deriv'.format(self._fieldType), :] = (
                np.zeros_like(f[src, self._fieldType, :])
            )

            for rx in src.rxList:
                PT_v[src, '{}Deriv'.format(rx.projField), :] = rx.evalDeriv(
                    src, self.mesh, self.timeMesh, f, Utils.mkvc(v[src, rx]),
                    adjoint=True
                ) # this is +=

                # PT_v = np.reshape(curPT_v,(len(curPT_v)/self.timeMesh.nN,
                # self.timeMesh.nN), order='F')
                df_duTFun = getattr(f, '_{}Deriv'.format(rx.projField), None)

                for tInd in range(self.nT+1):
                    cur = df_duTFun(
                        tInd, src, None, Utils.mkvc(
                            PT_v[src, '{}Deriv'.format(rx.projField), tInd]
                        ),
                        adjoint=True
                    )

                    df_duT_v[src, '{}Deriv'.format(self._fieldType), tInd] = (
                        df_duT_v[src, '{}Deriv'.format(self._fieldType), tInd] +
                        Utils.mkvc(cur[0], 2))
                    JTv = cur[1] + JTv

        del PT_v # no longer need this

        AdiagTinv = None

        # Do the back-solve through time
        # if the previous timestep is the same: no need to refactor the matrix
        # for tInd, dt in zip(range(self.nT), self.timeSteps):

        for tInd in reversed(range(self.nT)):
            # tInd = tIndP - 1
            if AdiagTinv is not None and (
                tInd <= self.nT and
                self.timeSteps[tInd] != self.timeSteps[tInd+1]
            ):
                AdiagTinv.clean()
                AdiagTinv = None

            # refactor if we need to
            if AdiagTinv is None:  # and tInd > -1:
                Adiag = self.getAdiag(tInd)
                AdiagTinv = self.Solver(Adiag.T, **self.solverOpts)

            if tInd < self.nT - 1:
                Asubdiag = self.getAsubdiag(tInd+1)

            for isrc, src in enumerate(self.survey.srcList):

                # solve against df_duT_v
                if tInd >= self.nT-1:
                    # last timestep (first to be solved)
                    ATinv_df_duT_v[isrc, :] = AdiagTinv * df_duT_v[
                        src, '{}Deriv'.format(self._fieldType), tInd+1]
                elif tInd > -1:
                    ATinv_df_duT_v[isrc, :] = AdiagTinv * (
                        Utils.mkvc(df_duT_v[
                            src, '{}Deriv'.format(self._fieldType), tInd+1
                        ]
                        ) - Asubdiag.T * Utils.mkvc(ATinv_df_duT_v[isrc, :]))

                if tInd < self.nT:
                    dAsubdiagT_dm_v = self.getAsubdiagDeriv(
                        tInd, f[src, ftype, tInd], ATinv_df_duT_v[isrc, :],
                        adjoint=True)
                else:
                    dAsubdiagT_dm_v = Utils.Zero()

                dRHST_dm_v = self.getRHSDeriv(
                        tInd+1, src, ATinv_df_duT_v[isrc, :], adjoint=True
                        )  # on nodes of time mesh

                un_src = f[src, ftype, tInd+1]
                # cell centered on time mesh
                dAT_dm_v = self.getAdiagDeriv(
                    tInd, un_src, ATinv_df_duT_v[isrc, :], adjoint=True
                )

                JTv = JTv + Utils.mkvc(
                    -dAT_dm_v - dAsubdiagT_dm_v + dRHST_dm_v
                )

        # del df_duT_v, ATinv_df_duT_v, A, Asubdiag
        if AdiagTinv is not None:
            AdiagTinv.clean()

        return Utils.mkvc(JTv).astype(float)

    def getSourceTerm(self, tInd):
        """
        Assemble the source term. This ensures that the RHS is a vector / array
        of the correct size
        """

        Srcs = self.survey.srcList

        if self._formulation == 'EB':
            s_m = np.zeros((self.mesh.nF, len(Srcs)))
            s_e = np.zeros((self.mesh.nE, len(Srcs)))
        elif self._formulation == 'HJ':
            s_m = np.zeros((self.mesh.nE, len(Srcs)))
            s_e = np.zeros((self.mesh.nF, len(Srcs)))

        for i, src in enumerate(Srcs):
            smi, sei = src.eval(self, self.times[tInd])
            s_m[:, i] = s_m[:, i] + smi
            s_e[:, i] = s_e[:, i] + sei

        return s_m, s_e

    def getInitialFields(self):
        """
        Ask the sources for initial fields
        """

        Srcs = self.survey.srcList

        if self._fieldType in ['b', 'j']:
            ifields = np.zeros((self.mesh.nF, len(Srcs)))
        elif self._fieldType in ['e', 'h']:
            ifields = np.zeros((self.mesh.nE, len(Srcs)))

        for i, src in enumerate(Srcs):
            ifields[:, i] = (
                ifields[:, i] + getattr(
                    src, '{}Initial'.format(self._fieldType), None
                )(self)
            )

        return ifields

    def getInitialFieldsDeriv(self, src, v, adjoint=False, f=None):
        if adjoint is False:
            if self._fieldType in ['b', 'j']:
                ifieldsDeriv = np.zeros(self.mesh.nF)
            elif self._fieldType in ['e', 'h']:
                ifieldsDeriv = np.zeros(self.mesh.nE)

        elif adjoint is True:
            ifieldsDeriv = np.zeros(self.mapping.nP)

        ifieldsDeriv = (Utils.mkvc(
            getattr(src, '{}InitialDeriv'.format(self._fieldType),
                    None)(self, v, adjoint, f)) + ifieldsDeriv
            )
        return ifieldsDeriv

###############################################################################
#                                                                             #
#                                E-B Formulation                              #
#                                                                             #
###############################################################################

# ------------------------------- Problem3D_b ------------------------------- #

class Problem3D_b(BaseTDEMProblem):
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
    surveyPair = SurveyTDEM

    def __init__(self, mesh, **kwargs):
        BaseTDEMProblem.__init__(self, mesh, **kwargs)

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
        I = Utils.speye(self.mesh.nF)

        A = 1./dt * I + (C * (MeSigmaI * (C.T * MfMui)))

        if self._makeASymmetric is True:
            return MfMui.T * A
        return A

    def getAdiagDeriv(self, tInd, u, v, adjoint=False):
        """
        Derivative of ADiag
        """
        C = self.mesh.edgeCurl

        def MeSigmaIDeriv(x):
            return self.MeSigmaIDeriv(x)

        MfMui = self.MfMui

        if adjoint:
            if self._makeASymmetric is True:
                v = MfMui * v
            return MeSigmaIDeriv(C.T * (MfMui * u)).T * (C.T * v)

        ADeriv = (C * (MeSigmaIDeriv(C.T * (MfMui * u)) * v))

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
        return Utils.Zero() * v

    def getRHS(self, tInd):
        """
        Assemble the RHS
        """
        C = self.mesh.edgeCurl
        MeSigmaI = self.MeSigmaI
        MfMui = self.MfMui

        s_m, s_e = self.getSourceTerm(tInd)

        rhs = (C * (MeSigmaI * s_e) + s_m)
        if self._makeASymmetric is True:
            return MfMui.T * rhs
        return rhs

    def getRHSDeriv(self, tInd, src, v, adjoint=False):
        """
        Derivative of the RHS
        """

        C = self.mesh.edgeCurl
        MeSigmaI = self.MeSigmaI

        def MeSigmaIDeriv(u):
            return self.MeSigmaIDeriv(u)

        MfMui = self.MfMui

        _, s_e = src.eval(self, self.times[tInd])
        s_mDeriv, s_eDeriv = src.evalDeriv(
            self, self.times[tInd], adjoint=adjoint
        )

        if adjoint:
            if self._makeASymmetric is True:
                v = self.MfMui * v
            if isinstance(s_e, Utils.Zero):
                MeSigmaIDerivT_v = Utils.Zero()
            else:
                MeSigmaIDerivT_v = MeSigmaIDeriv(s_e).T * C.T * v

            RHSDeriv = (
                MeSigmaIDerivT_v + s_eDeriv( MeSigmaI.T * (C.T * v)) +
                s_mDeriv(v)
            )

            return RHSDeriv

        if isinstance(s_e, Utils.Zero):
            MeSigmaIDeriv_v = Utils.Zero()
        else:
            MeSigmaIDeriv_v = MeSigmaIDeriv(s_e) * v

        RHSDeriv = (
            C * MeSigmaIDeriv_v + C * MeSigmaI * s_eDeriv(v) + s_mDeriv(v)
        )

        if self._makeASymmetric is True:
            return self.MfMui.T * RHSDeriv
        return RHSDeriv


# ------------------------------- Problem3D_e ------------------------------- #
class Problem3D_e(BaseTDEMProblem):
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
    surveyPair = SurveyTDEM
    Adcinv = None

    def __init__(self, mesh, **kwargs):
        BaseTDEMProblem.__init__(self, mesh, **kwargs)

    def Jtvec(self, m, v, f=None):

        """
            Jvec computes the adjoint of the sensitivity times a vector

        """

        if f is None:
            f = self.fields(m)

        self.model = m
        ftype = self._fieldType + 'Solution'  # the thing we solved for

        # Ensure v is a data object.
        if not isinstance(v, self.dataPair):
            v = self.dataPair(self.survey, v)

        df_duT_v = Fields_Derivs(self.mesh, self.survey)

        # same size as fields at a single timestep
        ATinv_df_duT_v = np.zeros(
            (
                len(self.survey.srcList),
                len(f[self.survey.srcList[0], ftype, 0])
            ),
            dtype=float
        )
        JTv = np.zeros(m.shape, dtype=float)

        # Loop over sources and receivers to create a fields object:
        # PT_v, df_duT_v, df_dmT_v
        # initialize storage for PT_v (don't need to preserve over sources)
        PT_v = Fields_Derivs(self.mesh, self.survey)
        for src in self.survey.srcList:
            # Looping over initializing field class is appending memory!
            # PT_v = Fields_Derivs(self.mesh, self.survey) # initialize storage
            # #for PT_v (don't need to preserve over sources)
            # initialize size
            df_duT_v[src, '{}Deriv'.format(self._fieldType), :] = (
                np.zeros_like(f[src, self._fieldType, :])
            )

            for rx in src.rxList:
                PT_v[src, '{}Deriv'.format(rx.projField), :] = rx.evalDeriv(
                    src, self.mesh, self.timeMesh, f, Utils.mkvc(v[src, rx]),
                    adjoint=True
                )
                # this is +=

                # PT_v = np.reshape(curPT_v,(len(curPT_v)/self.timeMesh.nN,
                # self.timeMesh.nN), order='F')
                df_duTFun = getattr(f, '_{}Deriv'.format(rx.projField), None)

                for tInd in range(self.nT+1):
                    cur = df_duTFun(
                        tInd, src, None, Utils.mkvc(
                            PT_v[src, '{}Deriv'.format(rx.projField), tInd]
                        ),
                        adjoint=True
                    )

                    df_duT_v[src, '{}Deriv'.format(self._fieldType), tInd] = (
                        df_duT_v[src, '{}Deriv'.format(self._fieldType), tInd]
                        + Utils.mkvc(cur[0], 2)
                        )
                    JTv = cur[1] + JTv

        # no longer need this
        del PT_v

        AdiagTinv = None

        # Do the back-solve through time
        # if the previous timestep is the same: no need to refactor the matrix
        # for tInd, dt in zip(range(self.nT), self.timeSteps):

        for tInd in reversed(range(self.nT)):
            # tInd = tIndP - 1
            if AdiagTinv is not None and (
                tInd <= self.nT and
                self.timeSteps[tInd] != self.timeSteps[tInd+1]
            ):
                AdiagTinv.clean()
                AdiagTinv = None

            # refactor if we need to
            if AdiagTinv is None:  # and tInd > -1:
                Adiag = self.getAdiag(tInd)
                AdiagTinv = self.Solver(Adiag.T, **self.solverOpts)

            if tInd < self.nT - 1:
                Asubdiag = self.getAsubdiag(tInd+1)

            for isrc, src in enumerate(self.survey.srcList):

                # solve against df_duT_v
                if tInd >= self.nT-1:
                    # last timestep (first to be solved)
                    ATinv_df_duT_v[isrc, :] = AdiagTinv * df_duT_v[
                        src, '{}Deriv'.format(self._fieldType), tInd+1]
                elif tInd > -1:
                    ATinv_df_duT_v[isrc, :] = AdiagTinv * (
                        Utils.mkvc(df_duT_v[
                            src, '{}Deriv'.format(self._fieldType), tInd+1
                        ]
                        ) - Asubdiag.T * Utils.mkvc(ATinv_df_duT_v[isrc, :]))

                dAsubdiagT_dm_v = self.getAsubdiagDeriv(
                    tInd, f[src, ftype, tInd], ATinv_df_duT_v[isrc, :],
                    adjoint=True)

                dRHST_dm_v = self.getRHSDeriv(
                        tInd+1, src, ATinv_df_duT_v[isrc, :], adjoint=True
                        )  # on nodes of time mesh

                un_src = f[src, ftype, tInd+1]
                # cell centered on time mesh
                dAT_dm_v = self.getAdiagDeriv(
                    tInd, un_src, ATinv_df_duT_v[isrc, :], adjoint=True
                )

                JTv = JTv + Utils.mkvc(
                    -dAT_dm_v - dAsubdiagT_dm_v + dRHST_dm_v
                )

        # Treating initial condition when a galvanic source is included
        tInd = -1
        Grad = self.mesh.nodalGrad

        for isrc, src in enumerate(self.survey.srcList):
            if src.srcType == "Galvanic":

                ATinv_df_duT_v[isrc, :] = Grad*(self.Adcinv*(Grad.T*(
                    Utils.mkvc(df_duT_v[
                        src, '{}Deriv'.format(self._fieldType), tInd+1
                    ]
                    ) - Asubdiag.T * Utils.mkvc(ATinv_df_duT_v[isrc, :]))
                ))

                dRHST_dm_v = self.getRHSDeriv(
                        tInd+1, src, ATinv_df_duT_v[isrc, :], adjoint=True
                        )  # on nodes of time mesh

                un_src = f[src, ftype, tInd+1]
                # cell centered on time mesh
                dAT_dm_v = (
                    self.MeSigmaDeriv(un_src).T * ATinv_df_duT_v[isrc, :]
                    )

                JTv = JTv + Utils.mkvc(
                    -dAT_dm_v + dRHST_dm_v
                )

        # del df_duT_v, ATinv_df_duT_v, A, Asubdiag
        if AdiagTinv is not None:
            AdiagTinv.clean()

        return Utils.mkvc(JTv).astype(float)

    def getAdiag(self, tInd):
        """
        Diagonal of the system matrix at a given time index
        """
        assert tInd >= 0 and tInd < self.nT

        dt = self.timeSteps[tInd]
        C = self.mesh.edgeCurl
        MfMui = self.MfMui
        MeSigma = self.MeSigma

        return C.T * (MfMui * C)+1./dt * MeSigma

    def getAdiagDeriv(self, tInd, u, v, adjoint=False):
        """
        Deriv of ADiag with respect to electrical conductivity
        """
        assert tInd >= 0 and tInd < self.nT

        dt = self.timeSteps[tInd]
        MeSigmaDeriv = self.MeSigmaDeriv(u)

        if adjoint:
            return 1./dt * MeSigmaDeriv.T * v

        return 1./dt * MeSigmaDeriv * v

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
            return - 1./dt * self.MeSigmaDeriv(u).T * v

        return - 1./dt * self.MeSigmaDeriv(u) * v

    def getRHS(self, tInd):
        """
        right hand side
        """
        # Omit this: Note input was tInd+1
        # if tInd == len(self.timeSteps):
        #     tInd = tInd - 1

        dt = self.timeSteps[tInd-1]
        s_m, s_e = self.getSourceTerm(tInd)
        _, s_en1 = self.getSourceTerm(tInd-1)
        return (-1./dt * (s_e - s_en1) +
                self.mesh.edgeCurl.T * self.MfMui * s_m)

    def getRHSDeriv(self, tInd, src, v, adjoint=False):
        # right now, we are assuming that s_e, s_m do not depend on the model.
        return Utils.Zero()

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
            print ("Calculating Initial fields")

        for i, src in enumerate(Srcs):
            # Check if the source is grounded
            if src.srcType == "Galvanic" and src.waveform.hasInitialFields:
                # Check self.Adcinv and clean
                if self.Adcinv is not None:
                    self.Adcinv.clean()
                # Factorize Adc matrix
                if self.verbose:
                    print ("Factorize system matrix for DC problem")
                Adc = self.getAdc()
                self.Adcinv = self.Solver(Adc)

            ifields[:, i] = (
                ifields[:, i] + getattr(
                    src, '{}Initial'.format(self._fieldType), None
                )(self)
            )

        return ifields

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
            return Grad.T*(self.MeSigmaDeriv(-u)*v)
        elif adjoint:
            return self.MeSigmaDeriv(-u).T * (Grad*v)
        return Adc

    def clean(self):
        """
        Clean factors
        """
        if self.Adcinv is not None:
            self.Adcinv.clean()

###############################################################################
#                                                                             #
#                                H-J Formulation                              #
#                                                                             #
###############################################################################

# ------------------------------- Problem3D_h ------------------------------- #

class Problem3D_h(BaseTDEMProblem):
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
    surveyPair = SurveyTDEM

    def __init__(self, mesh, **kwargs):
        BaseTDEMProblem.__init__(self, mesh, **kwargs)

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
        MfRhoDeriv = self.MfRhoDeriv(C * u)

        if adjoint:
            return MfRhoDeriv.T * C * v

        return C.T * (MfRhoDeriv * v)

    def getAsubdiag(self, tInd):
        assert tInd >= 0 and tInd < self.nT

        dt = self.timeSteps[tInd]

        return - 1./dt * self.MeMu

    def getAsubdiagDeriv(self, tInd, u, v, adjoint=False):
        return Utils.Zero()

    def getRHS(self, tInd):

        C = self.mesh.edgeCurl
        MfRho = self.MfRho
        s_m, s_e = self.getSourceTerm(tInd)

        return C.T * (MfRho * s_e) + s_m

    def getRHSDeriv(self, tInd, src, v, adjoint=False):
        C = self.mesh.edgeCurl
        s_m, s_e = src.eval(self, self.times[tInd])
        MfRhoDeriv = self.MfRhoDeriv(s_e)

        if adjoint is True:
            return MfRhoDeriv.T * (C * v)
        # assumes no source derivs
        return C.T * (MfRhoDeriv * v)


# ------------------------------- Problem3D_j ------------------------------- #

class Problem3D_j(BaseTDEMProblem):

    """
    Solve the H-J formulation for current density

    In this case, we eliminate :math:`\partial \mathbf{h} / \partial t` and
    solve for :math:`\mathbf{j}`

    """

    _fieldType = 'j'
    _formulation = 'HJ'
    fieldsPair = Fields3D_j  #: Fields object pair
    surveyPair = SurveyTDEM  #: survey

    def __init__(self, mesh, **kwargs):
        BaseTDEMProblem.__init__(self, mesh, **kwargs)

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
        MfRhoDeriv = self.MfRhoDeriv(u)
        MeMuI = self.MeMuI

        if adjoint:
            if self._makeASymmetric:
                v = MfRho * v
            return MfRhoDeriv.T * (C * (MeMuI.T * (C.T * v)))

        ADeriv = C * (MeMuI * (C.T * MfRhoDeriv * v))
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
        return Utils.Zero()

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
        return Utils.Zero()  # assumes no derivs on sources

