import numpy as np
import scipy.sparse as sp

from ...data import Data
from ...simulation import BaseTimeSimulation
from ...utils import mkvc, sdiag, speye, Zero, validate_type, validate_float
from ..base import BaseEMSimulation
from .survey import Survey
from .fields import (
    Fields3DMagneticFluxDensity,
    Fields3DElectricField,
    Fields3DMagneticField,
    Fields3DCurrentDensity,
    FieldsDerivativesEB,
    FieldsDerivativesHJ,
)


class BaseTDEMSimulation(BaseTimeSimulation, BaseEMSimulation):
    """
    We start with the first order form of Maxwell's equations, eliminate and
    solve the second order form. For the time discretization, we use backward
    Euler.
    """

    def __init__(self, mesh, survey=None, dt_threshold=1e-8, **kwargs):
        super().__init__(mesh=mesh, survey=survey, **kwargs)
        self.dt_threshold = dt_threshold
        if self.muMap is not None:
            raise NotImplementedError(
                "Time domain EM simulations do not support magnetic permeability "
                "inversion, yet."
            )

    @property
    def survey(self):
        """The survey for the simulation
        Returns
        -------
        SimPEG.electromagnetics.time_domain.survey.Survey
        """
        if self._survey is None:
            raise AttributeError("Simulation must have a survey set")
        return self._survey

    @survey.setter
    def survey(self, value):
        if value is not None:
            value = validate_type("survey", value, Survey, cast=False)
        self._survey = value

    @property
    def dt_threshold(self):
        """The threshold used to determine if a previous matrix factor can be reused.

        If the difference in time steps falls below this threshold, the factored matrix
        is re-used.

        Returns
        -------
        float
        """
        return self._dt_threshold

    @dt_threshold.setter
    def dt_threshold(self, value):
        self._dt_threshold = validate_float("dt_threshold", value, min_val=0.0)

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
        :rtype: SimPEG.electromagnetics.time_domain.fields.FieldsTDEM
        :return f: fields object
        """

        self.model = m

        f = self.fieldsPair(self)

        # set initial fields
        f[:, self._fieldType + "Solution", 0] = self.getInitialFields()

        if self.verbose:
            print("{}\nCalculating fields(m)\n{}".format("*" * 50, "*" * 50))

        # timestep to solve forward
        Ainv = None
        for tInd, dt in enumerate(self.time_steps):
            # keep factors if dt is the same as previous step b/c A will be the
            # same
            if Ainv is not None and (
                tInd > 0 and abs(dt - self.time_steps[tInd - 1]) > self.dt_threshold
            ):
                Ainv.clean()
                Ainv = None

            if Ainv is None:
                A = self.getAdiag(tInd)
                if self.verbose:
                    print("Factoring...   (dt = {:e})".format(dt))
                Ainv = self.solver(A, **self.solver_opts)
                if self.verbose:
                    print("Done")

            rhs = self.getRHS(tInd + 1)  # this is on the nodes of the time mesh
            Asubdiag = self.getAsubdiag(tInd)

            if self.verbose:
                print("    Solving...   (tInd = {:d})".format(tInd + 1))

            # taking a step
            sol = Ainv * (rhs - Asubdiag * f[:, (self._fieldType + "Solution"), tInd])

            if self.verbose:
                print("    Done...")

            if sol.ndim == 1:
                sol.shape = (sol.size, 1)
            f[:, self._fieldType + "Solution", tInd + 1] = sol

        if self.verbose:
            print("{}\nDone calculating fields(m)\n{}".format("*" * 50, "*" * 50))

        # clean factors and return
        Ainv.clean()
        return f

    def Jvec(self, m, v, f=None):
        r"""
        Jvec computes the sensitivity times a vector

        .. math::
            \mathbf{J} \mathbf{v} =
                \frac{d\mathbf{P}}{d\mathbf{F}}
                \left(
                    \frac{d\mathbf{F}}{d\mathbf{u}} \frac{d\mathbf{u}}{d\mathbf{m}}
                    + \frac{\partial\mathbf{F}}{\partial\mathbf{m}}
                \right)
                \mathbf{v}

        where

        .. math::
            \mathbf{A} \frac{d\mathbf{u}}{d\mathbf{m}}
            + \frac{\partial \mathbf{A} (\mathbf{u}, \mathbf{m})}
            {\partial\mathbf{m}} =
            \frac{d \mathbf{RHS}}{d \mathbf{m}}

        """

        if f is None:
            f = self.fields(m)

        ftype = self._fieldType + "Solution"  # the thing we solved for
        self.model = m

        # mat to store previous time-step's solution deriv times a vector for
        # each source
        # size: nu x nSrc

        # this is a bit silly

        # if self._fieldType == 'b' or self._fieldType == 'j':
        #     ifields = np.zeros((self.mesh.n_faces, len(Srcs)))
        # elif self._fieldType == 'e' or self._fieldType == 'h':
        #     ifields = np.zeros((self.mesh.n_edges, len(Srcs)))

        # for i, src in enumerate(self.survey.source_list):
        dun_dm_v = np.hstack(
            [
                mkvc(self.getInitialFieldsDeriv(src, v, f=f), 2)
                for src in self.survey.source_list
            ]
        )
        # can over-write this at each timestep
        # store the field derivs we need to project to calc full deriv
        df_dm_v = self.Fields_Derivs(self)

        Adiaginv = None

        for tInd, dt in zip(range(self.nT), self.time_steps):
            # keep factors if dt is the same as previous step b/c A will be the
            # same
            if Adiaginv is not None and (tInd > 0 and dt != self.time_steps[tInd - 1]):
                Adiaginv.clean()
                Adiaginv = None

            if Adiaginv is None:
                A = self.getAdiag(tInd)
                Adiaginv = self.solver(A, **self.solver_opts)

            Asubdiag = self.getAsubdiag(tInd)

            for i, src in enumerate(self.survey.source_list):
                # here, we are lagging by a timestep, so filling in as we go
                for projField in set([rx.projField for rx in src.receiver_list]):
                    df_dmFun = getattr(f, "_%sDeriv" % projField, None)
                    # df_dm_v is dense, but we only need the times at
                    # (rx.P.T * ones > 0)
                    # This should be called rx.footprint

                    df_dm_v[src, "{}Deriv".format(projField), tInd] = df_dmFun(
                        tInd, src, dun_dm_v[:, i], v
                    )

                un_src = f[src, ftype, tInd + 1]

                # cell centered on time mesh
                dA_dm_v = self.getAdiagDeriv(tInd, un_src, v)
                # on nodes of time mesh
                dRHS_dm_v = self.getRHSDeriv(tInd + 1, src, v)

                dAsubdiag_dm_v = self.getAsubdiagDeriv(tInd, f[src, ftype, tInd], v)

                JRHS = dRHS_dm_v - dAsubdiag_dm_v - dA_dm_v

                # step in time and overwrite
                if tInd != len(self.time_steps + 1):
                    dun_dm_v[:, i] = Adiaginv * (JRHS - Asubdiag * dun_dm_v[:, i])

        Jv = []
        for src in self.survey.source_list:
            for rx in src.receiver_list:
                Jv.append(
                    rx.evalDeriv(
                        src,
                        self.mesh,
                        self.time_mesh,
                        f,
                        mkvc(df_dm_v[src, "%sDeriv" % rx.projField, :]),
                    )
                )
        Adiaginv.clean()
        # del df_dm_v, dun_dm_v, Asubdiag
        # return mkvc(Jv)
        return np.hstack(Jv)

    def Jtvec(self, m, v, f=None):
        r"""
        Jvec computes the adjoint of the sensitivity times a vector

        .. math::

            \mathbf{J}^\top \mathbf{v} =
                \left(
                    \frac{d\mathbf{u}}{d\mathbf{m}} ^ \top
                    \frac{d\mathbf{F}}{d\mathbf{u}} ^ \top
                    + \frac{\partial\mathbf{F}}{\partial\mathbf{m}} ^ \top
                \right)
                \frac{d\mathbf{P}}{d\mathbf{F}} ^ \top
                \mathbf{v}

        where

        .. math::

            \frac{d\mathbf{u}}{d\mathbf{m}} ^\top \mathbf{A}^\top  +
            \frac{d\mathbf{A}(\mathbf{u})}{d\mathbf{m}} ^ \top =
            \frac{d \mathbf{RHS}}{d \mathbf{m}} ^ \top
        """

        if f is None:
            f = self.fields(m)

        self.model = m
        ftype = self._fieldType + "Solution"  # the thing we solved for

        # Ensure v is a data object.
        if not isinstance(v, Data):
            v = Data(self.survey, v)

        df_duT_v = self.Fields_Derivs(self)

        # same size as fields at a single timestep
        ATinv_df_duT_v = np.zeros(
            (
                len(self.survey.source_list),
                len(f[self.survey.source_list[0], ftype, 0]),
            ),
            dtype=float,
        )
        JTv = np.zeros(m.shape, dtype=float)

        # Loop over sources and receivers to create a fields object:
        # PT_v, df_duT_v, df_dmT_v
        # initialize storage for PT_v (don't need to preserve over sources)
        PT_v = self.Fields_Derivs(self)
        for src in self.survey.source_list:
            # Looping over initializing field class is appending memory!
            # PT_v = Fields_Derivs(self.mesh) # initialize storage
            # #for PT_v (don't need to preserve over sources)
            # initialize size
            df_duT_v[src, "{}Deriv".format(self._fieldType), :] = np.zeros_like(
                f[src, self._fieldType, :]
            )

            for rx in src.receiver_list:
                PT_v[src, "{}Deriv".format(rx.projField), :] = rx.evalDeriv(
                    src, self.mesh, self.time_mesh, f, mkvc(v[src, rx]), adjoint=True
                )  # this is +=

                # PT_v = np.reshape(curPT_v,(len(curPT_v)/self.time_mesh.nN,
                # self.time_mesh.nN), order='F')
                df_duTFun = getattr(f, "_{}Deriv".format(rx.projField), None)

                for tInd in range(self.nT + 1):
                    cur = df_duTFun(
                        tInd,
                        src,
                        None,
                        mkvc(PT_v[src, "{}Deriv".format(rx.projField), tInd]),
                        adjoint=True,
                    )

                    df_duT_v[src, "{}Deriv".format(self._fieldType), tInd] = df_duT_v[
                        src, "{}Deriv".format(self._fieldType), tInd
                    ] + mkvc(cur[0], 2)
                    JTv = cur[1] + JTv

        del PT_v  # no longer need this

        AdiagTinv = None

        # Do the back-solve through time
        # if the previous timestep is the same: no need to refactor the matrix
        # for tInd, dt in zip(range(self.nT), self.time_steps):

        for tInd in reversed(range(self.nT)):
            # tInd = tIndP - 1
            if AdiagTinv is not None and (
                tInd <= self.nT and self.time_steps[tInd] != self.time_steps[tInd + 1]
            ):
                AdiagTinv.clean()
                AdiagTinv = None

            # refactor if we need to
            if AdiagTinv is None:  # and tInd > -1:
                Adiag = self.getAdiag(tInd)
                AdiagTinv = self.solver(Adiag.T.tocsr(), **self.solver_opts)

            if tInd < self.nT - 1:
                Asubdiag = self.getAsubdiag(tInd + 1)

            for isrc, src in enumerate(self.survey.source_list):
                # solve against df_duT_v
                if tInd >= self.nT - 1:
                    # last timestep (first to be solved)
                    ATinv_df_duT_v[isrc, :] = (
                        AdiagTinv
                        * df_duT_v[src, "{}Deriv".format(self._fieldType), tInd + 1]
                    )
                elif tInd > -1:
                    ATinv_df_duT_v[isrc, :] = AdiagTinv * (
                        mkvc(df_duT_v[src, "{}Deriv".format(self._fieldType), tInd + 1])
                        - Asubdiag.T * mkvc(ATinv_df_duT_v[isrc, :])
                    )

                dAsubdiagT_dm_v = self.getAsubdiagDeriv(
                    tInd, f[src, ftype, tInd], ATinv_df_duT_v[isrc, :], adjoint=True
                )

                dRHST_dm_v = self.getRHSDeriv(
                    tInd + 1, src, ATinv_df_duT_v[isrc, :], adjoint=True
                )  # on nodes of time mesh

                un_src = f[src, ftype, tInd + 1]
                # cell centered on time mesh
                dAT_dm_v = self.getAdiagDeriv(
                    tInd, un_src, ATinv_df_duT_v[isrc, :], adjoint=True
                )

                JTv = JTv + mkvc(-dAT_dm_v - dAsubdiagT_dm_v + dRHST_dm_v)

        # Treat the initial condition

        # del df_duT_v, ATinv_df_duT_v, A, Asubdiag
        if AdiagTinv is not None:
            AdiagTinv.clean()

        return mkvc(JTv).astype(float)

    def getSourceTerm(self, tInd):
        """
        Assemble the source term. This ensures that the RHS is a vector / array
        of the correct size
        """

        Srcs = self.survey.source_list

        if self._formulation == "EB":
            s_m = np.zeros((self.mesh.n_faces, len(Srcs)))
            s_e = np.zeros((self.mesh.n_edges, len(Srcs)))
        elif self._formulation == "HJ":
            s_m = np.zeros((self.mesh.n_edges, len(Srcs)))
            s_e = np.zeros((self.mesh.n_faces, len(Srcs)))

        for i, src in enumerate(Srcs):
            smi, sei = src.eval(self, self.times[tInd])
            s_m[:, i] = s_m[:, i] + smi
            s_e[:, i] = s_e[:, i] + sei

        return s_m, s_e

    def getInitialFields(self):
        """
        Ask the sources for initial fields
        """

        Srcs = self.survey.source_list

        if self._fieldType in ["b", "j"]:
            ifields = np.zeros((self.mesh.n_faces, len(Srcs)))
        elif self._fieldType in ["e", "h"]:
            ifields = np.zeros((self.mesh.n_edges, len(Srcs)))

        if self.verbose:
            print("Calculating Initial fields")

        for i, src in enumerate(Srcs):
            ifields[:, i] = ifields[:, i] + getattr(
                src, "{}Initial".format(self._fieldType), None
            )(self)

        return ifields

    def getInitialFieldsDeriv(self, src, v, adjoint=False, f=None):
        ifieldsDeriv = mkvc(
            getattr(src, "{}InitialDeriv".format(self._fieldType), None)(
                self, v, adjoint, f
            )
        )

        # take care of any utils.zero cases
        if adjoint is False:
            if self._fieldType in ["b", "j"]:
                ifieldsDeriv += np.zeros(self.mesh.n_faces)
            elif self._fieldType in ["e", "h"]:
                ifieldsDeriv += np.zeros(self.mesh.n_edges)

        elif adjoint is True:
            if self._fieldType in ["b", "j"]:
                ifieldsDeriv += np.zeros(self.mesh.n_faces)
            elif self._fieldType in ["e", "h"]:
                ifieldsDeriv[0] += np.zeros(self.mesh.n_edges)
            ifieldsDeriv[1] += np.zeros_like(self.model)  # take care of a  Zero() case

        return ifieldsDeriv

    # Store matrix factors if we need to solve the DC problem to get the
    # initial condition
    @property
    def Adcinv(self):
        if not hasattr(self, "getAdc"):
            raise NotImplementedError(
                "Support for galvanic sources has not been implemented for "
                "{}-formulation".format(self._fieldType)
            )
        if getattr(self, "_Adcinv", None) is None:
            if self.verbose:
                print("Factoring the system matrix for the DC problem")
            Adc = self.getAdc()
            self._Adcinv = self.solver(Adc)
        return self._Adcinv

    @property
    def clean_on_model_update(self):
        items = super().clean_on_model_update
        return items + ["_Adcinv"]  #: clear DC matrix factors on any model updates


###############################################################################
#                                                                             #
#                                E-B Formulation                              #
#                                                                             #
###############################################################################

# ------------------------------- Simulation3DMagneticFluxDensity ------------------------------- #


class Simulation3DMagneticFluxDensity(BaseTDEMSimulation):
    r"""
    Starting from the quasi-static E-B formulation of Maxwell's equations
    (semi-discretized)

    .. math::

        \mathbf{C} \mathbf{e} + \frac{\partial \mathbf{b}}{\partial t} =
        \mathbf{s_m} \\
        \mathbf{C}^{\top} \mathbf{M_{\mu^{-1}}^f} \mathbf{b} -
        \mathbf{M_{\sigma}^e} \mathbf{e} = \mathbf{s_e}


    where :math:`\mathbf{s_e}` is an integrated quantity, we eliminate
    :math:`\mathbf{e}` using

    .. math::

        \mathbf{e} = \mathbf{M_{\sigma}^e}^{-1} \mathbf{C}^{\top}
        \mathbf{M_{\mu^{-1}}^f} \mathbf{b} -
        \mathbf{M_{\sigma}^e}^{-1} \mathbf{s_e}


    to obtain a second order semi-discretized system in :math:`\mathbf{b}`

    .. math::

        \mathbf{C} \mathbf{M_{\sigma}^e}^{-1} \mathbf{C}^{\top}
        \mathbf{M_{\mu^{-1}}^f} \mathbf{b}  +
        \frac{\partial \mathbf{b}}{\partial t} =
        \mathbf{C} \mathbf{M_{\sigma}^e}^{-1} \mathbf{s_e} + \mathbf{s_m}


    and moving everything except the time derivative to the rhs gives

    .. math::
        \frac{\partial \mathbf{b}}{\partial t} =
        -\mathbf{C} \mathbf{M_{\sigma}^e}^{-1} \mathbf{C}^{\top}
        \mathbf{M_{\mu^{-1}}^f} \mathbf{b} +
        \mathbf{C} \mathbf{M_{\sigma}^e}^{-1} \mathbf{s_e} + \mathbf{s_m}

    For the time discretization, we use backward euler. To solve for the
    :math:`n+1` th time step, we have

    .. math::

        \frac{\mathbf{b}^{n+1} - \mathbf{b}^{n}}{\mathbf{dt}} =
        -\mathbf{C} \mathbf{M_{\sigma}^e}^{-1} \mathbf{C}^{\top}
        \mathbf{M_{\mu^{-1}}^f} \mathbf{b}^{n+1} +
        \mathbf{C} \mathbf{M_{\sigma}^e}^{-1} \mathbf{s_e}^{n+1} +
        \mathbf{s_m}^{n+1}


    re-arranging to put :math:`\mathbf{b}^{n+1}` on the left hand side gives

    .. math::

        (\mathbf{I} + \mathbf{dt} \mathbf{C} \mathbf{M_{\sigma}^e}^{-1}
         \mathbf{C}^{\top} \mathbf{M_{\mu^{-1}}^f}) \mathbf{b}^{n+1} =
         \mathbf{b}^{n} + \mathbf{dt}(\mathbf{C} \mathbf{M_{\sigma}^e}^{-1}
         \mathbf{s_e}^{n+1} + \mathbf{s_m}^{n+1})

    """

    _fieldType = "b"
    _formulation = "EB"
    fieldsPair = Fields3DMagneticFluxDensity  #: A SimPEG.EM.TDEM.Fields3DMagneticFluxDensity object
    Fields_Derivs = FieldsDerivativesEB

    def getAdiag(self, tInd):
        r"""
        System matrix at a given time index

        .. math::

            (\mathbf{I} + \mathbf{dt} \mathbf{C} \mathbf{M_{\sigma}^e}^{-1}
            \mathbf{C}^{\top} \mathbf{M_{\mu^{-1}}^f})

        """
        assert tInd >= 0 and tInd < self.nT

        dt = self.time_steps[tInd]
        C = self.mesh.edge_curl
        MeSigmaI = self.MeSigmaI
        MfMui = self.MfMui
        I = speye(self.mesh.n_faces)

        A = 1.0 / dt * I + (C * (MeSigmaI * (C.T.tocsr() * MfMui)))

        if self._makeASymmetric is True:
            return MfMui.T.tocsr() * A
        return A

    def getAdiagDeriv(self, tInd, u, v, adjoint=False):
        """
        Derivative of ADiag
        """
        C = self.mesh.edge_curl

        # def MeSigmaIDeriv(x):
        #     return self.MeSigmaIDeriv(x)

        MfMui = self.MfMui

        if adjoint:
            if self._makeASymmetric is True:
                v = MfMui * v
            return self.MeSigmaIDeriv(C.T * (MfMui * u), C.T * v, adjoint)

        ADeriv = C * (self.MeSigmaIDeriv(C.T * (MfMui * u), v, adjoint))

        if self._makeASymmetric is True:
            return MfMui.T * ADeriv
        return ADeriv

    def getAsubdiag(self, tInd):
        """
        Matrix below the diagonal
        """

        dt = self.time_steps[tInd]
        MfMui = self.MfMui
        Asubdiag = -1.0 / dt * sp.eye(self.mesh.n_faces)

        if self._makeASymmetric is True:
            return MfMui.T * Asubdiag

        return Asubdiag

    def getAsubdiagDeriv(self, tInd, u, v, adjoint=False):
        return Zero() * v

    def getRHS(self, tInd):
        """
        Assemble the RHS
        """
        C = self.mesh.edge_curl
        MeSigmaI = self.MeSigmaI
        MfMui = self.MfMui

        s_m, s_e = self.getSourceTerm(tInd)

        rhs = C * (MeSigmaI * s_e) + s_m
        if self._makeASymmetric is True:
            return MfMui.T * rhs
        return rhs

    def getRHSDeriv(self, tInd, src, v, adjoint=False):
        """
        Derivative of the RHS
        """

        C = self.mesh.edge_curl
        MeSigmaI = self.MeSigmaI

        _, s_e = src.eval(self, self.times[tInd])
        s_mDeriv, s_eDeriv = src.evalDeriv(self, self.times[tInd], adjoint=adjoint)

        if adjoint:
            if self._makeASymmetric is True:
                v = self.MfMui * v
            if isinstance(s_e, Zero):
                MeSigmaIDerivT_v = Zero()
            else:
                MeSigmaIDerivT_v = self.MeSigmaIDeriv(s_e, C.T * v, adjoint)

            RHSDeriv = MeSigmaIDerivT_v + s_eDeriv(MeSigmaI.T * (C.T * v)) + s_mDeriv(v)

            return RHSDeriv

        if isinstance(s_e, Zero):
            MeSigmaIDeriv_v = Zero()
        else:
            MeSigmaIDeriv_v = self.MeSigmaIDeriv(s_e, v, adjoint)

        RHSDeriv = C * MeSigmaIDeriv_v + C * MeSigmaI * s_eDeriv(v) + s_mDeriv(v)

        if self._makeASymmetric is True:
            return self.MfMui.T * RHSDeriv
        return RHSDeriv


# ------------------------------- Simulation3DElectricField ------------------------------- #
class Simulation3DElectricField(BaseTDEMSimulation):
    r"""
    Solve the EB-formulation of Maxwell's equations for the electric field, e.

    Starting with

    .. math::

        \nabla \times \mathbf{e} + \frac{\partial \mathbf{b}}{\partial t} = \mathbf{s_m} \
        \nabla \times \mu^{-1} \mathbf{b} - \sigma \mathbf{e} = \mathbf{s_e}


    we eliminate :math:`\frac{\partial b}{\partial t}` using

    .. math::

        \frac{\partial \mathbf{b}}{\partial t} = - \nabla \times \mathbf{e} + \mathbf{s_m}


    taking the time-derivative of Ampere's law, we see

    .. math::

        \frac{\partial}{\partial t}\left( \nabla \times \mu^{-1} \mathbf{b} - \sigma \mathbf{e} \right) = \frac{\partial \mathbf{s_e}}{\partial t} \
        \nabla \times \mu^{-1} \frac{\partial \mathbf{b}}{\partial t} - \sigma \frac{\partial\mathbf{e}}{\partial t} = \frac{\partial \mathbf{s_e}}{\partial t}


    which gives us

    .. math::

        \nabla \times \mu^{-1} \nabla \times \mathbf{e} + \sigma \frac{\partial\mathbf{e}}{\partial t} = \nabla \times \mu^{-1} \mathbf{s_m} + \frac{\partial \mathbf{s_e}}{\partial t}


    """

    _fieldType = "e"
    _formulation = "EB"
    fieldsPair = Fields3DElectricField  #: A Fields3DElectricField
    Fields_Derivs = FieldsDerivativesEB

    # @profile
    def Jtvec(self, m, v, f=None):
        """
        Jvec computes the adjoint of the sensitivity times a vector
        """

        if f is None:
            f = self.fields(m)

        self.model = m
        ftype = self._fieldType + "Solution"  # the thing we solved for

        # Ensure v is a data object.
        if not isinstance(v, Data):
            v = Data(self.survey, v)

        df_duT_v = self.Fields_Derivs(self)

        # same size as fields at a single timestep
        ATinv_df_duT_v = np.zeros(
            (
                len(self.survey.source_list),
                len(f[self.survey.source_list[0], ftype, 0]),
            ),
            dtype=float,
        )
        JTv = np.zeros(m.shape, dtype=float)

        # Loop over sources and receivers to create a fields object:
        # PT_v, df_duT_v, df_dmT_v
        # initialize storage for PT_v (don't need to preserve over sources)
        PT_v = self.Fields_Derivs(self)
        for src in self.survey.source_list:
            # Looping over initializing field class is appending memory!
            # PT_v = Fields_Derivs(self.mesh) # initialize storage
            # #for PT_v (don't need to preserve over sources)
            # initialize size
            df_duT_v[src, "{}Deriv".format(self._fieldType), :] = np.zeros_like(
                f[src, self._fieldType, :]
            )

            for rx in src.receiver_list:
                PT_v[src, "{}Deriv".format(rx.projField), :] = rx.evalDeriv(
                    src, self.mesh, self.time_mesh, f, mkvc(v[src, rx]), adjoint=True
                )
                # this is +=

                # PT_v = np.reshape(curPT_v,(len(curPT_v)/self.time_mesh.nN,
                # self.time_mesh.nN), order='F')
                df_duTFun = getattr(f, "_{}Deriv".format(rx.projField), None)

                for tInd in range(self.nT + 1):
                    cur = df_duTFun(
                        tInd,
                        src,
                        None,
                        mkvc(PT_v[src, "{}Deriv".format(rx.projField), tInd]),
                        adjoint=True,
                    )

                    df_duT_v[src, "{}Deriv".format(self._fieldType), tInd] = df_duT_v[
                        src, "{}Deriv".format(self._fieldType), tInd
                    ] + mkvc(cur[0], 2)
                    JTv = cur[1] + JTv

        # no longer need this
        del PT_v

        AdiagTinv = None

        # Do the back-solve through time
        # if the previous timestep is the same: no need to refactor the matrix
        # for tInd, dt in zip(range(self.nT), self.time_steps):

        for tInd in reversed(range(self.nT)):
            # tInd = tIndP - 1
            if AdiagTinv is not None and (
                tInd <= self.nT and self.time_steps[tInd] != self.time_steps[tInd + 1]
            ):
                AdiagTinv.clean()
                AdiagTinv = None

            # refactor if we need to
            if AdiagTinv is None:  # and tInd > -1:
                Adiag = self.getAdiag(tInd)
                AdiagTinv = self.solver(Adiag.T, **self.solver_opts)

            if tInd < self.nT - 1:
                Asubdiag = self.getAsubdiag(tInd + 1)

            for isrc, src in enumerate(self.survey.source_list):
                # solve against df_duT_v
                if tInd >= self.nT - 1:
                    # last timestep (first to be solved)
                    ATinv_df_duT_v[isrc, :] = (
                        AdiagTinv
                        * df_duT_v[src, "{}Deriv".format(self._fieldType), tInd + 1]
                    )
                elif tInd > -1:
                    ATinv_df_duT_v[isrc, :] = AdiagTinv * (
                        mkvc(df_duT_v[src, "{}Deriv".format(self._fieldType), tInd + 1])
                        - Asubdiag.T * mkvc(ATinv_df_duT_v[isrc, :])
                    )

                dAsubdiagT_dm_v = self.getAsubdiagDeriv(
                    tInd, f[src, ftype, tInd], ATinv_df_duT_v[isrc, :], adjoint=True
                )

                dRHST_dm_v = self.getRHSDeriv(
                    tInd + 1, src, ATinv_df_duT_v[isrc, :], adjoint=True
                )  # on nodes of time mesh

                un_src = f[src, ftype, tInd + 1]
                # cell centered on time mesh
                dAT_dm_v = self.getAdiagDeriv(
                    tInd, un_src, ATinv_df_duT_v[isrc, :], adjoint=True
                )

                JTv = JTv + mkvc(-dAT_dm_v - dAsubdiagT_dm_v + dRHST_dm_v)

        # Treating initial condition when a galvanic source is included
        tInd = -1
        Grad = self.mesh.nodal_gradient

        for isrc, src in enumerate(self.survey.source_list):
            if src.srcType == "galvanic":
                ATinv_df_duT_v[isrc, :] = Grad * (
                    self.Adcinv
                    * (
                        Grad.T
                        * (
                            mkvc(
                                df_duT_v[
                                    src, "{}Deriv".format(self._fieldType), tInd + 1
                                ]
                            )
                            - Asubdiag.T * mkvc(ATinv_df_duT_v[isrc, :])
                        )
                    )
                )

                dRHST_dm_v = self.getRHSDeriv(
                    tInd + 1, src, ATinv_df_duT_v[isrc, :], adjoint=True
                )  # on nodes of time mesh

                un_src = f[src, ftype, tInd + 1]
                # cell centered on time mesh
                dAT_dm_v = self.MeSigmaDeriv(
                    un_src, ATinv_df_duT_v[isrc, :], adjoint=True
                )

                JTv = JTv + mkvc(-dAT_dm_v + dRHST_dm_v)

        # del df_duT_v, ATinv_df_duT_v, A, Asubdiag
        if AdiagTinv is not None:
            AdiagTinv.clean()

        return mkvc(JTv).astype(float)

    def getAdiag(self, tInd):
        """
        Diagonal of the system matrix at a given time index
        """
        assert tInd >= 0 and tInd < self.nT

        dt = self.time_steps[tInd]
        C = self.mesh.edge_curl
        MfMui = self.MfMui
        MeSigma = self.MeSigma

        return C.T.tocsr() * (MfMui * C) + 1.0 / dt * MeSigma

    def getAdiagDeriv(self, tInd, u, v, adjoint=False):
        """
        Deriv of ADiag with respect to electrical conductivity
        """
        assert tInd >= 0 and tInd < self.nT

        dt = self.time_steps[tInd]
        # MeSigmaDeriv = self.MeSigmaDeriv(u)

        if adjoint:
            return 1.0 / dt * self.MeSigmaDeriv(u, v, adjoint)

        return 1.0 / dt * self.MeSigmaDeriv(u, v, adjoint)

    def getAsubdiag(self, tInd):
        """
        Matrix below the diagonal
        """
        assert tInd >= 0 and tInd < self.nT

        dt = self.time_steps[tInd]

        return -1.0 / dt * self.MeSigma

    def getAsubdiagDeriv(self, tInd, u, v, adjoint=False):
        """
        Derivative of the matrix below the diagonal with respect to electrical
        conductivity
        """
        dt = self.time_steps[tInd]

        if adjoint:
            return -1.0 / dt * self.MeSigmaDeriv(u, v, adjoint)

        return -1.0 / dt * self.MeSigmaDeriv(u, v, adjoint)

    def getRHS(self, tInd):
        """
        right hand side
        """
        # Omit this: Note input was tInd+1
        # if tInd == len(self.time_steps):
        #     tInd = tInd - 1

        dt = self.time_steps[tInd - 1]
        s_m, s_e = self.getSourceTerm(tInd)
        _, s_en1 = self.getSourceTerm(tInd - 1)

        return -1.0 / dt * (s_e - s_en1) + self.mesh.edge_curl.T * self.MfMui * s_m

    def getRHSDeriv(self, tInd, src, v, adjoint=False):
        # right now, we are assuming that s_e, s_m do not depend on the model.
        return Zero()

    def getAdc(self):
        MeSigma = self.MeSigma
        Grad = self.mesh.nodal_gradient
        Adc = Grad.T.tocsr() * MeSigma * Grad
        # Handling Null space of A
        Adc[0, 0] = Adc[0, 0] + 1.0
        return Adc

    def getAdcDeriv(self, u, v, adjoint=False):
        Grad = self.mesh.nodal_gradient
        if not adjoint:
            return Grad.T * self.MeSigmaDeriv(-u, v, adjoint)
        else:
            return self.MeSigmaDeriv(-u, Grad * v, adjoint)

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

# ------------------------------- Simulation3DMagneticField ------------------------------- #


class Simulation3DMagneticField(BaseTDEMSimulation):
    r"""
    Solve the H-J formulation of Maxwell's equations for the magnetic field h.

    We start with Maxwell's equations in terms of the magnetic field and
    current density

    .. math::

        \nabla \times \rho \mathbf{j} + \mu \frac{\partial h}{\partial t} = \mathbf{s_m} \
        \nabla \times \mathbf{h} - \mathbf{j} = \mathbf{s_e}


    and eliminate :math:`\mathbf{j}` using

    .. math::

        \mathbf{j} = \nabla \times \mathbf{h} - \mathbf{s_e}


    giving

    .. math::

        \nabla \times \rho \nabla \times \mathbf{h} + \mu \frac{\partial h}{\partial t}
            = \nabla \times \rho \mathbf{s_e} + \mathbf{s_m}


    """

    _fieldType = "h"
    _formulation = "HJ"
    fieldsPair = Fields3DMagneticField  #: Fields object pair
    Fields_Derivs = FieldsDerivativesHJ

    def getAdiag(self, tInd):
        """
        System matrix at a given time index

        """
        assert tInd >= 0 and tInd < self.nT

        dt = self.time_steps[tInd]
        C = self.mesh.edge_curl
        MfRho = self.MfRho
        MeMu = self.MeMu

        return C.T * (MfRho * C) + 1.0 / dt * MeMu

    def getAdiagDeriv(self, tInd, u, v, adjoint=False):
        assert tInd >= 0 and tInd < self.nT

        C = self.mesh.edge_curl

        if adjoint:
            return self.MfRhoDeriv(C * u, C * v, adjoint)

        return C.T * self.MfRhoDeriv(C * u, v, adjoint)

    def getAsubdiag(self, tInd):
        assert tInd >= 0 and tInd < self.nT

        dt = self.time_steps[tInd]

        return -1.0 / dt * self.MeMu

    def getAsubdiagDeriv(self, tInd, u, v, adjoint=False):
        return Zero()

    def getRHS(self, tInd):
        C = self.mesh.edge_curl
        MfRho = self.MfRho
        s_m, s_e = self.getSourceTerm(tInd)

        return C.T * (MfRho * s_e) + s_m

    def getRHSDeriv(self, tInd, src, v, adjoint=False):
        C = self.mesh.edge_curl
        s_m, s_e = src.eval(self, self.times[tInd])

        if adjoint is True:
            return self.MfRhoDeriv(s_e, C * v, adjoint)
        # assumes no source derivs
        return C.T * self.MfRhoDeriv(s_e, v, adjoint)

    def getAdc(self):
        D = sdiag(self.mesh.cell_volumes) * self.mesh.face_divergence
        G = D.T
        MfRhoI = self.MfRhoI
        return D * MfRhoI * G

    def getAdcDeriv(self, u, v, adjoint=False):
        D = sdiag(self.mesh.cell_volumes) * self.mesh.face_divergence
        G = D.T

        if adjoint:
            # This is the same as
            #      self.MfRhoIDeriv(G * u, D.T * v, adjoint=True)
            return self.MfRhoIDeriv(G * u, G * v, adjoint=True)
        return D * self.MfRhoIDeriv(G * u, v)


# ------------------------------- Simulation3DCurrentDensity ------------------------------- #


class Simulation3DCurrentDensity(BaseTDEMSimulation):
    r"""
    Solve the H-J formulation for current density

    In this case, we eliminate :math:`\partial \mathbf{h} / \partial t` and
    solve for :math:`\mathbf{j}`

    """

    _fieldType = "j"
    _formulation = "HJ"
    fieldsPair = Fields3DCurrentDensity  #: Fields object pair
    Fields_Derivs = FieldsDerivativesHJ

    def getAdiag(self, tInd):
        """
        System matrix at a given time index

        """
        assert tInd >= 0 and tInd < self.nT

        dt = self.time_steps[tInd]
        C = self.mesh.edge_curl
        MfRho = self.MfRho
        MeMuI = self.MeMuI
        eye = sp.eye(self.mesh.n_faces)

        A = C * (MeMuI * (C.T * MfRho)) + 1.0 / dt * eye

        if self._makeASymmetric:
            return MfRho.T * A

        return A

    def getAdiagDeriv(self, tInd, u, v, adjoint=False):
        assert tInd >= 0 and tInd < self.nT

        C = self.mesh.edge_curl
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
        eye = sp.eye(self.mesh.n_faces)

        dt = self.time_steps[tInd]

        if self._makeASymmetric:
            return -1.0 / dt * self.MfRho.T
        return -1.0 / dt * eye

    def getAsubdiagDeriv(self, tInd, u, v, adjoint=False):
        return Zero()

    def getRHS(self, tInd):
        if tInd == len(self.time_steps):
            tInd = tInd - 1

        C = self.mesh.edge_curl
        MeMuI = self.MeMuI
        dt = self.time_steps[tInd]
        s_m, s_e = self.getSourceTerm(tInd)
        _, s_en1 = self.getSourceTerm(tInd - 1)

        rhs = -1.0 / dt * (s_e - s_en1) + C * MeMuI * s_m
        if self._makeASymmetric:
            return self.MfRho.T * rhs
        return rhs

    def getRHSDeriv(self, tInd, src, v, adjoint=False):
        return Zero()  # assumes no derivs on sources

    def getAdc(self):
        D = sdiag(self.mesh.cell_volumes) * self.mesh.face_divergence
        G = D.T
        MfRhoI = self.MfRhoI
        return D * MfRhoI * G

    def getAdcDeriv(self, u, v, adjoint=False):
        D = sdiag(self.mesh.cell_volumes) * self.mesh.face_divergence
        G = D.T

        if adjoint:
            # This is the same as
            #      self.MfRhoIDeriv(G * u, D.T * v, adjoint=True)
            return self.MfRhoIDeriv(G * u, G * v, adjoint=True)
        return D * self.MfRhoIDeriv(G * u, v)
