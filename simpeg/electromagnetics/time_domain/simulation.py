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
    r"""Base class for quasi-static TDEM simulation with finite volume.

    This class is used to define properties and methods necessary for solving
    3D time-domain EM problems. In the quasi-static regime, we ignore electric
    displacement, and Maxwell's equations are expressed as:

    .. math::
        \begin{align}
        \nabla \times \vec{e} + \frac{\partial \vec{b}}{\partial t} &= -\frac{\partial \vec{s}_m}{\partial t} \\
        \nabla \times \vec{h} - \vec{j} &= \vec{s}_e
        \end{align}

    where the constitutive relations between fields and fluxes are given by:

    * :math:`\vec{j} = \sigma \vec{e}`
    * :math:`\vec{b} = \mu \vec{h}`

    and:

    * :math:`\vec{s}_m` represents a magnetic source term
    * :math:`\vec{s}_e` represents a current source term

    Child classes of ``BaseTDEMSimulation`` solve the above expression numerically
    for various cases using mimetic finite volume and backward Euler time discretization.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
        The mesh.
    survey : .time_domain.survey.Survey
        The time-domain EM survey.
    dt_threshold : float
        Threshold used when determining the unique time-step lengths.
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
        """The TDEM survey object.

        Returns
        -------
        .time_domain.survey.Survey
            The TDEM survey object.
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
        """Threshold used when determining the unique time-step lengths.

        The number of linear systems that must be factored to solve the forward
        problem is equal to the number of unique time-step lengths. *dt_threshold*
        effectively sets the round-off error when determining the unique time-step
        lengths used by the simulation.

        Returns
        -------
        float
            Threshold used when determining the unique time-step lengths.
        """
        return self._dt_threshold

    @dt_threshold.setter
    def dt_threshold(self, value):
        self._dt_threshold = validate_float("dt_threshold", value, min_val=0.0)

    def fields(self, m):
        """Compute and return the fields for the model provided.

        Parameters
        ----------
        m : (n_param,) numpy.ndarray
            The model.

        Returns
        -------
        .time_domain.fields.FieldsTDEM
            The TDEM fields object.
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
        r"""Compute the sensitivity matrix times a vector.

        Where :math:`\mathbf{d}` are the data, :math:`\mathbf{m}` are the model parameters,
        and the sensitivity matrix is defined as:

        .. math::
            \mathbf{J} = \dfrac{\partial \mathbf{d}}{\partial \mathbf{m}}

        this method computes and returns the matrix-vector product:

        .. math::
            \mathbf{J v}

        for a given vector :math:`v`.

        Parameters
        ----------
        m : (n_param,) numpy.ndarray
            The model parameters.
        v : (n_param,) numpy.ndarray
            The vector.
        f : .time_domain.fields.FieldsTDEM, optional
            Fields solved for all sources.

        Returns
        -------
        (n_data,) numpy.ndarray
            The sensitivity matrix times a vector.
        """

        if f is None:
            f = self.fields(m)

        ftype = self._fieldType + "Solution"  # the thing we solved for
        self.model = m

        # mat to store previous time-step's solution deriv times a vector for
        # each source
        # size: nu x n_sources

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
        r"""Compute the adjoint sensitivity matrix times a vector.

        Where :math:`\mathbf{d}` are the data, :math:`\mathbf{m}` are the model parameters,
        and the sensitivity matrix is defined as:

        .. math::
            \mathbf{J} = \dfrac{\partial \mathbf{d}}{\partial \mathbf{m}}

        this method computes and returns the matrix-vector product:

        .. math::
            \mathbf{J^T v}

        for a given vector :math:`v`.

        Parameters
        ----------
        m : (n_param,) numpy.ndarray
            The model parameters.
        v : (n_data,) numpy.ndarray
            The vector.
        f : .time_domain.fields.FieldsTDEM, optional
            Fields solved for all sources.

        Returns
        -------
        (n_param,) numpy.ndarray
            The adjoint sensitivity matrix times a vector.
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
        r"""Return the discrete source terms for the time index provided.

        This method computes and returns the discrete magnetic and electric source terms for
        all soundings at the time index provided. The exact shape and implementation of source
        terms when solving for the fields at each time-step is formulation dependent.

        For definitions of the discrete magnetic (:math:`\mathbf{s_m}`) and electric
        (:math:`\mathbf{s_e}`) source terms for each simulation, see the *Notes* sections
        of the docstrings for:

        * :class:`.time_domain.Simulation3DElectricField`
        * :class:`.time_domain.Simulation3DMagneticField`
        * :class:`.time_domain.Simulation3DCurrentDensity`
        * :class:`.time_domain.Simulation3DMagneticFluxDensity`

        Parameters
        ----------
        tInd : int
            The time index. Value between ``[0, n_steps]``.

        Returns
        -------
        s_m : numpy.ndarray
            The magnetic sources terms. (n_faces, n_sources) for EB-formulations. (n_edges, n_sources) for HJ-formulations.
        s_e : numpy.ndarray
            The electric sources terms. (n_edges, n_sources) for EB-formulations. (n_faces, n_sources) for HJ-formulations.
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
        """Returns the fields for all sources at the initial time.

        Returns
        -------
        (n_edges or n_faces, n_sources) numpy.ndarray
            The fields for all sources at the initial time.
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
        r"""Derivative of the initial fields with respect to the model for a given source.

        For a given source object `src`, let :math:`\mathbf{u_0}` represent the initial
        fields discretized to the mesh. Where :math:`\mathbf{m}` are the model parameters
        and :math:`\mathbf{v}` is a vector, this method computes and returns:

        .. math::
            \dfrac{\partial \mathbf{u_0}}{\partial \mathbf{m}} \, \mathbf{v}

        or the adjoint operation:

        .. math::
            \dfrac{\partial \mathbf{u_0}}{\partial \mathbf{m}}^T \, \mathbf{v}

        Parameters
        ----------
        src : .time_domain.sources.BaseTDEMSrc
            A TDEM source.
        v : numpy.ndarray
            A vector of appropriate dimension. When `adjoint` is ``False``, `v` is a
            (n_param,) numpy.ndarray. When `adjoint` is ``True``, `v` is a (n_edges or n_faces,)
            numpy.ndarray.
        adjoint : bool
            Whether to perform the adjoint operation.
        f : .time_domain.fields.BaseTDEMFields, optional
            The TDEM fields object.

        Returns
        -------
        numpy.ndarray
            Derivatives of the initial fields with respect to the model for
            a given source.
            (n_edges or n_faces,) numpy.ndarray when ``adjoint`` is ``False``.
            (n_param,) numpy.ndarray when ``adjoint`` is ``True``.
        """
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
        r"""Inverse of the factored system matrix for the DC resistivity problem.

        The solution to the DC resistivity problem is necessary at the initial time for
        galvanic sources whose currents are non-zero at the initial time.
        This property is used to compute and store the inverse of the factored linear system
        matrix for the DC resistivity problem given by:

        .. math::
            \mathbf{A_{dc}} \, \boldsymbol{\phi_0} = \mathbf{q_{dc}}

        where :math:`\mathbf{A_{dc}}` is the system matrix, :math:`\boldsymbol{\phi_0}` represents the
        discrete solution for the electric potential and :math:`\mathbf{q_{dc}}` is the discrete
        right-hand side. Electric fields are computed by applying a discrete gradient operator
        to the discrete electric potential solution.

        Returns
        -------
        pymatsolver.solvers.Base
            Inver of the factored systems matrix for the DC resistivity problem.

        Notes
        -----
        See the docstrings for :class:`.resistivity.BaseDCSimulation`,
        :class:`.resistivity.Simulation3DCellCentered` and
        :class:`.resistivity.Simulation3DNodal` to learn
        more about how the DC resistivity problem is solved.
        """
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
        """List of model-dependent attributes to clean upon model update.

        Some of the TDEM simulation's attributes are model-dependent. This property specifies
        the model-dependent attributes that much be cleared when the model is updated.

        Returns
        -------
        list of str
            List of the model-dependent attributes to clean upon model update.
        """
        items = super().clean_on_model_update
        return items + ["_Adcinv"]  #: clear DC matrix factors on any model updates


###############################################################################
#                                                                             #
#                                E-B Formulation                              #
#                                                                             #
###############################################################################

# ------------------------------- Simulation3DMagneticFluxDensity ------------------------------- #


class Simulation3DMagneticFluxDensity(BaseTDEMSimulation):
    r"""3D TDEM simulation in terms of the magnetic flux density.

    This simulation solves for the magnetic flux density at each time-step.
    In this formulation, the electric fields are defined on mesh edges and the
    magnetic flux density is defined on mesh faces; i.e. it is an EB formulation.
    See the *Notes* section for a comprehensive description of the formulation.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
        The mesh.
    survey : .time_domain.survey.Survey
        The time-domain EM survey.
    dt_threshold : float
        Threshold used when determining the unique time-step lengths.

    Notes
    -----
    Here, we start with the quasi-static approximation for Maxwell's equations by neglecting
    electric displacement:

    .. math::
        &\nabla \times \vec{e} + \frac{\partial \vec{b}}{\partial t} = - \frac{\partial \vec{s}_m}{\partial t} \\
        &\nabla \times \vec{h} - \vec{j} = \vec{s}_e

    where :math:`\vec{s}_e` is an electric source term that defines a source current density,
    and :math:`\vec{s}_m` magnetic source term that defines a source magnetic flux density.
    We define the constitutive relations for the electrical conductivity :math:`\sigma`
    and magnetic permeability :math:`\mu` as:

    .. math::
        \vec{j} &= \sigma \vec{e} \\
        \vec{h} &= \mu^{-1} \vec{b}

    We then take the inner products of all previous expressions with a vector test function :math:`\vec{u}`.
    Through vector calculus identities and the divergence theorem, we obtain:

    .. math::
        & \int_\Omega \vec{u} \cdot (\nabla \times \vec{e}) \, dv
        + \int_\Omega \vec{u} \cdot \frac{\partial \vec{b}}{\partial t} \, dv
        = - \int_\Omega \vec{u} \cdot \frac{\partial \vec{s}_m}{\partial t} \, dv \\
        & \int_\Omega (\nabla \times \vec{u}) \cdot \vec{h} \, dv
        - \oint_{\partial \Omega} \vec{u} \cdot (\vec{h} \times \hat{n}) \, da
        - \int_\Omega \vec{u} \cdot \vec{j} \, dv = \int_\Omega \vec{u} \cdot \vec{s}_e \, dv \\
        & \int_\Omega \vec{u} \cdot \vec{j} \, dv = \int_\Omega \vec{u} \cdot \sigma \vec{e} \, dv \\
        & \int_\Omega \vec{u} \cdot \vec{h} \, dv = \int_\Omega \vec{u} \cdot \mu^{-1} \vec{b} \, dv

    Assuming natural boundary conditions, the surface integral is zero.

    The above expressions are discretized in space according to the finite volume method.
    The discrete electric fields :math:`\mathbf{e}` are defined on mesh edges,
    and the discrete magnetic flux densities :math:`\mathbf{b}` are defined on mesh faces.
    This implies :math:`\mathbf{j}` must be defined on mesh edges and :math:`\mathbf{h}` must
    be defined on mesh faces. Where :math:`\mathbf{u_e}` and :math:`\mathbf{u_f}` represent
    test functions discretized to edges and faces, respectively, we obtain the following
    set of discrete inner-products:

    .. math::
        &\mathbf{u_f^T C e} + \mathbf{u_f^T } \, \frac{\partial \mathbf{b}}{\partial t}
        = - \mathbf{u_f^T } \, \frac{\partial \mathbf{s_m}}{\partial t} \\
        &\mathbf{u_e^T C^T M_f h} - \mathbf{u_e^T M_e j} = \mathbf{u_e^T s_e} \\
        &\mathbf{u_e^T M_e j} = \mathbf{u_e^T M_{e\sigma} e} \\
        &\mathbf{u_f^T M_f h} = \mathbf{u_f^T M_{f \frac{1}{\mu}} b}

    where

    * :math:`\mathbf{C}` is the discrete curl operator
    * :math:`\mathbf{s_m}` and :math:`\mathbf{s_e}` are the integrated magnetic and electric source terms, respectively
    * :math:`\mathbf{M_e}` is the edge inner-product matrix
    * :math:`\mathbf{M_f}` is the face inner-product matrix
    * :math:`\mathbf{M_{e\sigma}}` is the inner-product matrix for conductivities projected to edges
    * :math:`\mathbf{M_{f\frac{1}{\mu}}}` is the inner-product matrix for inverse permeabilities projected to faces

    By cancelling like-terms and combining the discrete expressions in terms of the magnetic flux density, we obtain:

    .. math::
        \mathbf{C M_{e\sigma}^{-1} C^T M_{f\frac{1}{\mu}} b}
        + \frac{\partial \mathbf{b}}{\partial t}
        = \mathbf{C M_{e\sigma}^{-1}} \mathbf{s_e}
        - \frac{\partial \mathbf{s_m}}{\partial t}

    Finally, we discretize in time according to backward Euler. The discrete magnetic flux density
    on mesh faces at time :math:`t_k > t_0` is obtained by solving the following at each time-step:

    .. math::
        \mathbf{A}_k \mathbf{b}_k = \mathbf{q_k} - \mathbf{B}_k \mathbf{b}_{k-1}

    where :math:`\Delta t_k = t_k - t_{k-1}` and

    .. math::
        &\mathbf{A}_k = \mathbf{C M_{e\sigma}^{-1} C^T M_{f\frac{1}{\mu}}} + \frac{1}{\Delta t_k} \mathbf{I} \\
        &\mathbf{B}_k = -\frac{1}{\Delta t_k} \mathbf{I}\\
        &\mathbf{q}_k = \mathbf{C M_{e\sigma}^{-1}} \mathbf{s}_{\mathbf{e}, k} \;
        - \; \frac{1}{\Delta t_k} \big [ \mathbf{s}_{\mathbf{m}, k} - \mathbf{s}_{\mathbf{m}, k-1} \big ]

    Although the following system is never explicitly formed, we can represent
    the solution at all time-steps as:

    .. math::
        \begin{bmatrix}
        \mathbf{A_1} & & & & \\
        \mathbf{B_2} & \mathbf{A_2} & & & \\
        & & \ddots & & \\
        & & & \mathbf{B_n} & \mathbf{A_n}
        \end{bmatrix}
        \begin{bmatrix}
        \mathbf{b_1} \\ \mathbf{b_2} \\ \vdots \\ \mathbf{b_n}
        \end{bmatrix} =
        \begin{bmatrix}
        \mathbf{q_1} \\ \mathbf{q_2} \\ \vdots \\ \mathbf{q_n}
        \end{bmatrix} -
        \begin{bmatrix}
        \mathbf{B_1 b_0} \\ \mathbf{0} \\ \vdots \\ \mathbf{0}
        \end{bmatrix}

    where the magnetic flux densities at the initial time :math:`\mathbf{b_0}`
    are computed analytically or numerically depending on whether the source
    carries non-zero current at the initial time.

    """

    _fieldType = "b"
    _formulation = "EB"
    fieldsPair = Fields3DMagneticFluxDensity
    Fields_Derivs = FieldsDerivativesEB

    def getAdiag(self, tInd):
        r"""Diagonal system matrix for the given time-step index.

        This method returns the diagonal system matrix for the time-step index provided:

        .. math::
            \mathbf{A}_k = \mathbf{C M_{e\sigma}^{-1} C^T M_{f\frac{1}{\mu}}} + \frac{1}{\Delta t_k} \mathbf{I}

        where

        * :math:`\Delta t_k` is the step length
        * :math:`\mathbf{I}` is the identity matrix
        * :math:`\mathbf{C}` is the discrete curl operator
        * :math:`\mathbf{M_{e \sigma}}` is the conductivity inner-product matrix on edges
        * :math:`\mathbf{M_{f\frac{1}{\mu}}}` is the inverse permeability inner-product matrix on faces

        See the *Notes* section of the doc strings for :class:`Simulation3DMagneticFluxDensity`
        for a full description of the formulation.

        Parameters
        ----------
        tInd : int
            The time-step index; between ``[0, n_steps-1]``.

        Returns
        -------
        (n_faces, n_faces) sp.sparse.csr_matrix
            The diagonal system matrix.
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
        r"""Derivative operation for the diagonal system matrix times a vector.

        The diagonal system matrix for time-step index *k* is given by:

        .. math::
            \mathbf{A}_k = \mathbf{C M_{e\sigma}^{-1} C^T M_{f\frac{1}{\mu}}} + \frac{1}{\Delta t_k} \mathbf{I}

        where

        * :math:`\Delta t_k` is the step length
        * :math:`\mathbf{I}` is the identity matrix
        * :math:`\mathbf{C}` is the discrete curl operator
        * :math:`\mathbf{M_{e \sigma}}` is the conductivity inner-product matrix on edges
        * :math:`\mathbf{M_{f\frac{1}{\mu}}}` is the inverse permeability inner-product matrix on faces

        See the *Notes* section of the doc strings for :class:`Simulation3DMagneticFluxDensity`
        for a full description of the formulation.

        Where :math:`\mathbf{m}` are the set of model parameters, :math:`\mathbf{v}` is a vector
        and :math:`\mathbf{b_k}` is the discrete solution for time-step *k*, this method assumes
        the discrete solution is fixed and returns

        .. math::
            \frac{\partial (\mathbf{A_k \, b_k})}{\partial \mathbf{m}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial (\mathbf{A_k \, b_k})}{\partial \mathbf{m}}^T \, \mathbf{v}

        Parameters
        ----------
        tInd : int
            The time-step index; between ``[0, n_steps-1]``.
        u : (n_faces,) numpy.ndarray
            The solution for the fields for the current model; i.e. :math:`\mathbf{b_k}`.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_faces,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of system matrix times a vector. (n_faces,) for the standard operation.
            (n_param,) for the adjoint operation.
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
        r"""Sub-diagonal system matrix for the time-step index provided.

        This method returns the sub-diagonal system matrix for the time-step index provided:

        .. math::
            \mathbf{B}_k = -\frac{1}{\Delta t_k} \mathbf{I}

        where :math:`\Delta t_k` is the step length and :math:`\mathbf{I}` is the identity matrix.

        See the *Notes* section of the doc strings for :class:`Simulation3DMagneticFluxDensity`
        for a full description of the formulation.

        Parameters
        ----------
        tInd : int
            The time index; between ``[0, n_steps-1]``.

        Returns
        -------
        (n_faces, n_faces) sp.sparse.csr_matrix
            The sub-diagonal system matrix.
        """

        dt = self.time_steps[tInd]
        MfMui = self.MfMui
        Asubdiag = -1.0 / dt * sp.eye(self.mesh.n_faces)

        if self._makeASymmetric is True:
            return MfMui.T * Asubdiag

        return Asubdiag

    def getAsubdiagDeriv(self, tInd, u, v, adjoint=False):
        r"""Derivative operation for the sub-diagonal system matrix times a vector.

        The sub-diagonal system matrix for time-step index *k* is given by:

        .. math::
            \mathbf{B}_k = -\frac{1}{\Delta t_k} \mathbf{I}

        where :math:`\Delta t_k` is the step length and :math:`\mathbf{I}` is the identity matrix.

        See the *Notes* section of the doc strings for :class:`Simulation3DMagneticFluxDensity`
        for a full description of the formulation.

        Where :math:`\mathbf{m}` are the set of model parameters, :math:`\mathbf{v}` is a vector
        and :math:`\mathbf{b_{k-1}}` is the discrete solution for the previous time-step,
        this method assumes the discrete solution is fixed and returns

        .. math::
            \frac{\partial (\mathbf{B_k \, b_{k-1}})}{\partial \mathbf{m}} \, \mathbf{v} = \mathbf{0}

        Or the adjoint operation

        .. math::
            \frac{\partial (\mathbf{B_k \, b_{k-1}})}{\partial \mathbf{m}}^T \, \mathbf{v} = \mathbf{0}

        The derivative operation returns a vector of zeros because the sub-diagonal system matrix
        does not depend on the model!!!

        Parameters
        ----------
        tInd : int
            The time index; between ``[0, n_steps-1]``.
        u : (n_faces,) numpy.ndarray
            The solution for the fields for the current model for the previous time-step;
            i.e. :math:`\mathbf{b_{k-1}}`.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_faces,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of system matrix times a vector. (n_faces,) for the standard operation.
            (n_param,) for the adjoint operation.
        """
        return Zero() * v

    def getRHS(self, tInd):
        r"""Right-hand sides for the given time index.

        This method returns the right-hand sides for the time index provided.
        The right-hand side for each source is constructed according to:

        .. math::
            \mathbf{q}_k = \mathbf{C M_{e\sigma}^{-1}} \mathbf{s}_{\mathbf{e}, k} \;
            - \; \frac{1}{\Delta t_k} \big [ \mathbf{s}_{\mathbf{m}, k} - \mathbf{s}_{\mathbf{m}, k-1} \big ]

        where

        * :math:`\Delta t_k` is the step length
        * :math:`\mathbf{C}` is the discrete curl operator
        * :math:`\mathbf{s_m}` and :math:`\mathbf{s_e}` are the integrated magnetic and electric source terms, respectively
        * :math:`\mathbf{M_{e\sigma}}` is the inner-product matrix for conductivities projected to edges

        See the *Notes* section of the doc strings for :class:`Simulation3DMagneticFluxDensity`
        for a full description of the formulation.

        Parameters
        ----------
        tInd : int
            The time index; between ``[0, n_steps]``.

        Returns
        -------
        (n_faces, n_sources) numpy.ndarray
            The right-hand sides.
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
        r"""Derivative of the right-hand side times a vector for a given source and time index.

        The right-hand side for a given source at time index *k* is constructed according to:

        .. math::
            \mathbf{q}_k = \mathbf{C M_{e\sigma}^{-1}} \mathbf{s}_{\mathbf{e}, k} \;
            - \; \frac{1}{\Delta t_k} \big [ \mathbf{s}_{\mathbf{m}, k} - \mathbf{s}_{\mathbf{m}, k-1} \big ]

        where

        * :math:`\Delta t_k` is the step length
        * :math:`\mathbf{C}` is the discrete curl operator
        * :math:`\mathbf{s_m}` and :math:`\mathbf{s_e}` are the integrated magnetic and electric source terms, respectively
        * :math:`\mathbf{M_{e\sigma}}` is the inner-product matrix for conductivities projected to edges

         See the *Notes* section of the doc strings for :class:`Simulation3DMagneticFluxDensity`
        for a full description of the formulation.

        Where :math:`\mathbf{m}` are the set of model parameters and :math:`\mathbf{v}` is a vector,
        this method returns

        .. math::
            \frac{\partial \mathbf{q_k}}{\partial \mathbf{m}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial \mathbf{q_k}}{\partial \mathbf{m}}^T \, \mathbf{v}

        Parameters
        ----------
        tInd : int
            The time index; between ``[0, n_steps]``.
        src : .time_domain.sources.BaseTDEMSrc
            The TDEM source object.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_faces,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of the right-hand sides times a vector. (n_faces,) for the standard operation.
            (n_param,) for the adjoint operation.
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
    r"""3D TDEM simulation in terms of the electric field.

    This simulation solves for the electric field at each time-step.
    In this formulation, the electric fields are defined on mesh edges and the
    magnetic flux density is defined on mesh faces; i.e. it is an EB formulation.
    See the *Notes* section for a comprehensive description of the formulation.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
        The mesh.
    survey : .time_domain.survey.Survey
        The time-domain EM survey.
    dt_threshold : float
        Threshold used when determining the unique time-step lengths.

    Notes
    -----
    Here, we start with the quasi-static approximation for Maxwell's equations by neglecting
    electric displacement:

    .. math::
        &\nabla \times \vec{e} + \frac{\partial \vec{b}}{\partial t} = - \frac{\partial \vec{s}_m}{\partial t} \\
        &\nabla \times \vec{h} - \vec{j} = \vec{s}_e

    where :math:`\vec{s}_e` is an electric source term that defines a source current density,
    and :math:`\vec{s}_m` magnetic source term that defines a source magnetic flux density.
    We define the constitutive relations for the electrical conductivity :math:`\sigma`
    and magnetic permeability :math:`\mu` as:

    .. math::
        \vec{j} &= \sigma \vec{e} \\
        \vec{h} &= \mu^{-1} \vec{b}

    We then take the inner products of all previous expressions with a vector test function :math:`\vec{u}`.
    Through vector calculus identities and the divergence theorem, we obtain:

    .. math::
        & \int_\Omega \vec{u} \cdot (\nabla \times \vec{e}) \, dv
        + \int_\Omega \vec{u} \cdot \frac{\partial \vec{b}}{\partial t} \, dv
        = - \int_\Omega \vec{u} \cdot \frac{\partial \vec{s}_m}{\partial t} \, dv \\
        & \int_\Omega (\nabla \times \vec{u}) \cdot \vec{h} \, dv
        - \oint_{\partial \Omega} \vec{u} \cdot (\vec{h} \times \hat{n}) \, da
        - \int_\Omega \vec{u} \cdot \vec{j} \, dv = \int_\Omega \vec{u} \cdot \vec{s}_e \, dv \\
        & \int_\Omega \vec{u} \cdot \vec{j} \, dv = \int_\Omega \vec{u} \cdot \sigma \vec{e} \, dv \\
        & \int_\Omega \vec{u} \cdot \vec{h} \, dv = \int_\Omega \vec{u} \cdot \mu^{-1} \vec{b} \, dv

    Assuming natural boundary conditions, the surface integral is zero.

    The above expressions are discretized in space according to the finite volume method.
    The discrete electric fields :math:`\mathbf{e}` are defined on mesh edges,
    and the discrete magnetic flux densities :math:`\mathbf{b}` are defined on mesh faces.
    This implies :math:`\mathbf{j}` must be defined on mesh edges and :math:`\mathbf{h}` must
    be defined on mesh faces. Where :math:`\mathbf{u_e}` and :math:`\mathbf{u_f}` represent
    test functions discretized to edges and faces, respectively, we obtain the following
    set of discrete inner-products:

    .. math::
        &\mathbf{u_f^T C e} + \mathbf{u_f^T } \, \frac{\partial \mathbf{b}}{\partial t}
        = - \mathbf{u_f^T } \, \frac{\partial \mathbf{s_m}}{\partial t} \\
        &\mathbf{u_e^T C^T M_f h} - \mathbf{u_e^T M_e j} = \mathbf{u_e^T s_e} \\
        &\mathbf{u_e^T M_e j} = \mathbf{u_e^T M_{e\sigma} e} \\
        &\mathbf{u_f^T M_f h} = \mathbf{u_f^T M_{f \frac{1}{\mu}} b}

    where

    * :math:`\mathbf{C}` is the discrete curl operator
    * :math:`\mathbf{s_m}` and :math:`\mathbf{s_e}` are the integrated magnetic and electric source terms, respectively
    * :math:`\mathbf{M_e}` is the edge inner-product matrix
    * :math:`\mathbf{M_f}` is the face inner-product matrix
    * :math:`\mathbf{M_{e\sigma}}` is the inner-product matrix for conductivities projected to edges
    * :math:`\mathbf{M_{f\frac{1}{\mu}}}` is the inner-product matrix for inverse permeabilities projected to faces

    By cancelling like-terms and combining the discrete expressions in terms of the electric field, we obtain:

    .. math::
        \mathbf{C^T M_{f\frac{1}{\mu}} C e} + \mathbf{M_{e\sigma}}\frac{\partial \mathbf{e}}{\partial t}
        = \mathbf{C^T M_{f\frac{1}{\mu}}} \frac{\partial \mathbf{s_m}}{\partial t}
        - \frac{\partial \mathbf{s_e}}{\partial t}

    Finally, we discretize in time according to backward Euler. The discrete electric fields
    on mesh edges at time :math:`t_k > t_0` is obtained by solving the following at each time-step:

    .. math::
        \mathbf{A}_k \mathbf{b}_k = \mathbf{q}_k - \mathbf{B}_k \mathbf{b}_{k-1}

    where :math:`\Delta t_k = t_k - t_{k-1}` and

    .. math::
        &\mathbf{A}_k = \mathbf{C^T M_{f\frac{1}{\mu}} C} + \frac{1}{\Delta t_k} \mathbf{M_{e\sigma}} \\
        &\mathbf{B}_k = -\frac{1}{\Delta t_k} \mathbf{M_{e\sigma}} \\
        &\mathbf{q}_k = \frac{1}{\Delta t_k} \mathbf{C^T M_{f\frac{1}{\mu}}}
        \big [ \mathbf{s}_{\mathbf{m}, k} - \mathbf{s}_{\mathbf{m}, k-1} \big ]
        -\frac{1}{\Delta t_k} \big [ \mathbf{s}_{\mathbf{e}, k} - \mathbf{s}_{\mathbf{e}, k-1} \big ]

    Although the following system is never explicitly formed, we can represent
    the solution at all time-steps as:

    .. math::
        \begin{bmatrix}
        \mathbf{A_1} & & & & \\
        \mathbf{B_2} & \mathbf{A_2} & & & \\
        & & \ddots & & \\
        & & & \mathbf{B_n} & \mathbf{A_n}
        \end{bmatrix}
        \begin{bmatrix}
        \mathbf{e_1} \\ \mathbf{e_2} \\ \vdots \\ \mathbf{e_n}
        \end{bmatrix} =
        \begin{bmatrix}
        \mathbf{q_1} \\ \mathbf{q_2} \\ \vdots \\ \mathbf{q_n}
        \end{bmatrix} -
        \begin{bmatrix}
        \mathbf{B_1 e_0} \\ \mathbf{0} \\ \vdots \\ \mathbf{0}
        \end{bmatrix}

    where the electric fields at the initial time :math:`\mathbf{e_0}`
    are computed analytically or numerically depending on whether the
    source is galvanic and carries non-zero current at the initial time.

    """

    _fieldType = "e"
    _formulation = "EB"
    fieldsPair = Fields3DElectricField  #: A Fields3DElectricField
    Fields_Derivs = FieldsDerivativesEB

    # @profile
    def Jtvec(self, m, v, f=None):
        # Doctring inherited from parent class.
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
        r"""Diagonal system matrix for the time-step index provided.

        This method returns the diagonal system matrix for the time-step index provided:

        .. math::
            \mathbf{A}_k = \mathbf{C^T M_{f\frac{1}{\mu}} C} + \frac{1}{\Delta t_k} \mathbf{M_{e\sigma}}

        where

        * :math:`\Delta t_k` is the step length
        * :math:`\mathbf{C}` is the discrete curl operator
        * :math:`\mathbf{M_{e \sigma}}` is the conductivity inner-product matrix on edges
        * :math:`\mathbf{M_{f\frac{1}{\mu}}}` is the inverse permeability inner-product matrix on faces

        See the *Notes* section of the doc strings for :class:`Simulation3DElectricField`
        for a full description of the formulation.

        Parameters
        ----------
        tInd : int
            The time-step index; between ``[0, n_steps-1]``.

        Returns
        -------
        (n_edges, n_edges) sp.sparse.csr_matrix
            The diagonal system matrix.
        """
        assert tInd >= 0 and tInd < self.nT

        dt = self.time_steps[tInd]
        C = self.mesh.edge_curl
        MfMui = self.MfMui
        MeSigma = self.MeSigma

        return C.T.tocsr() * (MfMui * C) + 1.0 / dt * MeSigma

    def getAdiagDeriv(self, tInd, u, v, adjoint=False):
        r"""Derivative operation for the diagonal system matrix times a vector.

        The diagonal system matrix for time-step index *k* is given by:

        .. math::
            \mathbf{A}_k = \mathbf{C^T M_{f\frac{1}{\mu}} C} + \frac{1}{\Delta t_k} \mathbf{M_{e\sigma}}

        where

        * :math:`\Delta t_k` is the step length
        * :math:`\mathbf{C}` is the discrete curl operator
        * :math:`\mathbf{M_{e \sigma}}` is the conductivity inner-product matrix on edges
        * :math:`\mathbf{M_{f\frac{1}{\mu}}}` is the inverse permeability inner-product matrix on faces

        See the *Notes* section of the doc strings for :class:`Simulation3DElectricField`
        for a full description of the formulation.

        Where :math:`\mathbf{m}` are the set of model parameters, :math:`\mathbf{v}` is a vector
        and :math:`\mathbf{e_k}` is the discrete solution for time-step *k*, this method assumes
        the discrete solution is fixed and returns

        .. math::
            \frac{\partial (\mathbf{A_k \, e_k})}{\partial \mathbf{m}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial (\mathbf{A_k \, e_k})}{\partial \mathbf{m}}^T \, \mathbf{v}

        Parameters
        ----------
        tInd : int
            The time-step index; between ``[0, n_steps-1]``.
        u : (n_edges,) numpy.ndarray
            The solution for the fields for the current model for the specified time-step;
            i.e. :math:`\mathbf{e_k}`.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_edges,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of system matrix times a vector. (n_edges,) for the standard operation.
            (n_param,) for the adjoint operation.
        """
        assert tInd >= 0 and tInd < self.nT

        dt = self.time_steps[tInd]
        # MeSigmaDeriv = self.MeSigmaDeriv(u)

        if adjoint:
            return 1.0 / dt * self.MeSigmaDeriv(u, v, adjoint)

        return 1.0 / dt * self.MeSigmaDeriv(u, v, adjoint)

    def getAsubdiag(self, tInd):
        r"""Sub-diagonal system matrix for the time-step index provided.

        This method returns the sub-diagonal system matrix for the time-step index provided:

        .. math::
            \mathbf{B}_k = -\frac{1}{\Delta t_k} \mathbf{M_{e\sigma}}

        where :math:`\Delta t_k` is the step-length and :math:`\mathbf{M_{e \sigma}}` is the
        conductivity inner-product matrix on edges.

        See the *Notes* section of the doc strings for :class:`Simulation3DElectricField`
        for a full description of the formulation.

        Parameters
        ----------
        tInd : int
            The time-step index; between ``[0, n_steps-1]``.

        Returns
        -------
        (n_edges, n_edges) sp.sparse.csr_matrix
            The sub-diagonal system matrix.
        """
        assert tInd >= 0 and tInd < self.nT

        dt = self.time_steps[tInd]

        return -1.0 / dt * self.MeSigma

    def getAsubdiagDeriv(self, tInd, u, v, adjoint=False):
        r"""Derivative operation for the sub-diagonal system matrix times a vector.

        The sub-diagonal system matrix for time-step index *k* is given by:

        .. math::
            \mathbf{B}_k = -\frac{1}{\Delta t_k} \mathbf{M_{e\sigma}}

        where :math:`\Delta t_k` is the step-length and :math:`\mathbf{M_{e \sigma}}` is the
        conductivity inner-product matrix on edges.

        See the *Notes* section of the doc strings for :class:`Simulation3DElectricField`
        for a full description of the formulation.

        Where :math:`\mathbf{m}` are the set of model parameters, :math:`\mathbf{v}` is a vector
        and :math:`\mathbf{e_{k-1}}` is the discrete solution for the previous time-step,
        this method assumes the discrete solution is fixed and returns

        .. math::
            \frac{\partial (\mathbf{B_k \, e_{k-1}})}{\partial \mathbf{m}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial (\mathbf{B_k \, e_{k-1}})}{\partial \mathbf{m}}^T \, \mathbf{v}

        Parameters
        ----------
        tInd : int
            The time-step index; between ``[0, n_steps-1]``.
        u : (n_edges,) numpy.ndarray
            The solution for the fields for the current model for the previous time-step;
            i.e. :math:`\mathbf{e_{k-1}}`.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_edges,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of system matrix times a vector. (n_edges,) for the standard operation.
            (n_param,) for the adjoint operation.
        """
        dt = self.time_steps[tInd]

        if adjoint:
            return -1.0 / dt * self.MeSigmaDeriv(u, v, adjoint)

        return -1.0 / dt * self.MeSigmaDeriv(u, v, adjoint)

    def getRHS(self, tInd):
        r"""Right-hand sides for the given time index.

        This method returns the right-hand sides for the time index provided.
        The right-hand side for each source is constructed according to:

        .. math::
            \mathbf{q}_k =
            -\frac{1}{\Delta t_k} \big [ \mathbf{s}_{\mathbf{e}, k} - \mathbf{s}_{\mathbf{e}, k-1} \big ]
            - \frac{1}{\Delta t_k} \mathbf{C^T M_{f\frac{1}{\mu}}}
            \big [ \mathbf{s}_{\mathbf{m}, k} - \mathbf{s}_{\mathbf{m}, k-1} \big ]

        where

        * :math:`\Delta t_k` is the step length
        * :math:`\mathbf{C}` is the discrete curl operator
        * :math:`\mathbf{s_m}` and :math:`\mathbf{s_e}` are the integrated magnetic and electric source terms, respectively
        * :math:`\mathbf{M_{f\frac{1}{\mu}}}` is the inner-product matrices for inverse permeabilities projected to faces

        See the *Notes* section of the doc strings for :class:`Simulation3DElectricField`
        for a full description of the formulation.

        Parameters
        ----------
        tInd : int
            The time index; between ``[0, n_steps]``.

        Returns
        -------
        (n_edges, n_sources) numpy.ndarray
            The right-hand sides.
        """
        # Omit this: Note input was tInd+1
        # if tInd == len(self.time_steps):
        #     tInd = tInd - 1

        dt = self.time_steps[tInd - 1]
        s_m, s_e = self.getSourceTerm(tInd)
        _, s_en1 = self.getSourceTerm(tInd - 1)

        return -1.0 / dt * (s_e - s_en1) + self.mesh.edge_curl.T * self.MfMui * s_m

    def getRHSDeriv(self, tInd, src, v, adjoint=False):
        r"""Derivative of the right-hand side times a vector for a given source and time index.

        The right-hand side for a given source at time index *k* is constructed according to:

        .. math::
            \mathbf{q}_k =
            -\frac{1}{\Delta t_k} \big [ \mathbf{s}_{\mathbf{e}, k} - \mathbf{s}_{\mathbf{e}, k-1} \big ]
            - \frac{1}{\Delta t_k} \mathbf{C^T M_{f\frac{1}{\mu}} }
            \big [ \mathbf{s}_{\mathbf{m}, k} - \mathbf{s}_{\mathbf{m}, k-1} \big ]

        where

        * :math:`\Delta t_k` is the step length
        * :math:`\mathbf{C}` is the discrete curl operator
        * :math:`\mathbf{s_m}` and :math:`\mathbf{s_e}` are the integrated magnetic and electric source terms, respectively
        * :math:`\mathbf{M_{f\frac{1}{\mu}}}` is the inner-product matrices for inverse permeabilities projected to faces

        See the *Notes* section of the doc strings for :class:`Simulation3DElectricField`
        for a full description of the formulation.

        Where :math:`\mathbf{m}` are the set of model parameters and :math:`\mathbf{v}` is a vector,
        this method returns

        .. math::
            \frac{\partial \mathbf{q_k}}{\partial \mathbf{m}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial \mathbf{q_k}}{\partial \mathbf{m}}^T \, \mathbf{v}

        Parameters
        ----------
        tInd : int
            The time index; between ``[0, n_steps]``.
        src : .time_domain.sources.BaseTDEMSrc
            The TDEM source object.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_edges,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of the right-hand sides times a vector. (n_edges,) for the standard operation.
            (n_param,) for the adjoint operation.
        """
        # right now, we are assuming that s_e, s_m do not depend on the model.
        return Zero()

    def getAdc(self):
        r"""The system matrix for the DC resistivity problem.

        The solution to the DC resistivity problem is necessary at the initial time for
        galvanic sources whose currents are non-zero at the initial time.
        The discrete solution to the 3D DC resistivity problem is expressed as:

        .. math::
            \mathbf{A_{dc}}\,\boldsymbol{\phi_0} = \mathbf{q_{dc}}

        where :math:`\mathbf{A_{dc}}` is the DC resistivity system matrix, :math:`\boldsymbol{\phi_0}`
        is the discrete solution for the electric potentials at the initial time, and :math:`\mathbf{q_{dc}}`
        is the galvanic source term. This method returns the system matrix
        for the nodal formulation, i.e.:

        .. math::
            \mathbf{A_{dc}} = \mathbf{G^T \, M_{e\sigma} \, G}

        where :math:`\mathbf{G}` is the nodal gradient operator with imposed boundary conditions,
        and :math:`\mathbf{M_{e\sigma}}` is the inner product matrix for conductivities projected to edges.

        The electric fields at the initial time :math:`\mathbf{e_0}` are obtained by applying the
        nodal gradient operator. I.e.:

        .. math::
            \mathbf{e_0} = \mathbf{G} \, \boldsymbol{\phi_0}

        See the *Notes* section of the doc strings for :class:`.resistivity.Simulation3DNodal`
        for a full description of the nodal DC resistivity formulation.

        Returns
        -------
        (n_nodes, n_nodes) sp.sparse.csr_matrix
            The system matrix for the DC resistivity problem.
        """
        MeSigma = self.MeSigma
        Grad = self.mesh.nodal_gradient
        Adc = Grad.T.tocsr() * MeSigma * Grad
        # Handling Null space of A
        Adc[0, 0] = Adc[0, 0] + 1.0
        return Adc

    def getAdcDeriv(self, u, v, adjoint=False):
        r"""Derivative operation for the DC resistivity system matrix times a vector.

        The discrete solution to the 3D DC resistivity problem is expressed as:

        .. math::
            \mathbf{A_{dc}}\boldsymbol{\phi_0} = \mathbf{q_{dc}}

        where :math:`\mathbf{A_{dc}}` is the DC resistivity system matrix, :math:`\boldsymbol{\phi_0}`
        is the discrete solution for the electric potentials at the initial time, and :math:`\mathbf{q_{dc}}`
        is the galvanic source term. For a vector :math:`\mathbf{v}`, this method assumes
        the discrete solution is fixed and returns

        .. math::
            \frac{\partial (\mathbf{A_{dc}}\boldsymbol{\phi_0})}{\partial \mathbf{m}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial (\mathbf{A_{dc}}\boldsymbol{\phi_0})}{\partial \mathbf{m}}^T \, \mathbf{v}

        Parameters
        ----------
        u : (n_nodes,) numpy.ndarray
            The solution for the fields for the current model; i.e. electric potentials at nodes.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_nodes,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of the DC resistivity system matrix times a vector. (n_nodes,) for the standard operation.
            (n_param,) for the adjoint operation.
        """
        Grad = self.mesh.nodal_gradient
        if not adjoint:
            return Grad.T * self.MeSigmaDeriv(-u, v, adjoint)
        else:
            return self.MeSigmaDeriv(-u, Grad * v, adjoint)


###############################################################################
#                                                                             #
#                                H-J Formulation                              #
#                                                                             #
###############################################################################

# ------------------------------- Simulation3DMagneticField ------------------------------- #


class Simulation3DMagneticField(BaseTDEMSimulation):
    r"""3D TDEM simulation in terms of the magnetic field.

    This simulation solves for the magnetic field at each time-step.
    In this formulation, the magnetic fields are defined on mesh edges and the
    current density is defined on mesh faces; i.e. it is an HJ formulation.
    See the *Notes* section for a comprehensive description of the formulation.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
        The mesh.
    survey : .time_domain.survey.Survey
        The time-domain EM survey.
    dt_threshold : float
        Threshold used when determining the unique time-step lengths.

    Notes
    -----
    Here, we start with the quasi-static approximation for Maxwell's equations by neglecting
    electric displacement:

    .. math::
        &\nabla \times \vec{e} + \frac{\partial \vec{b}}{\partial t} = - \frac{\partial \vec{s}_m}{\partial t} \\
        &\nabla \times \vec{h} - \vec{j} = \vec{s}_e

    where :math:`\vec{s}_e` is an electric source term that defines a source current density,
    and :math:`\vec{s}_m` magnetic source term that defines a source magnetic flux density.
    We define the constitutive relations for the electrical resistivity :math:`\rho`
    and magnetic permeability :math:`\mu` as:

    .. math::
        \vec{e} &= \rho \vec{e} \\
        \vec{b} &= \mu \vec{h}

    We then take the inner products of all previous expressions with a vector test function :math:`\vec{u}`.
    Through vector calculus identities and the divergence theorem, we obtain:

    .. math::
        & \int_\Omega (\nabla \times \vec{u}) \cdot \vec{e} \; dv
        - \oint_{\partial \Omega} \vec{u} \cdot (\vec{e} \times \hat{n} ) \, da
        + \int_\Omega \vec{u} \cdot \frac{\partial \vec{b}}{\partial t} \, dv
        = - \int_\Omega \vec{u} \cdot \frac{\partial \vec{s}_m}{\partial t} \, dv \\
        & \int_\Omega \vec{u} \cdot (\nabla \times \vec{h} ) \, dv - \int_\Omega \vec{u} \cdot \vec{j} \, dv
        = \int_\Omega \vec{u} \cdot \vec{s}_e \, dv\\
        & \int_\Omega \vec{u} \cdot \vec{e} \, dv = \int_\Omega \vec{u} \cdot \rho \vec{j} \, dv \\
        & \int_\Omega \vec{u} \cdot \vec{b} \, dv = \int_\Omega \vec{u} \cdot \mu \vec{h} \, dv

    Assuming natural boundary conditions, the surface integral is zero.

    The above expressions are discretized in space according to the finite volume method.
    The discrete magnetic fields :math:`\mathbf{h}` are defined on mesh edges,
    and the discrete current densities :math:`\mathbf{j}` are defined on mesh faces.
    This implies :math:`\mathbf{b}` must be defined on mesh edges and :math:`\mathbf{e}` must
    be defined on mesh faces. Where :math:`\mathbf{u_e}` and :math:`\mathbf{u_f}` represent
    test functions discretized to edges and faces, respectively, we obtain the following
    set of discrete inner-products:

    .. math::
        &\mathbf{u_e^T C^T M_f \, e } + \mathbf{u_e^T M_e} \frac{\partial \mathbf{b}}{\partial t}
        = - \mathbf{u_e^T} \, \frac{\partial \mathbf{s_m}}{\partial t} \\
        &\mathbf{u_f^T C \, h} - \mathbf{u_f^T j} = \mathbf{u_f^T s_e} \\
        &\mathbf{u_f^T M_f e} = \mathbf{u_f^T M_{f\rho} \, j} \\
        &\mathbf{u_e^T M_e b} = \mathbf{u_e^T M_{e \mu} h}

    where

    * :math:`\mathbf{C}` is the discrete curl operator
    * :math:`\mathbf{s_m}` and :math:`\mathbf{s_e}` are the integrated magnetic and electric source terms, respectively
    * :math:`\mathbf{M_e}` is the edge inner-product matrix
    * :math:`\mathbf{M_f}` is the face inner-product matrix
    * :math:`\mathbf{M_{e\mu}}` is the inner-product matrix for permeabilities projected to edges
    * :math:`\mathbf{M_{f\rho}}` is the inner-product matrix for resistivities projected to faces

    Cancelling like-terms and combining the discrete expressions in terms of the magnetic field, we obtain:

    .. math::
        \mathbf{C^T M_{f\rho} C \, h} + \mathbf{M_{e\mu}} \frac{\partial \mathbf{h}}{\partial t}
        = \mathbf{C^T M_{f\rho} s_e} - \frac{\partial \mathbf{s_m}}{\partial t}

    Finally, we discretize in time according to backward Euler. The discrete magnetic field
    on mesh edges at time :math:`t_k > t_0` is obtained by solving the following at each time-step:

    .. math::
        \mathbf{A}_k \mathbf{h}_k = \mathbf{q}_k - \mathbf{B}_k \mathbf{h}_{k-1}

    where :math:`\Delta t_k = t_k - t_{k-1}` and

    .. math::
        &\mathbf{A}_k = \mathbf{C^T M_{f\rho} C} + \frac{1}{\Delta t_k} \mathbf{M_{e\mu}} \\
        &\mathbf{B}_k = -\frac{1}{\Delta t_k} \mathbf{M_{e\mu}}\\
        &\mathbf{q}_k = \mathbf{C^T M_{f\rho} s}_{\mathbf{e},k} \;
        - \frac{1}{\Delta t_k} \big [ \mathbf{s}_{\mathbf{m},k} + \mathbf{s}_{\mathbf{m},k-1} \big ]

    Although the following system is never explicitly formed, we can represent
    the solution at all time-steps as:

    .. math::
        \begin{bmatrix}
        \mathbf{A_1} & & & & \\
        \mathbf{B_2} & \mathbf{A_2} & & & \\
        & & \ddots & & \\
        & & & \mathbf{B_n} & \mathbf{A_n}
        \end{bmatrix}
        \begin{bmatrix}
        \mathbf{h_1} \\ \mathbf{h_2} \\ \vdots \\ \mathbf{h_n}
        \end{bmatrix} =
        \begin{bmatrix}
        \mathbf{q_1} \\ \mathbf{q_2} \\ \vdots \\ \mathbf{q_n}
        \end{bmatrix} -
        \begin{bmatrix}
        \mathbf{B_1 h_0} \\ \mathbf{0} \\ \vdots \\ \mathbf{0}
        \end{bmatrix}

    where the magnetic fields at the initial time :math:`\mathbf{h_0}`
    are computed analytically or numerically depending on whether the source
    carries non-zero current at the initial time.

    """

    _fieldType = "h"
    _formulation = "HJ"
    fieldsPair = Fields3DMagneticField  #: Fields object pair
    Fields_Derivs = FieldsDerivativesHJ

    def getAdiag(self, tInd):
        r"""Diagonal system matrix for the given time-step index.

        This method returns the diagonal system matrix for the time-step index provided:

        .. math::
            \mathbf{A}_k = \mathbf{C^T M_{f\rho} C} + \frac{1}{\Delta t_k} \mathbf{M_{e\mu}}

        where

        * :math:`\Delta t_k` is the step length
        * :math:`\mathbf{C}` is the discrete curl operator
        * :math:`\mathbf{M_{f \rho}}` is the resistivity inner-product matrix on faces
        * :math:`\mathbf{M_{e\mu}}` is the permeability inner-product matrix on edges

        See the *Notes* section of the doc strings for :class:`Simulation3DMagneticField`
        for a full description of the formulation.

        Parameters
        ----------
        tInd : int
            The time-step index; between ``[0, n_steps-1]``.

        Returns
        -------
        (n_edges, n_edges) sp.sparse.csr_matrix
            The diagonal system matrix.
        """
        assert tInd >= 0 and tInd < self.nT

        dt = self.time_steps[tInd]
        C = self.mesh.edge_curl
        MfRho = self.MfRho
        MeMu = self.MeMu

        return C.T * (MfRho * C) + 1.0 / dt * MeMu

    def getAdiagDeriv(self, tInd, u, v, adjoint=False):
        r"""Derivative operation for the diagonal system matrix times a vector.

        The diagonal system matrix for time-step index *k* is given by:

        .. math::
            \mathbf{A}_k = \mathbf{C^T M_{f\rho} C} + \frac{1}{\Delta t_k} \mathbf{M_{e\mu}}

        where

        * :math:`\Delta t_k` is the step length
        * :math:`\mathbf{C}` is the discrete curl operator
        * :math:`\mathbf{M_{f \rho}}` is the resistivity inner-product matrix on faces
        * :math:`\mathbf{M_{e\mu}}` is the permeability inner-product matrix on edges

        See the *Notes* section of the doc strings for :class:`Simulation3DMagneticField`
        for a full description of the formulation.

        Where :math:`\mathbf{m}` are the set of model parameters, :math:`\mathbf{v}` is a vector
        and :math:`\mathbf{h_k}` is the discrete solution for time-step *k*, this method assumes
        the discrete solution is fixed and returns

        .. math::
            \frac{\partial (\mathbf{A_k \, h_k})}{\partial \mathbf{m}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial (\mathbf{A_k \, h_k})}{\partial \mathbf{m}}^T \, \mathbf{v}

        Parameters
        ----------
        tInd : int
            The time-step index; between ``[0, n_steps-1]``.
        u : (n_edges,) numpy.ndarray
            The solution for the fields for the current model; i.e. :math:`\mathbf{h_k}`.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_edges,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of system matrix times a vector. (n_edges,) for the standard operation.
            (n_param,) for the adjoint operation.
        """
        assert tInd >= 0 and tInd < self.nT

        C = self.mesh.edge_curl

        if adjoint:
            return self.MfRhoDeriv(C * u, C * v, adjoint)

        return C.T * self.MfRhoDeriv(C * u, v, adjoint)

    def getAsubdiag(self, tInd):
        r"""Sub-diagonal system matrix for the time-step index provided.

        This method returns the sub-diagonal system matrix for the time-step index provided:

        .. math::
            \mathbf{B}_k = -\frac{1}{\Delta t_k} \mathbf{M_{e\mu}}

        where :math:`\Delta t_k` is the step-length and :math:`\mathbf{M_{e \mu}}` is the
        permeability inner-product matrix on edges.

        See the *Notes* section of the doc strings for :class:`Simulation3DMagneticField`
        for a full description of the formulation.

        Parameters
        ----------
        tInd : int
            The time-step index; between ``[0, n_steps-1]``.

        Returns
        -------
        (n_edges, n_edges) sp.sparse.csr_matrix
            The sub-diagonal system matrix.
        """
        assert tInd >= 0 and tInd < self.nT

        dt = self.time_steps[tInd]

        return -1.0 / dt * self.MeMu

    def getAsubdiagDeriv(self, tInd, u, v, adjoint=False):
        r"""Derivative operation for the sub-diagonal system matrix times a vector.

        The sub-diagonal system matrix for time-step index *k* is given by:

        .. math::
            \mathbf{B}_k = -\frac{1}{\Delta t_k} \mathbf{M_{e\mu}}

        where :math:`\Delta t_k` is the step-length and :math:`\mathbf{M_{e \mu}}` is the
        permeability inner-product matrix on edges.

        See the *Notes* section of the doc strings for :class:`Simulation3DMagneticField`
        for a full description of the formulation.

        Where :math:`\mathbf{m}` are the set of model parameters, :math:`\mathbf{v}` is a vector
        and :math:`\mathbf{h_{k-1}}` is the discrete solution for the previous time-step,
        this method assumes the discrete solution is fixed and returns

        .. math::
            \frac{\partial (\mathbf{B_k \, h_{k-1}})}{\partial \mathbf{m}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial (\mathbf{B_k \, h_{k-1}})}{\partial \mathbf{m}}^T \, \mathbf{v}

        Parameters
        ----------
        tInd : int
            The time-step index; between ``[0, n_steps-1]``.
        u : (n_edges,) numpy.ndarray
            The solution for the fields for the current model for the previous time-step;
            i.e. :math:`\mathbf{h_{k-1}}`.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_edges,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of system matrix times a vector. (n_edges,) for the standard operation.
            (n_param,) for the adjoint operation.
        """
        return Zero()

    def getRHS(self, tInd):
        r"""Right-hand sides for the given time index.

        This method returns the right-hand sides for the time index provided.
        The right-hand side for each source is constructed according to:

        .. math::
            \mathbf{q}_k = \mathbf{C^T M_{f\rho} s}_{\mathbf{e},k} \;
            - \frac{1}{\Delta t_k} \big [ \mathbf{s}_{\mathbf{m},k} + \mathbf{s}_{\mathbf{m},k-1} \big ]

        where

        * :math:`\Delta t_k` is the step length
        * :math:`\mathbf{C}` is the discrete curl operator
        * :math:`\mathbf{s_m}` and :math:`\mathbf{s_e}` are the integrated magnetic and electric source terms, respectively
        * :math:`\mathbf{M_{f\rho}}` is the inner-product matrix for resisitivites projected to faces

        See the *Notes* section of the doc strings for :class:`Simulation3DMagneticField`
        for a full description of the formulation.

        Parameters
        ----------
        tInd : int
            The time index; between ``[0, n_steps]``.

        Returns
        -------
        (n_edges, n_sources) numpy.ndarray
            The right-hand sides.
        """
        C = self.mesh.edge_curl
        MfRho = self.MfRho
        s_m, s_e = self.getSourceTerm(tInd)

        return C.T * (MfRho * s_e) + s_m

    def getRHSDeriv(self, tInd, src, v, adjoint=False):
        r"""Derivative of the right-hand side times a vector for a given source and time index.

        The right-hand side for a given source at time index *k* is constructed according to:

        .. math::
            \mathbf{q}_k = \mathbf{C^T M_{f\rho} s}_{\mathbf{e},k} \;
            - \frac{1}{\Delta t_k} \big [ \mathbf{s}_{\mathbf{m},k} + \mathbf{s}_{\mathbf{m},k-1} \big ]

        where

        * :math:`\Delta t_k` is the step length
        * :math:`\mathbf{C}` is the discrete curl operator
        * :math:`\mathbf{s_m}` and :math:`\mathbf{s_e}` are the integrated magnetic and electric source terms, respectively
        * :math:`\mathbf{M_{f\rho}}` is the inner-product matrix for resisitivites projected to faces

        See the *Notes* section of the doc strings for :class:`Simulation3DMagneticField`
        for a full description of the formulation.

        Where :math:`\mathbf{m}` are the set of model parameters and :math:`\mathbf{v}` is a vector,
        this method returns

        .. math::
            \frac{\partial \mathbf{q_k}}{\partial \mathbf{m}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial \mathbf{q_k}}{\partial \mathbf{m}}^T \, \mathbf{v}

        Parameters
        ----------
        tInd : int
            The time index; between ``[0, n_steps]``.
        src : .time_domain.sources.BaseTDEMSrc
            The TDEM source object.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_edges,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of the right-hand sides times a vector. (n_edges,) for the standard operation.
            (n_param,) for the adjoint operation.
        """
        C = self.mesh.edge_curl
        s_m, s_e = src.eval(self, self.times[tInd])

        if adjoint is True:
            return self.MfRhoDeriv(s_e, C * v, adjoint)
        # assumes no source derivs
        return C.T * self.MfRhoDeriv(s_e, v, adjoint)

    # I DON'T THINK THIS IS CURRENTLY USED BY THE H-FORMULATION.
    def getAdc(self):
        r"""The system matrix for the DC resistivity problem.

        The solution to the DC resistivity problem is necessary at the initial time for
        galvanic sources whose currents are non-zero at the initial time.
        The discrete solution to the 3D DC resistivity problem is expressed as:

        .. math::
            \mathbf{A_{dc}}\,\boldsymbol{\phi_0} = \mathbf{q_{dc}}

        where :math:`\mathbf{A_{dc}}` is the DC resistivity system matrix, :math:`\boldsymbol{\phi_0}`
        is the discrete solution for the electric potentials at the initial time, and :math:`\mathbf{q_{dc}}`
        is the galvanic source term. This method returns the system matrix
        for the cell-centered formulation, i.e.:

        .. math::
            \mathbf{D \, M_{f\rho}^{-1} \, G}

        where :math:`\mathbf{D}` is the face divergence operator, :math:`\mathbf{G}` is the cell gradient
        operator with imposed boundary conditions, and :math:`\mathbf{M_{f\rho}}` is the inner product
        matrix for resistivities projected to faces.

        See the *Notes* section of the doc strings for
        :class:`.resistivity.Simulation3DCellCentered`
        for a full description of the cell centered DC resistivity formulation.

        Returns
        -------
        (n_cells, n_cells) sp.sparse.csr_matrix
            The system matrix for the DC resistivity problem.
        """
        D = sdiag(self.mesh.cell_volumes) * self.mesh.face_divergence
        G = D.T
        MfRhoI = self.MfRhoI
        return D * MfRhoI * G

    def getAdcDeriv(self, u, v, adjoint=False):
        r"""Derivative operation for the DC resistivity system matrix times a vector.

        The discrete solution to the 3D DC resistivity problem is expressed as:

        .. math::
            \mathbf{A_{dc}}\boldsymbol{\phi_0} = \mathbf{q_{dc}}

        where :math:`\mathbf{A_{dc}}` is the DC resistivity system matrix, :math:`\boldsymbol{\phi_0}`
        is the discrete solution for the electric potentials at the initial time, and :math:`\mathbf{q_{dc}}`
        is the galvanic source term. For a vector :math:`\mathbf{v}`, this method assumes
        the discrete solution is fixed and returns

        .. math::
            \frac{\partial (\mathbf{A_{dc}}\boldsymbol{\phi_0})}{\partial \mathbf{m}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial (\mathbf{A_{dc}}\boldsymbol{\phi_0})}{\partial \mathbf{m}}^T \, \mathbf{v}

        Parameters
        ----------
        u : (n_cells,) numpy.ndarray
            The solution for the fields for the current model; i.e. electric potentials at cell centers.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_cells,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of the DC resistivity system matrix times a vector. (n_cells,) for the standard operation.
            (n_param,) for the adjoint operation.
        """
        D = sdiag(self.mesh.cell_volumes) * self.mesh.face_divergence
        G = D.T

        if adjoint:
            # This is the same as
            #      self.MfRhoIDeriv(G * u, D.T * v, adjoint=True)
            return self.MfRhoIDeriv(G * u, G * v, adjoint=True)
        return D * self.MfRhoIDeriv(G * u, v)


# ------------------------------- Simulation3DCurrentDensity ------------------------------- #


class Simulation3DCurrentDensity(BaseTDEMSimulation):
    r"""3D TDEM simulation in terms of the current density.

    This simulation solves for the current density at each time-step.
    In this formulation, the magnetic fields are defined on mesh edges and the
    current densities are defined on mesh faces; i.e. it is an HJ formulation.
    See the *Notes* section for a comprehensive description of the formulation.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
        The mesh.
    survey : .time_domain.survey.Survey
        The time-domain EM survey.
    dt_threshold : float
        Threshold used when determining the unique time-step lengths.

    Notes
    -----
    Here, we start with the quasi-static approximation for Maxwell's equations by neglecting
    electric displacement:

    .. math::
        &\nabla \times \vec{e} + \frac{\partial \vec{b}}{\partial t} = - \frac{\partial \vec{s}_m}{\partial t} \\
        &\nabla \times \vec{h} - \vec{j} = \vec{s}_e

    where :math:`\vec{s}_e` is an electric source term that defines a source current density,
    and :math:`\vec{s}_m` magnetic source term that defines a source magnetic flux density.
    We define the constitutive relations for the electrical resistivity :math:`\rho`
    and magnetic permeability :math:`\mu` as:

    .. math::
        \vec{e} &= \rho \vec{e} \\
        \vec{b} &= \mu \vec{h}

    We then take the inner products of all previous expressions with a vector test function :math:`\vec{u}`.
    Through vector calculus identities and the divergence theorem, we obtain:

    .. math::
        & \int_\Omega (\nabla \times \vec{u}) \cdot \vec{e} \; dv
        - \oint_{\partial \Omega} \vec{u} \cdot (\vec{e} \times \hat{n} ) \, da
        + \int_\Omega \vec{u} \cdot \frac{\partial \vec{b}}{\partial t} \, dv
        = - \int_\Omega \vec{u} \cdot \frac{\partial \vec{s}_m}{\partial t} \, dv \\
        & \int_\Omega \vec{u} \cdot (\nabla \times \vec{h} ) \, dv - \int_\Omega \vec{u} \cdot \vec{j} \, dv
        = \int_\Omega \vec{u} \cdot \vec{s}_e \, dv\\
        & \int_\Omega \vec{u} \cdot \vec{e} \, dv = \int_\Omega \vec{u} \cdot \rho \vec{j} \, dv \\
        & \int_\Omega \vec{u} \cdot \vec{b} \, dv = \int_\Omega \vec{u} \cdot \mu \vec{h} \, dv

    Assuming natural boundary conditions, the surface integral is zero.

    The above expressions are discretized in space according to the finite volume method.
    The discrete magnetic fields :math:`\mathbf{h}` are defined on mesh edges,
    and the discrete current densities :math:`\mathbf{j}` are defined on mesh faces.
    This implies :math:`\mathbf{b}` must be defined on mesh edges and :math:`\mathbf{e}` must
    be defined on mesh faces. Where :math:`\mathbf{u_e}` and :math:`\mathbf{u_f}` represent
    test functions discretized to edges and faces, respectively, we obtain the following
    set of discrete inner-products:

    .. math::
        &\mathbf{u_e^T C^T M_f \, e } + \mathbf{u_e^T M_e} \frac{\partial \mathbf{b}}{\partial t}
        = - \mathbf{u_e^T} \, \frac{\partial \mathbf{s_m}}{\partial t} \\
        &\mathbf{u_f^T C \, h} - \mathbf{u_f^T j} = \mathbf{u_f^T s_e} \\
        &\mathbf{u_f^T M_f e} = \mathbf{u_f^T M_{f\rho} \, j} \\
        &\mathbf{u_e^T M_e b} = \mathbf{u_e^T M_{e \mu} h}

    where

    * :math:`\mathbf{C}` is the discrete curl operator
    * :math:`\mathbf{s_m}` and :math:`\mathbf{s_e}` are the integrated magnetic and electric source terms, respectively
    * :math:`\mathbf{M_e}` is the edge inner-product matrix
    * :math:`\mathbf{M_f}` is the face inner-product matrix
    * :math:`\mathbf{M_{e\mu}}` is the inner-product matrix for permeabilities projected to edges
    * :math:`\mathbf{M_{f\rho}}` is the inner-product matrix for resistivities projected to faces

    Cancelling like-terms and combining the discrete expressions in terms of the current density, we obtain:

    .. math::
        \mathbf{C M_{e\mu}^{-1} C^T M_{f\rho} \, j} +
        \frac{\partial \mathbf{j}}{\partial t} =
        - \frac{\partial \mathbf{s_e}}{\partial t}
        - \mathbf{C M_{e\mu}^{-1}} \frac{\partial \mathbf{s_m}}{\partial t}

    Finally, we discretize in time according to backward Euler. The discrete current density
    on mesh edges at time :math:`t_k > t_0` is obtained by solving the following at each time-step:

    .. math::
        \mathbf{A}_k \mathbf{j}_k = \mathbf{q}_k - \mathbf{B}_k \mathbf{j}_{k-1}

    where :math:`\Delta t_k = t_k - t_{k-1}` and

    .. math::
        &\mathbf{A}_k = \mathbf{C M_{e\mu}^{-1} C^T M_{f\rho}} + \frac{1}{\Delta t_k} \mathbf{I} \\
        &\mathbf{B}_k = - \frac{1}{\Delta t_k} \mathbf{I}\\
        &\mathbf{q}_k = - \frac{1}{\Delta t_k} \big [ \mathbf{s}_{\mathbf{e},k} + \mathbf{s}_{\mathbf{e},k-1} \big ] \;
        - \; \frac{1}{\Delta t_k} \mathbf{C M_{e\mu}^{-1}} \big [ \mathbf{s}_{\mathbf{m},k} + \mathbf{s}_{\mathbf{m},k-1} \big ]

    Although the following system is never explicitly formed, we can represent
    the solution at all time-steps as:

    .. math::
        \begin{bmatrix}
        \mathbf{A_1} & & & & \\
        \mathbf{B_2} & \mathbf{A_2} & & & \\
        & & \ddots & & \\
        & & & \mathbf{B_n} & \mathbf{A_n}
        \end{bmatrix}
        \begin{bmatrix}
        \mathbf{j_1} \\ \mathbf{j_2} \\ \vdots \\ \mathbf{j_n}
        \end{bmatrix} =
        \begin{bmatrix}
        \mathbf{q_1} \\ \mathbf{q_2} \\ \vdots \\ \mathbf{q_n}
        \end{bmatrix} -
        \begin{bmatrix}
        \mathbf{B_1 j_0} \\ \mathbf{0} \\ \vdots \\ \mathbf{0}
        \end{bmatrix}

    where the current densities at the initial time :math:`\mathbf{j_0}`
    are computed analytically or numerically depending on whether the source
    carries non-zero current at the initial time.
    """

    _fieldType = "j"
    _formulation = "HJ"
    fieldsPair = Fields3DCurrentDensity  #: Fields object pair
    Fields_Derivs = FieldsDerivativesHJ

    def getAdiag(self, tInd):
        r"""Diagonal system matrix for the given time-step index.

        This method returns the diagonal system matrix for the time-step index provided:

        .. math::
            \mathbf{A}_k = \mathbf{C M_{e\mu}^{-1} C^T M_{f\rho}} + \frac{1}{\Delta t_k} \mathbf{I}

        where

        * :math:`\Delta t_k` is the step length
        * :math:`\mathbf{I}` is the identity matrix
        * :math:`\mathbf{C}` is the discrete curl operator
        * :math:`\mathbf{M_{e \mu}}` is the permeability inner-product matrix on faces
        * :math:`\mathbf{M_{f \rho}}` is the permeability inner-product matrix on edges

        See the *Notes* section of the doc strings for :class:`Simulation3DCurrentDensity`
        for a full description of the formulation.

        Parameters
        ----------
        tInd : int
            The time-step index; between ``[0, n_steps-1]``.

        Returns
        -------
        (n_faces, n_faces) sp.sparse.csr_matrix
            The diagonal system matrix.
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
        r"""Derivative operation for the diagonal system matrix times a vector.

        The diagonal system matrix for time-step index *k* is given by:

        .. math::
            \mathbf{A}_k = \mathbf{C M_{e\mu}^{-1} C^T M_{f\rho}} + \frac{1}{\Delta t_k} \mathbf{I}

        where

        * :math:`\Delta t_k` is the step length
        * :math:`\mathbf{I}` is the identity matrix
        * :math:`\mathbf{C}` is the discrete curl operator
        * :math:`\mathbf{M_{e \mu}}` is the permeability inner-product matrix on faces
        * :math:`\mathbf{M_{f \rho}}` is the permeability inner-product matrix on edges

        See the *Notes* section of the doc strings for :class:`Simulation3DCurrentDensity`
        for a full description of the formulation.

        Where :math:`\mathbf{m}` are the set of model parameters, :math:`\mathbf{v}` is a vector
        and :math:`\mathbf{j_k}` is the discrete solution for time-step *k*, this method assumes
        the discrete solution is fixed and returns

        .. math::
            \frac{\partial (\mathbf{A_k \, j_k})}{\partial \mathbf{m}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial (\mathbf{A_k \, j_k})}{\partial \mathbf{m}}^T \, \mathbf{v}

        Parameters
        ----------
        tInd : int
            The time-step index; between ``[0, n_steps-1]``.
        u : (n_faces,) numpy.ndarray
            The solution for the fields for the current model; i.e. :math:`\mathbf{j_k}`.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_faces,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of system matrix times a vector. (n_faces,) for the standard operation.
            (n_param,) for the adjoint operation.
        """
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
        r"""Sub-diagonal system matrix for the time-step index provided.

        This method returns the sub-diagonal system matrix for the time-step index provided:

        .. math::
            \mathbf{B}_k = -\frac{1}{\Delta t_k} \mathbf{I}

        where :math:`\Delta t_k` is the step-length and :math:`\mathbf{I}` is the
        identity matrix.

        See the *Notes* section of the doc strings for :class:`Simulation3DCurrentDensity`
        for a full description of the formulation.

        Parameters
        ----------
        tInd : int
            The time-step index; between ``[0, n_steps-1]``.

        Returns
        -------
        (n_faces, n_faces) sp.sparse.csr_matrix
            The sub-diagonal system matrix.
        """
        assert tInd >= 0 and tInd < self.nT
        eye = sp.eye(self.mesh.n_faces)

        dt = self.time_steps[tInd]

        if self._makeASymmetric:
            return -1.0 / dt * self.MfRho.T
        return -1.0 / dt * eye

    def getAsubdiagDeriv(self, tInd, u, v, adjoint=False):
        r"""Derivative operation for the sub-diagonal system matrix times a vector.

        The sub-diagonal system matrix for time-step index *k* is given by:

        .. math::
            \mathbf{B}_k = -\frac{1}{\Delta t_k} \mathbf{I}

        where :math:`\Delta t_k` is the step-length and :math:`\mathbf{I}` is the
        identity matrix.

        See the *Notes* section of the doc strings for :class:`Simulation3DCurrentDensity`
        for a full description of the formulation.

        Where :math:`\mathbf{m}` are the set of model parameters, :math:`\mathbf{v}` is a vector
        and :math:`\mathbf{j_{k-1}}` is the discrete solution for the previous time-step,
        this method assumes the discrete solution is fixed and returns

        .. math::
            \frac{\partial (\mathbf{B_k \, j_{k-1}})}{\partial \mathbf{m}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial (\mathbf{B_k \, j_{k-1}})}{\partial \mathbf{m}}^T \, \mathbf{v}

        Parameters
        ----------
        tInd : int
            The time-step index; between ``[0, n_steps-1]``.
        u : (n_faces,) numpy.ndarray
            The solution for the fields for the current model for the previous time-step;
            i.e. :math:`\mathbf{j_{k-1}}`.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_faces,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of system matrix times a vector. (n_faces,) for the standard operation.
            (n_param,) for the adjoint operation.
        """
        return Zero()

    def getRHS(self, tInd):
        r"""Right-hand sides for the given time index.

        This method returns the right-hand sides for the time index provided.
        The right-hand side for each source is constructed according to:

        .. math::
            \mathbf{q}_k = - \frac{1}{\Delta t_k} \big [ \mathbf{s}_{\mathbf{e},k} + \mathbf{s}_{\mathbf{e},k-1} \big ] \;
            - \; \frac{1}{\Delta t_k} \mathbf{C M_{e\mu}^{-1}} \big [ \mathbf{s}_{\mathbf{m},k} + \mathbf{s}_{\mathbf{m},k-1} \big ]

        where

        * :math:`\Delta t_k` is the step length
        * :math:`\mathbf{C}` is the discrete curl operator
        * :math:`\mathbf{s_m}` and :math:`\mathbf{s_e}` are the integrated magnetic and electric source terms, respectively
        * :math:`\mathbf{M_{e\mu}}` is the inner-product matrix for permeabilities projected to edges

        See the *Notes* section of the doc strings for :class:`Simulation3DCurrentDensity`
        for a full description of the formulation.

        Parameters
        ----------
        tInd : int
            The time index; between ``[0, n_steps]``.

        Returns
        -------
        (n_faces, n_sources) numpy.ndarray
            The right-hand sides.
        """
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
        r"""Derivative of the right-hand side times a vector for a given source and time index.

        The right-hand side for a given source at time index *k* is constructed according to:

        .. math::
            \mathbf{q}_k = - \frac{1}{\Delta t_k} \big [ \mathbf{s}_{\mathbf{e},k} + \mathbf{s}_{\mathbf{e},k-1} \big ] \;
            - \; \frac{1}{\Delta t_k} \mathbf{C M_{e\mu}^{-1}} \big [ \mathbf{s}_{\mathbf{m},k} + \mathbf{s}_{\mathbf{m},k-1} \big ]

        where

        * :math:`\Delta t_k` is the step length
        * :math:`\mathbf{C}` is the discrete curl operator
        * :math:`\mathbf{s_m}` and :math:`\mathbf{s_e}` are the integrated magnetic and electric source terms, respectively
        * :math:`\mathbf{M_{e\mu}}` is the inner-product matrix for permeabilities projected to edges

        See the *Notes* section of the doc strings for :class:`Simulation3DCurrentDensity`
        for a full description of the formulation.

        Where :math:`\mathbf{m}` are the set of model parameters and :math:`\mathbf{v}` is a vector,
        this method returns

        .. math::
            \frac{\partial \mathbf{q_k}}{\partial \mathbf{m}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial \mathbf{q_k}}{\partial \mathbf{m}}^T \, \mathbf{v}

        Parameters
        ----------
        tInd : int
            The time index; between ``[0, n_steps]``.
        src : .time_domain.sources.BaseTDEMSrc
            The TDEM source object.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_faces,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of the right-hand sides times a vector. (n_faces,) for the standard operation.
            (n_param,) for the adjoint operation.
        """
        return Zero()  # assumes no derivs on sources

    def getAdc(self):
        r"""The system matrix for the DC resistivity problem.

        The solution to the DC resistivity problem is necessary at the initial time for
        galvanic sources whose currents are non-zero at the initial time.
        The discrete solution to the 3D DC resistivity problem is expressed as:

        .. math::
            \mathbf{A_{dc}}\,\boldsymbol{\phi_0} = \mathbf{q_{dc}}

        where :math:`\mathbf{A_{dc}}` is the DC resistivity system matrix, :math:`\boldsymbol{\phi_0}`
        is the discrete solution for the electric potentials at the initial time, and :math:`\mathbf{q_{dc}}`
        is the galvanic source term. This method returns the system matrix
        for the cell-centered formulation, i.e.:

        .. math::
            \mathbf{D \, M_{f\rho}^{-1} \, G}

        where :math:`\mathbf{D}` is the face divergence operator, :math:`\mathbf{G}` is the cell gradient
        operator with imposed boundary conditions, and :math:`\mathbf{M_{f\rho}}` is the inner product
        matrix for resistivities projected to faces.

        The current density at the initial time :math:`\mathbf{j_0}` are obtained by applying:

        .. math::
            \mathbf{j_0} = \mathbf{M_{f\rho}^{-1} \, G} \, \boldsymbol{\phi_0}

        See the *Notes* section of the doc strings for :class:`.resistivity.Simulation3DCellCentered`
        for a full description of the cell centered DC resistivity formulation.

        Returns
        -------
        (n_cells, n_cells) sp.sparse.csr_matrix
            The system matrix for the DC resistivity problem.
        """
        D = sdiag(self.mesh.cell_volumes) * self.mesh.face_divergence
        G = D.T
        MfRhoI = self.MfRhoI
        return D * MfRhoI * G

    def getAdcDeriv(self, u, v, adjoint=False):
        r"""Derivative operation for the DC resistivity system matrix times a vector.

        The discrete solution to the 3D DC resistivity problem is expressed as:

        .. math::
            \mathbf{A_{dc}}\boldsymbol{\phi_0} = \mathbf{q_{dc}}

        where :math:`\mathbf{A_{dc}}` is the DC resistivity system matrix, :math:`\boldsymbol{\phi_0}`
        is the discrete solution for the electric potentials at the initial time, and :math:`\mathbf{q_{dc}}`
        is the galvanic source term. For a vector :math:`\mathbf{v}`, this method assumes
        the discrete solution is fixed and returns

        .. math::
            \frac{\partial (\mathbf{A_{dc}}\boldsymbol{\phi_0})}{\partial \mathbf{m}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial (\mathbf{A_{dc}}\boldsymbol{\phi_0})}{\partial \mathbf{m}}^T \, \mathbf{v}

        Parameters
        ----------
        u : (n_cells,) numpy.ndarray
            The solution for the fields for the current model; i.e. electric potentials at cell centers.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_cells,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of the DC resistivity system matrix times a vector. (n_cells,) for the standard operation.
            (n_param,) for the adjoint operation.
        """
        D = sdiag(self.mesh.cell_volumes) * self.mesh.face_divergence
        G = D.T

        if adjoint:
            # This is the same as
            #      self.MfRhoIDeriv(G * u, D.T * v, adjoint=True)
            return self.MfRhoIDeriv(G * u, G * v, adjoint=True)
        return D * self.MfRhoIDeriv(G * u, v)
