from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as sp
import time
import properties
import warnings

from SimPEG import Utils
from SimPEG import Problem
from SimPEG import Optimization
from SimPEG import Solver

from SimPEG.FLOW.Richards.RichardsSurvey import RichardsSurvey
from SimPEG.FLOW.Richards.Empirical import BaseHydraulicConductivity
from SimPEG.FLOW.Richards.Empirical import BaseWaterRetention


class RichardsProblem(Problem.BaseTimeProblem):
    """RichardsProblem"""

    hydraulic_conductivity = properties.Instance(
        'hydraulic conductivity function',
        BaseHydraulicConductivity
    )
    water_retention = properties.Instance(
        'water retention curve',
        BaseWaterRetention
    )

    # TODO: This can also be a function(time, u_ii)
    boundary_conditions = properties.Array('boundary conditions')
    initial_conditions = properties.Array('initial conditions')

    debug = properties.Bool('Show all messages', default=False)

    Solver = properties.Property('Numerical Solver', default=lambda: Solver)
    solverOpts = {}

    method = properties.StringChoice(
        'Formulation used, See notes in Celia et al., 1990',
        default='mixed',
        choices=['mixed', 'head']
    )

    do_newton = properties.Bool(
        'Do a Newton iteration vs. a Picard iteration',
        default=False
    )

    root_finder_max_iter = properties.Integer(
        'Maximum iterations for root_finder iteration',
        default=30
    )

    root_finder_tol = properties.Float(
        'tolerance of the root_finder',
        default=1e-4
    )

    @properties.observer('model')
    def _on_model_change(self, change):
        """Update the nested model functions when the
        model of the problem changes.

        Specifically :code:`hydraulic_conductivity` and
        :code:`water_retention` models are updated iff they have mappings.
        """

        if (
                not self.hydraulic_conductivity.needs_model and
                not self.water_retention.needs_model
           ):
            warnings.warn('There is no model to set.')
            return

        model = change['value']

        if self.hydraulic_conductivity.needs_model:
            self.hydraulic_conductivity.model = model

        if self.water_retention.needs_model:
            self.water_retention.model = model

    def getBoundaryConditions(self, ii, u_ii):
        if type(self.boundary_conditions) is np.ndarray:
            return self.boundary_conditions

        time = self.timeMesh.vectorCCx[ii]

        return self.boundary_conditions(time, u_ii)

    @properties.observer([
                          'do_newton',
                          'root_finder_max_iter',
                          'root_finder_tol'
                        ])
    def _on_root_finder_update(self, change):
        """Setting do_newton etc. will clear the root_finder,
        which will be reinitialized when called
        """
        if hasattr(self, '_root_finder'):
            del self._root_finder

    @property
    def root_finder(self):
        """Root-finding Algorithm"""
        if getattr(self, '_root_finder', None) is None:
            self._root_finder = Optimization.NewtonRoot(
                doLS=self.do_newton,
                maxIter=self.root_finder_max_iter,
                tol=self.root_finder_tol,
                Solver=self.Solver
            )
        return self._root_finder

    @Utils.timeIt
    def fields(self, m=None):
        if self.water_retention.needs_model or self.hydraulic_conductivity.needs_model:
            assert m is not None
        else:
            assert m is None

        tic = time.time()
        u = list(range(self.nT+1))
        u[0] = self.initial_conditions
        for ii, dt in enumerate(self.timeSteps):
            bc = self.getBoundaryConditions(ii, u[ii])
            u[ii+1] = self.root_finder.root(
                lambda hn1m, return_g=True: self.getResidual(
                    m, u[ii], hn1m, dt, bc, return_g=return_g
                ),
                u[ii]
            )
            if self.debug:
                print(
                    'Solving Fields ({0:4d}/{1:d} - {2:3.1f}% Done) {3:d} '
                    'Iterations, {4:4.2f} seconds'.format(
                        ii+1,
                        self.nT,
                        100.0*(ii+1)/self.nT,
                        self.root_finder.iter,
                        time.time() - tic
                    )
                )
        return u

    @property
    def Dz(self):
        if self.mesh.dim == 1:
            return self.mesh.faceDivx

        if self.mesh.dim == 2:
            mats = (
                Utils.spzeros(self.mesh.nC, self.mesh.vnF[0]),
                self.mesh.faceDivy
            )
        elif self.mesh.dim == 3:
            mats = (
                Utils.spzeros(self.mesh.nC, self.mesh.vnF[0]+self.mesh.vnF[1]),
                self.mesh.faceDivz
            )
        return sp.hstack(mats, format='csr')

    @Utils.timeIt
    def diagsJacobian(self, m, hn, hn1, dt, bc):
        """Diagonals and rhs of the jacobian system

        The matrix that we are computing has the form::

            .-                                      -. .-  -.   .-  -.
            |  Adiag                                 | | h1 |   | b1 |
            |   Asub    Adiag                        | | h2 |   | b2 |
            |            Asub    Adiag               | | h3 | = | b3 |
            |                 ...     ...            | | .. |   | .. |
            |                         Asub    Adiag  | | hn |   | bn |
            '-                                      -' '-  -'   '-  -'
        """
        if m is not None:
            self.model = m

        DIV = self.mesh.faceDiv
        GRAD = self.mesh.cellGrad
        BC = self.mesh.cellGradBC
        AV = self.mesh.aveF2CC.T
        Dz = self.Dz

        dT = self.water_retention.derivU(hn)
        dT1 = self.water_retention.derivU(hn1)
        dTm = self.water_retention.derivM(hn)
        dTm1 = self.water_retention.derivM(hn1)

        K1 = self.hydraulic_conductivity(hn1)
        dK1 = self.hydraulic_conductivity.derivU(hn1)
        dKm1 = self.hydraulic_conductivity.derivM(hn1)

        # Compute part of the derivative of:
        #
        #       DIV*diag(GRAD*hn1+BC*bc)*(AV*(1.0/K))^-1

        DdiagGh1 = DIV*Utils.sdiag(GRAD*hn1+BC*bc)
        diagAVk2_AVdiagK2 = (
            Utils.sdiag((AV*(1./K1))**(-2)) *
            AV*Utils.sdiag(K1**(-2))
        )

        Asub = (-1.0/dt)*dT

        Adiag = (
            (1.0/dt)*dT1 -
            DdiagGh1*diagAVk2_AVdiagK2*dK1 -
            DIV*Utils.sdiag(1./(AV*(1./K1)))*GRAD -
            Dz*diagAVk2_AVdiagK2*dK1
        )

        B = (
            DdiagGh1*diagAVk2_AVdiagK2*dKm1 +
            Dz*diagAVk2_AVdiagK2*dKm1 +
            (1.0/dt)*(dTm - dTm1)
        )

        return Asub, Adiag, B

    @Utils.timeIt
    def getResidual(self, m, hn, h, dt, bc, return_g=True):
        """Used by the root finder when going between timesteps

        Where h is the proposed value for the next time iterate (h_{n+1})
        """
        if m is not None:
            self.model = m

        DIV = self.mesh.faceDiv
        GRAD = self.mesh.cellGrad
        BC = self.mesh.cellGradBC
        AV = self.mesh.aveF2CC.T
        Dz = self.Dz

        T = self.water_retention(h)
        dT = self.water_retention.derivU(h)
        Tn = self.water_retention(hn)
        K = self.hydraulic_conductivity(h)
        dK = self.hydraulic_conductivity.derivU(h)

        aveK = 1./(AV*(1./K))

        RHS = DIV*Utils.sdiag(aveK)*(GRAD*h+BC*bc) + Dz*aveK
        if self.method == 'mixed':
            r = (T-Tn)/dt - RHS
        elif self.method == 'head':
            r = dT*(h - hn)/dt - RHS

        if not return_g:
            return r

        J = dT/dt - DIV*Utils.sdiag(aveK)*GRAD
        if self.do_newton:
            DDharmAve = Utils.sdiag(aveK**2)*AV*Utils.sdiag(K**(-2)) * dK
            J = J - DIV*Utils.sdiag(GRAD*h + BC*bc)*DDharmAve - Dz*DDharmAve

        return r, J

    @Utils.timeIt
    def Jfull(self, m=None, f=None):
        if f is None:
            f = self.fields(m)

        nn = len(f)-1
        Asubs, Adiags, Bs = list(range(nn)), list(range(nn)), list(range(nn))
        for ii in range(nn):
            dt = self.timeSteps[ii]
            bc = self.getBoundaryConditions(ii, f[ii])
            Asubs[ii], Adiags[ii], Bs[ii] = self.diagsJacobian(
                m, f[ii], f[ii+1], dt, bc
            )
        Ad = sp.block_diag(Adiags)
        zRight = Utils.spzeros(
            (len(Asubs)-1)*Asubs[0].shape[0], Adiags[0].shape[1]
        )
        zTop = Utils.spzeros(
            Adiags[0].shape[0], len(Adiags)*Adiags[0].shape[1]
        )
        As = sp.vstack((zTop, sp.hstack((sp.block_diag(Asubs[1:]), zRight))))
        A = As + Ad
        B = np.array(sp.vstack(Bs).todense())

        Ainv = self.Solver(A, **self.solverOpts)
        AinvB = Ainv * B
        z = np.zeros((self.mesh.nC, B.shape[1]))
        du_dm = np.vstack((z, AinvB))
        J = self.survey.deriv(f, du_dm_v=du_dm)  # not multiplied by v
        return J

    @Utils.timeIt
    def Jvec(self, m, v, f=None):
        if f is None:
            f = self.fields(m)

        JvC = list(range(len(f)-1))  # Cell to hold each row of the long vector

        # This is done via forward substitution.
        bc = self.getBoundaryConditions(0, f[0])
        temp, Adiag, B = self.diagsJacobian(
            m, f[0], f[1], self.timeSteps[0], bc
        )
        Adiaginv = self.Solver(Adiag, **self.solverOpts)
        JvC[0] = Adiaginv * (B*v)

        for ii in range(1, len(f)-1):
            bc = self.getBoundaryConditions(ii, f[ii])
            Asub, Adiag, B = self.diagsJacobian(
                m, f[ii], f[ii+1], self.timeSteps[ii], bc
            )
            Adiaginv = self.Solver(Adiag, **self.solverOpts)
            JvC[ii] = Adiaginv * (B*v - Asub*JvC[ii-1])

        du_dm_v = np.concatenate([np.zeros(self.mesh.nC)] + JvC)
        Jv = self.survey.deriv(f, du_dm_v=du_dm_v, v=v)
        return Jv

    @Utils.timeIt
    def Jtvec(self, m, v, f=None):
        if f is None:
            f = self.field(m)

        PTv, PTdv = self.survey.derivAdjoint(f, v=v)

        # This is done via backward substitution.
        minus = 0
        BJtv = 0
        for ii in range(len(f)-1, 0, -1):
            bc = self.getBoundaryConditions(ii-1, f[ii-1])
            Asub, Adiag, B = self.diagsJacobian(
                m, f[ii-1], f[ii], self.timeSteps[ii-1], bc
            )
            # select the correct part of v
            vpart = list(range((ii)*Adiag.shape[0], (ii+1)*Adiag.shape[0]))
            AdiaginvT = self.Solver(Adiag.T, **self.solverOpts)
            JTvC = AdiaginvT * (PTv[vpart] - minus)
            minus = Asub.T*JTvC  # this is now the super diagonal.
            BJtv = BJtv + B.T*JTvC

        return BJtv + PTdv
