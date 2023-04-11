import numpy as np
import os
from matplotlib import pyplot as plt
from discretize import TensorMesh

from SimPEG import maps
from SimPEG.electromagnetics import time_domain as tdem
from SimPEG.electromagnetics.utils.em1d_utils import plot_layer
import libaarhusxyz
import pandas as pd

import numpy as np
from scipy.spatial import cKDTree, Delaunay
import os, tarfile
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from discretize import TensorMesh, SimplexMesh
#from pymatsolver import PardisoSolver

from SimPEG.utils import mkvc
from SimPEG import (
    maps, data, data_misfit, inverse_problem, regularization, optimization,
    directives, inversion, utils
    )

from SimPEG.utils import mkvc
import SimPEG.electromagnetics.time_domain as tdem
import SimPEG.electromagnetics.utils.em1d_utils
from SimPEG.electromagnetics.utils.em1d_utils import get_2d_mesh,plot_layer, get_vertical_discretization_time
from SimPEG.regularization import LaterallyConstrained, RegularizationMesh

import scipy.stats


class XYZSystem(object):
    """This is a base class for system descriptions for moving EM
    acquisition platforms such as AEM (aerial EM), TTEM (towed time
    domain EM). The base assumption and simplification provided by
    this class is that the setup of receiver(s) and transmitter(s) is
    independent of the data, save for their absolute positions (but
    relative positions are still independent from data).

    Each subclass of this class, describes a particular setup of
    transmitters, receivers including dipole moments, waveforms,
    positions etc, as well as inversion parameters.

    A subclass can then be instantiated together with an XYZ file
    structure with raw data read using libaarhusxyz.XYZ(), to form an
    invertible object, or with a model read using the same library to
    do forward modelling.

    Basic usage:

    ```
    class MySystem(XYZSystem):
        def make_system(self, idx, location, times):
            # Your code here

    inv = MySystem(libaarhusxyz.XYZ("measured.xyz"))
    sparse, l2 = inv.invert()
    sparse.dump("sparse.xyz")
    l2.dump("l2.xyz")
    ```

    Not that any class level attribute, such as `n_layer`, can be
    overridden by a parameter when instantiating the class, e.g. 

    ```
    MySystem(libaarhusxyz.XYZ("measured.xyz"), n_layer=10)
    ```
    """
    
    n_layer=30
    start_res=100
    
    parallel = True
    n_cpu=3
    
    def __init__(self, xyz, **kw):
        self.xyz = xyz
        self.options = kw
    
    def __getattribute__(self, name):
        options = object.__getattribute__(self, "options")
        if name in options: return options[name]
        return object.__getattribute__(self, name)
        
    def make_system(self, idx, location, times):
        """This method should return a list of instances of some
        SimPEG.survey.BaseSrc subclass, such as
        SimPEG.electromagnetics.time_domain.sources.MagDipole.

        idx is an index into self.xyz.flightlines
        location is a tuple (x, y, z) corresponding to the coordinates
            found at that index in self.xyz.flightlines
        times is whatever is returned by self.times, typically a list
            of gate times, or for a multi channel system, a tuple of
            such lists, one for each channel.
        """
        raise NotImplementedError("You must subclass XYZInversion and override make_system() with your own method!")

    @property
    def times(self):
        return np.array(self.xyz.model_info['gate times for channel 1'])

    @property
    def data_array(self):
        return self.xyz.dbdt_ch1gt.values.flatten()

    @property
    def uncert_array(self):
        return self.xyz.dbdt_std_ch1gt.values.flatten()

    def make_thicknesses(self):
        if "dep_top" in self.xyz.layer_params:
            return np.diff(self.xyz.layer_params["dep_top"].values)
        return get_vertical_discretization_time(
            self.times,
            sigma_background=0.1,
            n_layer=self.n_layer-1
        )
        
    def make_survey(self):
        times = self.times
        systems = [
            self.make_system(
                idx,
                self.xyz.flightlines.loc[
                    idx, [self.xyz.x_column, self.xyz.y_column, self.xyz.alt_column]
                ].astype(float).values,
                times)
            for idx in range(0, len(self.xyz.flightlines))]
        return tdem.Survey([
            source
            for sources in systems
            for source in sources])

    def n_param(self, thicknesses):
        return (len(thicknesses)+1)*len(self.xyz.flightlines)
    
    def make_simulation(self, survey, thicknesses):
        return tdem.Simulation1DLayeredStitched(
            survey=survey,
            thicknesses=thicknesses,
            sigmaMap=maps.ExpMap(nP=self.n_param(thicknesses)), 
            parallel=self.parallel,
            n_cpu=self.n_cpu,
            n_layer=self.n_layer)
    
    def make_data(self, survey):
        return data.Data(
            survey,
            dobs=self.data_array,
            standard_deviation=self.uncert_array)
    
    def make_misfit_weights(self, thicknesses):
        return 1./self.uncert_array

    def make_misfit(self, thicknesses):
        survey = self.make_survey()

        dmis = data_misfit.L2DataMisfit(
            simulation=self.make_simulation(survey, thicknesses),
            data=self.make_data(survey))
        dmis.W = self.make_misfit_weights(thicknesses)
        return dmis
    
    alpha_s = 1e-10
    alpha_r = 1.
    alpha_z = 1.
    def make_regularization(self, thicknesses):
        if False:
            assert False, "LCI is currently broken"
            hz = np.r_[thicknesses, thicknesses[-1]]
            reg = LaterallyConstrained(
                get_2d_mesh(len(self.xyz.flightlines), hz),
                mapping=maps.IdentityMap(nP=self.n_param(thicknesses)),
                alpha_s = 0.01,
                alpha_r = 1.,
                alpha_z = 1.)
            # reg.get_grad_horizontal(self.xyz.flightlines[["x", "y"]], hz, dim=2, use_cell_weights=True)
            # ps, px, py = 0, 0, 0
            # reg.norms = np.c_[ps, px, py, 0]
            reg.mref = np.log(np.ones(self.n_param(thicknesses)) * 1/self.start_res)
            # reg.mrefInSmooth = False
            return reg
        else:
            coords = self.xyz.flightlines[[self.xyz.x_column, self.xyz.y_column]].astype(float).values
            # FIXME: Triangulation fails if all coords are on a line, as in a typical synthetic case...
            # Ideally, shouldn't this degrade gracefully to an LCI?
            coords[:,1] += np.random.randn(len(coords)) * 5 #1e-10
            tri = Delaunay(coords)
            hz = np.r_[thicknesses, thicknesses[-1]]

            mesh_radial = SimplexMesh(tri.points, tri.simplices)
            mesh_vertical = SimPEG.electromagnetics.utils.em1d_utils.set_mesh_1d(hz)
            mesh_reg = [mesh_radial, mesh_vertical]
            n_param = int(mesh_radial.n_nodes * mesh_vertical.nC)
            reg_map = SimPEG.maps.IdentityMap(nP=n_param)    # Mapping between the model and regularization
            reg = SimPEG.regularization.LaterallyConstrained(
                mesh_reg, mapping=reg_map,
                alpha_s = self.alpha_s,
                alpha_r = self.alpha_r,
                alpha_z = self.alpha_z,
            )
            reg.mref = np.log(np.ones(self.n_param(thicknesses)) * 1/self.start_res)
            return reg
            
    def make_directives(self):
        return [
            directives.BetaEstimate_ByEig(beta0_ratio=10),
            SimPEG.directives.BetaSchedule(coolingFactor=2, coolingRate=1),
            SimPEG.directives.TargetMisfit()

#            directives.SaveOutputEveryIteration(save_txt=False),
            # directives.Update_IRLS(
            #     max_irls_iterations=30,
            #     minGNiter=1,
            #     fix_Jmatrix=True,
            #     f_min_change = 1e-3,
            #     coolingRate=1),
            # directives.UpdatePreconditioner()

        ]

    def make_optimizer(self):
        return optimization.InexactGaussNewton(maxIter = 40, maxIterCG=20)
    
    def make_inversion(self):
        thicknesses = self.make_thicknesses()

        return inversion.BaseInversion(
            inverse_problem.BaseInvProblem(
                self.make_misfit(thicknesses),
                self.make_regularization(thicknesses),
                self.make_optimizer()),
            self.make_directives())

    def make_forward(self):
        return self.make_simulation(self.make_survey(), self.make_thicknesses())
        
    def inverted_model_to_xyz(self, model, thicknesses):
        xyzsparse = libaarhusxyz.XYZ()
        xyzsparse.model_info.update(self.xyz.model_info)
        xyzsparse.flightlines = self.xyz.flightlines
        xyzsparse.layer_data["resistivity"] = 1 / np.exp(pd.DataFrame(
            model.reshape((len(self.xyz.flightlines),
                           len(model) // len(self.xyz.flightlines)))))

        dep_top = np.cumsum(np.concatenate(([0], thicknesses)))
        dep_bot = np.concatenate((dep_top[1:], [np.inf]))

        xyzsparse.layer_data["dep_top"] = pd.DataFrame(np.meshgrid(dep_top, self.xyz.flightlines.index)[0])
        xyzsparse.layer_data["dep_bot"] = pd.DataFrame(np.meshgrid(dep_bot, self.xyz.flightlines.index)[0])

        return xyzsparse
    
    def invert(self):
        """Invert the data from the XYZ file using this system description and
        inversion parameters.

        Returns a sparse model and an l2 (smooth model), both in xyz format.
        """
        
        self.inv = self.make_inversion()
        
        recovered_model = self.inv.run(self.inv.invProb.reg.mref)
        
        thicknesses = self.inv.invProb.dmisfit.simulation.thicknesses
        
        self.sparse = self.inverted_model_to_xyz(recovered_model, thicknesses)
        self.l2 = None
        if hasattr(self.inv.invProb, "l2model"):
            self.l2 = self.inverted_model_to_xyz(self.inv.invProb.l2model, thicknesses)
        
        return self.sparse, self.l2

    def forward_data_to_xyz(self, resp):
        xyzresp = libaarhusxyz.XYZ()
        xyzresp.flightlines = self.xyz.flightlines
        xyzresp.layer_data = {
            "dbdt_ch1gt": pd.DataFrame(resp)
        }

        # XYZ assumes all receivers have the same times
        xyzresp.model_info["gate times for channel 1"] = list(self.times)

        return xyzresp
    
    def forward(self):
        """Does a forward modelling of the model in the XYZ file using
        this system description. Returns data in xyz format."""
        # self.inv.invProb.dmisfit.simulation
        self.sim = self.make_forward()

        model_cond=np.log(1/self.xyz.resistivity.values)
        resp = self.sim.dpred(model_cond.flatten())

        resp = resp.reshape((len(self.xyz.flightlines), len(resp) // len(self.xyz.flightlines)))
        return self.forward_data_to_xyz(resp)
