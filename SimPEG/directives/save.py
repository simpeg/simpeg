import numpy as np
import matplotlib.pyplot as plt
import os
from ..regularization import (
    Sparse,
)
from ..utils import (
    validate_string,
)
from ..utils.code_utils import (
    validate_type,
)
from .base import InversionDirective


class SaveEveryIteration(InversionDirective):
    """SaveEveryIteration

    This directive saves an array at each iteration. The default
    directory is the current directory and the models are saved as
    ``InversionModel-YYYY-MM-DD-HH-MM-iter.npy``
    """

    def __init__(self, directory=".", name="InversionModel", **kwargs):
        super().__init__(**kwargs)
        self.directory = directory
        self.name = name

    @property
    def directory(self):
        """Directory to save results in.

        Returns
        -------
        str
        """
        return self._directory

    @directory.setter
    def directory(self, value):
        value = validate_string("directory", value)
        fullpath = os.path.abspath(os.path.expanduser(value))

        if not os.path.isdir(fullpath):
            os.mkdir(fullpath)
        self._directory = value

    @property
    def name(self):
        """Root of the filename to be saved.

        Returns
        -------
        str
        """
        return self._name

    @name.setter
    def name(self, value):
        self._name = validate_string("name", value)

    @property
    def fileName(self):
        if getattr(self, "_fileName", None) is None:
            from datetime import datetime

            self._fileName = "{0!s}-{1!s}".format(
                self.name, datetime.now().strftime("%Y-%m-%d-%H-%M")
            )
        return self._fileName


class SaveModelEveryIteration(SaveEveryIteration):
    """SaveModelEveryIteration

    This directive saves the model as a numpy array at each iteration. The
    default directory is the current directoy and the models are saved as
    ``InversionModel-YYYY-MM-DD-HH-MM-iter.npy``
    """

    def initialize(self):
        print(
            "SimPEG.SaveModelEveryIteration will save your models as: "
            "'{0!s}###-{1!s}.npy'".format(self.directory + os.path.sep, self.fileName)
        )

    def endIter(self):
        np.save(
            "{0!s}{1:03d}-{2!s}".format(
                self.directory + os.path.sep, self.opt.iter, self.fileName
            ),
            self.opt.xc,
        )


class SaveOutputEveryIteration(SaveEveryIteration):
    """SaveOutputEveryIteration"""

    def __init__(self, save_txt=True, **kwargs):
        super().__init__(**kwargs)

        self.save_txt = save_txt

    @property
    def save_txt(self):
        """Whether to save the output as a text file.

        Returns
        -------
        bool
        """
        return self._save_txt

    @save_txt.setter
    def save_txt(self, value):
        self._save_txt = validate_type("save_txt", value, bool)

    def initialize(self):
        if self.save_txt is True:
            print(
                "SimPEG.SaveOutputEveryIteration will save your inversion "
                "progress as: '###-{0!s}.txt'".format(self.fileName)
            )
            f = open(self.fileName + ".txt", "w")
            header = "  #     beta     phi_d     phi_m   phi_m_small     phi_m_smoomth_x     phi_m_smoomth_y     phi_m_smoomth_z      phi\n"
            f.write(header)
            f.close()

        # Create a list of each

        self.beta = []
        self.phi_d = []
        self.phi_m = []
        self.phi_m_small = []
        self.phi_m_smooth_x = []
        self.phi_m_smooth_y = []
        self.phi_m_smooth_z = []
        self.phi = []

    def endIter(self):
        phi_s, phi_x, phi_y, phi_z = 0, 0, 0, 0

        for reg in self.reg.objfcts:
            if isinstance(reg, Sparse):
                i_s, i_x, i_y, i_z = 0, 1, 2, 3
            else:
                i_s, i_x, i_y, i_z = 0, 1, 3, 5
            if getattr(reg, "alpha_s", None):
                phi_s += reg.objfcts[i_s](self.invProb.model) * reg.alpha_s
            if getattr(reg, "alpha_x", None):
                phi_x += reg.objfcts[i_x](self.invProb.model) * reg.alpha_x

            if reg.regularization_mesh.dim > 1 and getattr(reg, "alpha_y", None):
                phi_y += reg.objfcts[i_y](self.invProb.model) * reg.alpha_y
            if reg.regularization_mesh.dim > 2 and getattr(reg, "alpha_z", None):
                phi_z += reg.objfcts[i_z](self.invProb.model) * reg.alpha_z

        self.beta.append(self.invProb.beta)
        self.phi_d.append(self.invProb.phi_d)
        self.phi_m.append(self.invProb.phi_m)
        self.phi_m_small.append(phi_s)
        self.phi_m_smooth_x.append(phi_x)
        self.phi_m_smooth_y.append(phi_y)
        self.phi_m_smooth_z.append(phi_z)
        self.phi.append(self.opt.f)

        if self.save_txt:
            f = open(self.fileName + ".txt", "a")
            f.write(
                " {0:3d} {1:1.4e} {2:1.4e} {3:1.4e} {4:1.4e} {5:1.4e} "
                "{6:1.4e}  {7:1.4e}  {8:1.4e}\n".format(
                    self.opt.iter,
                    self.beta[self.opt.iter - 1],
                    self.phi_d[self.opt.iter - 1],
                    self.phi_m[self.opt.iter - 1],
                    self.phi_m_small[self.opt.iter - 1],
                    self.phi_m_smooth_x[self.opt.iter - 1],
                    self.phi_m_smooth_y[self.opt.iter - 1],
                    self.phi_m_smooth_z[self.opt.iter - 1],
                    self.phi[self.opt.iter - 1],
                )
            )
            f.close()

    def load_results(self):
        results = np.loadtxt(self.fileName + str(".txt"), comments="#")
        self.beta = results[:, 1]
        self.phi_d = results[:, 2]
        self.phi_m = results[:, 3]
        self.phi_m_small = results[:, 4]
        self.phi_m_smooth_x = results[:, 5]
        self.phi_m_smooth_y = results[:, 6]
        self.phi_m_smooth_z = results[:, 7]

        self.phi_m_smooth = (
            self.phi_m_smooth_x + self.phi_m_smooth_y + self.phi_m_smooth_z
        )

        self.f = results[:, 7]

        self.target_misfit = self.invProb.dmisfit.simulation.survey.nD / 2.0
        self.i_target = None

        if self.invProb.phi_d < self.target_misfit:
            i_target = 0
            while self.phi_d[i_target] > self.target_misfit:
                i_target += 1
            self.i_target = i_target

    def plot_misfit_curves(
        self,
        fname=None,
        dpi=300,
        plot_small_smooth=False,
        plot_phi_m=True,
        plot_small=False,
        plot_smooth=False,
    ):
        self.target_misfit = (
            np.sum([dmis.nD for dmis in self.invProb.dmisfit.objfcts]) / 2.0
        )
        self.i_target = None

        if self.invProb.phi_d < self.target_misfit:
            i_target = 0
            while self.phi_d[i_target] > self.target_misfit:
                i_target += 1
            self.i_target = i_target

        fig = plt.figure(figsize=(5, 2))
        ax = plt.subplot(111)
        ax_1 = ax.twinx()
        ax.semilogy(
            np.arange(len(self.phi_d)), self.phi_d, "k-", lw=2, label=r"$\phi_d$"
        )

        if plot_phi_m:
            ax_1.semilogy(
                np.arange(len(self.phi_d)), self.phi_m, "r", lw=2, label=r"$\phi_m$"
            )

        if plot_small_smooth or plot_small:
            ax_1.semilogy(
                np.arange(len(self.phi_d)), self.phi_m_small, "ro", label="small"
            )
        if plot_small_smooth or plot_smooth:
            ax_1.semilogy(
                np.arange(len(self.phi_d)), self.phi_m_smooth_x, "rx", label="smooth_x"
            )
            ax_1.semilogy(
                np.arange(len(self.phi_d)), self.phi_m_smooth_y, "rx", label="smooth_y"
            )
            ax_1.semilogy(
                np.arange(len(self.phi_d)), self.phi_m_smooth_z, "rx", label="smooth_z"
            )

        ax.legend(loc=1)
        ax_1.legend(loc=2)

        ax.plot(
            np.r_[ax.get_xlim()[0], ax.get_xlim()[1]],
            np.ones(2) * self.target_misfit,
            "k:",
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$\phi_d$")
        ax_1.set_ylabel(r"$\phi_m$", color="r")
        ax_1.tick_params(axis="y", which="both", colors="red")

        plt.show()
        if fname is not None:
            fig.savefig(fname, dpi=dpi)

    def plot_tikhonov_curves(self, fname=None, dpi=200):
        self.target_misfit = self.invProb.dmisfit.simulation.survey.nD / 2.0
        self.i_target = None

        if self.invProb.phi_d < self.target_misfit:
            i_target = 0
            while self.phi_d[i_target] > self.target_misfit:
                i_target += 1
            self.i_target = i_target

        fig = plt.figure(figsize=(5, 8))
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313)

        ax1.plot(self.beta, self.phi_d, "k-", lw=2, ms=4)
        ax1.set_xlim(np.hstack(self.beta).min(), np.hstack(self.beta).max())
        ax1.set_xlabel(r"$\beta$", fontsize=14)
        ax1.set_ylabel(r"$\phi_d$", fontsize=14)

        ax2.plot(self.beta, self.phi_m, "k-", lw=2)
        ax2.set_xlim(np.hstack(self.beta).min(), np.hstack(self.beta).max())
        ax2.set_xlabel(r"$\beta$", fontsize=14)
        ax2.set_ylabel(r"$\phi_m$", fontsize=14)

        ax3.plot(self.phi_m, self.phi_d, "k-", lw=2)
        ax3.set_xlim(np.hstack(self.phi_m).min(), np.hstack(self.phi_m).max())
        ax3.set_xlabel(r"$\phi_m$", fontsize=14)
        ax3.set_ylabel(r"$\phi_d$", fontsize=14)

        if self.i_target is not None:
            ax1.plot(self.beta[self.i_target], self.phi_d[self.i_target], "k*", ms=10)
            ax2.plot(self.beta[self.i_target], self.phi_m[self.i_target], "k*", ms=10)
            ax3.plot(self.phi_m[self.i_target], self.phi_d[self.i_target], "k*", ms=10)

        for ax in [ax1, ax2, ax3]:
            ax.set_xscale("linear")
            ax.set_yscale("linear")
        plt.tight_layout()
        plt.show()
        if fname is not None:
            fig.savefig(fname, dpi=dpi)


class SaveOutputDictEveryIteration(SaveEveryIteration):
    """
    Saves inversion parameters at every iteration.
    """

    # Initialize the output dict
    def __init__(self, saveOnDisk=False, **kwargs):
        super().__init__(**kwargs)
        self.saveOnDisk = saveOnDisk

    @property
    def saveOnDisk(self):
        """Whether to save the output dict to disk.

        Returns
        -------
        bool
        """
        return self._saveOnDisk

    @saveOnDisk.setter
    def saveOnDisk(self, value):
        self._saveOnDisk = validate_type("saveOnDisk", value, bool)

    def initialize(self):
        self.outDict = {}
        if self.saveOnDisk:
            print(
                "SimPEG.SaveOutputDictEveryIteration will save your inversion progress as dictionary: '###-{0!s}.npz'".format(
                    self.fileName
                )
            )

    def endIter(self):
        # regCombo = ["phi_ms", "phi_msx"]

        # if self.simulation[0].mesh.dim >= 2:
        #     regCombo += ["phi_msy"]

        # if self.simulation[0].mesh.dim == 3:
        #     regCombo += ["phi_msz"]

        # Initialize the output dict
        iterDict = {}

        # Save the data.
        iterDict["iter"] = self.opt.iter
        iterDict["beta"] = self.invProb.beta
        iterDict["phi_d"] = self.invProb.phi_d
        iterDict["phi_m"] = self.invProb.phi_m

        # for label, fcts in zip(regCombo, self.reg.objfcts[0].objfcts):
        #     iterDict[label] = fcts(self.invProb.model)

        iterDict["f"] = self.opt.f
        iterDict["m"] = self.invProb.model
        iterDict["dpred"] = self.invProb.dpred

        for reg in self.reg.objfcts:
            if isinstance(reg, Sparse):
                for reg_part, norm in zip(reg.objfcts, reg.norms):
                    reg_name = f"{type(reg_part).__name__}"
                    if hasattr(reg_part, "orientation"):
                        reg_name = reg_part.orientation + " " + reg_name
                    iterDict[reg_name + ".irls_threshold"] = reg_part.irls_threshold
                    iterDict[reg_name + ".norm"] = norm

        # Save the file as a npz
        if self.saveOnDisk:
            np.savez("{:03d}-{:s}".format(self.opt.iter, self.fileName), iterDict)

        self.outDict[self.opt.iter] = iterDict


class SimilarityMeasureSaveOutputEveryIteration(SaveEveryIteration):
    """
    SaveOutputEveryIteration for Joint Inversions.
    Saves information on the tradeoff parameters, data misfits, regularizations,
    coupling term, number of CG iterations, and value of cost function.
    """

    header = None
    save_txt = True
    betas = None
    phi_d = None
    phi_m = None
    phi_sim = None
    phi = None

    def initialize(self):
        if self.save_txt is True:
            print(
                "CrossGradientSaveOutputEveryIteration will save your inversion "
                "progress as: '###-{0!s}.txt'".format(self.fileName)
            )
            f = open(self.fileName + ".txt", "w")
            self.header = "  #          betas            lambda         joint_phi_d                joint_phi_m            phi_sim       iterCG     phi    \n"
            f.write(self.header)
            f.close()

        # Create a list of each
        self.betas = []
        self.lambd = []
        self.phi_d = []
        self.phi_m = []
        self.phi = []
        self.phi_sim = []

    def endIter(self):
        self.betas.append(["{:.2e}".format(elem) for elem in self.invProb.betas])
        self.phi_d.append(["{:.3e}".format(elem) for elem in self.invProb.phi_d_list])
        self.phi_m.append(["{:.3e}".format(elem) for elem in self.invProb.phi_m_list])
        self.lambd.append("{:.2e}".format(self.invProb.lambd))
        self.phi_sim.append(self.invProb.phi_sim)
        self.phi.append(self.opt.f)

        if self.save_txt:
            f = open(self.fileName + ".txt", "a")
            i = self.opt.iter
            f.write(
                " {0:2d}  {1}  {2}  {3}  {4}  {5:1.4e}  {6:d}  {7:1.4e}\n".format(
                    i,
                    self.betas[i - 1],
                    self.lambd[i - 1],
                    self.phi_d[i - 1],
                    self.phi_m[i - 1],
                    self.phi_sim[i - 1],
                    self.opt.cg_count,
                    self.phi[i - 1],
                )
            )
            f.close()

    def load_results(self):
        results = np.loadtxt(self.fileName + str(".txt"), comments="#")
        self.betas = results[:, 1]
        self.lambd = results[:, 2]
        self.phi_d = results[:, 3]
        self.phi_m = results[:, 4]
        self.phi_sim = results[:, 5]
        self.f = results[:, 7]
