from SimPEG import *
from SimPEG.EM import mu_0, FDEM, Analytics
from SimPEG.EM.Utils import omega
try:
    from pymatsolver import MumpsSolver as Solver
except ImportError:
    try:
        from pymatsolver import PardisoSolver as Solver
    except ImportError:
        Solver = SolverLU
import matplotlib.pyplot as plt
import time

np.random.seed(42)


class PrimSecCasingExample(object):

    NAME = 'PrimSec_5e6Casing_50Mu_05Hz_LargeCondBody'

    solvePrimary = True  # else JPRIMARY needs to be provided

    # -------------- SETUP MODEL PARAMS ---------------------------- #

    sigmaair = 1e-8  # air
    sigmaback = 1e-2  # background

    sigmacasing = 5.5e6  # casing
    sigmainside = 1  # inside the casing
    mucasing = 50  # casing permeability

    casing_l = 1000   # length of the casing
    casing_d = 10e-2  # 10cm diameter casing
    casing_t = 1e-2   # 1cm thickness

    # layer
    sigmalayer = 1./10.
    layer_z = np.r_[-1000., -900.]

    # 3D body
    sigmablock =  2.
    block_x = np.r_[75., 475.]
    block_y = np.r_[-125, 125.]
    block_z = layer_z

    # Survey Params
    freqs = np.r_[0.5]  # frequencies 0.5
    dsz = -950.  # down-hole z source location

    src_a = np.r_[0., 0., dsz]
    src_b = np.r_[1e4, 0.,  0.]  # return electrode is a ring, 200

    def __init__(self):
        # translate casing parameters to radial distances to outer casing wall,
        # inner casing wall, and casing length to z locations
        self.casing_r = self.casing_d/2.
        self.casing_a = self.casing_r - self.casing_t/2.  # inner radius
        self.casing_b = self.casing_r + self.casing_t/2.  # outer radius
        self.casing_z = np.r_[-self.casing_l, 0.]

        # Display skin depth so we can ensure our mesh goes further.
        print('\nSkin Depth: {}'.format(
                [(500./np.sqrt(self.sigmaback*_)) for _ in self.freqs])
              )

    @property
    def mtrue(self):
        # -------------- Model --------------------------------- #
        # This is the model we are using to compute the sensitivity. Each of
        # these parameters would be considered unknown in an inversion. This
        # model is of a parametrized block in a layer
        #
        #   +--------------------------------------+
        #   |                                      |
        #   |                                      |
        #   |            background                |
        #   |                                      |
        #   |                                      |
        #   +--------------------+-------+---------+
        #   |                    |       |         |
        #   |      layer         | block |         |
        #   |                    |       |         |
        #   +--------------------+-------+---------+
        #   |                                      |
        #   |                                      |
        #   |            background                |
        #   |                                      |
        #   |                                      |
        #   +--------------------------------------+

        return np.hstack(np.r_[np.log(self.sigmaback),  # value in the background
                               np.log(self.sigmalayer),  # value in the layer
                               np.log(self.sigmablock),  # value in the block
                               self.layer_z.mean(),  # layer center
                               self.layer_z[1] - self.layer_z[0],  # layer thickness
                               self.block_x.mean(),  # block x_0
                               self.block_y.mean(),  # block y_0
                               self.block_x[1] - self.block_x[0],  # block dx
                               self.block_y[1] - self.block_y[0]  # block dy
                               ])

    # ----------------------------------------------------------------- #
    # -------------- PRIMARY PROBLEM ---------------------------------- #
    # ----------------------------------------------------------------- #

    @property
    def meshp(self):
        if getattr(self, '_meshp', None) is None:
            # -------------- Mesh Parameters ------------------ #
            # x-direction
            csx1, csx2 = 2.5e-3, 25. # fine cells near well bore
            pfx1, pfx2 = 1.3, 1.4  # padding factors: fine -> uniform, pad to infinity
            ncx1 = np.ceil(self.casing_b/csx1+2)  # number of fine cells (past casing wall)
            dx2 = 1000.  # uniform mesh out to here
            npadx2 = 21  # padding out to infinity

            # z-direction
            csz = 0.05  # finest z-cells
            nza = 10  # number of fine cells above air-earth interface
            pfz = pfx2 # padding factor in z-direction

            # ------------- Assemble the Cyl Mesh ------------- #
            # pad nicely to second cell size
            npadx1 = np.floor(np.log(csx2/csx1) / np.log(pfx1))
            hx1a = Utils.meshTensor([(csx1, ncx1)])
            hx1b = Utils.meshTensor([(csx1, npadx1, pfx1)])
            dx1 = sum(hx1a)+sum(hx1b)
            dx1 = np.floor(dx1/csx2)
            hx1b *= (dx1*csx2 - sum(hx1a))/sum(hx1b)

            # second chunk of mesh
            ncx2 = np.ceil((dx2 - dx1)/csx2)
            hx2a = Utils.meshTensor([(csx2, ncx2)])
            hx2b = Utils.meshTensor([(csx2, npadx2, pfx2)])
            hx = np.hstack([hx1a, hx1b, hx2a, hx2b])

            # cell size, number of core cells, number of padding cells in the
            # x-direction
            ncz, npadzu, npadzd = np.int(np.ceil(np.diff(self.casing_z)[0]/csz))+10, 43, 43

            # vector of cell widths in the z-direction
            hz = Utils.meshTensor([(csz, npadzd, -pfz), (csz, ncz), (csz, npadzu, pfz)])

            # primary mesh
            self._meshp = Mesh.CylMesh([hx, 1., hz], [0., 0., -np.sum(hz[:npadzu+ncz-nza])])

        return self._meshp

    @property
    def indActivePrimary(self):
        return self.meshp.gridCC[:, 2] <= 0.  # air cells

    @property
    def projectionMapPrimary(self):
        return Maps.Projection(np.r_[0, 1, 2, 3], np.r_[0, 1, 3, 4], (4, 9))

    @property
    def primaryMapping(self):
        # Setup Pimary Maps:
        # we want to simulate on a physical property model that
        # consists of casing in a layered background. Air cells are included.
        # Our "model", that we are considering when computing the sensitivity,
        # consists of the layered background and block, so the casing and air cells
        # are inactive parts of the model and need to be appropriately injected
        # during the construction of the primary model

        if getattr(self, '_primaryMapping', None) is None:

            # inject parameters we want to invert for into the full casing model


            valInactive = np.r_[np.log(self.sigmacasing),  # log conductivity of the casing
                                np.log(self.sigmainside),  # log conductivity of fluid inside casing
                                self.casing_r,  # radius of the casing (to its center)
                                self.casing_t,  # casing thickness
                                self.casing_z[0],  # bottom of casing (at depth)
                                self.casing_z[1]   # top of casing (at surface)
                                ]

            # inject casing parameters so they are included in the construction of the
            # layered background + casing
            injectCasingParams = Maps.InjectActiveCells(
                None, indActive=np.r_[0, 1, 4, 5], valInactive=valInactive,
                nC=10
                )

            # maps a list of casing parameters to the cyl mesh (below the
            # subsurface)
            paramMapPrimary = Maps.ParametrizedCasingAndLayer(
                self.meshp, indActive=self.indActivePrimary, slopeFact=1e4
                )

            # inject air cells
            injActMapPrimary = Maps.InjectActiveCells(
                self.meshp, self.indActivePrimary, np.log(self.sigmaair)
                )

            # map from log conductivity to conductivity
            expMapPrimary = Maps.ExpMap(self.meshp)

            # assemble the primary mapping
            primaryMapping = (
                expMapPrimary *  # log(sigma) --> sigma
                injActMapPrimary *  # log(sigma) below surface --> include air
                paramMapPrimary *  # parametric --> casing + layered earth
                injectCasingParams *  # parametric layered earth --> parametric layered earth + casing
                self.projectionMapPrimary  # grab relevant parameters from full model (eg. ignore block)
                )

            self._paramMapPrimary = paramMapPrimary
            self._primaryMapping = primaryMapping

        return self._primaryMapping

    @property
    def muModel(self):
        # Mu Model
        # here, we want to consider variable magnetic permeability in the
        # simulation. The only permeable item in the domain is the casing.
        if getattr(self, '_muModel', None) is None:
            if getattr(self, '_paramMapPrimary', None) is None:
                self.primaryMapping

            muMap = (Maps.InjectActiveCells(
                        self.meshp, self.indActivePrimary, mu_0) *
                     self._paramMapPrimary)

            muModel = muMap * np.hstack(
                np.r_[mu_0,  # val Background
                      mu_0,  # val Layer
                      mu_0*self.mucasing,  # val Casing
                      mu_0,  # val inside Casing
                      self.layer_z.mean(),  # layer center
                      self.layer_z[1] - self.layer_z[0], # layer thickness
                      self.casing_r,  # casing radius
                      self.casing_t,  # casing thickness
                      self.casing_z[0],  # casing bottom
                      self.casing_z[1]  # casing top
                      ]
                )
            self._muModel = muModel
        return self._muModel

    # --------------- CONSTRUCT SOURCE -------------------------------- #
    def setupPrimarySource(self, plotIt=False):

        # Construct a downhole source that is coupled to the casing
        meshp = self.meshp
        src_a = self.src_a
        src_b = self.src_b
        casing_a = self.casing_a

        # downhole source
        dg_x = np.zeros(meshp.vnF[0], dtype=complex)
        dg_y = np.zeros(meshp.vnF[1], dtype=complex)
        dg_z = np.zeros(meshp.vnF[2], dtype=complex)

        # vertically directed wire in borehole
        # go through the center of the well
        dgv_indx = (meshp.gridFz[:, 0] < meshp.hx.min())
        dgv_indz = ((meshp.gridFz[:, 2] >= src_a[2])
                    & (meshp.gridFz[:, 2] <= src_b[2]))
        dgv_ind = dgv_indx & dgv_indz
        dg_z[dgv_ind] = -1.

        # couple to the casing downhole - top part
        dgh_indx = meshp.gridFx[:, 0] <= casing_a + meshp.hx.min()*2

        # couple to the casing downhole - bottom part
        dgh_indz2 = ((meshp.gridFx[:, 2] <= src_a[2])  &
                     (meshp.gridFx[:, 2] > src_a[2] - meshp.hz.min()))
        dgh_ind2 = dgh_indx & dgh_indz2
        dg_x[dgh_ind2] = 1.

        # horizontally directed wire
        sgh_indx = (meshp.gridFx[:, 0] <= src_b[0])
        sgh_indz = ((meshp.gridFx[:, 2] > meshp.hz.min())
                    & (meshp.gridFx[:, 2] < 2*meshp.hz.min()))
        sgh_ind = sgh_indx & sgh_indz
        dg_x[sgh_ind] = -1.

        # return electrode
        sgv_indx = ((meshp.gridFz[:, 0] > src_b[0]*0.9)
                    & (meshp.gridFz[:, 0] < src_b[0]*1.1))
        sgv_indz = ((meshp.gridFz[:, 2] >= -meshp.hz.min())
                    & (meshp.gridFz[:, 2] < 2*meshp.hz.min()))
        sgv_ind = sgv_indx & sgv_indz
        dg_z[sgv_ind] = 1.

        # assemble the source (downhole grounded primary)
        dg = np.hstack([dg_x, dg_y, dg_z])
        dg_p = [FDEM.Src.RawVec_e([], _, dg/meshp.area) for _ in self.freqs]

        # if plotIt:
        #     # Plot the source to make sure the path is infact connected

        #     fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        #     meshp.plotGrid(ax=ax)
        #     ax.plot(meshp.gridFz[dgv_ind, 0], meshp.gridFz[dgv_ind, 2], 'rd')
        #     ax.plot(meshp.gridFx[dgh_ind2, 0], meshp.gridFx[dgh_ind2, 2], 'rd')
        #     ax.plot(meshp.gridFz[sgv_ind, 0], meshp.gridFz[sgv_ind, 2], 'rd')
        #     ax.plot(meshp.gridFx[sgh_ind, 0], meshp.gridFx[sgh_ind, 2], 'rd')

        #     ax.set_title('downhole casing source on mesh')

        #     ax.set_xlim([0, 1.1e4])
        #     ax.set_ylim([-1100., 0.5])

        #     plt.show()

        return dg_p

    def plotPrimaryMesh(self):
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        self.meshp.plotGrid(ax=ax)
        plt.title('Cyl Mesh')
        return ax

    def plotPrimaryProperties(self):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        f = self.meshp.plotImage(
                self.muModel/mu_0, ax=ax[0],
                pcolorOpts={'cmap':plt.get_cmap('viridis')}, grid=False
                )
        plt.colorbar(f[0], ax=ax[0])
        ax[0].set_xlim([0, 1.])
        ax[0].set_ylim([-1.5e3, 500])
        ax[0].set_title('mu_r')

        f = self.meshp.plotImage(
            np.log10(self.primaryMapping*self.mtrue), ax=ax[1],
            pcolorOpts={'cmap':plt.get_cmap('viridis')}, grid=False
                     )
        plt.colorbar(f[0], ax=ax[1])
        ax[1].set_xlim([0, 1.])
        ax[1].set_ylim([-1.5e3, 500])
        ax[1].set_title('log10 sigma')

        plt.tight_layout()
        plt.show()
        return ax

    # -------------- MESH  -------------------------------------------- #

    def setupPrimaryProblem(self, mapping=None, muModel=mu_0, plotIt=False):

        # define a custom prop map to include variable mu that we are not inverting
        # for
        class CasingEMPropMap(Maps.PropMap):

            sigma = Maps.Property("Electrical Conductivity", defaultInvProp=True,
                                  propertyLink=('rho', Maps.ReciprocalMap))
            mu = Maps.Property("Inverse Magnetic Permeability", defaultVal=muModel,
                               propertyLink=('mui', Maps.ReciprocalMap))

            rho = Maps.Property("Electrical Resistivity",
                                propertyLink=('sigma', Maps.ReciprocalMap))
            mui = Maps.Property("Inverse Magnetic Permeability",
                                defaultVal=1./muModel,
                                propertyLink=('mu', Maps.ReciprocalMap))

        if mapping is None:
            mapping = Maps.IdentityMap(meshp)

        FDEM.Problem3D_h.PropMap = CasingEMPropMap
        primaryProblem = FDEM.Problem3D_h(self.meshp, mapping=mapping)
        # primaryProblem.PropMap = CasingEMPropMap
        primaryProblem.Solver = Solver
        return primaryProblem

    def setupSecondaryProblem(self, mapping=None):

        if mapping is None:
            mapping = [('sigma', Maps.IdentityMap(self.meshs))]
        sec_problem = FDEM.Problem3D_e(self.meshs, mapping=mapping)
        sec_problem.Solver = Solver
        return sec_problem

    def setupSecondarySurvey(self, primaryProblem, primarySurvey,
                             map2meshSecondary):

        # -------------- PROBLEM and SURVEY ---------------------------- #
        rxlocs = Utils.ndgrid([np.linspace(-2050, 2050, 41),
                               np.linspace(-2050, 2050, 41),
                               np.r_[-1]])

        rx_x = FDEM.Rx.Point_e(rxlocs, orientation='x', component='real')
        rx_y = FDEM.Rx.Point_e(rxlocs, orientation='y', component='real')

        RxList = [rx_x, rx_y]

        sec_src = [FDEM.Src.PrimSecMappedSigma(RxList, freq, primaryProblem,
                                               primarySurvey,
                                               map2meshSecondary=map2meshSecondary)
                   for freq in self.freqs]

        return FDEM.Survey(sec_src)

    def solveSecondary(sec_problem, sec_survey, m, plotIt=False):

        if not sec_problem.ispaired:
            sec_problem.pair(sec_survey)

        # -------------- SOLVE ---------------------------- #

        print 'Solving Secondary'
        t0 = time.time()
        fields = sec_problem.fields(m)
        dpred = sec_survey.dpred(m, f=fields)
        t1 = time.time()
        print '   secondary time ', t1-t0

        return fields, dpred

    # ----------------------------------------------------------------- #
    # -------------- SECONDARY PROBLEM -------------------------------- #
    # ----------------------------------------------------------------- #

    # -------------- MESH  -------------------------------------------- #


    @property
    def meshs(self):
        if getattr(self, '_meshs', None) is None:
            csx, ncx, npadx = 50, 21, 12
            csy, ncy, npady = 50, 21, 12
            csz, ncz, npadz = 25, 40, 14
            pf = 1.5

            hx = Utils.meshTensor([(csx, npadx, -pf), (csx, ncx), (csx, npadx, pf)])
            hy = Utils.meshTensor([(csy, npady, -pf), (csy, ncy), (csy, npady, pf)])
            hz = Utils.meshTensor([(csz, npadz, -pf), (csz, ncz), (csz, npadz, pf)])

            x0 = np.r_[-hx.sum()/2., -hy.sum()/2., -hz[:npadz+ncz].sum()]
            meshs = Mesh.TensorMesh([hx, hy, hz], x0=x0)

            self._meshs = meshs
        return self._meshs

    @property
    def indActive(self):
        return self.meshs.gridCC[:, 2] <= 0.  # air cells

    @property
    def injActMap(self):
        return Maps.InjectActiveCells(
            self.meshs, self.indActive, np.log(self.sigmaair)
            )

    @property
    def expMap(self):
        return Maps.ExpMap(self.meshs)

    @property
    def mapping(self):
        # secondary mapping
        # here, we construct the parametric mapping to take the parameters
        # describing the block in a layered space and map it to a conductivity
        # model on our mesh
        paramMap = Maps.ParametrizedBlockInLayer(
            self.meshs, indActive=self.indActive
            )

        mapping = (self.expMap *  # log sigma --> sigma
                   self.injActMap *  # inject air cells
                   paramMap)  # block in a layered space (subsurface)

    @property
    def primaryMap2meshs(self):
        # map the primary model to the secondary mesh (layer without the block)
        paramMapPrimaryMeshs = Maps.ParametrizedLayer(
            self.meshs, indActive=self.indActive
            )

        # primary map to the secondary mesh
        return (self.expMap *  # log sigma --> sigma
                self.injActMap *  # include air cells
                paramMapPrimaryMeshs *  # parametrized layer
                self.projectionMapPrimary)  # grab correct indices

    def run(self, plotIt=False, runTests=False, reRun=False, verbose=False,
            savePrimaryFields=False):

        # ---------------- Primary Mesh --------------------------------- #
        if verbose is True:
            print('Cyl Mesh Extent xmax: {},: zmin: {}, zmax: {}'.format(
                    self.meshp.vectorCCx.max(), self.meshp.vectorCCz.min(),
                    self.meshp.vectorCCz.max()
                      ))

        # -------------- Primary --------------------------------- #
        srcList = self.setupPrimarySource(plotIt)  # put primary source on mesh
        primarySurvey = FDEM.Survey(srcList)

        # Test the derivs on the primary mapping
        if runTests is True:
            self.primaryMapping.test()

        # Plot the Primary Model
        if plotIt is True:
            # self.plotPrimaryMesh() # plot the mesh
            self.plotPrimaryProperties() # plot mu, sigma

        # Primary Problem
        primaryProblem = self.setupPrimaryProblem(
                                             mapping=self.primaryMapping,
                                             muModel=self.muModel
                                             )

        primaryProblem.pair(primarySurvey)

        if savePrimaryFields is True:
            print('solving primary')
            t0 = time.time()
            primfields = primaryProblem.fields(mtrue)
            np.save('primaryfields_' + NAME, primfields[:, :])
            t1 = time.time()
            print('   fields ', t1-t0)
            print('   saved %s' % 'primaryfields_' + NAME)


        # -------------- Secondary --------------------------------- #
        # Construct Secondary Mesh
        # meshs = setupCartMesh(plotIt)
        # if plotIt is True:
        #     self.meshs.plotGrid()

        if verbose is True:
            print('Secondary Mesh ... ')
            print(' xmin, xmax, zmin, zmax: ', self.meshs.vectorCCx.min(),
                  self.meshs.vectorCCx.max(), self.meshs.vectorCCy.min(),
                  self.meshs.vectorCCy.max(), self.meshs.vectorCCz.min(),
                  self.meshs.vectorCCz.max())
            print(' nC, vnC', self.meshs.nC, self.meshs.vnC)



        # make a copy so we can simulate the background (without the conductive
        # block)
        mback = self.mtrue.copy()
        mback[2] = np.log(self.sigmalayer)

        # Secondary Problem and Survey
        sec_problem = self.setupSecondaryProblem(mapping=self.mapping)

        sec_survey = self.setupSecondarySurvey(
                                          primaryProblem,
                                          primarySurvey,
                                          self.primaryMap2meshs)
        sec_problem.pair(sec_survey)

        # layered earth only (background)
        background_problem = self.setupSecondaryProblem(
            mapping=primaryMap2meshs
            )
        background_survey = self.setupSecondarySurvey(
                                                 primaryProblem,
                                                 primarySurvey,
                                                 self.primaryMap2meshs)
        background_problem.pair(background_survey)


        # -------------- Test the sensitivity --------------------------------- #
        if runTests:
            x0 = mtrue

            # Test Block Model
            fun = lambda x: [sec_survey.dpred(x),
                             lambda x: sec_problem.Jvec(mtrue, x)]
            Tests.checkDerivative(fun, self.mtrue, num=2, plotIt=False)

        # # -------------- Calculate Fields --------------------------------- #
        # # Background
        # t0 = time.time()
        # print('solving background ... ')
        # fieldsback, dpredback = solveSecondary(background_problem,
        #                                        background_survey,
        #                                        mtrue)
        # t1 = time.time()
        # print('   dpred_back {}'.format(t1-t0))

        # if savePrimaryFields:
        #     np.save('dpred_' + NAME + '_back', dpredback)
        #     np.save('fields_' + NAME + '_back', fieldsback[:, :])

        #     print('   saved {}'.format(NAME + '_back'))

        # # with Block
        # t0 = time.time()
        # print('solving with block ... ')
        # fields, dpred = solveSecondary(sec_problem, sec_survey, mtrue)
        # print('   dpred {}'.format(t1-t0))
        # if savePrimaryFields:
        #     np.save('dpred_' + NAME, dpred)
        #     np.save('fields_' + NAME, fields[:, :])

        # t1 = time.time()

        # print('   saved {}'.format(NAME))


        # -------------- Calculate J --------------------------------- #

        # # Calculate J for background
        # t0 = time.time()

        # print 'starting J back'
        # J = []
        # for i in range(len(mtrue)):
        #     ei = np.zeros_like(mtrue)
        #     ei[i] = 1.
        #     J.append(background_problem.Jvec(mtrue, ei, f=fieldsback))

        # J = np.vstack(J)

        # t1 = time.time()
        # print '   J ', t1-t0

        # np.save('J_'+ NAME  + '_back', J)
        # print '   saved %s' % 'J_' + NAME + '_back'

        # # Calculate J with block
        # print 'starting J with block'
        # t0 = time.time()

        # J = []
        # for i in range(len(mtrue)):
        #     ei = np.zeros_like(mtrue)
        #     ei[i] = 1.
        #     J.append(sec_problem.Jvec(mtrue, ei, f=fields))

        # J = np.vstack(J)

        # t1 = time.time()
        # print '   J ', t1-t0

        # np.save('J_'+ NAME, J)
        # print '   saved %s' % 'J_' + NAME

def downloadStoredResults(basePath=None):
    import os
    from SimPEG.Utils.io_utils import remoteDownload
    # download the results from where they are stored on google app engine
    url = 'https://storage.googleapis.com/simpeg/papers/Heagyetal2016/'
    cloudfiles = ['dpred_PrimSec_5e6Casing_50Mu_05Hz_LargeCondBody_back.npy',
                  'dpred_PrimSec_5e6Casing_50Mu_05Hz_LargeCondBody.npy',
                  'J_PrimSec_5e6Casing_50Mu_05Hz_LargeCondBody.npy']

    if basePath is None:
        # download to a SimPEGtemp folder in Downloads
        basePath = os.path.sep.join(
            os.path.abspath(os.getenv('HOME')).split(os.path.sep)+
            ['Downloads']+['SimPEGtemp']
            )

    return os.path.abspath(remoteDownload(url, cloudfiles,
                           basePath=basePath+os.path.sep))


def removeStoredResults(basePath):
    import shutil
    print('Removing {}'.format(basePath))
    shutil.rmtree(basePath)


def plotPrimaryFields(dataDict):
    primaryFields = primaryProblem.fieldsPair(meshp, primarySurvey)
    primaryFields[primarySurvey.srcList[0], 'hSolution'] = dict(primaryFieldsH.tolist())['hSolution']


def run(plotIt=False, runTests=False, reRun=False):

    casingExample = PrimSecCasingExample()

    if reRun is True:
        dataDict = casingExample.run(plotIt=plotIt, runTests=runTests)

    elif reRun is False:
    # Start by downloading files from the remote repository
        basePath = downloadStoredResults()
        # plotResults()
        removeStoredResults(basePath)

if __name__ == '__main__':
    run(plotIt=False, runTests=False, reRun=True)
