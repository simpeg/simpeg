from SimPEG import *
import simpegDC as DC
import matplotlib.pyplot as plt


def getTxList(nElecs, aSpacing, in2D=False, plotIt=False):

    elocs = np.arange(0,aSpacing*nElecs,aSpacing)
    space = 1
    WENNER = np.zeros((0,),dtype=int)
    for ii in range(nElecs):
        for jj in range(nElecs):
            test = np.r_[jj,jj+space,jj+space*2,jj+space*3]
            if np.any(test >= nElecs):
                break
            WENNER = np.r_[WENNER, test]
        space += 1
    WENNER = WENNER.reshape((-1,4))


    if plotIt:
        for i, s in enumerate('rbkg'):
            plt.plot(elocs[WENNER[:,i]],s+'.')
        plt.show()

    # Create transmitters and receivers
    i = 0
    if in2D:
        getLoc = lambda ii, abmn: np.r_[elocs[WENNER[ii,abmn]],0]
    else:
        getLoc = lambda ii, abmn: np.r_[elocs[WENNER[ii,abmn]],0, 0]
    txList = []
    for i in range(WENNER.shape[0]):
        rx = DC.DipoleRx(getLoc(i,1),getLoc(i,2))
        tx = DC.DipoleTx(getLoc(i,0),getLoc(i,3), [rx])
        txList += [tx]

    return txList



def example(aSpacing=2.5, nElecs=10, plotIt=False):

    surveySize = nElecs*aSpacing - aSpacing
    cs = surveySize/nElecs/4

    mesh = Mesh.TensorMesh([
            [(cs,10, -1.3),(cs,surveySize/cs),(cs,10, 1.3)],
            [(cs,3, -1.3),(cs,3,1.3)],
    #         [(cs,5, -1.3),(cs,10)]
        ],'CN')
    if plotIt:
        mesh.plotGrid(showIt=True)

    txList = getTxList(nElecs, aSpacing, in2D=True)
    survey = DC.SurveyDC(txList)
    problem = DC.ProblemDC(mesh)
    problem.pair(survey)

    return mesh, survey, problem
