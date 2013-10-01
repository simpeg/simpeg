from SimPEG import LogicallyOrthogonalMesh, utils
import matplotlib.pyplot as plt
X, Y = utils.exampleLomGird([3,3],'rotate')
M = LogicallyOrthogonalMesh([X, Y])
M.plotGrid()
plt.show()
