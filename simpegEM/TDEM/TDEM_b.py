from BaseTDEM import ProblemBaseTDEM


class ProblemTDEM_b(ProblemBaseTDEM):
    """
        docstring for ProblemTDEM_b
    """
    def __init__(self, mesh, model, **kwargs):
        ProblemBaseTDEM.__init__(self, mesh, model, **kwargs)

    solType = 'b'

    def getA(self, i):
        pass
        
    def getRHS(self, i, F):
        pass
