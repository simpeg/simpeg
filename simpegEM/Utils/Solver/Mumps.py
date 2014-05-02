from mumps import DMumpsContext

class Mumps():
    A = None
    ctx = None
    x = None

    def __init__(self, A, **kwagrs):
        
        self.ctx = DMumpsContext(sym=0, par=1)  

        if self.ctx.myid ==0:
            self.A = A
            self.ctx.set_icntl(14, 60)            
            self.ctx.set_centralized_sparse(A)

        self.ctx.set_silent()    
        self.ctx.run(job=4) # Factorization            

    def solve(self,b):               
        self.x = b.copy()
        self.ctx.set_rhs(self.x)
        self.ctx.run(job=3) # Solve 

        return self.x

    def clean(self):
    	self.ctx.destroy()