import numpy as np
from scipy import sparse
from utils import mkvc
from sputils import ddx, sdiag, speye, kron3

def getvol(h):   
    
    # Cell sizes in each direction 
    h1 = h[0]
    h2 = h[1]
    h3 = h[2]

    # Compute cell volumes                        
    v12 = h1.T*h2       
    V = mkvc(v12.reshape(-1,1)*h3)
    
    return V
        
def getarea(h):   
    
    # Cell sizes in each direction 
    h1 = h[0]
    h2 = h[1]
    h3 = h[2]

    # The number of cell centers in each direction
    n1 = np.size(h1)
    n2 = np.size(h2)
    n3 = np.size(h3)       
    # Compute areas of cell faces
    area1 = np.ones((n1+1,1))*mkvc(h2.T*h3) 
    area2 = h1.T*mkvc(np.ones((n2+1,1))*h3)
    area3 = h1.T*mkvc(h2.T*np.ones(n3+1))               
    area = np.hstack((np.hstack((mkvc(area1), mkvc(area2))), mkvc(area3)))
    
    return area

def getDivMatrix(h):    
    """Consturct the 3D divergence operator on Faces."""
           
    # Cell sizes in each direction 
    h1 = h[0]
    h2 = h[1]
    h3 = h[2]

    # The number of cell centers in each direction
    n1 = np.size(h1)
    n2 = np.size(h2)
    n3 = np.size(h3)  
        
    # Compute areas of cell faces
    #area1 = np.ones((n1+1,1))*mkvc(h2.T*h3) 
    #area2 = h1.T*mkvc(np.ones((n2+1,1))*h3)
    #area3 = h1.T*mkvc(h2.T*np.ones(n3+1))               
    #area = np.hstack((np.hstack((mkvc(area1), mkvc(area2))), mkvc(area3)))
    area = getarea(h)
            
    S = sdiag(area)                   
        
    # Compute cell volumes                        
    #v12 = h1.T*h2       
    #V = mkvc(v12.reshape(-1,1)*h3)
    V = getvol(h)
        
    # Compute divergence operator on faces                       
    d1 = ddx(n1)
    d2 = ddx(n2)
    d3 = ddx(n3)
    D1 = kron3(speye(n3), speye(n2), d1)
    D2 = kron3(speye(n3), d2, speye(n1))
    D3 = kron3(d3, speye(n2), speye(n1))       
        
    D = sparse.hstack((sparse.hstack((D1, D2)), D3))
    return sdiag(1/V)*D*S

