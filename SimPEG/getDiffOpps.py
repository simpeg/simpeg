import numpy as np
from scipy import sparse
from utils import mkvc
from sputils import *
#from sputils import ddx, sdiag, speye, kron3, spzeros, appendBottom3, 

def getvol(h):   
    
    # Cell sizes in each direction 
    h1 = h[0]
    h2 = h[1]
    h3 = h[2]

    # Compute cell volumes                        
    V = mkvc(np.outer(mkvc(np.outer(h1,h2)),h3))       
    
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
    area1 = mkvc(np.outer(np.ones(n1+1),np.outer(h2,h3))) 
    area2 = mkvc(np.outer(h1,mkvc(np.outer(np.ones(n2+1),h3))))
    area3 = mkvc(np.outer(h1,mkvc(np.outer(h2,np.ones(n3+1)))))               
    area = np.hstack((np.hstack((area1, area2)), area3))
    
    return area
    
def getLength(h):    

    h1 = h[0]
    h2 = h[1]
    h3 = h[2]

    # The number of cell centers in each direction
    n1 = np.size(h1)
    n2 = np.size(h2)
    n3 = np.size(h3)

    # compute the length of each edge
    Length1 = mkvc(np.outer(h1,mkvc(np.outer(np.ones(n2+1),np.ones(n3+1))))) 
    Length2 = mkvc(np.outer(np.ones(n1+1),mkvc(np.outer(h2,np.ones(n3+1)))))
    Length3 = mkvc(np.outer(np.ones(n1+1),mkvc(np.outer(np.ones(n2+1),h3))))
    
    Length = np.hstack((np.hstack((Length1, Length2)), Length3))
    
    return Length
    
    
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
        
    area = getarea(h)          
    S = sdiag(area)                   
        
    # Compute cell volumes                        
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


def getCurlMatrix(h):
    """Edge CURL """

    # Cell sizes in each direction 
    h1 = h[0]; h2 = h[1]; h3 = h[2]

    # The number of cell centers in each direction
    n1 = np.size(h1); n2 = np.size(h2); n3 = np.size(h3) 
    
    d1 = ddx(n1); d2 = ddx(n2); d3 = ddx(n3)
    # derivatives on x-edge variables
    D32 = kron3(d3, speye(n2), speye(n1+1))
    D23 = kron3(speye(n3), d2, speye(n1+1))
    D31 = kron3(d3, speye(n2+1), speye(n1))
    D13 = kron3(speye(n3), speye(n2+1), d1)
    D21 = kron3(speye(n3+1), d2, speye(n1))
    D12 = kron3(speye(n3+1), speye(n2), d1)

    O1 = spzeros(np.shape(D32)[0], np.shape(D31)[1])
    O2 = spzeros(np.shape(D31)[0], np.shape(D32)[1])
    O3 = spzeros(np.shape(D21)[0], np.shape(D13)[1])

    CURL = appendBottom3(
           appendRight3(O1,    -D32,  D23),
           appendRight3(D31,    O2,  -D13),
           appendRight3(-D21,  D12,   O3))


    area = getarea(h)          
    S = sdiag(1/area)                   
        
    # Compute edge length                       
    lngth = getLength(h)
    L = sdiag(lngth)
    
    return S*(CURL*L)


def getNodalGradient(h):
    """Nodal Gradients"""

    # Cell sizes in each direction 
    h1 = h[0]; h2 = h[1]; h3 = h[2]

    # The number of cell centers in each direction
    n1 = np.size(h1); n2 = np.size(h2); n3 = np.size(h3) 
    

    D1 = kron3(speye(n3+1), speye(n2+1), ddx(n1))
    D2 = kron3(speye(n3+1), ddx(n2), speye(n1+1))
    D3 = kron3(ddx(n3), speye(n2+1), speye(n1+1))

    # topological gradient
    GRAD = appendBottom3(D1, D2, D3)

    # scale for non-uniform mesh
    # Compute edge length                       
    lngth = getLength(h)
    L = sdiag(1/lngth)

    return L*GRAD
