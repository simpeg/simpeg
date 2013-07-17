import numpy as np
from scipy import sparse
from utils import mkvc
from sputils import ddx, sdiag, speye, kron3, spzeros, av

def getvol(h):   
    """Construct cell volumes of the 3D model as 1d array."""        
    # Cell sizes in each direction 
    h1 = h[0]
    h2 = h[1]
    h3 = h[2]

    # Compute cell volumes                        
    v12 = h1.T*h2       
    V = mkvc(v12.reshape(-1,1)*h3)
    
    return V
        
def getarea(h):   
    """Construct face areas of the 3D model as 1d array."""        
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
    area = np.concatenate((mkvc(area1), mkvc(area2), mkvc(area3)), axis=0)
    
    return area

def getlength_e(h):   
    """Construct edge legnths of the 3D model as 1d array."""    
    # Cell sizes in each direction 
    h1 = h[0]
    h2 = h[1]
    h3 = h[2]

    # The number of cell centers in each direction
    n1 = np.size(h1)
    n2 = np.size(h2)
    n3 = np.size(h3)       
    # Compute areas of cell faces
    l1 = h1.T*mkvc(np.ones((n2+1,1))*np.ones(n3+1))
    l2 = np.ones((n1+1,1))*mkvc(h2.T*np.ones(n3+1))
    l3 = np.ones((n1+1,1))*mkvc(np.ones((n2+1,1))*h3)               
    #l = np.hstack((np.hstack((mkvc(area1), mkvc(area2))), mkvc(area3)))
    l = np.concatenate((mkvc(l1), mkvc(l2), mkvc(l3)), axis=0)
    
    return l

def getDivMatrix(h):    
    """Construct the 3D divergence operator on Faces."""
           
    # Cell sizes in each direction 
    h1 = h[0]
    h2 = h[1]
    h3 = h[2]

    # The number of cell centers in each direction
    n1 = np.size(h1)
    n2 = np.size(h2)
    n3 = np.size(h3)  
        
    # Compute areas of cell faces
    S = getarea(h)            
        
    # Compute cell volumes                        
    V = getvol(h)
        
    # Compute divergence operator on faces                       
    d1 = ddx(n1)
    d2 = ddx(n2)
    d3 = ddx(n3)
    D1 = kron3(speye(n3), speye(n2), d1)
    D2 = kron3(speye(n3), d2, speye(n1))
    D3 = kron3(d3, speye(n2), speye(n1))       
        
    D = sparse.hstack((D1, D2, D3), format="csr")
    return sdiag(1/V)*D*sdiag(S)

def getGradMatrix(h):    
    """Construct the 3D nodal gradient operator."""
           
    # Cell sizes in each direction 
    h1 = h[0]
    h2 = h[1]
    h3 = h[2]

    # The number of cell centers in each direction
    n1 = np.size(h1)
    n2 = np.size(h2)
    n3 = np.size(h3)  
        
    # Compute lengths of cell edges
    L = getlength_e(h)                
    
    # Compute divergence operator on faces                       
    d1 = ddx(n1)
    d2 = ddx(n2)
    d3 = ddx(n3)
    D1 = kron3(speye(n3+1), speye(n2+1), d1)
    D2 = kron3(speye(n3+1), d2, speye(n1+1))
    D3 = kron3(d3, speye(n2+1), speye(n1+1))       
        
    G = sparse.vstack((D1, D2, D3), format="csr")
    return sdiag(1/L)*G
    
def getCurlMatrix(h):    
    """Construct the 3D curl operator."""
           
    # Cell sizes in each direction 
    h1 = h[0]
    h2 = h[1]
    h3 = h[2]

    # The number of cell centers in each direction
    n1 = np.size(h1)
    n2 = np.size(h2)
    n3 = np.size(h3)  
        
    # Compute lengths of cell edges
    L = getlength_e(h)                

    # Compute areas of cell faces
    S = getarea(h)            
    
    # Compute divergence operator on faces                       
    d1 = ddx(n1)
    d2 = ddx(n2)
    d3 = ddx(n3)
    
    D32 = kron3(d3, speye(n2), speye(n1+1))
    D23 = kron3(speye(n3), d2, speye(n1+1))
    D31 = kron3(d3, speye(n2+1), speye(n1))
    D13 = kron3(speye(n3), speye(n2+1), d1)
    D21 = kron3(speye(n3+1), d2, speye(n1))            
    D12 = kron3(speye(n3+1), speye(n2), d1)      
    
    O1 = spzeros(np.shape(D32)[0], np.shape(D31)[1])
    O2 = spzeros(np.shape(D31)[0], np.shape(D32)[1])
    O3 = spzeros(np.shape(D21)[0], np.shape(D13)[1])
    
    C = sparse.vstack((sparse.hstack((O1,-D32, D23)), 
                       sparse.hstack((D31,O2, -D13)), 
                       sparse.hstack((-D21,D12, O3))), format="csr") 
    
    return sdiag(1/S)*(C*sdiag(L))

def getAverageMatrixF(h):
    """Construct the 3D averaging operator on cell faces."""    

    # Cell sizes in each direction 
    h1 = h[0]
    h2 = h[1]
    h3 = h[2]

    # The number of cell centers in each direction
    n1 = np.size(h1)
    n2 = np.size(h2)
    n3 = np.size(h3)  
    
    av1 = av(n1)
    av2 = av(n2)
    av3 = av(n3)
    
    AvF = sparse.hstack(kron3(speye(n3), speye(n2), av1),
                        kron3(speye(n3), av2, speye(n3)),
                        kron3(av3, speye(n2), speye(n3)), format="csr")
    return AvF                      
        
def getAverageMatrixE(h):    
    """Construct the 3D averaging operator on cell edges."""        
    
    # Cell sizes in each direction 
    h1 = h[0]
    h2 = h[1]
    h3 = h[2]

    # The number of cell centers in each direction
    n1 = np.size(h1)
    n2 = np.size(h2)
    n3 = np.size(h3)      

    av1 = av(n1)
    av2 = av(n2)
    av3 = av(n3)
    
    AvE = sparse.hstack(kron3(av3, av2, speye(n1)),
                        kron3(av3, speye(n2), av1),
                        kron3(speye(n3), av2, av1), format="csr")
    return AvE