#============= Nodal Gradients ===========================	
def getNodalGradient(h1,h2,h3):
    
    n1 = size(h1) 
    n2 = size(h2) 
    n3 = size(h3) 	 	
    D1  = kron3(speye(n3+1),speye(n2+1),ddx(n1))
    D2  = kron3(speye(n3+1),ddx(n2),speye(n1+1))
    D3  = kron3(ddx(n3),speye(n2+1),speye(n1+1))
    
    # topological gradient
    GRAD = appendBottom3(D1,D2,D3)
    
    # scale for non-uniform mesh
    L  = blkDiag3(kron3(speye(n3+1),speye(n2+1),sdiag(1/h1)),
                  kron3(speye(n3+1),sdiag(1/h2),speye(n1+1)),
                  kron3(sdiag(1/h3),speye(n2+1),speye(n1+1)))  
        
    return L*GRAD	

#============= Edge CURL ===========================	
def getCurlMatrix(h1,h2,h3):
    
    n1 = size(h1) 
    n2 = size(h2) 
    n3 = size(h3)

    d1 = ddx(n1)
    d2 = ddx(n2)
    d3 = ddx(n3)
    # derivatives on x-edge variables
    D32  = kron3(d3,speye(n2),speye(n1+1))
    D23  = kron3(speye(n3),d2,speye(n1+1))  	 	 	 	
    D31  = kron3(d3,speye(n2+1),speye(n1))
    D13  = kron3(speye(n3),speye(n2+1),d1)
    D21  = kron3(speye(n3+1),d2,speye(n1))
    D12  = kron3(speye(n3+1),speye(n2),d1)

    O1 = spzeros(shape(D32)[0],shape(D31)[1])
    O2 = spzeros(shape(D31)[0],shape(D32)[1])
    O3 = spzeros(shape(D21)[0],shape(D13)[1])

    CURL = appendBottom3(
        appendRight3(O1,    -D32,  D23),
        appendRight3(D31,    O2,  -D13),
        appendRight3(-D21,  D12,   O3))
        
    # scale for non-uniform mesh     
    F  = blkDiag3(kron3(sdiag(1/h3),sdiag(1/h2),speye(n1+1)),
             kron3(sdiag(1/h3),speye(n2+1),sdiag(1/h1)),
             kron3(speye(n3+1),sdiag(1/h2),sdiag(1/h1)))
             
    L  = blkDiag3(kron3(speye(n3+1),speye(n2+1),sdiag(h1)),
                  kron3(speye(n3+1),sdiag(h2),speye(n1+1)),
                  kron3(sdiag(h3),speye(n2+1),speye(n1+1)))  
                 
    
    return F*(CURL*L)	

#============= Face DIV ===========================	
def getDivMatrix(h1,h2,h3):
      
    n1 = size(h1) 
    n2 = size(h2) 
    n3 = size(h3)

    d1 = ddx(n1)
    d2 = ddx(n2)
    d3 = ddx(n3)
    D1  = kron3(speye(n3),speye(n2),d1)
    D2  = kron3(speye(n3),d2,speye(n1))
    D3  = kron3(d3,speye(n2),speye(n1))

    # divergence on faces
    D = appendRight3(D1, D2, D3)

    # scale for non-uniform mesh     
    F  = blkDiag3(kron3(sdiag(h3),sdiag(h2),speye(n1+1)),
             kron3(sdiag(h3),speye(n2+1),sdiag(h1)),
             kron3(speye(n3+1),sdiag(h2),sdiag(h1)))

    V  = kron3(sdiag(1/h3),sdiag(1/h2),sdiag(1/h1))
    
    return V*(D*F)
   
#====== Face Averageing =================
def getFaceAverage(n1,n2,n3):

    av1 = av(n1)
    av2 = av(n2)
    av3 = av(n3)

    Af = appendRight3(kron3(speye(n3),speye(n2),av1),
                      kron3(speye(n3),av2,speye(n1)),
                      kron3(av3,speye(n2),speye(n1)))
    return Af

#====== Edge Averageing =================
def getEdgeAverage(n1,n2,n3):

    av1 = av(n1)
    av2 = av(n2)
    av3 = av(n3)
            
    Ae = appendRight3(kron3(av3,av2,speye(n1)),
                      kron3(av3,speye(n2),av1),
                      kron3(speye(n3),av2,av1))
    return Ae

#====== Node Averageing =================
def getNodeAverage(n1,n2,n3):

    av1 = av(n1)
    av2 = av(n2)
    av3 = av(n3)
  
    return kron3(av3,av2,av1)  
    
      
