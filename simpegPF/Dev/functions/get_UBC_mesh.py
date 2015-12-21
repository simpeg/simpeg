'''
Created on Jul 17, 2013

@author: dominiquef
'''
def get_UBC_mesh(meshfile):
    """ Read UBC mesh file and extract parameters
         Works for the condenced version (20 * 3) --> [20 20 20] """ 

    fid = open(meshfile,'r')
    from numpy import zeros
    
    # Go through the log file and extract data and the last achieved misfit
    for ii in range (1, 6): 
                        
        line = fid.readline()
        line = line.split(' ')    
        
            # First line: number of cells in i, j, k 
        if ii == 1:
            
            numcell=[]
                
            for jj in range(len(line)):
                t = int(line[jj])
                numcell.append(t)
                
            nX = numcell[0]
            nY = numcell[1]
            nZ = numcell[2] 
            # Second line: origin coordinate (X,Y,Z)
        elif ii==2:
                
            origin = []
            
            for jj in range(len(line)):
                t = float(line[jj])
                origin.append(t)
                
            
        # Other lines for the xn, yn, zn (nodes location)
        elif ii==3:
            
            xn=zeros((nX+1,1), dtype=float)
            xn[0] = origin[0]

            count_entry = 0;
            count = 0;
            while (count<nX):

                if line[count_entry].find('*') != -1:
                    
                    ndx = line[count_entry].split('*')
                    
                    for kk in range(int(ndx[0])):
                        xn[count+1] = xn[count] + (ndx[1])
                        count = count+1
                    count_entry=count_entry+1
                         
                else:
                    
                    t = float(line[count_entry])
                    xn[count+1]= xn[count] +t
                    count = count+1;  
                    count_entry=count_entry+1
                        
        elif ii==4:
            
            yn=zeros((nY+1,1), dtype=float)
            yn[0] = origin[0]

            count_entry = 0;
            count = 0;
            while (count<nY):

                if line[count_entry].find('*') != -1:
                    
                    ndx = line[count_entry].split('*')
                    
                    for kk in range(int(ndx[0])):
                        yn[count+1] = yn[count] + (ndx[1])
                        count = count+1
                    count_entry=count_entry+1
                         
                else:
                    
                    t = float(line[count_entry])
                    yn[count+1]= yn[count] +t
                    count = count+1;  
                    count_entry=count_entry+1
                    
        elif ii==5:
            
            zn=zeros((nZ+1,1), dtype=float)
            zn[0] = origin[0]

            count_entry = 0;
            count = 0;
            while (count<nZ):

                if line[count_entry].find('*') != -1:
                    
                    ndx = line[count_entry].split('*')
                    
                    for kk in range(int(ndx[0])):
                        zn[count+1] = zn[count] + (ndx[1])
                        count = count+1
                    count_entry=count_entry+1
                         
                else:
                    
                    t = float(line[count_entry])
                    zn[count+1]= zn[count] +t
                    count = count+1;  
                    count_entry=count_entry+1
    fid.close();                
    return xn,yn,zn
