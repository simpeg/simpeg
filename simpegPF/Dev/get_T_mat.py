'''
Created on Sep 27, 2015

@author: dominiquef
'''
def get_T_mat(xn,yn,zn,rxLoc):
    """ 
    Load in the nodes of a tensor mesh and computes the magnetic tensor 
    for a given observation location [obsx, obsy, obsz]
    OUTPUT:
    Tx = [Txx Txy Txz]
    Ty = [Tyx Tyy Tyz]
    Tz = [Tzx Tzy Tzz]

    where each elements have dimension 1-by-mcell.
    Only the upper half 5 elements have to be computed since symetric.
    Currently done as for-loops but will eventually be changed to vector
    indexing, once the topography has been figured out.
     """ 

    from SimPEG import np, mkvc
    
    ncx = len(xn)-1
    ncy = len(yn)-1
    ncz = len(zn)-1
    
    mcell = ncx*ncy*ncz
        
    # Pre-allocate space for 1D array
    Tx = np.zeros((1,3*mcell))
    Ty = np.zeros((1,3*mcell))
    Tz = np.zeros((1,3*mcell))
    
    yn2,xn2,zn2 = np.meshgrid(yn[1:], xn[1:], zn[1:])
    yn1,xn1,zn1 = np.meshgrid(yn[0:ncy], xn[0:ncx], zn[0:ncz])
    
    yn2 = mkvc(yn2)
    yn1 = mkvc(yn1)
    
    zn2 = mkvc(zn2)
    zn1 = mkvc(zn1)
    
    xn2 = mkvc(xn2)
    xn1 = mkvc(xn1)
    #%%
    #==============================================================================

    
    dz2 = rxLoc[2] - zn1;
    dz1 = rxLoc[2] - zn2;
     
    
    dy2 = yn2 - rxLoc[1];
    dy1 = yn1 - rxLoc[1];
    
    
    dx2 = xn2 - rxLoc[0];
    dx1 = xn1 - rxLoc[0];
    
    R1 = ( dy2**2 + dx2**2 );
    R2 = ( dy2**2 + dx1**2 );
    R3 = ( dy1**2 + dx2**2 );
    R4 = ( dy1**2 + dx1**2 );
    
    
    arg1 = np.sqrt( dz2**2 + R2 );
    arg2 = np.sqrt( dz2**2 + R1 );
    arg3 = np.sqrt( dz1**2 + R1 );
    arg4 = np.sqrt( dz1**2 + R2 );
    arg5 = np.sqrt( dz2**2 + R3 );
    arg6 = np.sqrt( dz2**2 + R4 );
    arg7 = np.sqrt( dz1**2 + R4 );
    arg8 = np.sqrt( dz1**2 + R3 );
    
            
    
    Tx[0,0:mcell] = np.arctan2( dy1 * dz2 , ( dx2 * arg5 ) ) +\
                - np.arctan2( dy2 * dz2 , ( dx2 * arg2 ) ) +\
                np.arctan2( dy2 * dz1 , ( dx2 * arg3 ) ) +\
                - np.arctan2( dy1 * dz1 , ( dx2 * arg8 ) ) +\
                np.arctan2( dy2 * dz2  , ( dx1 * arg1 ) ) +\
                - np.arctan2( dy1 * dz2 , ( dx1 * arg6 ) ) +\
                np.arctan2( dy1 * dz1 , ( dx1 * arg7 ) ) +\
                - np.arctan2( dy2 * dz1 , ( dx1 * arg4 ) );
    
    
    Ty[0,0:mcell] = np.log( ( dz2 + arg2 ) / (dz1 + arg3 ) ) +\
                -np.log( ( dz2 + arg1 ) / (dz1 + arg4 ) ) +\
                np.log( ( dz2 + arg6 ) / (dz1 + arg7 ) ) +\
                -np.log( ( dz2 + arg5 ) / (dz1 + arg8 ) );
    
    Ty[0,mcell:2*mcell] = np.arctan2( dx1 * dz2 , ( dy2 * arg1 ) ) +\
                    - np.arctan2( dx2 * dz2 , ( dy2 * arg2 ) ) +\
                    np.arctan2( dx2 * dz1 , ( dy2 * arg3 ) ) +\
                    - np.arctan2( dx1 * dz1 , ( dy2 * arg4 ) ) +\
                    np.arctan2( dx2 * dz2 , ( dy1 * arg5 ) ) +\
                    - np.arctan2( dx1 * dz2 , ( dy1 * arg6 ) ) +\
                    np.arctan2( dx1 * dz1 , ( dy1 * arg7 ) ) +\
                    - np.arctan2( dx2 * dz1 , ( dy1 * arg8 ) );
    
    R1 = (dy2**2 + dz1**2);
    R2 = (dy2**2 + dz2**2);
    R3 = (dy1**2 + dz1**2);
    R4 = (dy1**2 + dz2**2);
    
    Ty[0,2*mcell:] = np.log( ( dx1 + np.sqrt( dx1**2 + R1 ) ) / (dx2 + np.sqrt( dx2**2 + R1 ) ) ) +\
                        -np.log( ( dx1 + np.sqrt( dx1**2 + R2 ) ) / (dx2 + np.sqrt( dx2**2 + R2 ) ) ) +\
                        np.log( ( dx1 + np.sqrt( dx1**2 + R4 ) ) / (dx2 + np.sqrt( dx2**2 + R4 ) ) ) +\
                        -np.log( ( dx1 + np.sqrt( dx1**2 + R3 ) ) / (dx2 + np.sqrt( dx2**2 + R3 ) ) );
    
    R1 = (dx2**2 + dz1**2);
    R2 = (dx2**2 + dz2**2);
    R3 = (dx1**2 + dz1**2);
    R4 = (dx1**2 + dz2**2);
    
    Tx[0,2*mcell:] = np.log( ( dy1 + np.sqrt( dy1**2 + R1 ) ) / (dy2 + np.sqrt( dy2**2 + R1 ) ) ) +\
                        -np.log( ( dy1 + np.sqrt( dy1**2 + R2 ) ) / (dy2 + np.sqrt( dy2**2 + R2 ) ) ) +\
                        np.log( ( dy1 + np.sqrt( dy1**2 + R4 ) ) / (dy2 + np.sqrt( dy2**2 + R4 ) ) ) +\
                        -np.log( ( dy1 + np.sqrt( dy1**2 + R3 ) ) / (dy2 + np.sqrt( dy2**2 + R3 ) ) );
    
    Tz[0,2*mcell:] = -( Ty[0,mcell:2*mcell] + Tx[0,0:mcell] );
    Tz[0,mcell:2*mcell] = Ty[0,2*mcell:];
    Tx[0,mcell:2*mcell] = Ty[0,0:mcell];
    Tz[0,0:mcell] = Tx[0,2*mcell:];
    
    
    
    Tx = Tx/(4*np.pi);
    Ty = Ty/(4*np.pi);
    Tz = Tz/(4*np.pi);   
       
                   
    return Tx,Ty,Tz
