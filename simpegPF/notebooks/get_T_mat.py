'''
Created on Sep 27, 2015

@author: dominiquef
'''
def get_T_mat(xn,yn,zn,obsx,obsy,obsz):
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

    from numpy import empty, pi, log, arctan, sqrt, shape
    
    ncx = len(xn)-1
    ncy = len(yn)-1
    ncz = len(zn)-1
    
    mcell = ncx*ncy*ncz;

    Tx = empty([1,3*mcell], dtype=float)
    Ty = empty([1,3*mcell], dtype=float)
    Tz = empty([1,3*mcell], dtype=float)

    count = 0

    
    for ii in range(ncz):

        print ii, ncz

        dz2 = zn[ii] - obsz;
        dz1 = zn[ii+1] - obsz;

        for jj in range(ncy): 

            dy2 = yn[jj] - obsy;
            dy1 = yn[jj+1] - obsy;

            for kk in range(ncx):   

                dx2 = xn[kk] - obsx;
                dx1 = xn[kk+1] - obsx;

                R1 = ( dy2**2 + dx2**2 );
                R2 = ( dy2**2 + dx1**2 );
                R3 = ( dy1**2 + dx2**2 );
                R4 = ( dy1**2 + dx1**2 );

                arg1 = sqrt( dz2**2 + R2 );
                arg2 = sqrt( dz2**2 + R1 );
                arg3 = sqrt( dz1**2 + R1 );
                arg4 = sqrt( dz1**2 + R2 );
                arg5 = sqrt( dz2**2 + R3 );
                arg6 = sqrt( dz2**2 + R4 );
                arg7 = sqrt( dz1**2 + R4 );
                arg8 = sqrt( dz1**2 + R3 );

                

                Tx[0,count] = arctan( dy1 * dz2 / ( dx2 * arg5 ) ) +\
                            - arctan( dy2 * dz2 / ( dx2 * arg2 ) ) +\
                            arctan( dy2 * dz1 / ( dx2 * arg3 ) ) +\
                            - arctan( dy1 * dz1 / ( dx2 * arg8 ) ) +\
                            arctan( dy2 * dz2 / ( dx1 * arg1 ) ) +\
                            - arctan( dy1 * dz2 / ( dx1 * arg6 ) ) +\
                            arctan( dy1 * dz1 / ( dx1 * arg7 ) ) +\
                            - arctan( dy2 * dz1 / ( dx1 * arg4 ) );


                Ty[0,count] = log( ( dz2 + arg2 ) / (dz1 + arg3 ) ) +\
                            -log( ( dz2 + arg1 ) / (dz1 + arg4 ) ) +\
                            log( ( dz2 + arg6 ) / (dz1 + arg7 ) ) +\
                            -log( ( dz2 + arg5 ) / (dz1 + arg8 ) );


                Ty[0,mcell+count] = arctan( dx1 * dz2 / ( dy2 * arg1 ) ) +\
                                - arctan( dx2 * dz2 / ( dy2 * arg2 ) ) +\
                                arctan( dx2 * dz1 / ( dy2 * arg3 ) ) +\
                                - arctan( dx1 * dz1 / ( dy2 * arg4 ) ) +\
                                arctan( dx2 * dz2 / ( dy1 * arg5 ) ) +\
                                - arctan( dx1 * dz2 / ( dy1 * arg6 ) ) +\
                                arctan( dx1 * dz1 / ( dy1 * arg7 ) ) +\
                                - arctan( dx2 * dz1 / ( dy1 * arg8 ) );

                R1 = (dy2**2 + dz1**2);
                R2 = (dy2**2 + dz2**2);
                R3 = (dy1**2 + dz1**2);
                R4 = (dy1**2 + dz2**2);

                Ty[0,2*mcell+count] = log( ( dx1 + sqrt( dx1**2 + R1 ) ) / (dx2 + sqrt( dx2**2 + R1 ) ) ) +\
                                    -log( ( dx1 + sqrt( dx1**2 + R2 ) ) / (dx2 + sqrt( dx2**2 + R2 ) ) ) +\
                                    log( ( dx1 + sqrt( dx1**2 + R4 ) ) / (dx2 + sqrt( dx2**2 + R4 ) ) ) +\
                                    -log( ( dx1 + sqrt( dx1**2 + R3 ) ) / (dx2 + sqrt( dx2**2 + R3 ) ) );

                R1 = (dx2**2 + dz1**2);
                R2 = (dx2**2 + dz2**2);
                R3 = (dx1**2 + dz1**2);
                R4 = (dx1**2 + dz2**2);

                Tx[0,2*mcell+count] = log( ( dy1 + sqrt( dy1**2 + R1 ) ) / (dy2 + sqrt( dy2**2 + R1 ) ) ) +\
                                    -log( ( dy1 + sqrt( dy1**2 + R2 ) ) / (dy2 + sqrt( dy2**2 + R2 ) ) ) +\
                                    log( ( dy1 + sqrt( dy1**2 + R4 ) ) / (dy2 + sqrt( dy2**2 + R4 ) ) ) +\
                                    -log( ( dy1 + sqrt( dy1**2 + R3 ) ) / (dy2 + sqrt( dy2**2 + R3 ) ) );

                Tz[0,2*mcell+count] = -( Ty[0,mcell+count] + Tx[0,count] );
                Tz[0,mcell+count] = Ty[0,2*mcell+count];
                Tx[0,mcell+count] = Ty[0,count];
                Tz[0,count] = Tx[0,2*mcell+count];



                count = count + 1

    Tx = Tx/(4*pi);
    Ty = Ty/(4*pi);
    Tz = Tz/(4*pi);        
                   
    return Tx,Ty,Tz
