def interpFFT(x,y,m):
    """ 
    Load in a 2D grid and resample
    
    OUTPUT:
    m_out


     """ 

    from SimPEG import np, sp
    import scipy.signal as sn
    
    
    # Add padding values by reflection (2**n)    
    lenx = np.round( np.log2( 2*len(x) ) )
    npadx = int(np.floor( ( 2**lenx - len(x) ) /2. ))
    
    #Create hemming taper
    if np.mod(npadx*2+len(x),2) != 0:
        oddx = 1
        
    else:
        oddx = 0
        
    tap0 = sn.hamming(npadx*2)    
    tapl = sp.spdiags(tap0[0:npadx],0,npadx,npadx)
    tapr = sp.spdiags(tap0[npadx:],0,npadx,npadx+oddx)
     
    # Mirror the 2d data over the half lenght and apply 0-taper
    mpad = np.hstack([np.fliplr(m[:,0:npadx]) * tapl, m, np.fliplr(m[:,-npadx:]) * tapr])    
    
    # Repeat along the second dimension
    leny = np.round( np.log2( 2*len(y) ) )
    npady = int(np.floor( ( 2**leny - len(y) ) /2. ))
    
    #Create hemming taper
    if np.mod(npady*2+len(y),2) != 0:
        oddy = 1
        
    else:
        oddy = 0
        
    tap0 = sn.hamming(npady*2)
    tapu = sp.spdiags(tap0[0:npady],0,npady,npady)
    tapd = sp.spdiags(tap0[npady:],0,npady+oddy,npady)
    
    mpad = np.vstack([tapu*np.flipud(mpad[0:npady,:]), mpad, tapd*np.flipud(mpad[-npady:,:])])
    
    # Compute FFT
    FFTm = np.fft.fft2(mpad)    
    
    # Do an FFT shift
    FFTshift = np.fft.fftshift(FFTm)
    
    # Pad high frequencies with zeros to increase the sampling rate
    py = int(FFTm.shape[0]/2)
    px = int(FFTm.shape[1]/2)
    
    FFTshift = np.hstack([np.zeros((FFTshift.shape[0],px)),FFTshift,np.zeros((FFTshift.shape[0],px))])
    FFTshift = np.vstack([np.zeros((py,FFTshift.shape[1])),FFTshift,np.zeros((py,FFTshift.shape[1]))])
    
    # Inverse shift
    FFTm = np.fft.ifftshift(FFTshift)
    
    
    # Compute inverse FFT
    IFFTm = np.fft.ifft2(FFTm)*FFTm.size/mpad.size
    
    
    m_out = np.real(IFFTm)
    # Extract core
    #m_out = np.real(IFFTm[npady*2:-(npady*2+oddy+1),npadx*2:-(npadx*2+oddx+1)])

        
        
    m_out = m_out[npady*2:-(npady+oddy)*2,npadx*2:-(npadx+oddx)*2]
     
    if np.mod(m.shape[0],2) != 0:
        m_out = m_out[:-1,:]
        
    if np.mod(m.shape[1],2) != 0:
        m_out = m_out[:,:-1]
                   
    return m_out
