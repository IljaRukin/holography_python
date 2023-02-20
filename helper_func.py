##### Holography helper functions
#import with: from helper_func import *

import numpy as np
from math import floor,ceil
from PIL import Image

sq = lambda x: (x*np.conj(x)).astype(np.float)
modulo = lambda x: np.mod(x+pi,2*pi)-pi
convolve2dfft = lambda a,b: np.fft.fftshift(np.fft.ifft2( (np.fft.fft2(np.fft.fftshift( a ))) * (np.fft.fft2(np.fft.fftshift( b ))) ))

global pi,i
pi = np.pi
i = np.complex(0,1)

### array manipulation

def scale(img,minv,maxv):
    #rescale real array
    mini, maxi = np.min(img), np.max(img)
    if (mini==maxi):
        return np.zeros(np.shape(img))+mini
    else:
        return (img-mini)/(maxi-mini) *(maxv-minv) +minv

def scaleComplex(img,minv,maxv):
    #rescale amplitudes of complex array
    mini, maxi = np.min(np.abs(img)), np.max(np.abs(img))
    if (mini==maxi):
        return np.zeros(np.shape(img))+mini
    else:
        return (img-mini)/(maxi-mini) *(maxv-minv) +minv

def average(img,binsize):
    #average over bins, array size should be multiple of binsize
    My,Nx = img.shape
    return img.reshape((My//binsize, binsize, Nx//binsize, binsize)).mean(3).mean(1)

### unwrapping

def unwrapping(xw,period=2*pi,diff=pi):
    xu = xw;
    for kk in range(1,np.size(xw)):
        difference = xw[kk]-xw[kk-1];
        if difference > diff:
            xu[kk:] = xu[kk:] - period;
        elif difference < -diff:
            xu[kk:] = xu[kk:] + period;
    return xu

### 1D fitting

def linfit1D(x,f,w):
    #interpolate function linearly f=ax*x+b
    #with weights w for the importance of every datapoint
    w_bar = (w).sum();
    x_bar = (w*x).sum();
    f_bar = (w*f).sum();
    xx_bar = (w*x**2).sum();
    xf_bar = (w*x*f).sum();

    a = (w_bar*xf_bar-x_bar*f_bar)/(w_bar*xx_bar-x_bar**2);
    b = (f_bar - a*x_bar)/w_bar;
    return a,b

def lsq(x,y):
    #solve min(sum(y-Xb))
    '''
    (x1,1) (a) (y1)
    (x2,1)@(b)=(y2)
    (x3,1)     (y3)
    '''
    n = x.size
    X = np.ones((n,2))
    X[:,0] = x.reshape(n)
    #X[:,1] = 1
    Y = y.reshape(n)
    return np.linalg.inv(X.T@X)@X.T@Y

### 2D fitting

def linfit2D(x,y,f,w):
    #interpolate function linearly f=ax*x+ay*y+b
    #with weights w for the importance of every datapoint
    w_bar = (w).sum();
    x_bar = (w*x).sum();
    y_bar = (w*y).sum();
    f_bar = (w*f).sum();
    xx_bar = (w*x**2).sum();
    yy_bar = (w*y**2).sum();
    xy_bar = (w*x*y).sum();
    xf_bar = (w*x*f).sum();
    yf_bar = (w*y*f).sum();

    ax = (y_bar**2*xf_bar+xy_bar*w_bar*yf_bar+f_bar*x_bar*yy_bar-f_bar*xy_bar*y_bar-x_bar*y_bar*yf_bar-w_bar*xf_bar*yy_bar)/(x_bar**2*yy_bar+y_bar**2*xx_bar+xy_bar**2*w_bar-2*xy_bar*x_bar*y_bar-w_bar*xx_bar*yy_bar);
    ay = (x_bar**2*yf_bar+xy_bar*w_bar*xf_bar+f_bar*y_bar*xx_bar-f_bar*xy_bar*x_bar-x_bar*y_bar*xf_bar-w_bar*xx_bar*yf_bar)/(x_bar**2*yy_bar+y_bar**2*xx_bar+xy_bar**2*w_bar-2*xy_bar*x_bar*y_bar-w_bar*xx_bar*yy_bar);
    b = (f_bar - ax*x_bar - ay*y_bar)/w_bar;
    return ax,ay,b

def lsq2D(x1,x2,y):
    #solve min(sum(y-Xb))
    n = x1.size
    X = np.ones((n,3))
    X[:,0] = x1.reshape(n)
    X[:,1] = x2.reshape(n)
    #X[:,2] = 1
    Y = y.reshape(n)
    return np.linalg.inv(X.T@X)@X.T@Y

### wavefield propagation

def angular_spectrum(U,z,N,M,Dn,Dm,lamb):
    yy,xx = np.mgrid[-M/2:M/2,-N/2:N/2]; #discrete: k,l,m,n
    dfx = 1/(N*Dn); #frequency steps = 1/SLM_width
    freq2 = xx**2*dfx**2 + yy**2*dfx**2; #frequencies squared
    mask = freq2*lamb**2<1;
    k = np.sqrt( (1 - freq2*lamb**2) * mask );
    k[np.where(np.isnan(k))]=0
    ft_spherical = np.exp(i*2*pi/lamb*z* k ) #* np.exp(-freq2*Dm);
    #propagate
    E = np.fft.fftshift(np.fft.ifft2( (np.fft.fft2(np.fft.fftshift( U ))) * np.fft.fftshift(ft_spherical) ));
    #propagate + zeropadding M/2,N/2
    #U = np.pad(U,[[M//2,M//2],[N//2,N//2]])
    #ft_spherical = np.pad(ft_spherical,[[M//2,M//2],[N//2,N//2]])
    #E = np.fft.fftshift(np.fft.ifft2( (np.fft.fft2(np.fft.fftshift( U ))) * np.fft.fftshift(ft_spherical) ))[M//2:M+M//2,N//2:N+N//2]
    #propagate + zeropadding M,N
    #U = np.pad(U,[[M,M],[N,N]])
    #ft_spherical = np.pad(ft_spherical,[[M,M],[N,N]])
    #E = np.fft.fftshift(np.fft.ifft2( (np.fft.fft2(np.fft.fftshift( U ))) * np.fft.fftshift(ft_spherical) ))[M:M+M,N:N+N]
    return E

def plane_wave(U,z,N,M,Dn,Dm,lamb):
    yy,xx = np.mgrid[-M/2:M/2,-N/2:N/2]; #discrete: k,l,m,n
    coord2 = xx**2*Dn**2 + yy**2*Dm**2; #SLM coordinates squared
    #antialiasing
    if Dn==Dm:
        p = lamb/4;
        x_NDn = (lamb*z)/(4*Dn); #approximation
        mask = xx**2+yy**2 < (x_NDn/Dn)**2;
    else:
        p = lamb/4;
        x_NDn = (lamb*z)/(4*Dn); #approximation
        y_NDm = (lamb*z)/(4*Dm); #approximation
        mask = xx**2*(Dn/x_NDn)**2 + yy**2*(Dm/y_NDm)**2 < 1;
    spherical = np.exp(i*2*pi/lamb*np.sign(z)* np.sqrt( z**2 + coord2 ) ) *mask;
    #propagate
    #from scipy.signal import convolve2d
    #slow! #E = convolve2d(U,spherical, mode='same', boundary='fill', fillvalue=0);
    #---
    #import cv2
    #only float! #E = cv2.filter2D(src=U, ddepth=-1, kernel=spherical)
    #---
    E = convolve2dfft(U,spherical)
    return E

### save wavefield for SLM

def crop1080(img):
    #crop middle to 1080x1920 SLM
    M,N,_ = img.shape
    s,n,w,o = floor((M-1080)/2), ceil((M-1080)/2), floor((N-1920)/2), ceil((N-1920)/2)
    if s>0:
        img = img[s:n+1080,:,:]
    elif s<0:
        img = np.pad(img, [[abs(s),abs(n)],[0,0],[0,0]])
    if w>0:
        img = img[:,w:o+1920,:]
    elif w<0:
        img = np.pad(img, [[0,0],[abs(w),abs(o)],[0,0]])
    return img

def phases4F(E,offset):
    #seperate modulation of phase and amplitude
    #r is amplitude, g/b are phases
    #amplitude is modulated using phasemodulation by interference of the light with itself
    
    #wave field
    #E0 = ...
    #ampli = scale( np.abs(E0) ,0,1)
    #phase = np.angle(E0)
    #E0 = ampli*np.exp(i*phase)
    
    #modulation
    #phi = +-2*np.arccos(ampli)
    #E = np.exp(i*(phase+phi/2))
    #E = E*(0.5 + 0.5*np.exp(-i*phi))
    
    #phase encoding
    phi2 = np.angle(E);
    
    #amplitude encoding
    if offset<=0.5:
        phi1 = offset + np.flipud(np.fliplr( 2*np.arccos( np.abs(E) ) ))
    else:
        phi1 = offset - np.flipud(np.fliplr( 2*np.arccos( np.abs(E) ) ))
    
    #correct phase shift which was produced by amplitude modulation
    phi2 -= np.flipud(np.fliplr( 0.5*phi1 ))
    
    phi1 = np.mod(phi1,2*pi);
    phi2 = np.mod(phi2,2*pi);
    return phi1,phi2

def phasesRELPH(E):
    #divide into phases for two slms in michelson interferometer setup
    phi1 = np.angle(E) + np.arctan(np.sqrt(4./np.abs(E**2)-1));
    phi2 = np.angle(E) - np.arctan(np.sqrt(4./np.abs(E**2)-1));
    phi1 = np.real(phi1);
    phi2 = np.real(phi2);
    phi1 = np.mod(phi1,2*pi);
    phi2 = np.mod(phi2,2*pi);
    return phi1,phi2

def savePhase(phi1,phi2,phi3,lamb,name):
    M,N = max(np.shape(phi1) , np.shape(phi2) , np.shape(phi3))
    phases = np.zeros((M,N,3))
    phases[:,:,0] = phi1
    phases[:,:,1] = phi2
    phases[:,:,2] = phi3
    
    #phases = phases *lamb/0.000633; #correction
    phases = np.mod( phases/(2*pi) ,1)
    phases = np.uint8( np.clip( phases * 255 ,0,255) )
    ###phases = crop1080( phases )
    image = Image.fromarray( phases )
    image.save("./"+name)
    return None

### data manipulation

def compress_real_fft_coeff(fft2_coeff):
    N,M = np.shape(fft2_coeff)
    N2 = int(np.ceil(N/2))
    M2 = int(np.ceil(M/2))
    N_even = (np.mod(N,2)==0)
    M_even = (np.mod(M,2)==0)

    compressed = np.zeros((N,M),dtype=np.float)

    compressed[0,:] = np.concatenate(( np.real(fft2_coeff[0,0:(M2+M_even)]), np.imag(fft2_coeff[0,(M2+M_even):M]) ))
    compressed[1:N2,:] = np.real(fft2_coeff[1:N2,:])
    if N_even == 1:
        compressed[N2+N_even-1,:] = np.concatenate(( np.real(fft2_coeff[N2+N_even-1,0:(M2+N_even*M_even)]), np.imag(fft2_coeff[N2+N_even-1,(M2+N_even*M_even):M]) ))

    compressed[(N2+N_even):N,:] = np.imag(fft2_coeff[(N2+N_even):N,:])
    return compressed

def decompress_real_fft_coeff(compressed):
    N,M = np.shape(compressed)
    N2 = int(np.ceil(N/2))
    M2 = int(np.ceil(M/2))
    N_even = (np.mod(N,2)==0)
    M_even = (np.mod(M,2)==0)

    decompressed = np.zeros((N,M),dtype=np.complex)

    if N_even == 1:
        if M_even == 1:
            # M even && N even
            decompressed[0,:] = np.concatenate(( compressed[0,0:(M2+M_even)], compressed[0,M2-1:0:-1] )) + i* np.concatenate(( [0] , -1*compressed[0,M-1:(M2+M_even-1):-1] , [0] , compressed[0,(M2+M_even):M] ))
            decompressed[1:,0] = np.concatenate(( compressed[1:(N2+N_even),0].T, compressed[N2-1:0:-1,0].T )) + i* np.concatenate(( -1*compressed[N-1:(N2+N_even-1):-1,0].T , [0] , compressed[(N2+N_even):N,0].T ))
            decompressed[N2+N_even-1,1:] = np.concatenate(( compressed[N2+N_even-1,1:(M2+M_even)], compressed[N2+N_even-1,M2-1:0:-1] )) + i* np.concatenate(( -1*compressed[N2+N_even-1,M-1:(M2+M_even-1):-1] , [0] , compressed[N2+N_even-1,(M2+M_even):M] ))
        else:
            # M odd && N even
            decompressed[0,:] = np.concatenate(( compressed[0,0:(M2+M_even)], compressed[0,M2-1:0:-1] )) + i* np.concatenate(( [0] , -1*compressed[0,M-1:(M2+M_even-1):-1] , compressed[0,(M2+M_even):M] ))
            decompressed[1:,0] = np.concatenate(( compressed[1:(N2+N_even),0].T, compressed[N2-1:0:-1,0].T )) + i* np.concatenate(( -1*compressed[N-1:(N2+N_even-1):-1,0].T , [0] , compressed[(N2+N_even):N,0].T ))
            decompressed[N2+N_even-1,1:] = np.concatenate(( compressed[N2+N_even-1,1:(M2+M_even)], compressed[N2+N_even-1,M2-1:0:-1] )) + i* np.concatenate(( -1*compressed[N2+N_even-1,M-1:(M2+M_even-1):-1] , compressed[N2+N_even-1,(M2+M_even):M] ))
    else:
        if M_even == 1:
            # M even && N odd
            decompressed[0,:] = np.concatenate(( compressed[0,0:(M2+M_even)], compressed[0,M2-1:0:-1] )) + i* np.concatenate(( [0] , -1*compressed[0,M-1:(M2+M_even-1):-1] , [0] , compressed[0,(M2+M_even):M] ))
            decompressed[1:,0] = np.concatenate(( compressed[1:(N2+N_even),0].T, compressed[N2-1:0:-1,0].T )) + i* np.concatenate(( -1*compressed[N-1:(N2+N_even-1):-1,0].T , compressed[(N2+N_even):N,0].T ))
        else:
            # M odd && N odd
            decompressed[0,:] = np.concatenate(( compressed[0,0:(M2+M_even)], compressed[0,M2-1:0:-1] )) + i* np.concatenate(( [0] , -1*compressed[0,M-1:(M2+M_even-1):-1] , compressed[0,(M2+M_even):M] ))
            decompressed[1:,0] = np.concatenate(( compressed[1:(N2+N_even),0].T, compressed[N2-1:0:-1,0].T )) + i* np.concatenate(( -1*compressed[N-1:(N2+N_even-1):-1,0].T , compressed[(N2+N_even):N,0].T ))

    decompressed[1:N2,1:] = compressed[1:N2,1:] - i*np.rot90(compressed[(N2+N_even):,1:],2)
    decompressed[(N2+N_even):,1:] = np.rot90(compressed[1:N2,1:],2) + i*compressed[(N2+N_even):,1:]
    return decompressed

def bandlimit(x,percent):
    minlim,maxlim = 0.5*(1-percent) , 0.5*(1+percent)
    N,M = np.shape(x)
    x[:int(minlim*N),:] = 0
    x[:,:int(minlim*M)] = 0
    x[int(maxlim*N):,:] = 0
    x[:,int(maxlim*M):] = 0
    return x
