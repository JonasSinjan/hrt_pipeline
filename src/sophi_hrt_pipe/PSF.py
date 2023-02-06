import numpy as np
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
from scipy.optimize import leastsq
from scipy import signal as sig

def Zernike_polar(coefficients, r, u):
   #Z= np.insert(np.array([0,0,0]),3,coefficients)  
   Z =  coefficients
   #Z1  =  Z[0]  * 1*(np.cos(u)**2+np.sin(u)**2)
   #Z2  =  Z[1]  * 2*r*np.cos(u)
   #Z3  =  Z[2]  * 2*r*np.sin(u)

   Z4  =  Z[0]  * np.sqrt(3)*(2*r**2-1)  #defocus

   Z5  =  Z[1]  * np.sqrt(6)*r**2*np.sin(2*u) #astigma
   Z6  =  Z[2]  * np.sqrt(6)*r**2*np.cos(2*u)

   Z7  =  Z[3]  * np.sqrt(8)*(3*r**2-2)*r*np.sin(u) #coma
   Z8  =  Z[4]  * np.sqrt(8)*(3*r**2-2)*r*np.cos(u)

   Z9  =  Z[5]  * np.sqrt(8)*r**3*np.sin(3*u) #trefoil
   Z10=  Z[6] * np.sqrt(8)*r**3*np.cos(3*u)

   Z11 =  Z[7] * np.sqrt(5)*(1-6*r**2+6*r**4) #secondary spherical

   Z12 =  Z[8] * np.sqrt(10)*(4*r**2-3)*r**2*np.cos(2*u)  #2 astigma
   Z13 =  Z[9] * np.sqrt(10)*(4*r**2-3)*r**2*np.sin(2*u)

   Z14 =  Z[10] * np.sqrt(10)*r**4*np.cos(4*u) #tetrafoil
   Z15 =  Z[11] * np.sqrt(10)*r**4*np.sin(4*u)

   Z16 =  Z[12] * np.sqrt(12)*(10*r**4-12*r**2+3)*r*np.cos(u) #secondary coma
   Z17 =  Z[13] * np.sqrt(12)*(10*r**4-12*r**2+3)*r*np.sin(u)

   Z18 =  Z[14] * np.sqrt(12)*(5*r**2-4)*r**3*np.cos(3*u) #secondary trefoil
   Z19 =  Z[15] * np.sqrt(12)*(5*r**2-4)*r**3*np.sin(3*u)

   Z20 =  Z[16] * np.sqrt(12)*r**5*np.cos(5*u) #pentafoil
   Z21 =  Z[17] * np.sqrt(12)*r**5*np.sin(5*u)

   Z22 =  Z[18] * np.sqrt(7)*(20*r**6-30*r**4+12*r**2-1) #spherical

   Z23 =  Z[19] * np.sqrt(14)*(15*r**4-20*r**2+6)*r**2*np.sin(2*u) #astigma
   Z24 =  Z[20] * np.sqrt(14)*(15*r**4-20*r**2+6)*r**2*np.cos(2*u)

   Z25 =  Z[21] * np.sqrt(14)*(6*r**2-5)*r**4*np.sin(4*u)#trefoil
   Z26 =  Z[22] * np.sqrt(14)*(6*r**2-5)*r**4*np.cos(4*u)

   Z27 =  Z[23] * np.sqrt(14)*r**6*np.sin(6*u) #hexafoil 
   Z28 =  Z[24] * np.sqrt(14)*r**6*np.cos(6*u)

   Z29 =  Z[25] * 4*(35*r**6-60*r**4+30*r**2-4)*r*np.sin(u) #coma
   Z30 =  Z[26] * 4*(35*r**6-60*r**4+30*r**2-4)*r*np.cos(u)

   Z31 =  Z[27] * 4*(21*r**4-30*r**2+10)*r**3*np.sin(3*u)#trefoil
   Z32 =  Z[28] * 4*(21*r**4-30*r**2+10)*r**3*np.cos(3*u)

   Z33 =  Z[29] * 4*(7*r**2-6)*r**5*np.sin(5*u) #pentafoil
   Z34 =  Z[30] * 4*(7*r**2-6)*r**5*np.cos(5*u)

   Z35 =  Z[31] * 4*r**7*np.sin(7*u) #heptafoil
   Z36 =  Z[32] * 4*r**7*np.cos(7*u)

   Z37 =  Z[33] * 3*(70*r**8-140*r**6+90*r**4-20*r**2+1) #spherical

   ZW = Z4+Z5+Z6+Z7+Z8+Z9+Z10+Z11+Z12+Z13+Z14+Z15+Z16+ Z17+Z18+Z19+Z20+Z21+Z22+Z23+ Z24+Z25+Z26+Z27+Z28+ Z29+ Z30+ Z30+ Z31+ Z32+ Z33+ Z34+ Z35+ Z36+ Z37
   return ZW


def pupil_size(D,lam,pix,size):
    pixrad = pix*np.pi/(180*3600)  # Pixel-size in radians
    nu_cutoff = D/lam      # Cutoff frequency in rad^-1
    deltanu = 1./(size*pixrad)     # Sampling interval in rad^-1
    rpupil = nu_cutoff/(2*deltanu) #pupil size in pixels
    return int(rpupil)


## function for making the phase in a unit circle
def phase(coefficients,rpupil):
   r = 1
   x = np.linspace(-r, r, 2*rpupil)
   y = np.linspace(-r, r, 2*rpupil)

   [X,Y] = np.meshgrid(x,y) 
   R = np.sqrt(X**2+Y**2)
   theta = np.arctan2(Y, X)
      
   Z = Zernike_polar(coefficients,R,theta)
   Z[R>1] = 0
   return Z


def pupil_foc(coefficients,size,rpupil):
    #rpupil = pupil_size(D,lam,pix,size)
    A = np.zeros([size,size])
    A[size//2-rpupil+1:size//2+rpupil+1,size//2-rpupil+1:size//2+rpupil+1]= phase(coefficients,rpupil)
    aberr =  np.exp(1j*A)
    return aberr



def mask(rpupil, size):
   r = 1
   x = np.linspace(-r, r, 2*rpupil)
   y = np.linspace(-r, r, 2*rpupil) 

   [X,Y] = np.meshgrid(x,y) 
   R = np.sqrt(X**2+Y**2)
   theta = np.arctan2(Y, X)
   M = 1*(np.cos(theta)**2+np.sin(theta)**2)
   M[R>1] = 0
   Mask =  np.zeros([size,size])
   Mask[size//2-rpupil+1:size//2+rpupil+1,size//2-rpupil+1:size//2+rpupil+1]= M
   return Mask

def PSF(mask,abbe):
   ## making zero where the aberration is equal to 1 (the zero background)
   abbe_z = np.zeros((len(abbe),len(abbe)),dtype=complex)
   abbe_z = mask*abbe
   PSF = ifftshift(fft2(fftshift(abbe_z))) #from brandon
   PSF = (np.abs(PSF))**2 #or PSF*PSF.conjugate()
   #PSF = PSF/PSF.sum()
   return PSF


## function to compute the OTF from PSF (to be used in PD fit )
def OTF(psf):
    otf = ifftshift(psf)
    otf = fft2(otf)
    otf = otf/np.real(otf[0,0])
    #otf = otf/otf.max() # or otf_max = otf[size/2,size/2] if max is shifted to center
   
    return otf


def noise_mask_high(size,cut_off):
   X = np.linspace(-0.5,0.5,size)
   x,y = np.meshgrid(X,X)
   mask = np.zeros((size,size))
   m = x * x + y * y <= cut_off**2
   mask[m] = 1
   return mask



def apo2d(masi,perc):
   s = masi.shape
   # edge = 100./perc
   mean = np.mean(masi)
   masi = masi-mean
   # xmask = np.ones(s[1])
   # ymask = np.ones(s[0])
   # smooth_x = int(s[1]/edge)
   # smooth_y = int(s[0]/edge)

   # for i in range(0,smooth_x):
   #    xmask[i] = (1.-np.cos(np.pi*float(i)/float(smooth_x)))/2.
   #    ymask[i] = (1.-np.cos(np.pi*float(i)/float(smooth_y)))/2.
    
   # xmask[s[1] - smooth_x:s[1]] = (xmask[0:smooth_x])[::-1]
   # ymask[s[0] - smooth_y:s[0]] = (ymask[0:smooth_y])[::-1]

   mask = sig.tukey(s[0], alpha=perc*2/100)[np.newaxis]*sig.tukey(s[1], alpha=perc*2/100)[:,np.newaxis]
   masi *= mask

   # #mask_x = np.outer(xmask,xmask)
   # #mask_y = np.outer(ymask,ymask)
   # for i in range(0,s[1]):
   #    masi[:,i] = masi[:,i]*xmask[i]
   # for i in range(0,s[0]):
   #    masi[i,:] = masi[i,:]*ymask[i]
   masi = masi+mean
   return masi


def Wienerfilter(img,t0,reg,cut_off,ap,size):
    noise_filter = fftshift(noise_mask_high(size,cut_off))

    im0 = apo2d(img,ap)
    d0 = fft2(im0)
    scene = noise_filter*d0*(np.conj(t0)/(np.abs(t0)**2+reg))
    scene2 = ifft2(scene).real  
    return scene2


def Wienerfilter_th(img,t0,reg,cut_off,ap,size,t0_th):
    noise_filter = fftshift(noise_mask_high(size,cut_off))

    im0 = apo2d(img,ap)
    d0 = fft2(im0)
    scene = t0_th*noise_filter*d0*(np.conj(t0)/(np.abs(t0)**2+reg))
    scene2 = ifft2(scene).real
    return scene2


def combine_all_PD():
   coefficients = np.zeros(38)
   coefficients[:23] = np.array([ 1.94428601,  0.36762832,  0.61874354, -0.24504269, -0.25386458,
         0.18834626, -1.26666803,  1.15594472,  0.02213881,  0.14563544,
         -0.29933957,  0.01619363, -0.15828751, -0.04142371,  0.07867531,
         -0.17719113,  0.04658804,  0.1191974 ,  0.04424823,  0.05749467,
         0.03511377, -0.0163282 , -0.12770855])
      
   coef_stp = np.zeros(38)
   coef_stp[:10] = np.array([ 0.32725408,  -0.01148539,  0.46752924,  0.00413511, 0.01964055,
                              0.13448377,  -0.59294403,  0.38801043,  0.03050344,  -0.07010066])

   coef_phi5 = np.zeros(38)
   coef_phi5[:10] = np.array([-0.16508714,  -0.10259014,  0.35210216,  0.22433325, 0.14908674,
                              0.06424761,  -0.3348285 ,  0.22482951, -0.02742681,  -0.08395544])
   coef_rsw = coefficients

   Z = np.zeros((10,3))
   for i in range(10):
      Z[i] = np.array([coef_phi5[i], coef_stp[i],coef_rsw[i]])
   Z = Z/(2*np.pi)

   return Z


def func(params, x):
   a, b, c = params
   return a * x * x + b * x + c

def error(params, X, Y):
   return func(params, X) - Y

def slovePara(X,Y):
   p0 = [1, 1, 1]

   Para = leastsq(error, p0, args=(X, Y))
   return Para

def solution(X,Y,zernike,d_in):
   Para = slovePara(X,Y)
   a, b, c = Para[0]
   aberr = a * d_in**2 + b * d_in + c
   return aberr*2*np.pi



def build_zernikes(Z,d_in):
   n = np.array([0.82,0.52,0.334])
   coefficients = np.zeros(38)
   defocus = solution(n,Z[0],'defocus',d_in)
# Output the final result
   Y_trefoil = solution(n,Z[5],'Y-Trefoil',d_in)

   X_trefoil =  solution(n,Z[6],'X-Trefoil',d_in)
   SA =solution(n,Z[7],'Spherical aberration',d_in)
   coefficients[:23] = np.array([defocus,  0.36762832,  0.61874354, -0.24504269, -0.25386458,
         Y_trefoil, X_trefoil,  SA,  0.02213881,  0.14563544,
         -0.29933957,  0.01619363, -0.15828751, -0.04142371,  0.07867531,
         -0.17719113,  0.04658804,  0.1191974 ,  0.04424823,  0.05749467,
         0.03511377, -0.0163282 , -0.12770855])
   return coefficients


def make_wf(size,coefficients):

   D = 140
   lam = 617.3*10**(-6)
   pix = 0.5
   f = 4125.3
   ap = 10
   rpupil = pupil_size(D,lam,pix,size)
   Mask = mask(rpupil,size)
   A_f = pupil_foc(coefficients,size,rpupil)
   psf_foc = PSF(Mask,A_f)
   t0 = OTF(psf_foc)
   return t0

def make_wf_th(size):
   coe = np.zeros(38)
   D = 140
   lam = 617.3*10**(-6)
   pix = 0.5
   f = 4125.3
   ap = 10
   rpupil = pupil_size(D,lam,pix,size)
   Mask = mask(rpupil,size)
   A_f = pupil_foc(coe,size,rpupil)
   psf_foc = PSF(Mask,A_f)
   t0 = OTF(psf_foc)
   return t0

def restore_stokes_cube(stokes_data, header, orbit = 'perihelion',aberr_cor=False):

   size = stokes_data[:,:,0,0].shape[0]
   if size < 2048:
      edge_mask = np.zeros((size,size))
      edge_mask[3:-3,3:-3] = 1
   else:
      edge_mask = np.ones((size,size))
   pad_width = int(size*10/(100-10*2))
   res_stokes = np.zeros((size+pad_width*2,size+pad_width*2,4,6))
   pad_size = size+pad_width*2
   d_in = header['DSUN_AU']

   if orbit=='perihelion':
    Z = combine_all_PD()
    coefficients = build_zernikes(Z,d_in)
    t0 = make_wf(pad_size,coefficients)
    if aberr_cor:
     t0_th = make_wf_th(pad_size)
     for i in range(4):
       for j in range(6):
         im0 = stokes_data[:,:,i,j] * edge_mask
         im0 = np.pad(im0, pad_width=((pad_width, pad_width), (pad_width, pad_width)), mode='symmetric')

         res_stokes[:,:,i,j] = Wienerfilter_th(im0,t0,0.01,0.5,10,pad_size,t0_th)

    else:
     for i in range(4):
       for j in range(6):
         im0 = stokes_data[:,:,i,j] * edge_mask
         im0 = np.pad(im0, pad_width=((pad_width, pad_width), (pad_width, pad_width)), mode='symmetric')

         res_stokes[:,:,i,j] = Wienerfilter(im0,t0,0.01,0.5,10,pad_size)


                         

   elif orbit == '0.5':
       coefficients = np.zeros(38)
       coefficients[:10] = np.array([ 0.32725408,  -0.01148539,  0.46752924,  0.00413511, 0.01964055, 0.13448377,  -0.59294403,  0.38801043,  0.03050344,  -0.07010066])                     
       t0 = make_wf(pad_size,coefficients)
        
       if aberr_cor:
        t0_th = make_wf_th(pad_size)
        for i in range(4):
         for j in range(6):
          im0 = stokes_data[:,:,i,j] * edge_mask
          im0 = np.pad(im0, pad_width=((pad_width, pad_width), (pad_width, pad_width)), mode='symmetric')

          res_stokes[:,:,i,j] = Wienerfilter_th(im0,t0,0.01,0.5,10,pad_size,t0_th)

       else:
        for i in range(4):
          for j in range(6):
           im0 = stokes_data[:,:,i,j] * edge_mask
           im0 = np.pad(im0, pad_width=((pad_width, pad_width), (pad_width, pad_width)), mode='symmetric')

           res_stokes[:,:,i,j] = Wienerfilter(im0,t0,0.01,0.5,10,pad_size)
   print('-->>>>>>> PSF deconvolution is done with Z='+str(coefficients)+'\nAberration correction is set to '+str(aberr_cor))

   res_stokes = res_stokes[pad_width:-pad_width,pad_width:-pad_width] * edge_mask[:,:,np.newaxis,np.newaxis]
   return res_stokes
