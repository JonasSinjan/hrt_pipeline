from astropy.io import fits
import numpy as np
import os
import matplotlib.pyplot as plt

class bcolors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'
  RESET = '\u001b[0m'

def printc(*args, color = bcolors.RESET, **kwargs):
  """My custom print() function."""
  print(u"\u001b"+f"{color}", end='\r')
  print(*args, **kwargs)
  print(u"\u001b"+f"{bcolors.RESET}", end='\r')
  return 

def load_fits(path):
  """
  load the fits file
  
  Parameters
  ----------
  path: string, location of the fits file
  
  Output
  ------
  data: numpy array, of stokes images in (row, col, wv, pol) 
  header: hdul header object, header of the fits file
  """

  hdul_tmp = fits.open(f'{path}')
  
  data = np.asarray(hdul_tmp[0].data, dtype = np.float32)

  header = hdul_tmp[0].header
  
  return data, header 


def get_data(path, scaling = True, bit_convert_scale = True, scale_data = True):
    """
    load science data from path
    """
    try:
        data, header = load_fits(path)

        if bit_convert_scale: #conversion from 24.8bit to 32bit
            data /=  256.

        if scaling:
            
            accu = header['ACCACCUM']*header['ACCROWIT']*header['ACCCOLIT'] #getting the number of accu from header

            data /= accu

            printc(f"Dividing by number of accumulations: {accu}",color=bcolors.OKGREEN)

        if scale_data: #not for commissioning data

            try:    
                maxRange = fits.open(path)[9].data['PHI_IMG_maxRange']
            
                data *= maxRange[0]/maxRange[-1]
            except IndexError:
                data *= 81920/128
                
        return data, header

    except Exception:
        printc("ERROR, Unable to open fits file: {}",path,color=bcolors.FAIL)
        raise ValueError()
       

def fits_get_sampling(file,verbose = False):
    '''
    wave_axis,voltagesData,tunning_constant,cpos = fits_get_sampling(file)
    No S/C velocity corrected!!!
    cpos = 0 if continuum is at first wavelength and = 5 if continuum is at the end

    From SPGPylibs PHITools
    '''
    #print('-- Obtaining voltages......')
    fg_head = 3
    #try:
    with fits.open(file) as hdu_list:
        header = hdu_list[fg_head].data
        j = 0
        dummy = 0
        voltagesData = np.zeros((6))
        tunning_constant = 0.0
        ref_wavelength = 0.0
        for v in header:
            #print(v)
            if (j < 6):
                if tunning_constant == 0:
                    tunning_constant = float(v[4])/1e9
                if ref_wavelength == 0:
                    ref_wavelength = float(v[5])/1e3
                #print(dummy, v[2], type(dummy), type(v[2]))
                if np.abs(np.abs(float(v[2])) - np.abs(dummy)) > 5: #check that the next voltage is more than 5 from the previous, as voltages change slightly
                    #print(dummy, v[2])
                    voltagesData[j] = float(v[2])
                    dummy = voltagesData[j] 
                    j += 1

    #except Exception:
    #   print("Unable to open fits file: {}",file)     
    #print(voltagesData)
    d1 = voltagesData[0] - voltagesData[1]
    d2 = voltagesData[4] - voltagesData[5]
    if np.abs(d1) > np.abs(d2):
        cpos = 0
    else:
        cpos = 5
    if verbose:
        print('Continuum position at wave: ', cpos)
    wave_axis = voltagesData*tunning_constant + ref_wavelength  #6173.3356
    #print(wave_axis)
    return wave_axis,voltagesData,tunning_constant,cpos


def check_filenames(data_f):
    """
    checks if the science scans have the same DID - this would cause an issue for naming the output demod files
    """

    scan_name_list = [str(scan.split('.fits')[0][-10:]) for scan in data_f]

    seen = set()
    uniq_scan_DIDs = [x for x in scan_name_list if x in seen or seen.add(x)] #creates list of unique DIDs from the list

    #print(uniq_scan_DIDs)
    #print(scan_name_list)S
    if uniq_scan_DIDs == []:
        print("The scans' DIDs are all unique")

    else:

        for x in uniq_scan_DIDs:
            number = scan_name_list.count(x)
            if number > 1: #if more than one
                print(f"The DID: {x} is repeated {number} times")
                i = 1
                for index, name in enumerate(scan_name_list):
                    if name == x:
                        scan_name_list[index] = name + f"_{i}" #add _1, _2, etc to the file name, so that when written to output file not overwriting
                        i += 1

        print("The New DID list is: ", scan_name_list)

    return scan_name_list


def check_size(data_arr):
    """
    checks if science scans have same dimensions
    """
    first_shape = data_arr[0].shape
    result = all(element.shape == first_shape for element in data_arr)
    if (result):
        print("All the scan(s) have the same dimension")

    else:
        print("The scans have different dimensions! \n Ending process")

        exit()


def check_cpos(cpos_arr):
    """
    checks if the science scans have the same continuum positions
    """
    first_cpos = cpos_arr[0]
    result = all(c_position == first_cpos for c_position in cpos_arr)
    if (result):
        print("All the scan(s) have the same continuum wavelength position")

    else:
        print("The scans have different continuum_wavelength postitions! Please fix \n Ending Process")

        exit()


def compare_cpos(data,cpos,cpos_ref):
    """
    checks if flat continuum same as data, if not try to move flat around - this assumes that there was a mistake with the continuum position in the flat
    """
    if cpos != cpos_ref:
        print("The flat field continuum position is not the same as the data, trying to correct.")

        if cpos == 5 and cpos_ref == 0:

            return np.roll(data, 1, axis = -1)

        elif cpos == 0 and cpos_ref == 5:

            return np.roll(data, -1, axis = -1)

        else:
            print("Cannot reconcile the different continuum positions. \n Ending Process.")

            exit()
    else:
        return data


def check_pmp_temp(hdr_arr):
    """
    check science scans have same PMP temperature set point
    """
    first_pmp_temp = hdr_arr[0]['HPMPTSP1']
    result = all(hdr['HPMPTSP1'] == first_pmp_temp for hdr in hdr_arr)
    if (result):
        print(f"All the scan(s) have the same PMP Temperature Set Point: {first_pmp_temp}")
        pmp_temp = str(first_pmp_temp)
        return pmp_temp
    else:
        print("The scans have different PMP Temperatures! Please fix \n Ending Process")

        exit()


def check_IMGDIRX(hdr_arr):
    """
    check if all scans contain imgdirx keyword
    """
    if all('IMGDIRX' in hdr for hdr in hdr_arr):
        header_imgdirx_exists = True
        first_imgdirx = hdr_arr[0]['IMGDIRX']
        result = all(hdr['IMGDIRX'] == first_imgdirx for hdr in hdr_arr)
        if (result):
            print(f"All the scan(s) have the same IMGDIRX keyword: {first_imgdirx}")
            imgdirx_flipped = str(first_imgdirx)
            
            return header_imgdirx_exists, imgdirx_flipped
        else:
            print("The scans have different IMGDIRX keywords! Please fix \n Ending Process")
            exit()
    else:
        header_imgdirx_exists = False
        print("Not all the scan(s) contain the 'IMGDIRX' keyword! Assuming all not flipped - Proceed with caution")
        return header_imgdirx_exists, False


def compare_IMGDIRX(flat,header_imgdirx_exists,imgdirx_flipped,header_fltdirx_exists,fltdirx_flipped):
    """
    returns flat that matches the orientation of the science data
    """
    if header_imgdirx_exists and imgdirx_flipped == 'YES': 
        #if science is flipped
        if header_fltdirx_exists:
            if fltdirx_flipped == 'YES':
                return flat
            else:
                return flat[:,:,::-1]
        else:
            return flat[:,:,::-1]
    elif (header_imgdirx_exists and imgdirx_flipped == 'NO') or not header_imgdirx_exists: 
        #if science is not flipped, or keyword doesnt exist, then assumed not flipped
        if header_fltdirx_exists:
            if fltdirx_flipped == 'YES':
                return flat[:,:,::-1] #flip flat back to match science
            else:
                return flat
        else:
            return flat
    else:
        return flat


def stokes_reshape(data):
    """
    converting science to [y,x,pol,wv,scans]
    """
    data_shape = data.shape
    if data_shape[2] == 25:
        data = data[:,:,:24]
        data_shape = data.shape
    data = data.reshape(data_shape[0],data_shape[1],6,4,data_shape[-1]) #separate 24 images, into 6 wavelengths, with each 4 pol states
    data = np.moveaxis(data, 2,-2)
    
    return data
    

def fix_path(path,dir='forward',verbose=False):
    """
    From SPGPylibs PHITools
    """
    path = repr(path)
    if dir == 'forward':
        path = path.replace(")", "\)")
        path = path.replace("(", "\(")
        path = path.replace(" ", "\ ")
        path = os.path.abspath(path).split("'")[1]
        if verbose == True:
            print('forward')
            print(path)
        return path
    elif dir == 'backward':
        path = path.replace("\\\\", "")
        path = path.split("'")[1]
        if verbose == True:
            print('backward')
            print(path)
        return path
    else:
        pass   


def filling_data(arr, thresh, mode, axis = -1):
    from scipy.interpolate import CubicSpline
    
    a0 = np.zeros(arr.shape)
    a0 = arr.copy()
    if mode == 'max':
        a0[a0>thresh] = np.nan
    if mode == 'min':
        a0[a0<thresh] = np.nan
    if mode == 'abs':
        a0[np.abs(a0)>thresh] = np.nan
    if 'exact rows' in mode.keys():
        rows = mode['exact rows']
        for r in rows:
            a0[r] = np.nan
        axis = 1
    if 'exact columns' in mode.keys():
        cols = mode['exact columns']
        for c in cols:
            a0[:,r] = np.nan
        axis = 0
    
    N = arr.shape[axis]; n = arr.shape[axis-1]
    
    with np.errstate(divide='ignore'):
        for i in range(N):
            a1 = a0.take(i, axis=axis)
            nans, index = np.isnan(a1), lambda z: z.nonzero()[0]
            if nans.sum()>0:
                a1[nans] = CubicSpline(np.arange(n)[~nans], a1[~nans])(np.arange(n))[nans]
                if axis == 0:
                    a0[i] = a1
                else:
                    a0[:,i] = a1
    return a0
    

    """
    Returns a boolean array with True if points are outliers and False 
        otherwise.
        Parameters:
        -----------
            points : An numobservations by numdimensions array of observations
            thresh : The modified z-score to use as a threshold. Observations with
                a modified z-score (based on the median absolute deviation) greater
                than this value will be classified as outliers.
        Returns:
        --------
            mask : A numobservations-length boolean array.
        References:
        ----------
            Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
            Handle Outliers", The ASQC Basic References in Quality Control:
            Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
        """

def limb_fitting(img, hdr, mar=200):
    def _residuals(p,x,y):
        xc,yc,R = p
        return R**2 - (x-xc)**2 - (y-yc)**2
    
    def _is_outlier(points, thresh=3):
        if len(points.shape) == 1:
            points = points[:,None]
        median = np.median(points, axis=0)
        diff = np.sum((points - median)**2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation

        return modified_z_score > thresh
    
    def _interp(y, m, kind='cubic',fill_value='extrapolate'):
        
        from scipy.interpolate import interp1d
        x = np.arange(np.size(y))
        fn = interp1d(x, y, kind=kind,fill_value=fill_value)
        x_new = np.arange(len(y), step=1./m)
        return fn(x_new)
    
    def _circular_mask(h, w, center, radius):

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius
        return mask
    
    def _image_derivative(d):
        import numpy as np
        from scipy.signal import convolve
        kx = np.asarray([[1,0,-1], [1,0,-1], [1,0,-1]])
        ky = np.asarray([[1,1,1], [0,0,0], [-1,-1,-1]])

        kx=kx/3.
        ky=ky/3.

        SX = convolve(d, kx,mode='same')
        SY = convolve(d, ky,mode='same')

        A=SX+SY

        return A

    from scipy import optimize
    
    Rpix=(hdr['RSUN_ARC']/hdr['CDELT1'])
    center=[hdr['CRPIX1']-hdr['CRVAL1']/hdr['CDELT1']-1,hdr['CRPIX2']-hdr['CRVAL2']/hdr['CDELT2']-1]
    wcs_mask = _circular_mask(img.shape[0],img.shape[1],center,Rpix)
    wcs_grad = _image_derivative(wcs_mask)
    
    x_p = img.shape[1] - (center[0]+Rpix)
    y_p = img.shape[0] - (center[1]+Rpix)
    x_n = center[0]-Rpix
    y_n = center[1]-Rpix
    
    side = ''
    if x_p > 0: 
        side += 'W'
    elif x_n > 0:
        side += 'E'
    if y_p > 0:
        side += 'N'
    if y_n > 0:
        side += 'S'
    
    if side == '':
        print('Limb is not in the FoV according to WCS keywords')

        return None, None
    
    if 'W' in side or 'E' in side:
        mode = 'rows'
    else:
        mode = 'columns'
    
    if 'W' in side or 'N' in side:
        norm = -1
    else:
        norm = 1
    
    print('Limb:',side)
    
    if mode == 'columns':
        xi = np.arange(100,2000,50)
        yi = []
        m = 10
        for c in xi:
            wcs_col = wcs_grad[1:,c]*norm
            mm = wcs_col.mean(); ss = wcs_col.std()
            try:
                y_start = np.where(wcs_col>mm+5*ss)[0][0]+1
            except:
                y_start = wcs_col.argmax()+1
            
            col = img[y_start-mar:y_start+mar,c]
            g = np.gradient(col*norm)
            gi = _interp(g,m)
            
            yi += [gi.argmax()/m+y_start-mar]
        yi = np.asarray(yi)
        xi = xi[~_is_outlier(yi)]
        yi = yi[~_is_outlier(yi)]
    
    elif mode == 'rows':
        yi = np.arange(100,2000,50)
        xi = []
        m = 10
        for r in yi:
            wcs_row = wcs_grad[r,1:]*norm
            mm = wcs_row.mean(); ss = wcs_row.std()
            try:
                x_start = np.where(wcs_row>mm+5*ss)[0][0]+1
            except:
                x_start = wcs_row.argmax()+1
                
            row = img[r,x_start-mar:x_start+mar]
            g = np.gradient(row*norm)
            gi = _interp(g,m)
            
            xi += [gi.argmax()/m+x_start-mar]
        xi = np.asarray(xi)
        out_one = _is_outlier(xi)
        out_two = ~out_one
        yi = yi[~_is_outlier(xi)]
        xi = xi[~_is_outlier(xi)]


    p = optimize.least_squares(_residuals,x0 = [center[0],center[1],Rpix], args=(xi,yi))
        
    mask80 = _circular_mask(img.shape[0],img.shape[1],[p.x[0],p.x[1]],p.x[2]*.8)
    return _circular_mask(img.shape[0],img.shape[1],[p.x[0],p.x[1]],p.x[2]), mask80


def auto_norm(file_name):
    d = fits.open(file_name)
    try:
        print('PHI_IMG_maxRange 0:',d[9].data['PHI_IMG_maxRange'][0])
        print('PHI_IMG_maxRange -1:',d[9].data['PHI_IMG_maxRange'][-1])
        norm = d[9].data['PHI_IMG_maxRange'][0]/ \
        d[9].data['PHI_IMG_maxRange'][-1]/256/ \
        (d[0].header['ACCCOLIT']*d[0].header['ACCROWIT']*d[0].header['ACCACCUM'])
    except:
        norm = 1/256/ \
        (d[0].header['ACCCOLIT']*d[0].header['ACCROWIT']*d[0].header['ACCACCUM'])
    print('accu:',(d[0].header['ACCCOLIT']*d[0].header['ACCROWIT']*d[0].header['ACCACCUM']))
    return norm

# new functions by DC ######################################

def fft_shift(img,shift):
    """
    im: 2D-image to be shifted
    shift = [dy,dx] shift in pixel
    """
    
    try:
        import pyfftw.interfaces.numpy_fft as fft
    except:
        import np.fft as fft
    sz = img.shape
    ky = fft.ifftshift(np.linspace(-np.fix(sz[0]/2),np.ceil(sz[0]/2)-1,sz[0]))
    kx = fft.ifftshift(np.linspace(-np.fix(sz[1]/2),np.ceil(sz[1]/2)-1,sz[1]))

    img_fft = fft.fft2(img)
    shf = np.exp(-2j*np.pi*(ky[:,np.newaxis]*shift[0]/sz[0]+kx[np.newaxis]*shift[1]/sz[1]))
    
    img_fft *= shf
    img_shf = fft.ifft2(img_fft).real
    
    return img_shf
    
def SPG_shifts_FFT(data,norma=True,prec=100,coarse_prec = 1.5,sequential = False):

    """
    From SPGPylibs. Same function used for FDT pipeline, adapted by DC
    At least two images should be provided!
    s_y, s_x, simage = PHI_shifts_FFT(image_cropped,prec=500,verbose=True,norma=False)
    (row_shift, column_shift) deficed as  center = center + (y,x) 
    """
    
    def sampling(N):
        """
        From SPGPylibs. Same function used for FDT pipeline.
        This function creates a grid of points with NxN dimensions for calling the
        Zernike polinomials.
        Output:
            X,Y: X and Y meshgrid of the detector
        """
        if N%2 != 0:
            print('Number of pixels must be an even integer!')
            return
        x=np.linspace(-N/2,N/2,N)
        y=np.copy(x)
        X,Y=np.meshgrid(x,y)
        return X,Y 

    def aperture(X,Y,N,R):
        """
        From SPGPylibs. Same function used for FDT pipeline.
        This function calculates a simple aperture function that is 1 within
        a circle of radius R, takes and intermediate value between 0
        and 1 in the edge and 0 otherwise. The values in the edges are calculated
        according to the percentage of area corresponding to the intersection of the
        physical aperture and the edge pixels.
        http://photutils.readthedocs.io/en/stable/aperture.html
        Input:
            X,Y: meshgrid with the coordinates of the detector ('sampling.py')
            R: radius (in pixel units) of the mask
        Output:
            A: 2D array with 0s and 1s
        """
        from photutils import CircularAperture
        A=CircularAperture((N/2,N/2),r=R) #Circular mask (1s in and 0s out)
        A=A.to_mask(method='exact') #Mask with exact value in edge pixels
        A=A.to_image(shape=(N,N)) #Conversion from mask to image
        return A
        
    def dft_fjbm(F,G,kappa,dftshift,nr,nc,Nr,Nc,kernr,kernc):
        """
        From SPGPylibs. Same function used for FDT pipeline.
        Calculates the shift between a couple of images 'f' and 'g' with subpixel
        accuracy by calculating the IFT with the matrix multiplication tecnique.
        Shifts between images must be kept below 1.5 'dftshift' for the algorithm
        to work.
        Input: 
            F,G: ffts of images 'f' and 'g' without applying any fftshift
            kappa: inverse of subpixel precision (kappa=20 > 0.005 pixel precision)
        Output:
        """
        #DFT by matrix multiplication
        M=F*np.conj(G) #Cross-correlation
        CC=kernr @ M @ kernc
        CCabs=np.abs(CC)
        ind = np.unravel_index(np.argmax(CCabs, axis=None), CCabs.shape)
        CCmax=CC[ind]
        rloc,cloc=ind-dftshift
        row_shift=-rloc/kappa
        col_shift=-cloc/kappa
        rg00=np.sum(np.abs(F)**2)
        rf00=np.sum(np.abs(G)**2)
        error=np.sqrt(1-np.abs(CCmax)**2/(rg00*rf00))
        Nc,Nr=np.meshgrid(Nc,Nr)

        Gshift=G*np.exp(1j*2*np.pi*(-row_shift*Nr/nr-col_shift*Nc/nc)) 
        return error,row_shift,col_shift,Gshift


    #Normalization for each image
    sz,sy,sx = data.shape
    f=np.copy(data)
    if norma == True:
        norm=np.zeros(sz)
        for i in range(sz):
            norm[i]=np.mean(data[i,:,:])
            f[i,:,:]=data[i,:,:]/norm[i]

    #Frequency cut
    wvl=617.3e-9
    D = 0.14  #HRT
    foc = 4.125 #HRT
    fnum = foc / D
    nuc=1/(wvl*fnum) #Critical frequency (1/m)
    N=sx #Number of pixels per row/column (max. 2048)
    deltax = 10e-6 #Pixel size
    deltanu=1/(N*deltax)
    R=(1/2)*nuc/deltanu
    nuc=2*R#Max. frequency [pix]

    #Mask
    X,Y = sampling(N)
    mask = aperture(X,Y,N,R)

    #Fourier transform
    f0=f[0,:,:]
    #pf.movie(f0-f,'test.mp4',resol=1028,axis=0,fps=5,cbar='yes',cmap='seismic')
    F=np.fft.fft2(f0)

    #Masking
    F=np.fft.fftshift(F)
    F*=mask
    F=np.fft.ifftshift(F)

    #FJBM algorithm
    kappa=prec
    n_out=np.ceil(coarse_prec*2.*kappa)
    dftshift=np.fix(n_out/2)
    nr,nc=f0.shape
    Nr=np.fft.ifftshift(np.arange(-np.fix(nr/2),np.ceil(nr/2)))
    Nc=np.fft.ifftshift(np.arange(-np.fix(nc/2),np.ceil(nc/2)))
    kernc=np.exp((-1j*2*np.pi/(nc*kappa))*np.outer(\
    np.fft.ifftshift(np.arange(0,nc).T-np.floor(nc/2)),np.arange(0,n_out)-dftshift))
    kernr=np.exp((-1j*2*np.pi/(nr*kappa))*np.outer(\
    np.arange(0,n_out)-dftshift,np.fft.ifftshift(np.arange(0,nr).T-np.floor(nr/2))))

    row_shift=np.zeros(sz)
    col_shift=np.zeros(sz)
    shifted_image = np.zeros_like(data)

    if sequential == False:
        for i in np.arange(1,sz):
            g=f[i,:,:]
            G=np.fft.fft2(g)
            #Masking
            G=np.fft.fftshift(G)
            G*=mask
            G=np.fft.ifftshift(G)

            error,row_shift[i],col_shift[i],Gshift=dft_fjbm(F,G,kappa,dftshift,nr,\
            nr,Nr,Nc,kernr,kernc)
            shifted_image[i,:,:] = np.real(np.fft.ifft2(Gshift)) 
    if sequential == True:
        print('No fastidies')
        for i in np.arange(1,sz):
            g=f[i,:,:]
            G=np.fft.fft2(g)
            #Masking
            G=np.fft.fftshift(G)
            G*=mask
            G=np.fft.ifftshift(G)

            error,row_shift[i],col_shift[i],Gshift=dft_fjbm(F,G,kappa,dftshift,nr,\
            nr,Nr,Nc,kernr,kernc)
            shifted_image[i,:,:] = np.real(np.fft.ifft2(Gshift)) 
            F = np.copy(G) #Sequencial
            row_shift[i] = row_shift[i] + row_shift[i-1]
            col_shift[i] = col_shift[i] + col_shift[i-1]
 
    return row_shift,col_shift,shifted_image














