from astropy.io import fits
import numpy as np
import os

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
                
            maxRange = fits.open(path)[9].data['PHI_IMG_maxRange']
            
            data *= maxRange[0]/maxRange[-1] #conversion factor if 16 bits
                
        return data, header

    except Exception:
       printc("ERROR, Unable to open fits file: {}",path,color=bcolors.FAIL)


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
        return header_imgdirx_exists, None


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
    
    N = arr.shape[axis]
    
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
    if 'exact columns' in mode.keys():
        cols = mode['exact columns']
        for c in cols:
            a0[:,r] = np.nan
    
    with np.errstate(divide='ignore'):
        for i in range(N):
            a1 = a0.take(i, axis=axis)
            nans, index = np.isnan(a1), lambda z: z.nonzero()[0]
            if nans.sum()>0:
                a1[nans] = CubicSpline(np.arange(N)[~nans], a1[~nans])(np.arange(N))[nans]
                if axis == 0:
                    a0[i] = a1
                else:
                    a0[:,i] = a1
    return a0

def limb_fitting(img, mode = 'columns', switch = False):
    def _residuals(p,x,y):
        xc,yc,R = p
        return R**2 - (x-xc)**2 - (y-yc)**2
    
    def _is_outlier(points, thresh=3.5):
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

    from scipy import optimize
    
    if switch:
        norm = -1
    else:
        norm = 1
        
    if mode == 'columns':
        xi = np.arange(100,2000,50)
        yi = []
        m = 10
        for c in xi:
            col = img[:,c]
            g = np.gradient(col*norm)
            gi = _interp(g,m)
            
            yi += [gi.argmax()/m]
        yi = np.asarray(yi)
        xi = xi[~_is_outlier(yi)]
        yi = yi[~_is_outlier(yi)]
    
    elif mode == 'rows':
        yi = np.arange(100,2000,50)
        xi = []
        m = 10
        for r in yi:
            row = img[r,:]
            g = np.gradient(row*norm)
            gi = _interp(g,m)
            
            xi += [gi.argmax()/m]
        xi = np.asarray(xi)
        yi = yi[~_is_outlier(xi)]
        xi = xi[~_is_outlier(xi)]
        
    p = optimize.least_squares(_residuals,x0 = [1024,1024,3000], args=(xi,yi))
       
    mask80 = _circular_mask(img.shape[0],img.shape[1],[p.x[0],p.x[1]],p.x[2]*.8)
    return _circular_mask(img.shape[0],img.shape[1],[p.x[0],p.x[1]],p.x[2]), mask80

