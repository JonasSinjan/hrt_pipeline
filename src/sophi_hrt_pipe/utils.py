from astropy.io import fits
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.optimize as spo
import scipy.signal as sps
from datetime import datetime as dt
import datetime

from astropy.constants import c, R_sun
from scipy.ndimage import map_coordinates
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

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
        hdr = fits.open(path)
        data = hdr[0].data
#         data, header = load_fits(path)
        if 'L2' in hdr[0].header['LEVEL']:
            return hdr[0].data, hdr[0].header
        if np.size(hdr) > 9:
            ex = 9
        else:
            ex = 7
        
        if bit_convert_scale: #conversion from 24.8bit to 32bit
            try:
                IMGformat = hdr[ex].data['PHI_IMG_format'][-1]
            except:
                print("Most likely file does not have 9th Image extension")
                IMGformat = 'IMGFMT_16'
            if IMGformat != 'IMGFMT_24_8':
                data /=  256.
            else:
                print("Dataset downloaded as raw: no bit convert scaling needed")
        if scaling:
            
            accu = hdr[0].header['ACCACCUM']*hdr[0].header['ACCROWIT']*hdr[0].header['ACCCOLIT'] #getting the number of accu from header

            data /= accu

            printc(f"Dividing by number of accumulations: {accu}",color=bcolors.OKGREEN)

        if scale_data: #not for commissioning data

            try:    
                maxRange = hdr[ex].data['PHI_IMG_maxRange']
            
                data *= int(maxRange[0])/int(maxRange[-1])
            except IndexError:
                data *= 81920/128
                
        return data, hdr[0].header

    except Exception:
        printc("ERROR, Unable to open fits file: {}",path,color=bcolors.FAIL)
        raise ValueError()

def fits_get_sampling(file,num_wl = 6, TemperatureCorrection = False, verbose = False):
    '''
    wave_axis,voltagesData,tunning_constant,cpos = fits_get_sampling(file,num_wl = 6, TemperatureCorrection = False, verbose = False)
    No S/C velocity corrected!!!
    cpos = 0 if continuum is at first wavelength and = num_wl - 1 (usually 5) if continuum is at the end
    '''
    fg_head = 3

    with fits.open(file) as hdu_list:
        header = hdu_list[fg_head].data
        tunning_constant = float(header[0][4])/1e9
        ref_wavelength = float(header[0][5])/1e3
        Tfg = hdu_list[0].header['FGOV1PT1'] #temperature of the FG
        
        try:
            voltagesData = np.zeros(num_wl)
            hi = np.histogram(header['PHI_FG_voltage'],bins=num_wl+1)
            yi = hi[0]; xi = hi[1]
            j = 0        
            for i in range(num_wl + 1):
                if yi[i] != 0 :
                    if i < num_wl:
                        idx = np.logical_and(header['PHI_FG_voltage']>=xi[i],header['PHI_FG_voltage']<xi[i+1])
                    else:
                        idx = np.logical_and(header['PHI_FG_voltage']>=xi[i],header['PHI_FG_voltage']<=xi[i+1])
                    voltagesData[j] = int(np.median(header['PHI_FG_voltage'][idx]))
                    j += 1
        except:
            printc('WARNING: Running fits_get_sampling_SPG',color=bcolors.WARNING)
            return fits_get_sampling_SPG(file, False)
    
    d1 = voltagesData[0] - voltagesData[1]
    d2 = voltagesData[num_wl-2] - voltagesData[num_wl-1]
    if np.abs(d1) > np.abs(d2):
        cpos = 0
    else:
        cpos = num_wl-1
    if verbose:
        print('Continuum position at wave: ', cpos)
    wave_axis = voltagesData*tunning_constant + ref_wavelength  #6173.341
    #print(wave_axis)
    
    if TemperatureCorrection:
        if verbose:
            printc('-->>>>>>> If FG temperature is not 61, the relation wl = wlref + V * tunning_constant is not valid anymore',color=bcolors.WARNING)
            printc('          Use instead: wl =  wlref + V * tunning_constant + temperature_constant_new*(Tfg-61)',color=bcolors.WARNING)
        temperature_constant_old = 40.323e-3 # old temperature constant, still used by Johann
        temperature_constant_new = 37.625e-3 # new and more accurate temperature constant
        # wave_axis += temperature_constant_old*(Tfg-61)
        wave_axis += temperature_constant_new*(Tfg-61) # 20221123 see cavity_maps.ipynb with example
        # voltagesData += np.round((temperature_constant_old-temperature_constant_new)*(Tfg-61)/tunning_constant,0)

    return wave_axis,voltagesData,tunning_constant,cpos

def fits_get_sampling_SPG(file,verbose = False):
    '''
    wave_axis,voltagesData,tunning_constant,cpos = fits_get_sampling_SPG(file)
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
    try:
        scan_name_list = [fits.getheader(scan)['PHIDATID'] for scan in data_f]
    except:
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
    if data_shape[0] == 25:
        data = data[:24]
        data_shape = data.shape
    if data.ndim == 4: # [24,y,x,scans]
        data = np.moveaxis(data,0,2).reshape(data_shape[1],data_shape[2],6,4,data_shape[-1]) #separate 24 images, into 6 wavelengths, with each 4 pol states
        data = np.moveaxis(data, 2,3)
    elif data.ndim == 3: # [24,y,x]
        data = np.moveaxis(data,0,2).reshape(data_shape[1],data_shape[2],6,4) #separate 24 images, into 6 wavelengths, with each 4 pol states
        data = np.moveaxis(data, 2,3)
    elif data.ndim == 5: # it means that it is already [y,x,pol,wv,scans]
        pass
    return data
    

def fix_path(path,dir='forward',verbose=False):
    """
    From SPGPylibs PHITools
    """
    path = repr(path)
    if dir == 'forward':
        path = path.replace(")", r"\)")
        path = path.replace("(", r"\(")
        path = path.replace(" ", r"\ ")
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
def mu_angle(hdr,coord=None):
    """
    input
    hdr: header or filename
    coord: pixel for which the mu angle is found (if None: center of the FoV)
    
    output
    mu = cosine of the heliocentric angle
    """
    if type(hdr) is str:
        hdr = fits.getheader(hdr)
    
    center=center_coord(hdr)
    Rpix=(hdr['RSUN_ARC']/hdr['CDELT1'])
    
    if coord is None:
        coord = np.asarray([(hdr['PXEND1']-hdr['PXBEG1'])/2,
                            (hdr['PXEND2']-hdr['PXBEG2'])/2])
    
    coord -= center[:2]
    mu = np.sqrt(Rpix**2 - (coord[0]**2 + coord[1]**2)) / Rpix
    return mu

def center_coord(hdr):
    """
    input
    hdr: header
    
    output
    center: [x,y,1] coordinates of the solar disk center (units: pixel)
    """
    pxbeg1 = hdr['PXBEG1']
    pxend1 = hdr['PXEND1']
    pxbeg2 = hdr['PXBEG2']
    pxend2 = hdr['PXEND2']
    coord=np.asarray([hdr['CRPIX1'],
            hdr['CRPIX2'],
           1])

    angle = hdr['CROTA'] # positive angle = clock-wise rotation of the reference system axes 
    rad = angle * np.pi/180
    rot = np.asarray([[np.cos(rad),-np.sin(rad),0],[np.sin(rad),np.cos(rad),0],[0,0,1]])
    rc = [(pxend1-pxbeg1)/2,(pxend2-pxbeg2)/2] # CRPIX from 1 to 2048, so 1024.5 is the center

    tr = np.asarray([[1,0,rc[0]],[0,1,rc[1]],[0,0,1]])
    invtr = np.asarray([[1,0,-rc[0]],[0,1,-rc[1]],[0,0,1]])
    M = tr @ rot @ invtr

    coord = (M @ coord)[:2]

    # center of the sun in the rotated reference system
    center=np.asarray([coord[0]-hdr['CRVAL1']/hdr['CDELT1']-1,
                       coord[1]-hdr['CRVAL2']/hdr['CDELT2']-1,
                       1])
    # rotation of the sun center back to the original reference system
    angle = -hdr['CROTA'] # positive angle = clock-wise rotation of the reference system axes 
    rad = angle * np.pi/180
    rot = np.asarray([[np.cos(rad),-np.sin(rad),0],[np.sin(rad),np.cos(rad),0],[0,0,1]])
    
    tr = np.asarray([[1,0,rc[0]],[0,1,rc[1]],[0,0,1]])
    invtr = np.asarray([[1,0,-rc[0]],[0,1,-rc[1]],[0,0,1]])
    M = tr @ rot @ invtr

    center = (M @ center)
    
    return center

def circular_mask(h, w, center, radius):

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def limb_side_finder(img, hdr,verbose=True,outfinder=False):
    Rpix=(hdr['RSUN_ARC']/hdr['CDELT1'])
    # center=[hdr['CRPIX1']-hdr['CRVAL1']/hdr['CDELT1']-1,hdr['CRPIX2']-hdr['CRVAL2']/hdr['CDELT2']-1]
    center = center_coord(hdr)[:2] - 1
    limb_wcs = circular_mask(hdr['PXEND2']-hdr['PXBEG2']+1,
                             hdr['PXEND1']-hdr['PXBEG1']+1,center,Rpix)
    
    f = 16
    fract = int(limb_wcs.shape[0]//f)
    
    finder = np.zeros((f,f))
    for i in range(f):
        for j in range(f):
            finder[i,j] = np.sum(~limb_wcs[fract*i:fract*(i+1),fract*j:fract*(j+1)])

    sides = dict(E=0,N=0,W=0,S=0)

    sides['E'] = np.sum(finder[:,0:int(f//3-1)])
    sides['W'] = np.sum(finder[:,f-int(f//3-1):])
    sides['S'] = np.sum(finder[0:int(f//3-1)])
    sides['N'] = np.sum(finder[f-int(f//3-1):])
    finder_original = finder.copy()
    
    finder[:int(f//3-1),:int(f//6)] = 0
    finder[:int(f//3-1),-int(f//3-1):] = 0
    finder[-int(f//3-1):,:int(f//3-1)] = 0
    finder[-int(f//3-1):,-int(f//3-1):] = 0

    if np.any(finder) > 0:
        side = max(sides,key=sides.get)
        if verbose:
            print('Limb side:',side)
    else:
        side = ''
        if verbose:
            print('Limb is not in the FoV according to WCS keywords')
    
    ds = 256
    if hdr['DSUN_AU'] < 0.4:
        if side == '':
            ds = 384
    dx = 0; dy = 0
    if 'N' in side and img.shape[0]//2 - ds > img.shape[0]//4:
        dy = -img.shape[0]//4
    if 'S' in side and img.shape[0]//2 - ds > img.shape[0]//4:
        dy = img.shape[0]//4
    if 'W' in side and img.shape[1]//2 - ds > img.shape[1]//4:
        dx = -img.shape[1]//4
    if 'E' in side and img.shape[1]//2 - ds > img.shape[1]//4:
        dx = img.shape[1]//4

    if img.shape[0] > 2*ds:
        sly = slice(img.shape[0]//2 - ds + dy, img.shape[0]//2 + ds + dy)
    else:
        sly = slice(0,img.shape[0])
    if img.shape[1] > 2*ds:
        slx = slice(img.shape[1]//2 - ds + dx, img.shape[1]//2 + ds + dx)
    else:
        slx = slice(0,img.shape[1])
    
    if outfinder:
        return side, center, Rpix, sly, slx, finder_original
    else:
        return side, center, Rpix, sly, slx

def limb_fitting(img, hdr, field_stop, verbose=True):
    def _residuals(p,x,y):
        xc,yc,R = p
        return R**2 - (x-xc)**2 - (y-yc)**2
    
    def _is_outlier(points, thresh=2):
        if len(points.shape) == 1:
            points = points[:,None]
        median = np.median(points, axis=0)
        diff = np.sum((points - median)**2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation

        return modified_z_score > thresh
        
    def _image_derivative(d):
        import numpy as np
        from scipy.signal import convolve
        kx = np.asarray([[1,0,-1], [1,0,-1], [1,0,-1]])
        ky = np.asarray([[1,1,1], [0,0,0], [-1,-1,-1]])

        kx=kx/3.
        ky=ky/3.

        SX = convolve(d, kx,mode='same')
        SY = convolve(d, ky,mode='same')

        return SX, SY

    from scipy.optimize import least_squares
    from scipy.ndimage import binary_erosion

    side, center, Rpix, sly, slx, finder_small = limb_side_finder(img,hdr,verbose=verbose,outfinder=True)
    f = 16
    fract = int(img.shape[0]//f)
    finder = np.zeros(img.shape)
    for i in range(f):
        for j in range(f):
            finder[fract*i:fract*(i+1),fract*j:fract*(j+1)] = finder_small[i,j]

#     wcs_mask = circular_mask(img.shape[0],img.shape[1],center,Rpix)
#     wcs_grad = _image_derivative(wcs_mask)
        
    if side == '':
        return None, sly, slx, side#, None, None
    
    if 'N' in side or 'S' in side:
        img = np.moveaxis(img,0,1)
        finder = np.moveaxis(finder,0,1)
        center = center[::-1]
    
    s = 5
    thr = 3
    
    diff = _image_derivative(img)[0][s:-s,s:-s]
    rms = np.sqrt(np.mean(diff[field_stop[s:-s,s:-s]>0]**2))
    yi, xi = np.where(np.abs(diff*binary_erosion(field_stop,np.ones((2,2)),iterations=20)[s:-s,s:-s])>rms*thr)
    tyi = yi.copy(); txi = xi.copy()
    yi = []; xi = []
    for i,j in zip(tyi,txi):
        if finder[i,j]:
            yi += [i+s]; xi += [j+s]
    yi = np.asarray(yi); xi = np.asarray(xi)
    
    out = _is_outlier(xi)

    yi = yi[~out]
    xi = xi[~out]

    p = least_squares(_residuals,x0 = [center[0],center[1],Rpix], args=(xi,yi),
                              bounds = ([center[0]-150,center[1]-150,Rpix-50],[center[0]+150,center[1]+150,Rpix+50]))
        
#     mask80 = circular_mask(img.shape[0],img.shape[1],[p.x[0],p.x[1]],p.x[2]*.8)
    mask100 = circular_mask(img.shape[0],img.shape[1],[p.x[0],p.x[1]],p.x[2])
    
    if 'N' in side or 'S' in side:
        return np.moveaxis(mask100,0,1), sly, slx, side
    else:
        return mask100, sly, slx, side

def fft_shift(img,shift):
    """
    im: 2D-image to be shifted
    shift = [dy,dx] shift in pixel
    """
    
    try:
        import pyfftw.interfaces.numpy_fft as fft
    except:
        import numpy.fft as fft
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

#plotting functions for quick data analysis for communal use


def find_nearest(array, value):
    """
    return index of nearest value in array to the desired value
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def gaus(x,a,x0,sigma):
    """
    return Gauss function
    """
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def gaussian_fit(a,show=True):
    """
    gaussian fit for data 'a' from np.histogram
    """
#    a=np.histogram(data.flat,density=True,bins=100)
    xx=a[1][:-1] + (a[1][1]-a[1][0])/2
    y=a[0][:]
    p0=[0.,sum(xx*y)/sum(y),np.sqrt(sum(y * (xx - sum(xx*y)/sum(y))**2) / sum(y))] #weighted avg of bins for avg and sigma inital values
    p0[0]=y[find_nearest(xx,p0[1])-5:find_nearest(xx,p0[1])+5].mean() #find init guess for ampltiude of gauss func
    try:
        p,cov=spo.curve_fit(gaus,xx,y,p0=p0)
        if show:
            lbl = '{:.2e} $\pm$ {:.2e}'.format(p[1],p[2])
            plt.plot(xx,gaus(xx,*p),'r--', label=lbl)
            plt.legend(fontsize=9)
        return p
    except:
        printc("Gaussian fit failed: return initial guess",color=bcolors.WARNING)
        return p0
        
def iter_noise(temp, p = [1,0,1e-1], eps = 1e-6):
    p_old = [1,0,10]; count = 0
    it = 0
    while np.abs(p[2] - p_old[2])>eps:
        p_old = p; count += 1
        hi = np.histogram(temp, bins=np.linspace(p[1] - 3*p[2],p[1] + 3*p[2],200),density=False);
        p = gaussian_fit(hi, show=False)
        if it == 50:
            break
        it += 1
    return p, hi

  
def blos_noise(blos_file, fs = None):
    """
    plot blos on left panel, and blos hist + Gaussian fit (w/ iterative option)
    """

    blos = fits.getdata(blos_file)
    hdr = fits.getheader(blos_file)
    #first get the pixels that we want (central 512x512 and limb handling)
    limb_side, center, Rpix, sly, slx = limb_side_finder(blos, hdr)
    values = blos[sly,slx]
#     if limb_side == '':#not a limb image
#         values = blos[512:1536, 512:1536]

#     else:
#         data_size = np.shape(blos)
        # ds = 386 #?
        # dx = 0; dy = 0
        # if 'N' in limb_side and data_size[0]//2 - ds > 512:
        #     dy = -512
        # if 'S' in limb_side and data_size[0]//2 - ds > 512:
        #     dy = 512
        # if 'W' in limb_side and data_size[1]//2 - ds > 512:
        #     dx = -512
        # if 'E' in limb_side and data_size[1]//2 - ds > 512:
        #     dx = 512

        # sly = slice(data_size[0]//2 - ds + dy, data_size[0]//2 + ds + dy)
        # slx = slice(data_size[1]//2 - ds + dx, data_size[1]//2 + ds + dx)
        # values = blos[sly, slx]


    fig, ax = plt.subplots(1,2, figsize = (14,6))
    if fs is not None:
        idx = np.where(fs<1)
        blos[idx] = -300
    im1 = ax[0].imshow(blos, cmap = "gray", origin = "lower", vmin = -200, vmax = 200)
    fig.colorbar(im1, ax = ax[0], fraction=0.046, pad=0.04)
    hi = ax[1].hist(values.flatten(), bins=np.linspace(-2e2,2e2,200))
    #print(hi)
    tmp = [0,0]
    tmp[0] = hi[0].astype('float64')
    tmp[1] = hi[1].astype('float64')


    #guassian fit + label
    p = gaussian_fit(tmp, show = False)    
    p_iter, hi_iter = iter_noise(values,[1.,0.,1.],eps=1e-4)
    xx=hi[1][:-1] + (hi[1][1]-hi[1][0])/2
    lbl = f'{p[1]:.2e} $\pm$ {p[2]:.2e} G'
    ax[1].plot(xx,gaus(xx,*p),'r--', label=lbl)
    ax[1].scatter(0,0, color = 'white', s = 0, label = f"Iter Fit: {p_iter[1]:.2e} $\pm$ {p_iter[2]:.2e} G")
    ax[1].legend(fontsize=15)

    date = blos_file.split('blos_')[1][:15]
    dt_str = dt.strptime(date, "%Y%m%dT%H%M%S")
    fig.suptitle(f"Blos {dt_str}")

    plt.tight_layout()
    plt.show()

def stokes_noise(stokes_file):
    """
    plot stokes V on left panel, and Stokes V hist + Gaussian fit (w/ iterative option)
    """

    stokes = fits.getdata(stokes_file)
    hdr = fits.getheader(stokes_file)
    out = fits_get_sampling(stokes_file)
    cpos = out[3]
    #first get the pixels that we want (central 512x512 and limb handling)
    limb_side, center, Rpix, sly, slx = limb_side_finder(stokes[:,:,3,cpos], hdr)
    values = stokes[sly,slx,3,cpos]
#     if limb_side == '':#not a limb image
#         values = stokes[512:1536, 512:1536,3,cpos]

#     else:
#         data_size = np.shape(stokes[:,:,0,0])
        # ds = 386 #?
        # dx = 0; dy = 0
        # if 'N' in limb_side and data_size[0]//2 - ds > 512:
        #     dy = -512
        # if 'S' in limb_side and data_size[0]//2 - ds > 512:
        #     dy = 512
        # if 'W' in limb_side and data_size[1]//2 - ds > 512:
        #     dx = -512
        # if 'E' in limb_side and data_size[1]//2 - ds > 512:
        #     dx = 512

        # sly = slice(data_size[0]//2 - ds + dy, data_size[0]//2 + ds + dy)
        # slx = slice(data_size[1]//2 - ds + dx, data_size[1]//2 + ds + dx)
        # values = blos[sly, slx]


    fig, ax = plt.subplots(1,2, figsize = (14,6))
    im1 = ax[0].imshow(stokes[:,:,3,cpos], cmap = "gist_heat", origin = "lower", vmin = -1e-2, vmax = 1e-2)
    fig.colorbar(im1, ax = ax[0], fraction=0.046, pad=0.04)
    hi = ax[1].hist(values.flatten(), bins=np.linspace(-1e-2,1e-2,200))
    #print(hi)
    tmp = [0,0]
    tmp[0] = hi[0].astype('float64')
    tmp[1] = hi[1].astype('float64')


    #guassian fit + label
    p = gaussian_fit(tmp, show = False)    
    p_iter, hi_iter = iter_noise(values,eps=1e-6)
    xx=hi[1][:-1] + (hi[1][1]-hi[1][0])/2
    lbl = f'{p[1]:.2e} $\pm$ {p[2]:.2e} Ic'
    ax[1].plot(xx,gaus(xx,*p),'r--', label=lbl)
    ax[1].scatter(0,0, color = 'white', s = 0, label = f"Iter Fit: {p_iter[1]:.2e} $\pm$ {p_iter[2]:.2e} Ic")
    ax[1].legend(fontsize=15)

    date = stokes_file.split('stokes_')[1][:15]
    dt_str = dt.strptime(date, "%Y%m%dT%H%M%S")
    fig.suptitle(f"Stokes {dt_str}")

    plt.tight_layout()
    plt.show()


"""vsnr = iter_noise(img[sly,slx,3,5].ravel())[2]
    print(data_date[i]+': Stokes V SNR (iterative, iss off): {:.2e}'.format(vsnr))"""

########### new WCS script 3/6/2022 ###########
def image_derivative(d):
    kx = np.asarray([[1,0,-1], [1,0,-1], [1,0,-1]])
    ky = np.asarray([[1,1,1], [0,0,0], [-1,-1,-1]])
    kx=kx/3.
    ky=ky/3.

    SX = sps.convolve(d, kx,mode='same')
    SY = sps.convolve(d, ky,mode='same')

    A=SX**2+SY**2

    return A

def Inv2(x_c,y_c,x_u,y_u,k):
    """
    undistortion model
    """
    r_u = np.sqrt((x_u-x_c)**2+(y_u-y_c)**2) 
    x_d = x_c+(x_u-x_c)*(1-k*r_u**2)
    y_d = y_c+(y_u-y_c)*(1-k*r_u**2)
    return x_d,y_d

def und(hrt, order=1, flip = True):
    """
    spherical undistortion function 
    by F. Kahil (MPS)
    """
    if flip:
        hrt = hrt[:,::-1]
    Nx = Ny=2048
    x = y = np.arange(Nx)
    X,Y = np.meshgrid(x,y)
    x_c =1016
    y_c =982
    k=8e-09
    hrt_und = np.zeros((Nx,Ny))
    x_d, y_d = Inv2(x_c,y_c,X,Y,k)
    hrt_und = map_coordinates(hrt,[y_d,x_d],order=order)
    if flip:
        return hrt_und[:,::-1]
    else:
        return hrt_und

def rotate_header(h,angle):
    h['CROTA'] -= angle
    h['PC1_1'] = np.cos(h['CROTA']*np.pi/180)
    h['PC1_2'] = -np.sin(h['CROTA']*np.pi/180)
    h['PC2_1'] = np.sin(h['CROTA']*np.pi/180)
    h['PC2_2'] = np.cos(h['CROTA']*np.pi/180)
    rad = angle * np.pi/180
    rot = np.asarray([[np.cos(rad),-np.sin(rad),0],[np.sin(rad),np.cos(rad),0],[0,0,1]])
    coords = np.asarray([h['CRPIX1'],h['CRPIX2'],1])
    center = [1024.5,1024.5] # CRPIX from 1 to 2048, so 1024.5 is the center
    tr = np.asarray([[1,0,center[0]],[0,1,center[1]],[0,0,1]])
    invtr = np.asarray([[1,0,-center[0]],[0,1,-center[1]],[0,0,1]])

    M = tr @ rot @ invtr
    bl = M @ np.asarray([0,0,1])
    tl = M @ np.asarray([0,2048,1])
    br = M @ np.asarray([2048,0,1])
    tr = M @ np.asarray([2048,2048,1])

    O = -np.asarray([bl,tl,br,tr]).min(axis=0)[:-1]
    newO = np.asarray([[1,0,O[0]+1],[0,1,O[1]+1],[0,0,1]])
    newM = newO @ M
    new_coords = newM @ coords
    h['CRPIX1'] = round(new_coords[0],4)
    h['CRPIX2'] = round(new_coords[1],4)
    
    return h
    
def translate_header(h,tvec):
    tr = np.asarray([[1,0,-tvec[1]*h['CDELT1']],[0,1,-tvec[0]*h['CDELT2']],[0,0,1]])
    coords = np.asarray([h['CRVAL1'],h['CRVAL2'],1])
    new_coords = tr @ coords
    h['CRVAL1'] = round(new_coords[0],4)
    h['CRVAL2'] = round(new_coords[1],4)
    
    return h

def remap(hrt_map, hmi_map, out_shape = (1024,1024), verbose = False):
    import sunpy.map
    from reproject import reproject_adaptive
    
    # plot of the maps
    if verbose:
        plt.figure(figsize=(9,5))
        plt.subplot(121,projection=hmi_map)
        hmi_map.plot()
        hmi_map.draw_limb()
        top_right = hmi_map.world_to_pixel(hrt_map.top_right_coord)
        bottom_left = hmi_map.world_to_pixel(hrt_map.bottom_left_coord)
        hmi_map.draw_quadrangle(np.array([bottom_left.x.value,bottom_left.y.value])*u.pix,
                          top_right=np.array([top_right.x.value,top_right.y.value])*u.pix, edgecolor='yellow')

        plt.subplot(122,projection=hrt_map)
        hrt_map.plot()
        hrt_map.draw_limb()

        plt.show()
    
    # define new header for hmi map using hrt observer coordinates
    out_header = sunpy.map.make_fitswcs_header(
        out_shape,
        hrt_map.reference_coordinate.replicate(rsun=hmi_map.reference_coordinate.rsun),
        scale=u.Quantity(hrt_map.scale),
        instrument="HMI",
        observatory="SDO",
        wavelength=hmi_map.wavelength
    )

    out_header['dsun_obs'] = hrt_map.coordinate_frame.observer.radius.to(u.m).value
    out_header['hglt_obs'] = hrt_map.coordinate_frame.observer.lat.value
    out_header['hgln_obs'] = hrt_map.coordinate_frame.observer.lon.value
    out_header['crpix1'] = hrt_map.fits_header['CRPIX1']
    out_header['crpix2'] = hrt_map.fits_header['CRPIX2']
    out_header['crval1'] = hrt_map.fits_header['CRVAL1']
    out_header['crval2'] = hrt_map.fits_header['CRVAL2']
    
    out_header['crota2'] = hrt_map.fits_header['CROTA']
    out_header['PC1_1'] = hrt_map.fits_header['PC1_1']
    out_header['PC1_2'] = hrt_map.fits_header['PC1_2']
    out_header['PC2_1'] = hrt_map.fits_header['PC2_1']
    out_header['PC2_2'] = hrt_map.fits_header['PC2_2']

    out_wcs = WCS(out_header)
    
    # reprojection
    hmi_origin = hmi_map
    output, footprint = reproject_adaptive(hmi_origin, out_wcs, out_shape)
    hmi_map = sunpy.map.Map(output, out_header)
    hmi_map.plot_settings = hmi_origin.plot_settings

    # plot reprojected maps
    if verbose:
        fig = plt.figure(figsize=(10,6))
        ax1 = fig.add_subplot(1, 2, 1, projection=hmi_map)
        hmi_map.plot(axes=ax1, title='SDO/HMI image as seen from PHI/HRT')
        hmi_map.draw_limb(color='blue')
        ax2 = fig.add_subplot(1, 2, 2, projection=hrt_map)
        hrt_map.plot(axes=ax2)
        # Set the HPC grid color to black as the background is white
        ax1.coords[0].grid_lines_kwargs['edgecolor'] = 'k'
        ax1.coords[1].grid_lines_kwargs['edgecolor'] = 'k'
        ax2.coords[0].grid_lines_kwargs['edgecolor'] = 'k'
        ax2.coords[1].grid_lines_kwargs['edgecolor'] = 'k'
    
    return hmi_map

def WCS_correction(file_name,jsoc_email,dir_out='./',allDID=False,verbose=False):
    """
    This function saves new version of the fits file with updated WCS.
    It works correlating HRT data on remap HMI data. Not validated on limb data. Not tested on data with different viewing angle.
    icnt, stokes or ilam files are expected as input.
    if allDID is True, all the fits file with the same DID in the directory of the input file will be saved with the new WCS.
    return new header, if dir_out is None it does not save any fits file
    """
    import sunpy, drms, imreg_dft
    import sunpy.map
    from reproject import reproject_interp, reproject_adaptive
    from sunpy.coordinates import get_body_heliographic_stonyhurst

    from sunpy.coordinates import frames
    import warnings, sunpy
    warnings.filterwarnings("ignore", category=sunpy.util.SunpyMetadataWarning)

    
    print('This is a preliminary procedure, not fully tested')
    print('It has been optimized on raw and continuum data')
    print('This script is based on sunpy routines and examples')
    
    phi, h_phi = fits.getdata(file_name,header=True)
    start_row = int(h_phi['PXBEG2']-1)
    start_col = int(h_phi['PXBEG1']-1)
    _,_,_,cpos = fits_get_sampling(file_name)
    
    if phi.ndim == 3:
        phi = phi[cpos*4]
    elif phi.ndim == 4:
        phi = phi[:,:,0,cpos]
    
    if phi.shape[0] == 2048:
        und_phi = phi # und(phi)
        # h_phi['CRPIX1'],h_phi['CRPIX2'] = Inv2(1016,982,h_phi['CRPIX1'],h_phi['CRPIX2'],8e-9)
        phi_map = sunpy.map.Map((und_phi,h_phi))
    else:
        phi = np.pad(phi,[(start_row,2048-(start_row+phi.shape[0])),(start_col,2048-(start_row+phi.shape[1]))])
        h_phi['NAXIS1'] = 2048; h_phi['NAXIS2'] = 2048
        h_phi['CRPIX1'] += start_col; h_phi['CRPIX2'] += start_row
        und_phi = phi # und(phi)
        # h_phi['CRPIX1'],h_phi['CRPIX2'] = Inv2(1016,982,h_phi['CRPIX1'],h_phi['CRPIX2'],8e-9)
        phi_map = sunpy.map.Map((und_phi,h_phi))
    
    if verbose:
        phi_map.peek()
    
    client = drms.Client(email=jsoc_email, verbose=True)
    kwlist = ['T_REC','T_OBS','DATE-OBS','CADENCE','DSUN_OBS']
    ht = phi_map.fits_header
    # lt = (hmi_map.dsun - phi_map.dsun).to(u.m)/299792458*u.s/u.m

    t_obs = datetime.datetime.fromisoformat(ht['DATE-AVG'])
    
    dtai = datetime.timedelta(seconds=37) # datetime.timedelta(seconds=94)
    dcad = datetime.timedelta(seconds=35) # half HMI cadence (23) + margin
    
    dltt = datetime.timedelta(seconds=ht['EAR_TDEL']) # difference in light travel time S/C-Earth

    keys = client.query('hmi.ic_45s['+(t_obs+dtai-dcad+dltt).strftime('%Y.%m.%d_%H:%M:%S')+'-'+
                      (t_obs+dtai+dcad+dltt).strftime('%Y.%m.%d_%H:%M:%S')+']',seg=None,key=kwlist,n=2)
    
    lt = (np.mean(keys['DSUN_OBS'])*u.m - phi_map.dsun).to(u.m)/c
    dltt = datetime.timedelta(seconds=lt.value) # difference in light travel time S/C-SDO

    ind = np.argmin([np.abs((datetime.datetime.strptime(t,'%Y.%m.%d_%H:%M:%S_TAI') - dtai - dltt - t_obs).total_seconds())
                     for t in keys['T_OBS']])
    name_h = 'hmi.ic_45s['+keys['T_REC'][ind]+']{Continuum}'

    if np.abs((datetime.datetime.strptime(keys['T_OBS'][ind],'%Y.%m.%d_%H:%M:%S_TAI') - dtai - dltt - t_obs).total_seconds()) > 23:
        print('WARNING: Closer file exists but has not been found.')
        print(name_h)
        print('T_OBS:',datetime.datetime.strptime(keys['T_OBS'][ind],'%Y.%m.%d_%H:%M:%S_TAI') - dtai - dltt)
        print('DATE-AVG:',t_obs)
        print('')

    s45 = client.export(name_h,protocol='fits')
    hmi_map = sunpy.map.Map(s45.urls.url[0])
    
    if verbose:
        hmi_map.peek()
    
    sly = slice(128*4,128*12)
    slx = slice(128*4,128*12)

    ht = h_phi.copy()
    ht['DATE-BEG'] = ht['DATE-AVG']; ht['DATE-OBS'] = ht['DATE-AVG']
    ht['DATE-BEG'] = datetime.datetime.isoformat(datetime.datetime.fromisoformat(ht['DATE-BEG']) + dltt)
    ht['DATE-OBS'] = datetime.datetime.isoformat(datetime.datetime.fromisoformat(ht['DATE-OBS']) + dltt)
    shift = [1,1]
    i = 0
    angle = 1
    
    while np.any(np.abs(shift)>5e-2):
        phi_map = sunpy.map.Map((und_phi,ht))

        bl = phi_map.pixel_to_world(slx.start*u.pix, sly.start*u.pix)
        tr = phi_map.pixel_to_world((slx.stop-1)*u.pix, (sly.stop-1)*u.pix)
        phi_submap = phi_map.submap(np.asarray([slx.start, sly.start])*u.pix,
                              top_right=np.asarray([slx.stop-1, sly.stop-1])*u.pix)

        hmi_map_remap = remap(phi_map, hmi_map, out_shape = (2048,2048), verbose=False)

        top_right = hmi_map_remap.world_to_pixel(tr)
        bottom_left = hmi_map_remap.world_to_pixel(bl)
        tr_hmi_map = np.array([top_right.x.value,top_right.y.value])
        bl_hmi_map = np.array([bottom_left.x.value,bottom_left.y.value])
        hmi_map_wcs = hmi_map_remap.submap(bl_hmi_map*u.pix,top_right=tr_hmi_map*u.pix)

        ref = phi_submap.data.copy()
        temp = hmi_map_wcs.data.copy(); temp[np.isinf(temp)] = 0; temp[np.isnan(temp)] = 0
        s = [1,1]
        shift = [0,0]
        it = 0

        if abs(angle>1e-2):
            r = imreg_dft.similarity(ref.copy(),temp.copy(),numiter=3,constraints=dict(scale=(1,0)))
            shift = r['tvec']; angle = r['angle']
            hmi_map_shift = imreg_dft.transform_img(hmi_map_wcs.data,scale=1,angle=angle,tvec=shift)
            hmi_map_shift = sunpy.map.Map((hmi_map_shift,hmi_map_wcs.fits_header))
            print('logpol transform shift (x,y):',round(shift[1],2),round(shift[0],2),'angle (deg):',round(angle,3))

            ht = translate_header(rotate_header(ht.copy(),-angle),shift)

        else:
            while np.any(np.abs(s)>1e-3):
                sr, sc, _ = SPG_shifts_FFT(np.asarray([ref,temp])); s = [sr[1],sc[1]]
                shift = [shift[0]+s[0],shift[1]+s[1]]
                temp = fft_shift(hmi_map_wcs.data.copy(), shift); temp[np.isinf(temp)] = 0; temp[np.isnan(temp)] = 0
                it += 1
                if it == 10:
                    break
            hmi_map_shift = sunpy.map.Map((temp,hmi_map_wcs.fits_header))

            ht = translate_header(ht.copy(),np.asarray(shift))
            print(it,'iterations shift (x,y):',round(shift[1],2),round(shift[0],2))

        i+=1
        if i == 10:
            print('Maximum iterations reached:',i)
            break
            
    phi_map = sunpy.map.Map((und_phi,ht))
    phi_submap = phi_map.submap(np.asarray([slx.start, sly.start])*u.pix,
                               top_right=np.asarray([slx.stop-1, sly.stop-1])*u.pix)
    
    
    if verbose:
        fig = plt.figure(figsize=(10,6))
        ax1 = fig.add_subplot(1, 2, 1, projection=hmi_map_wcs)
        hmi_map_wcs.plot(axes=ax1, title='SDO/HMI image as seen from PHI/HRT')
        ax2 = fig.add_subplot(1, 2, 2, projection=phi_submap)
        phi_submap.plot(axes=ax2)
        # Set the HPC grid color to black as the background is white
        ax1.coords[0].grid_lines_kwargs['edgecolor'] = 'k'
        ax1.coords[1].grid_lines_kwargs['edgecolor'] = 'k'
        ax2.coords[0].grid_lines_kwargs['edgecolor'] = 'k'
        ax2.coords[1].grid_lines_kwargs['edgecolor'] = 'k'
    
    name = file_name.split('/')[-1]
    new_name = name.split('L2')[0]+'L2.WCS'+name.split('L2')[1]
    
    if dir_out is not None:
        if allDID:
            did = h_phi['PHIDATID']
            directory = file_name[:-len(name)]
            file_n = os.listdir(directory)
            if type(did) != str:
                did = str(did)
            did_n = [directory+i for i in file_n if did in i]
            l2_n = ['stokes','icnt','bmag','binc','bazi','vlos','blos']
            for n in l2_n:
                f = [i for i in did_n if n in i][0]
                name = f.split('/')[-1]
                new_name = name.split('L2')[0]+'L2.WCS'+name.split('L2')[1]
                with fits.open(f) as h:
                    h[0].header['CROTA'] = ht['CROTA']
                    h[0].header['CRPIX1'] = ht['CRPIX1']
                    h[0].header['CRPIX2'] = ht['CRPIX2']
                    h[0].header['CRVAL1'] = ht['CRVAL1']
                    h[0].header['CRVAL2'] = ht['CRVAL2']
                    h[0].header['PC1_1'] = ht['PC1_1']
                    h[0].header['PC1_2'] = ht['PC1_2']
                    h[0].header['PC2_1'] = ht['PC2_1']
                    h[0].header['PC2_2'] = ht['PC2_2']
                    h[0].header['HISTORY'] = 'WCS corrected via HRTundistorted - HMI cross correlation (continuum intensity)'
                    h.writeto(dir_out+new_name, overwrite=True)        
        else:
            with fits.open(file_name) as h:
                h[0].header['CROTA'] = ht['CROTA']
                h[0].header['CRPIX1'] = ht['CRPIX1']
                h[0].header['CRPIX2'] = ht['CRPIX2']
                h[0].header['CRVAL1'] = ht['CRVAL1']
                h[0].header['CRVAL2'] = ht['CRVAL2']
                h[0].header['PC1_1'] = ht['PC1_1']
                h[0].header['PC1_2'] = ht['PC1_2']
                h[0].header['PC2_1'] = ht['PC2_1']
                h[0].header['PC2_2'] = ht['PC2_2']
                h[0].header['HISTORY'] = 'WCS corrected via HRTundistorted - HMI cross correlation '
                h.writeto(dir_out+new_name, overwrite=True)
    return ht
###############################################

def cavity_shifts(cavity_f, wave_axis,rows,cols):
    cavityMap, _ = load_fits(cavity_f) # cavity maps
    if cavityMap.ndim == 3:
        cavityWave = cavityMap[:,rows,cols].mean(axis=0)
    else:
        cavityWave = cavityMap[rows,cols]
     
    new_wave_axis = wave_axis[np.newaxis,np.newaxis] - cavityWave[...,np.newaxis]

    return new_wave_axis

def load_l2_rte(directory,did,version=None):
    file_n = os.listdir(directory)
    if type(did) != str:
        did = str(did)
    if version is None:
        did_n = [directory+i for i in file_n if did in i]
    else:
        did_n = [directory+i for i in file_n if (did in i and version in i)]
    rte_n = ['icnt','bmag','binc','bazi','vlos','blos','chi2']
    rte_out = []
    for n in rte_n:
        try:
            rte_out += [fits.getdata([i for i in did_n if n in i][0])]
        except:
            print(n+' not found')
    
    rte_out = np.asarray(rte_out)
    
    return rte_out

def center_coord(hdr):
    """
    input
    hdr: header
    
    output
    center: [x,y,1] coordinates of the solar disk center (units: pixel)
    """
    pxbeg1 = hdr['PXBEG1']
    pxend1 = hdr['PXEND1']
    pxbeg2 = hdr['PXBEG2']
    pxend2 = hdr['PXEND2']
    coord=np.asarray([hdr['CRPIX1']-1,
            hdr['CRPIX2']-1,
           1])
    
    angle = hdr['CROTA'] # positive angle = clock-wise rotation of the reference system axes 
    rad = angle * np.pi/180
    rot = np.asarray([[np.cos(rad),-np.sin(rad),0],[np.sin(rad),np.cos(rad),0],[0,0,1]])
    rc = [(pxend1-pxbeg1)/2,(pxend2-pxbeg2)/2] # CRPIX from 1 to 2048, so 1024.5 is the center

    tr = np.asarray([[1,0,rc[0]],[0,1,rc[1]],[0,0,1]])
    invtr = np.asarray([[1,0,-rc[0]],[0,1,-rc[1]],[0,0,1]])
    M = tr @ rot @ invtr

    coord = (M @ coord)[:2]

    # center of the sun in the rotated reference system
    center=np.asarray([coord[0]-hdr['CRVAL1']/hdr['CDELT1'],
                       coord[1]-hdr['CRVAL2']/hdr['CDELT2'],
                       1])
    # rotation of the sun center back to the original reference system
    angle = -hdr['CROTA'] # positive angle = clock-wise rotation of the reference system axes 
    rad = angle * np.pi/180
    rot = np.asarray([[np.cos(rad),-np.sin(rad),0],[np.sin(rad),np.cos(rad),0],[0,0,1]])
    
    tr = np.asarray([[1,0,rc[0]],[0,1,rc[1]],[0,0,1]])
    invtr = np.asarray([[1,0,-rc[0]],[0,1,-rc[1]],[0,0,1]])
    M = tr @ rot @ invtr

    center = (M @ center)
    
    return center

def mu_angle(hdr,coord=None):
    """
    input
    hdr: header or filename
    coord: pixel for which the mu angle is found (if None: center of the FoV)
    
    output
    mu = cosine of the heliocentric angle
    """
    if type(hdr) is str:
        hdr = fits.getheader(hdr)
    
    center=center_coord(hdr)
    Rpix=(hdr['RSUN_ARC']/hdr['CDELT1'])
    
    if coord is None:
        coord = np.asarray([(hdr['PXEND1']-hdr['PXBEG1'])/2,
                            (hdr['PXEND2']-hdr['PXBEG2'])/2])
    else:
        coord = np.asarray(coord,dtype=float)
    
    coord -= center[:2]
    mu = np.sqrt(Rpix**2 - (coord[0]**2 + coord[1]**2)) / Rpix
    return mu

def ccd2HPC(file,coords=None):
    """
    from CCD frame to Helioprojective Cartesian
    
    Input
    file: file_name, sunpy map or header
    coords: (x,y) or np.asarray([[x0,y0],[x1,y1],...])
            if None: built the coordinates map
            
    Output
    HPCx, HPCy, HPCd
    """
    import sunpy.map
    if type(file) == str:
        hdr = fits.getheader(file)
    elif type(file) == sunpy.map.mapbase.GenericMap or type(file) == sunpy.map.sources.sdo.HMIMap:
        hdr = file.fits_header
    elif type(file) == fits.header.Header:
        hdr = file
    
    if coords is not None:
        if type(coords) == list or type(coords) == tuple:
            coords = np.asarray([coords[0],coords[1],1])
        elif type(coords) == np.ndarray:
            if coords.ndim == 1:
                if coords.shape[0] == 2:
                    coords = np.append(coords,1)
            else:
                if coords.shape[1] == 2:
                    coords = np.append(coords,np.ones((coords.shape[0],1)),axis=1)
        if coords.ndim == 1:
            coords = coords[np.newaxis]
        
    pxsc = hdr['CDELT1']
    sun_dist_m=(hdr['DSUN_AU']*u.AU).to(u.m).value #Earth
    sun_dist_AU=hdr['DSUN_AU'] #Earth
    rsun = hdr['RSUN_REF'] # m
    pxbeg1 = hdr['PXBEG1']
    pxend1 = hdr['PXEND1']
    pxbeg2 = hdr['PXBEG2']
    pxend2 = hdr['PXEND2']
    
    dx = 2 * (sun_dist_m-rsun) * np.tan(pxsc/2/3600*np.pi/180) # m/px
    
    center = center_coord(hdr)
    
    # translation of the reference system from (0,0) to the disk center
    tr = np.asarray([[1,0,-center[0]],[0,1,-center[1]],[0,0,1]])
    if coords is None:
        X,Y = np.meshgrid(np.arange(0,pxend1-pxbeg1+1),np.arange(0,pxend2-pxbeg2+1))
        tr3 = np.tile(tr, (X.shape[0],X.shape[1],1,1))
        temp = np.moveaxis(np.moveaxis(np.asarray([[X,Y,np.ones(X.shape)],[X,Y,np.ones(X.shape)],[X,Y,np.ones(X.shape)]]),0,-1),0,-2)

        new_coords = (tr3 @ temp)
        del temp
    else:
        new_coords = tr @ np.moveaxis(coords,0,1)
    
    # rotation of the coordinate
    angle = hdr['CROTA'] # positive angle = clock-wise rotation of the reference system axes 
    rad = angle * np.pi/180
    rot = np.asarray([[np.cos(rad),-np.sin(rad),0],[np.sin(rad),np.cos(rad),0],[0,0,1]])
    rc = [0,0] # center is in (0,0) now

    tr = np.asarray([[1,0,rc[0]],[0,1,rc[1]],[0,0,1]])
    invtr = np.asarray([[1,0,-rc[0]],[0,1,-rc[1]],[0,0,1]])
    M = tr @ rot @ invtr
    
    if coords is None:
        M3 = np.tile(M, (X.shape[0],X.shape[1],1,1))
        new_coords = M3 @ new_coords
        new_coords = np.moveaxis(new_coords[:,:,:2,0],-1,0)
    else:
        new_coords = M @ new_coords
    
    new_coords = (new_coords)*pxsc
    th = np.arctan(np.sqrt(np.cos(new_coords[1]/3600*np.pi/180)**2*np.sin(new_coords[0]/3600*np.pi/180)**2+np.sin(new_coords[1]/3600*np.pi/180)**2/
                          (np.cos(new_coords[1]/3600*np.pi/180)*np.cos(new_coords[0]/3600*np.pi/180))))
    b = np.arcsin(sun_dist_m/rsun*np.sin(th)) - th
    # g = np.pi - th - b
    d = (sun_dist_m-rsun*np.cos(b))/np.cos(th)

    return new_coords[0],new_coords[1], d

def ccd2HCC(file,coords = None):
    """
    coordinate center in the center of the Sun
    x is pointing westward, y toward the north pole and z toward the observer (max for all should be Rsun)
    
    Input
    file: file_name, sunpy map or header
    coords: (x,y) or np.asarray([[x0,y0],[x1,y1],...])
            if None: built the coordinates map
            
    Output
    HCCx, HCCy, HCCz
    """
    import sunpy.map
    if type(file) == str:
#         smap = sunpy.map.Map(file)
        hdr = fits.getheader(file)
    elif type(file) == sunpy.map.mapbase.GenericMap or type(file) == sunpy.map.sources.sdo.HMIMap:
#         smap = file
        hdr = file.fits_header
    elif type(file) == fits.header.Header:
        hdr = file
    
    pxsc = hdr['CDELT1']
    sun_dist_m=(hdr['DSUN_AU']*u.AU).to(u.m).value #Earth
    sun_dist_AU=hdr['DSUN_AU'] #Earth
    rsun = hdr['RSUN_REF'] # m
    pxbeg1 = hdr['PXBEG1']
    pxend1 = hdr['PXEND1']
    pxbeg2 = hdr['PXBEG2']
    pxend2 = hdr['PXEND2']
    
    HPCx, HPCy, HPCd = ccd2HPC(file,coords)
    
    HCCx = HPCd * np.cos(HPCy/3600*np.pi/180) * np.sin(HPCx/3600*np.pi/180)
    HCCy = HPCd * np.sin(HPCy/3600*np.pi/180)
    HCCz = sun_dist_m - HPCd * np.cos(HPCy/3600*np.pi/180) * np.cos(HPCx/3600*np.pi/180)
    
    return HCCx,HCCy,HCCz

def ccd2HGS(file, coords = None):
    """
    From CCD frame to Heliographic Stonyhurst coordinates
    
    Input
    file: file_name, sunpy map or header
    coords: (x,y) or np.asarray([[x0,y0],[x1,y1],...])
            if None: built the coordinates map
            
    Output
    r, THETA, PHI
    """
    import sunpy.map
    if type(file) == str:
        hdr = fits.getheader(file)
    elif type(file) == sunpy.map.mapbase.GenericMap or type(file) == sunpy.map.sources.sdo.HMIMap:
        hdr = file.fits_header
    elif type(file) == fits.header.Header:
        hdr = file
    
    pxbeg1 = hdr['PXBEG1']
    pxend1 = hdr['PXEND1']
    pxbeg2 = hdr['PXBEG2']
    pxend2 = hdr['PXEND2']
    pxsc = hdr['CDELT1']
    B0 = hdr['HGLT_OBS']*np.pi/180
    PHI0 = hdr['HGLN_OBS']*np.pi/180
    sun_dist_m=(hdr['DSUN_AU']*u.AU).to(u.m).value #Earth
    sun_dist_AU=hdr['DSUN_AU'] #Earth
    rsun = hdr['RSUN_REF'] # m
    
    HCCx, HCCy, HCCz = ccd2HCC(file,coords)
        
    r = np.sqrt(HCCx**2 + HCCy**2 + HCCz**2)
    THETA = np.arcsin((HCCy*np.cos(B0) + HCCz*np.sin(B0))/r)*180/np.pi
    PHI = PHI0*180/np.pi + np.arctan(HCCx/(HCCz*np.cos(B0) - HCCy*np.sin(B0)))*180/np.pi
    
    # THETA == LAT; PHI == LON
    return r, THETA, PHI
  
def phi_disambig(bazi,bamb,method=2):
    """
    input
    bazi: magnetic field azimut. Type: str or array
    bamb: disambiguation fits. Type: str or array
    method: method selected for the disambiguation (0, 1 or 2). Type: int (2 as Default)
    
    output
    disbazi: disambiguated azimut. Type: array
    """
    # from astropy.io import fits
    if type(bazi) is str:
        bazi = fits.getdata(bazi)
    if type(bamb) is str:
        bamb = fits.getdata(bamb)
    
    disambig = bamb[0]/2**method
    disbazi = bazi.copy()
    disbazi[disambig%2 != 0] += 180
    
    return disbazi
