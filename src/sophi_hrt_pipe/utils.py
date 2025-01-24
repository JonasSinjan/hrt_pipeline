from astropy.io import fits
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.optimize as spo
import scipy.signal as sps
from datetime import datetime as dt
import datetime
import time
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

from scipy.ndimage import map_coordinates
from astropy import units as u
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
    """My custom print() function.

    Parameters 
    ----------
    *args:
        arguments to be printed
    color: string
        color of the text
    **kwargs:
        keyword arguments to be passed to print()

    Returns
    -------
    None

    From SPGPyLib PHITools
    """
    print(u"\u001b"+f"{color}", end='\r')
    print(*args, **kwargs)
    print(u"\u001b"+f"{bcolors.RESET}", end='\r')
    return 


def load_fits(path):
    """load a fits file

    Parameters
    ----------
    path: string
    location of the fits file

    Output
    ------
    data: numpy array, of stokes images in (row, col, wv, pol) 
    header: hdul header object, header of the fits file
    """
    with fits.open(f'{path}') as hdul_tmp:
        data = np.asarray(hdul_tmp[0].data, dtype = np.float32)
        header = hdul_tmp[0].header

    return data, header 


def get_data(path, scaling = True, bit_convert_scale = True, scale_data = True):
    """load science data from path and scale it if needed

    Parameters
    ----------
    path: string
        location of the fits file
    scaling: bool
        if True, divide by number of accumulations
    bit_convert_scale: bool
        if True, divide by 256 if the data is in 24.8bit format
    scale_data: bool
        if True, scale the data to the maximum range of the detector
    
    Returns
    -------
    data: numpy array
        stokes images in (row, col, wv, pol)
    header: hdul header object
        header of the fits file
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
       

def fits_get_sampling(file,num_wl = 6, TemperatureCorrection = True, TemperatureConstant = 40.323e-3, verbose = False):
    '''Open fits file, extract the wavelength axis and the continuum position, from Voltages in header

    Parameters
    ----------
    file: string
        location of the fits file
    num_wl: int
        number of wavelength
    TemperatureCorrection: bool
        if True, apply temperature correction to the wavelength axis
    TemperatureConstant: float
        Temperature constant to be used when TemperatureCorrection is True. Default: 40.323e-3 Å/K. Suggested (old) value: 36.46e-3 Å/K
    verbose: bool
        if True, print the continuum position
    
    Returns
    -------
    wave_axis: numpy array
        wavelength axis
    voltagesData: numpy array
        voltages of the wavelength axis
    tunning_constant: float
        tunning constant of the etalon
    cpos: int
        continuum position
    
    Adapted from SPGPyLib

    Usage: wave_axis,voltagesData,tunning_constant,cpos = fits_get_sampling(file,num_wl = 6, TemperatureCorrection = False, verbose = False)
    No S/C velocity corrected!!!
    cpos = 0 if continuum is at first wavelength and = num_wl - 1 (usually 5) if continuum is at the end
    '''
    fg_head = 3
    with fits.open(file) as hdu_list:
        header = hdu_list[fg_head].data
        tunning_constant = float(header[0][4])/1e9
        ref_wavelength = float(header[0][5])/1e3
        Tfg = hdu_list[0].header['FGOV1PT1'] # ['FGH_TSP1'] #temperature of the FG
        
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
    
    if TemperatureCorrection:
        if verbose:
            printc('-->>>>>>> If FG temperature is not 61, the relation wl = wlref + V * tunning_constant is not valid anymore',color=bcolors.WARNING)
            printc('          Use instead: wl =  wlref + V * tunning_constant + temperature_constant_new*(Tfg-61)',color=bcolors.WARNING)
        # temperature_constant_old = 40.323e-3 # old temperature constant, still used by Johann
        # temperature_constant_new = 37.625e-3 # new and more accurate temperature constant
        # temperature_constant_new = 36.46e-3 # value from HS
        wave_axis += TemperatureConstant*(Tfg-61) # 20221123 see cavity_maps.ipynb with example
        
    return wave_axis,voltagesData,tunning_constant,cpos


def fits_get_sampling_SPG(file,verbose = False):
    '''
    Obtains the wavelength and voltages from  fits header

    Parameters
    ----------
    file : str
        fits file path
    verbose : bool, optional
        More info printed. The default is False.

    Returns
    -------
    wave_axis : array
        wavelength axis
    voltagesData : array
        voltages
    tunning_constant : float
        tunning constant of etalon (FG)
    cpos : int
        continuum position
    
    From SPGPylibs PHITools
    '''
    fg_head = 3
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
                if np.abs(np.abs(float(v[2])) - np.abs(dummy)) > 5: #check that the next voltage is more than 5 from the previous, as voltages change slightly
                    voltagesData[j] = float(v[2])
                    dummy = voltagesData[j] 
                    j += 1

    d1 = voltagesData[0] - voltagesData[1]
    d2 = voltagesData[4] - voltagesData[5]
    if np.abs(d1) > np.abs(d2):
        cpos = 0
    else:
        cpos = 5
    if verbose:
        print('Continuum position at wave: ', cpos)
    wave_axis = voltagesData*tunning_constant + ref_wavelength  #6173.3356

    return wave_axis,voltagesData,tunning_constant,cpos


def check_filenames(data_f):
    """checks if the science scans have the same DID - this would otherwise cause an issue for naming the output demod files

    Parameters
    ----------
    data_f : list
        list of science scan file names
    
    Returns
    -------
    scan_name_list : list
        list of science scan file names with unique DIDs
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
    """check if science scans have same dimensions

    Parameters
    ----------
    data_arr : list
        list of science scan data arrays
    
    Returns
    -------
    None
    """
    first_shape = data_arr[0].shape
    result = all(element.shape == first_shape for element in data_arr)
    if (result):
        print("All the scan(s) have the same dimension")

    else:
        print("The scans have different dimensions! \n Ending process")

        exit()


def check_cpos(cpos_arr):
    """checks if the science scans have the same continuum positions

    Parameters
    ----------
    cpos_arr : list
        list of continuum positions

    Returns
    -------
    None
    """
    first_cpos = cpos_arr[0]
    result = all(c_position == first_cpos for c_position in cpos_arr)
    if (result):
        print("All the scan(s) have the same continuum wavelength position")

    else:
        print("The scans have different continuum_wavelength postitions! Please fix \n Ending Process")

        exit()


def compare_cpos(flat,cpos,cpos_ref):
    """checks if flat continuum same as data, if not try to move flat around - this assumes that there was a mistake with the continuum position in the flat

    Parameters
    ----------
    flat : array
        flat field data array
    cpos : int
        continuum position of flat field
    cpos_ref : int
        continuum position of science scan

    Returns
    -------
    flat : array
        flat field data array with continuum position corrected
    """
    if cpos != cpos_ref:
        print("The flat field continuum position is not the same as the data, trying to correct.")

        if cpos == 5 and cpos_ref == 0:

            return np.roll(flat, 1, axis = -1)

        elif cpos == 0 and cpos_ref == 5:

            return np.roll(flat, -1, axis = -1)

        else:
            print("Cannot reconcile the different continuum positions. \n Ending Process.")

            exit()
    else:
        return flat


def check_pmp_temp(hdr_arr):
    """check science scans have same PMP temperature set point

    Parameters
    ----------
    hdr_arr : list
        list of science scan header arrays
    
    Returns
    -------
    pmp_temp : str
    """
    first_pmp_temp = int(hdr_arr[0]['HPMPTSP1'])
    result = all(hdr['HPMPTSP1'] == first_pmp_temp for hdr in hdr_arr)
    if (result):
        t0 = time.strptime('2023-03-28T00:10:00','%Y-%m-%dT%H:%M:%S')
        t1 = time.strptime('2023-03-30T00:10:00','%Y-%m-%dT%H:%M:%S')
        tobs = time.strptime(hdr_arr[0]['DATE-OBS'][:-4],'%Y-%m-%dT%H:%M:%S')
        
        if (tobs > t0 and tobs < t1):
            first_pmp_temp = 50
            printc('WARNING: Data acquired on 2023-03-28 and 2023-03-29 have a PMP temperature setting to 40 deg, but the PMP are fluctuating at ~45 deg \nException to HRT pipeline to use the 50 deg demodulation matrix.',color=bcolors.WARNING)
        print(f"All the scan(s) have the same PMP Temperature Set Point: {first_pmp_temp}")
        pmp_temp = str(first_pmp_temp)
        return pmp_temp
    else:
        print("The scans have different PMP Temperatures! Please fix \n Ending Process")

        exit()


def check_IMGDIRX(hdr_arr):
    """check if all scans contain imgdirx keyword

    Parameters
    ----------
    hdr_arr : list
        list of science scan header arrays
    
    Returns
    -------
    header_imgdirx_exists : bool
    imgdirx_flipped : str or bool
        OPTIONS: 'YES' or 'NO' or False
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
    """returns flat that matches the orientation of the science data

    Parameters
    ----------
    flat : array
        flat field data array
    header_imgdirx_exists : bool
        if all scans contain imgdirx keyword
    imgdirx_flipped : str or bool
        OPTIONS: 'YES' or 'NO' or False
    header_fltdirx_exists : bool
        if flat contains fltdirx keyword
    fltdirx_flipped : str or bool
        OPTIONS: 'YES' or 'NO' or False

    Returns
    -------
    flat : array
        flat field data array with orientation corrected
    """
    if header_imgdirx_exists and imgdirx_flipped == 'YES': 
        #if science is flipped
        if header_fltdirx_exists:
            if fltdirx_flipped == 'YES':
                return flat
            else:
                print('Flipping the calibration dataset')
                return flat[:,:,::-1]
        else:
            print('Flipping the calibration dataset')
            return flat[:,:,::-1]
    elif (header_imgdirx_exists and imgdirx_flipped == 'NO') or not header_imgdirx_exists: 
        #if science is not flipped, or keyword doesnt exist, then assumed not flipped
        if header_fltdirx_exists:
            if fltdirx_flipped == 'YES':
                print('Flipping the calibration dataset')
                return flat[:,:,::-1] #flip flat back to match science
            else:
                return flat
        else:
            return flat
    else:
        return flat


def stokes_reshape(data):
    """converting science to [y,x,pol,wv,scans]
    
    Parameters
    ----------
    data : array
        science data array
    
    Returns
    -------
    data : array
        science data array with shape [y,x,pol,wv,scans]
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
    """This function is used to fix the path for windows and linux systems

    Parameters
    ----------
    path : str
        path to be fixed
    dir : str, optional
        direction of the path, by default 'forward'
    verbose : bool, optional
        print the path, by default False

    Returns
    -------
    path : str
        fixed path

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
    """filling the data with cubic spline interpolation

    Parameters
    ----------
    arr : array
        array to be filled
    thresh : float
        threshold for filling
    mode : str
        mode for filling, 'max', 'min', 'abs', 'exact rows', 'exact columns'
    axis : int, optional
        axis to be filled, by default -1

    Returns
    -------
    array
        filled array
    """
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
    
def ARmasking(stk, initial_mask, cpos = 0, bin_lim = 7, mask_lim = 5, erosion_iter = 3, dilation_iter = 3):
    """Creates a mask to cover active parts of the FoV
    Parameters
    ----------
    stk : array
        Stokes Vector
    Initial_mask : array
        Mask with off-limb or field_Stop excluded
    cpos : int
        continuum position (DEFAULT: 0)
    bin_lim : float
        number to be multiplied to the polarized std to se the maximum limit of the bins (DEFAULT: 7)
    mask_lim : float
        number of std that defines the contour of the mask (DEFAULT: 5)
    erosion_iter : int
        number of iterations for the erosion of the mask (DEFAULT: 3)
    dilation_iter : int
        number of iterations for the dilation of the mask (DEFAULT: 3)    

    Returns
    -------
    array
        AR_mask
    """

    AR_mask = initial_mask.copy()
    # automatic bins looking at max std of the continuum polarization
    if stk.shape[1] == 4:
        stk = np.einsum('lpyx->yxpl',stk.copy())
    lim = np.max((stk[:,:,1:,cpos]).std(axis=(0,1)))*bin_lim
    bins = np.linspace(-lim,lim,150)

    for p in range(1,4):
        hi = np.histogram(stk[:,:,p].flatten(),bins=bins)
        gval = gaussian_fit(hi, show = False)
        AR_mask *= np.max(np.abs(stk[:,:,p] - gval[1]),axis=-1) < mask_lim*abs(gval[2])

    AR_mask = np.asarray(AR_mask, dtype=bool)

    # erosion and dilation to remove small scale masked elements
    AR_mask = ~binary_dilation(binary_erosion(~AR_mask.copy(),generate_binary_structure(2,2), iterations=erosion_iter),
                               generate_binary_structure(2,2), iterations=dilation_iter)
    
    return AR_mask

def auto_norm(file_name):
    """This function is used to normalize the data from the fits extensions

    Parameters
    ----------
    file_name : str
        path to file

    Returns
    -------
    norm : float
        normalization factor
    """
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
    """get mu angle for a pixel

    Parameters
    ----------
    hdr : header or filename
        header of the fits file or filename path
    coord : array, optional
        pixel location (x,y) for which the mu angle is found (if None: center of the FoV), by default None.
        Shape has to be (2,Npix)

    Returns
    -------
    mu : float
        cosine of the heliocentric angle
    """
    if type(hdr) is str:
        hdr = fits.getheader(hdr)
    
    center=center_coord(hdr)
    Rpix=(hdr['RSUN_ARC']/hdr['CDELT1'])
    
    if coord is None:
        coord = np.asarray([(hdr['PXEND1']-hdr['PXBEG1'])/2,
                            (hdr['PXEND2']-hdr['PXBEG2'])/2]) - center[:2]
    else:
        coord -= center[:2,np.newaxis]
    mu = np.sqrt(Rpix**2 - (coord[0]**2 + coord[1]**2)) / Rpix
    
    return mu

def center_coord(hdr):
    """calculate the center of the solar disk in the rotated reference system

    Parameters
    ----------
    hdr : header
        header of the fits file

    Returns
    -------
    center: [x,y,1] coordinates of the solar disk center (units: pixel)
    """
    pxsc = hdr['CDELT1']
    # sun_dist_m=(hdr['DSUN_AU']*u.AU).to(u.m).value #Earth
    # sun_dist_AU=hdr['DSUN_AU'] #Earth
    # rsun = hdr['RSUN_REF'] # m
    # pxbeg1 = hdr['PXBEG1']
    # pxend1 = hdr['PXEND1']
    # pxbeg2 = hdr['PXBEG2']
    # pxend2 = hdr['PXEND2']
    crval1 = hdr['CRVAL1']
    crval2 = hdr['CRVAL2']
    crpix1 = hdr['CRPIX1']
    crpix2 = hdr['CRPIX2']
    PC1_1 = hdr['PC1_1']
    PC1_2 = hdr['PC1_2']
    PC2_1 = hdr['PC2_1']
    PC2_2 = hdr['PC2_2']
        
    HPC1 = 0
    HPC2 = 0
    
    x0 = crpix1 + 1/pxsc * (PC1_1*(HPC1-crval1) - PC1_2*(HPC2-crval2)) - 1
    y0 = crpix2 + 1/pxsc * (PC2_2*(HPC2-crval2) - PC2_1*(HPC1-crval1)) - 1
    
    return np.asarray([x0,y0,1])



def circular_mask(h, w, center, radius):
    """create a circular mask

    Parameters
    ----------
    h : int
        height of the mask
    w : int
        width of the mask
    center : [x,y]
        center of the mask
    radius : float
        radius of the mask

    Returns
    -------
    mask: 2D array
        mask with 1 inside the circle and 0 outside
    """
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def limb_side_finder(img, hdr,verbose=True,outfinder=False):
    """find the limb in the image

    Parameters
    ----------
    img : 2D array
        data array
    hdr : header
        header of the fits file
    verbose : bool, optional
        print the limb side, by default True
    outfinder : bool, optional
        return the finder array, by default False
    
    Returns
    -------
    side: str
        limb side
    center: [x,y] 
        coordinates of the solar disk center (units: pixel)
    Rpix: float
        Radius of solar disk in pixels
    sly: slice
        slice in y direction to be used for normalisation
    slx: slice
        slice in x direction to be used for normalisation
    finder: 2D array
        finder array, optional, only returned if outfinder is True
    """
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


def limb_fitting(img, hdr, field_stop, verbose=True, percent=False, fit_results=False):
    """Fits limb to the image using least squares method.

    Parameters
    ----------
    img : numpy.ndarray
        Image to fit limb to.
    hdr : astropy.io.fits.header.Header
        header of fits file
    field_stop : array
        field stop array
    verbose : bool, optional
        Print limb fitting results, by default True
    percent : bool, optional
        return mask with 96% of the readius, by default False
    fit_results : bool, optional
        return results of the circular fit, by default False

    Returns
    -------
    mask100: numpy.ndarray
        masked array (ie off disc region) with 100% of the radius
    sly: slice
        slice in y direction to be used for normalisation (ie good pixels on disc)
    slx: slice
        slice in x direction to be used for normalisation (ie good pixels on disc)
    side: str
        limb side
    mask96: numpy.ndarray
        masked array (ie off disc region) with 96% of the radius (only if percent = True)
    """
    def _residuals(p,x,y):
        """
        Finding the residuals of the fit
        
        Parameters
        ----------
        p : list
            [xc,yc,R] - coordinates of the centre and radius of the circle
        x : float
            test x coordinate
        y : float
            test y coordinate

        Returns
        -------
        residual = R**2 - (x-xc)**2 - (y-yc)**2
        """
        xc,yc,R = p
        residual = R**2 - (x-xc)**2 - (y-yc)**2
        return residual
    
    def _is_outlier(points, thresh=2):
        """Returns a boolean array with True if points are outliers and False otherwise
        
        Parameters
        ----------
        points : numpy.ndarray
            1D array of points
        thresh : int, optional
            threshold for outlier detection, by default 2
        """
        if len(points.shape) == 1:
            points = points[:,None]
        median = np.median(points, axis=0)
        diff = np.sum((points - median)**2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation

        return modified_z_score > thresh
        
    def _image_derivative(d):
        """Calculates the image derivative in x and y using a 3x3 kernel
        
        Parameters
        ----------
        d : numpy.ndarray
            image to calculate derivative of
        
        Returns
        -------
        SX : numpy.ndarray
            derivative in x direction
        SY : numpy.ndarray
            derivative in y direction
        """
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
        
    if side == '':
        output = [None,sly,slx,side]
        
        if percent:
            output += [None]
        if fit_results:
            output += [None]    

        return output
    
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
        
    mask100 = circular_mask(img.shape[0],img.shape[1],[p.x[0],p.x[1]],p.x[2])

    mask96 = circular_mask(img.shape[0],img.shape[1],[p.x[0],p.x[1]],p.x[2]*.96)
    
    output = [sly,slx,side]
    if 'N' in side or 'S' in side:
        output = [np.moveaxis(mask100,0,1)] + output
        if percent:
            output += [np.moveaxis(mask96,0,1)]
    else:
        output = [mask100] + output
        if percent:
            output += [mask96]
    if fit_results:
        output += [p]    

    return output

#### New Limb fitting ###
def double_gaus(x,a0,x0,sigma0,a1,x1,sigma1):
    """return Gauss function

    Parameters
    ----------
    x : array
        x values
    a0 : float
        gaussian nr.1 amplitude
    x0 : float
        gaussian nr.1 mean x value
    sigma0 : float
        gaussian nr.1 standard deviation
    a1 : float
        gaussian nr.2 amplitude
    x1 : float
        gaussian nr.2 mean x value
    sigma1 : float
        gaussian nr.2 standard deviation

    Returns
    -------
    Double Gauss Function : array
    """
    return a0*np.exp(-(x-x0)**2/(2*sigma0**2)) + a1*np.exp(-(x-x1)**2/(2*sigma1**2))

def double_gaussian_fit(a,show=True,covariance=False):
    """Two Gaussian fit for data 'a' from np.histogram or plt.hist
    The gaussian must be complitely separated and on opposite sides of the distribution
    Parameters
    ----------
    a : array
        output from np.histogram or plt.hist
    show : bool, optional
        show plot of fit, by default True
    covariance: bool, optional
        if True, reutn the covariance matrix (Default: False)
    Returns
    -------
    p : array
        fitted coefficients for Double Gaussian function
    """
    xx=a[1][:-1] + (a[1][1]-a[1][0])/2
    y=a[0][:]
    # p0 = np.ones(6)
    xx1 = xx[:xx.size//2]; xx2 = xx[xx.size//2:]
    y1 = y[:y.size//2]; y2 = y[y.size//2:]
    p0=[max(y1),sum(xx1*y1)/sum(y1),np.sqrt(sum(y1 * (xx1 - sum(xx1*y1)/sum(y1))**2) / sum(y1)),max(y2),sum(xx2*y2)/sum(y2),np.sqrt(sum(y2 * (xx2 - sum(xx2*y2)/sum(y2))**2) / sum(y2))] #weighted avg of bins for avg and sigma inital values
    # p0[0]=y1[find_nearest(xx1,p0[1])-5:find_nearest(xx1,p0[1])+5].mean() #find init guess for ampltiude of gauss func
    # p0[3]=y2[find_nearest(xx2,p0[1])-5:find_nearest(xx2,p0[1])+5].mean() #find init guess for ampltiude of gauss func
    
    try:
        p,cov=spo.curve_fit(double_gaus,xx,y,p0=p0)
        if show:
            lbl = '{:.2e} $\pm$ {:.2e}\n{:.2e} $\pm$ {:.2e}'.format(p[1],p[2],p[4],p[5])
            plt.plot(xx,double_gaus(xx,*p),'r--', label=lbl)
            plt.legend(fontsize=9)
        if covariance:
            return p,cov
        else:
            return p
    except:
        printc("Gaussian fit failed: return initial guess",color=bcolors.WARNING)
        return p0
    
def elliptical_mask(shape,p):
    """
    Ellipse mask

    Parameters
    ----------
    shape : tuple
            shape of the mask
    p : list
        [a,b,h,k,A] - ellipse axes (x,y), centers (x,y) and angle

    Returns
    -------
    mask: numpy.ndarray
          Boolean elliptical mask with 1 inside the ellipse and 0 outside
    """    
    a,b,h,k,A = p
    x,y = np.meshgrid(np.arange(shape[1]),np.arange(shape[0]))
    mask = np.zeros(shape)
    mask[((x-h)*np.cos(A)+(y-k)*np.sin(A))**2/a**2+((x-h)*np.sin(A)-(y-k)*np.cos(A))**2/b**2 <= 1] = 1

    return mask

def fit_plane(data, mask=None, order=1):

    """
    Fit 2D plane to data. Will be replazed by a global one (comming from had-hoc branch) at some point.
    """
    def _polyfit2d(a,XY,order):
        from scipy.optimize import curve_fit
        sz = a.shape
        try:
            X,Y = XY
        except:
            X,Y = np.meshgrid(np.arange(sz[1])-sz[1]//2,np.arange(sz[0])-sz[0]//2)
        fit_func = lambda x, *p : poly2d(order,x,*p)
        
        try:
            X = X[~X.mask]; Y = Y[~Y.mask]; a = a[~a.mask]
        except:
            pass
        N = np.arange(1,order+2).sum()
        popt, pcov = curve_fit(fit_func, (X,Y), a.ravel(),p0=np.ones(N))
        
        return popt#, np.reshape(poly2d(order,(X,Y), *popt), sz),

    def poly2d(m, X, *p):
        x,y = X
        z = np.zeros(x.shape)
        n = 0
        for k in range(1,m+1):
            for i in range(k+1):
                z += x**(k-i) * y**i * p[n]
                n += 1
        z += p[-1]
        return z.ravel()
    
    yd, xd = data.shape
    x = np.arange(xd)
    y = np.arange(yd)
    X, Y = np.meshgrid(x, y)

    if mask is not None:
        X_masked = X[mask]
        Y_masked = Y[mask]
        Z_masked = data[mask]
    else:
        X_masked = X
        Y_masked = Y
        Z_masked = data

    p = _polyfit2d(Z_masked.flatten(),(X_masked.flatten(),Y_masked.flatten()),order)
    P = np.reshape(poly2d(order,(X,Y), *p), (yd,xd))

    return (P, p)

def subROIconstrast(img, img_mask, windowSize, windowSeparation):
    """Align the mod (pol) states 2,3,4 with state 1 for a given wavelength
    loop through all wavelengths

    Parameters
    ----------
    img: ndarray
        2D input image array
    img_mask: boolean ndarray
        mask that defines where to compute the contrast
    windowSize: int
        half size of the sub regions where to compute the contrast
    windowSeparation: int
        sepration between the center of the sub regions where to compute the contrast
    
    Returns
    -------
    contrast: ndarray
        values of the contrast (all zeros except for the pixels corresponding to the center of the sub regions)

    """

    data_size = img.shape
    contrast = np.zeros((img.shape))
    # shift_raw = np.zeros((2,pn*wln))

    for i in list(range(int(50),int(data_size[0]-50),windowSeparation))+list(range(int(data_size[0]-50),int(50),-windowSeparation)):
        for j in list(range(int(50),int(data_size[1]-50),windowSeparation))+list(range(int(data_size[1]-50),int(50),-windowSeparation)):
             # print(f'({i}/{data_size[0]}, {j}/{data_size[1]})')#\r',end='')
            roi = (slice(i-windowSize,i+windowSize),slice(j-windowSize,j+windowSize))
            if img_mask[roi].sum() == 4*windowSize**2:
                temp = img[roi].copy()
                # detrend
                temp /= fit_plane(temp.copy(),order=5)[0]
                contrast[i,j] = np.nanstd(temp)/np.nanmean(temp)
            
    return contrast

def limb_ellipse(img, hdr, field_stop, AR_mask, verbose=True, percent=False, fit_results=False, high_contrast = True, debug=False):
    """Fits limb to the image using least squares method.

    Parameters
    ----------
    img : numpy.ndarray
        Image to fit limb to.
    hdr : astropy.io.fits.header.Header
        header of fits file
    field_stop : array
        field stop array
    verbose : bool, optional
        Print limb fitting results, by default True
    percent : bool, optional
        return mask with 96% of the readius, by default False
    fit_results : bool, optional
        return results of the circular fit, by default False
    high_contrast : bool, optional
        if true it returns slices from the region with higher contrast instead of those from limb_side_finder, by default True
    debug: bool, optional
        if True, return dictionary with all the variables, by default False
    Returns
    -------
    mask100: numpy.ndarray
        masked array (ie off disc region) with 100% of the radius
    sly: slice
        slice in y direction to be used for normalisation (ie good pixels on disc)
    slx: slice
        slice in x direction to be used for normalisation (ie good pixels on disc)
    side: str
        limb side
    mask96: numpy.ndarray
        masked array (ie off disc region) with 96% of the radius (only if percent = True)
    p: scipy.optimize._optimize.OptimizeResult
        Result of the least sqaure ellipse fit
    """

    def _residuals(p,x,y):
        """
        Finding the residuals of the fit

        Parameters
        ----------
        p : list
            [a,b,h,k,A] - ellipse axes (x,y), centers (x,y) and angle
        x : float
            test x coordinate
        y : float
            test y coordinate

        Returns
        -------
        residual = ((x-h)*np.cos(A)+(y-k)*np.sin(A))**2/a**2 + (-(x-h)*np.sin(A)+(y-k)*np.cos(A))**2/b**2 - 1
        """

        a,b,h,k,A = p
        residual = ((x-h)*np.cos(A)+(y-k)*np.sin(A))**2/a**2 + (-(x-h)*np.sin(A)+(y-k)*np.cos(A))**2/b**2 - 1

        return residual

    from scipy.optimize import least_squares
    from scipy.ndimage import binary_erosion, binary_dilation

    side, center, Rpix, sly, slx = limb_side_finder(img,hdr,verbose=verbose,outfinder=False)
    
    s = 5

    hi = np.histogram(img[s:-s,s:-s][AR_mask[s:-s,s:-s]>0].flatten(),bins=100);
    gres, cov = double_gaussian_fit(hi,False,True)
    
    if side == '' and (np.any((np.sqrt(np.diagonal(cov))/gres)[:3] > 10) or np.any(np.isnan(cov))): # sometimes south pole limb is not found, so extra condition on fit
        output = [None,sly,slx,side]
        
        if percent:
            output += [None]
        if fit_results:
            output += [None]    

        return output

    xx=hi[1][:-1] + (hi[1][1]-hi[1][0])/2
    thr = xx[find_nearest(xx,min(gres[1],gres[4]))+np.argmin(hi[0][find_nearest(xx,min(gres[1],gres[4])):find_nearest(xx,max(gres[1],gres[4]))])]

    limb_mask = img[s:-s,s:-s]>thr;
    
    # dilation and erosion to remove any possible zeros coming from umbrae
    # border_value=1 in erosion to avoid black edges
    limb_mask = binary_erosion(binary_dilation(limb_mask,[[0,1,0],[1,1,1],[0,1,0]],iterations=20),[[0,1,0],[1,1,1],[0,1,0]],iterations=20,border_value=1)
    
    # erosion of field stop to avoid edges from there
    limb_edge = image_derivative(limb_mask)*binary_erosion(field_stop,[[0,1,0],[1,1,1],[0,1,0]],iterations=20)[s:-s,s:-s]
    yi, xi = np.where(limb_edge>0.9)

    p = least_squares(_residuals,x0 = [Rpix,Rpix,center[0],center[1],0], args=(xi,yi),
                              bounds = ([Rpix-100,Rpix-100,center[0]-300,center[1]-300,-np.pi/2],[Rpix+100,Rpix+100,center[0]+300,center[1]+300,np.pi/2]))

    mask100 = elliptical_mask(img.shape,p.x)
    mask98 = elliptical_mask(img.shape,[p.x[0]*.98,p.x[1]*.98,p.x[2],p.x[3],p.x[4]])
    mask96 = elliptical_mask(img.shape,[p.x[0]*.96,p.x[1]*.96,p.x[2],p.x[3],p.x[4]])
    
    if high_contrast:
        # if hdr['DSUN_AU'] < 0.4:
        #     windowSize = 384
        # else:
        windowSize = 256
        contrast256 = subROIconstrast(img.copy(), (field_stop*mask98)>0, windowSize, windowSize)
        i,j = np.unravel_index(np.argmax(contrast256),contrast256.shape)
        sly,slx = slice(i-windowSize,i+windowSize), slice(j-windowSize,j+windowSize)
        print('\nHigh contrast slices: ',sly,slx,'\n')

    output = [sly,slx,side]

    output = [mask100] + output
    
    if debug:
        return {'mask100':mask100,'mask96':mask96,'hi':hi,'gres':gres,'thr':thr,'xx':xx,'limb_mask':limb_mask,'limb_edge':limb_edge,'yi':yi,'xi':xi,'p':p}
    if percent:
        output += [mask96]
    if fit_results:
        output += [p]    

    return output
    ######

def fft_shift(img,shift):
    """Shift an image in the Fourier domain and return the shifted image (non fourier domain)

    Parameters
    ----------
    img : 2D-image
        2D-image to be shifted
    shift : list
        [dy,dx] shift in pixel

    Returns
    -------
    img_shf : 2D-image
        shifted image
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
    """FFT shifting function from SPGPylibs as used in FDT pipeline.

    Parameters
    ----------
    data : 3D-array
        [z,y,x] 3D-array of images to be shifted, images stacked along the first (z) axis.
    norma : bool, optional
        If True, the images are normalized before shifting. The default is True.
    prec : int, optional
        Precision of the shift. The default is 100.
    coarse_prec : float, optional
        Coarse precision of the shift. The default is 1.5.
    sequential : bool, optional
        if True, adds shifts to the previous one. The default is False.

    Returns
    -------
    row_shift : 1D-array
        row shifts
    column_shift : 1D-array
        column shifts
    shifted_image: 3D-array
        shifted images

    From SPGPylibs. Same function used for FDT pipeline, adapted by DC
    At least two images should be provided!
    usage: s_y, s_x, simage = PHI_shifts_FFT(image_cropped,prec=500,verbose=True,norma=False)
    (row_shift, column_shift) defined as  center = center + (y,x) 
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
    """return index of nearest value in array to the desired value

    Parameters
    ----------
    array : array
        array to search
    value : float
        value to search for

    Returns
    -------
    idx : int
        index of nearest value in array to the desired value
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def gaus(x,a,x0,sigma):
    """return Gauss function

    Parameters
    ----------
    x : array
        x values
    a : float
        amplitude
    x0 : float
        mean x value
    sigma : float
        standard deviation

    Returns
    -------
    Gauss Function : array
    """
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def gaussian_fit(a,show=True):
    """Gaussian fit for data 'a' from np.histogram or plt.hist

    Parameters
    ----------
    a : array
        output from np.histogram
    show : bool, optional
        show plot of fit, by default True
    
    Returns
    -------
    p : array
        fitted coefficients for Gaussian function
    """
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
    """Iterative Gaussian fit for noise estimate

    Parameters
    ----------
    temp : array
        data to fit
    p : array, optional
        initial guess for Gaussian fit, by default [1,0,1e-1]
    eps : float, optional
        convergence criteria, by default 1e-6
    
    Returns
    -------
    p : array
        fitted coefficients for Gaussian function
    hi : array
        output from np.histogram
    """
    p_old = [1,0,100]; count = 0
    it = 0
    while np.abs(p[2] - p_old[2])>eps:
        p_old = p; count += 1
        hi = np.histogram(temp, bins=np.linspace(p[1] - 3*p[2],p[1] + 3*p[2],200),density=False);
        p = gaussian_fit(hi, show=False)
        if it == 50:
            break
        it += 1
    return p, hi

  
def blos_noise(blos_file, iter=True, fs = None):
    """plot blos on left panel, and blos hist + Gaussian fit (w/ iterative fit option - only shown in legend)

    Parameters
    ----------
    blos_file : str
        path to blos file
    iter : bool, optional
        performs iterative Gaussian fit, by default True
    fs : array, optional
        field stop mask, by default None

    Returns
    -------
    p or p_iter: fit coefficients for Gaussian function
    """
    blos = fits.getdata(blos_file)
    hdr = fits.getheader(blos_file)
    #get the pixels that we want to consider (central 512x512 and limb handling)
    _, _, _, sly, slx = limb_side_finder(blos, hdr)
    values = blos[sly,slx]

    fig, ax = plt.subplots(1,2, figsize = (14,6))
    if fs is not None:
        idx = np.where(fs<1)
        blos[idx] = -300
    im1 = ax[0].imshow(blos, cmap = "gray", origin = "lower", vmin = -200, vmax = 200)
    fig.colorbar(im1, ax = ax[0], fraction=0.046, pad=0.04)
    hi = ax[1].hist(values.flatten(), bins=np.linspace(-2e2,2e2,200))
    tmp = [0,0]
    tmp[0] = hi[0].astype('float64')
    tmp[1] = hi[1].astype('float64')

    #guassian fit + label
    p = gaussian_fit(tmp, show = False)    
    xx=hi[1][:-1] + (hi[1][1]-hi[1][0])/2
    lbl = f'{p[1]:.2e} $\pm$ {p[2]:.2e} G'
    
    if iter:
        ax[1].plot(xx,gaus(xx,*p),'r--', label=lbl)
        try:
            p_iter, hi_iter = iter_noise(values,[1.,0.,10.],eps=1e-4); p_iter[0] = p[0]
            ax[1].plot(xx,gaus(xx,*p_iter),'g--', label= f"Iter Fit: {p_iter[1]:.2e} $\pm$ {p_iter[2]:.2e} G")
            # ax[1].scatter(0,0, color = 'white', s = 0, label = lbl) #also display the original fit in legend
        except:
            print("Iterative Gauss Fit failed")
            p_iter = p

    else:
        ax[1].plot(xx,gaus(xx,*p),'r--', label=lbl)

    ax[1].legend(fontsize=15)

    date = blos_file.split('blos_')[1][:15]
    dt_str = dt.strptime(date, "%Y%m%dT%H%M%S")
    fig.suptitle(f"Blos {dt_str}")

    plt.tight_layout()
    plt.show()

    if iter:
        return p_iter
    else:
        return p


def blos_noise_arr(blos, fs = None):
    """
    plot blos on left panel, and blos hist + Gaussian fit (w/ iterative option)

    DEPRACATED - use blos_noise instead
    """

    fig, ax = plt.subplots(1,2, figsize = (14,6))
    if fs is not None:
        idx = np.where(fs<1)
        blos[idx] = -300
    im1 = ax[0].imshow(blos, cmap = "gray", origin = "lower", vmin = -200, vmax = 200)
    fig.colorbar(im1, ax = ax[0], fraction=0.046, pad=0.04)
    hi = ax[1].hist(blos.flatten(), bins=np.linspace(-2e2,2e2,200))
    #print(hi)
    tmp = [0,0]
    tmp[0] = hi[0].astype('float64')
    tmp[1] = hi[1].astype('float64')


    #guassian fit + label
    p = gaussian_fit(tmp, show = False)  
    try:  
        p_iter, hi_iter = iter_noise(blos,[1.,0.,1.],eps=1e-4)
        ax[1].scatter(0,0, color = 'white', s = 0, label = f"Iter Fit: {p_iter[1]:.2e} $\pm$ {p_iter[2]:.2e} G")
    except:
        print("Iterative Gauss Fit failed")
    xx=hi[1][:-1] + (hi[1][1]-hi[1][0])/2
    lbl = f'{p[1]:.2e} $\pm$ {p[2]:.2e} G'
    ax[1].plot(xx,gaus(xx,*p),'r--', label=lbl)
    ax[1].legend(fontsize=15)

    plt.tight_layout()
    plt.show()
    

def stokes_noise(stokes_file, iter=True):
    """plot stokes V on left panel, and Stokes V hist + Gaussian fit (w/ iterative option)

    Parameters
    ----------
    stokes_file : str
        path to stokes file
    iter : bool, optional
        whether to use iterative Gaussian fit, by default True

    Returns
    -------
    p or p_iter: array
        Gaussian fit parameters
    """
    stokes = fits.getdata(stokes_file)
    if stokes.shape[0] == 6:
        stokes = np.einsum('lpyx->yxpl',stokes)
    hdr = fits.getheader(stokes_file)
    out = fits_get_sampling(stokes_file)
    cpos = out[3]
    #first get the pixels that we want (central 512x512 and limb handling)
    _, _, _, sly, slx = limb_side_finder(stokes[:,:,3,cpos], hdr)
    values = stokes[sly,slx,3,cpos]

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
    xx=hi[1][:-1] + (hi[1][1]-hi[1][0])/2
    lbl = f'{p[1]:.2e} $\pm$ {p[2]:.2e}'
    
    if iter:
        ax[1].plot(xx,gaus(xx,*p),'r--', label=lbl)
        try:
            p_iter, hi_iter = iter_noise(values,[1.,0.,.1],eps=1e-6); p_iter[0] = p[0]
            ax[1].plot(xx,gaus(xx,*p_iter),'g--', label= f"Iter Fit: {p_iter[1]:.2e} $\pm$ {p_iter[2]:.2e}")
            # ax[1].scatter(0,0, color = 'white', s = 0, label = lbl) #also display the original fit in legend
        except:
            print("Iterative Gauss Fit failed")
            ax[1].plot(xx,gaus(xx,*p),'r--', label=lbl)
            p_iter = p

    else:
        ax[1].plot(xx,gaus(xx,*p),'r--', label=lbl)

    ax[1].legend(fontsize=15)

    date = stokes_file.split('stokes_')[1][:15]
    dt_str = dt.strptime(date, "%Y%m%dT%H%M%S")
    fig.suptitle(f"Stokes {dt_str}")

    plt.tight_layout()
    plt.show()

    if iter:
        return p_iter
    else:
        return p


########### new WCS script 3/6/2022 ###########
def image_derivative(d):
    """Calculates the total image derivative (x**2 + y**2) using a 3x3 kernel
    
    Parameters
    ----------
    d : numpy.ndarray
        image to calculate derivative of
    
    Returns
    -------
    A : numpy.ndarray
        image derivative (combined X and Y)
    """
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
    by F. Kahil (MPS)
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


def rotate_header(h,angle,center = [1024.5,1024.5]):
    """calculate new header when image is rotated by a fixed angle

    Parameters
    ----------
    h : astropy.io.fits.header.Header
        header of image to be rotated
    angle : float
        angle to rotate image by (in degrees)
    center : list or numpy.array
        center of rotation (x,y) in pixel, Default is [1024.5,1024.5]

    Returns
    -------
    h: astropy.io.fits.header.Header
        new header
    """
    h['CROTA'] -= angle
    h['PC1_1'] = np.cos(h['CROTA']*np.pi/180)
    h['PC1_2'] = -np.sin(h['CROTA']*np.pi/180)
    h['PC2_1'] = np.sin(h['CROTA']*np.pi/180)
    h['PC2_2'] = np.cos(h['CROTA']*np.pi/180)
    rad = angle * np.pi/180
    rot = np.asarray([[np.cos(rad),-np.sin(rad),0],[np.sin(rad),np.cos(rad),0],[0,0,1]])
    coords = np.asarray([h['CRPIX1'],h['CRPIX2'],1])
#     center = [1024.5,1024.5] # CRPIX from 1 to 2048, so 1024.5 is the center
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
    
def translate_header(h,tvec,mode='crpix'):
    """calculate new header when image is translated by a fixed vector

    Parameters
    ----------
    h : astropy.io.fits.header.Header
        header of image to be translated
    tvec : list
        vector to translate image by (in pixels) [x,y]
    mode : str
        if 'crpix' (Default) the shift will be applied to CRPIX*, if 'crval' the shift will be applied to CRVAL*

    Returns
    -------
    h: astropy.io.fits.header.Header
        new header
    """
    if mode == 'crval':
        tr = np.asarray([[1,0,-tvec[1]],[0,1,-tvec[0]],[0,0,1]])
        angle = h['CROTA'] # positive angle = clock-wise rotation of the reference system axes 
        rad = angle * np.pi/180
        vec = np.asarray([tvec[1],tvec[0],1])
        rot = np.asarray([[np.cos(rad),-np.sin(rad),0],[np.sin(rad),np.cos(rad),0],[0,0,1]])
        shift = rot @ vec
        shift[0] *= h['CDELT1']
        shift[1] *= h['CDELT2']
        h['CRVAL1'] = round(h['CRVAL1']-shift[0],4)
        h['CRVAL2'] = round(h['CRVAL2']-shift[1],4)
    elif mode == 'crpix':
        tr = np.asarray([[1,0,tvec[1]],[0,1,tvec[0]],[0,0,1]])
        coords = np.asarray([h['CRPIX1'],h['CRPIX2'],1])
        new_coords = tr @ coords
        h['CRPIX1'] = round(new_coords[0],4)
        h['CRPIX2'] = round(new_coords[1],4)
    else:
        print('mode not valid\nreturn old header')
    return h

def image_register(ref,im,subpixel=True,deriv=False,d=50):
    """
    credits: Marco Stangalini (2010, IDL version). Adapted for Python by Daniele Calchetti.
    """
    try:
        import pyfftw.interfaces.numpy_fft as fft
    except:
        import numpy.fft as fft
        
    def _image_derivative(d):
        import numpy as np
        from scipy.signal import convolve
        kx = np.asarray([[1,0,-1], [1,0,-1], [1,0,-1]])
        ky = np.asarray([[1,1,1], [0,0,0], [-1,-1,-1]])
        kx=kx/3.
        ky=ky/3.
        SX = convolve(d, kx,mode='same')
        SY = convolve(d, ky,mode='same')
#         A=SX+SY
        # DC change on 12/07/2022
        A=SX**2+SY**2
        return A

    def _g2d(X, offset, amplitude, sigma_x, sigma_y, xo, yo, theta):
        import numpy as np
        (x, y) = X
        xo = float(xo)
        yo = float(yo)
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                                + c*((y-yo)**2)))
        return g.ravel()

    def _gauss2dfit(a,mask):
        import numpy as np
        from scipy.optimize import curve_fit
        sz = np.shape(a)
        X,Y = np.meshgrid(np.arange(sz[1])-sz[1]//2,np.arange(sz[0])-sz[0]//2)
        c = np.unravel_index(a.argmax(),sz)
        Xf = X[mask>0]; Yf = Y[mask>0]; af = a[mask>0]

        # y = a[c[0],:]
        # x = X[c[0],:]
        stdx = .5 #np.sqrt(abs(sum(y * (x - sum(x*y)/sum(y))**2) / sum(y)))
        # y = a[:,c[1]]
        # x = Y[:,c[1]]
        stdy = .5 #np.sqrt(abs(sum(y * (x - sum(x*y)/sum(y))**2) / sum(y)))
        initial_guess = [np.median(a), np.max(a), stdx, stdy, c[1] - sz[1]//2, c[0] - sz[0]//2, 0]
        bounds = ([-1,-1,0,0,initial_guess[4]-1,initial_guess[5]-1,-180],
                  [1,1,initial_guess[2]*4,initial_guess[2]*4,initial_guess[4]+1,initial_guess[5]+1,180])

        popt, pcov = curve_fit(_g2d, (Xf, Yf), af.ravel(), p0=initial_guess,bounds=bounds)
        return np.reshape(_g2d((X,Y), *popt), sz), popt
    
    def _one_power(array):
        return array/np.sqrt((np.abs(array)**2).mean())

    if deriv:
        ref = _image_derivative(ref - np.mean(ref))
        im = _image_derivative(im - np.mean(im))
        
    shifts=np.zeros(2)
    FT1=fft.fftn(ref - np.mean(ref))
    FT2=fft.fftn(im - np.mean(im))
    ss=np.shape(ref)
    r=np.real(fft.ifftn(_one_power(FT1) * _one_power(FT2.conj())))
    r = fft.fftshift(r)
    rmax=np.max(r)
    ppp = np.unravel_index(np.argmax(r),ss)
    shifts = [ppp[0]-ss[0]//2,ppp[1]-ss[1]//2]
    if subpixel:
        dd = [d,75,100,30]
        for d1 in dd:
            try:
                if d1>0:
                    mask = circular_mask(ss[0],ss[1],[ppp[1],ppp[0]],d1)
                else:
                    mask = np.ones(ss,dtype=bool); d = ss[0]//2
                g, A = _gauss2dfit(r,mask)
                break
            except RuntimeError as e:
                print(f"Issue with gaussian fitting using mask with radius {d1}\nTrying new value...")
                if d1 == dd[-1]:
                    raise RuntimeError(e)
        shifts[0] = A[5]
        shifts[1] = A[4]
        del g
    del FT1, FT2
    return r, shifts

def remap(hrt_map, hmi_map, out_shape = (1024,1024), verbose = False):
    """reproject hmi map onto hrt with hrt pixel size and observer coordinates
    
    Parameters
    ----------
    hrt_map : sunpy.map.GenericMap
        hrt map
    hmi_map : sunpy.map.GenericMap
        hmi map
    out_shape : tuple
        shape of output map, default is (1024,1024) (default is only true near HRT = 0.5 au)
    verbose : bool
        if True, plot of the maps will be shown
    
    Returns
    -------
    hmi_map : sunpy.map.GenericMap
        reprojected hmi map
    """
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
    output, footprint = reproject_adaptive(hmi_origin, out_wcs, out_shape,kernel='Hann',boundary_mode='ignore')
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

def subregion_selection(ht,start_row,start_col,original_shape,dsmax = 512,edge = 20):
    intcrpix1 = int(round(ht['CRPIX1']))
    intcrpix2 = int(round(ht['CRPIX2']))
    ds = min(dsmax,
             intcrpix2-start_row-edge,
             intcrpix1-start_col-edge,
             original_shape[0]+start_row-intcrpix2-edge,
             original_shape[1]+start_col-intcrpix1-edge)
    sly = slice(intcrpix2-ds,intcrpix2+ds)
    slx = slice(intcrpix1-ds,intcrpix1+ds)
    
    return sly, slx

def downloadClosestHMI(ht,t_obs,jsoc_email,verbose=False,path=False,cad='45'):
    """
    Script to download the HMI m_45 or ic_45 cosest in time to the provided SO/PHI observation.
    TAI convention and light travel time are taken into consideration.
    
    Parameters
    ----------
    ht: astropy.io.fits.header.Header
        header of the SO/PHI observation
    t_obs: str, datetime.datetime
        observation time of the SO/PHI observation. A string can be provided, but it is expected to be isoformat
    jsoc_email: str
        email address to be used for JSOC connection
    verbose: bool
        if True, plot of the HMI map will be shown (DEFAULT: False)
    path: bool
        if True, the path of the cache directory and of the HMI dataset will return as output (DEFAULT: False)
    """
    
    import drms
    import sunpy, sunpy.map
    from astropy.constants import c
    
    if type(t_obs) == str:
        t_obs = datetime.datetime.fromisoformat(t_obs)
    dtai = datetime.timedelta(seconds=37) # datetime.timedelta(seconds=94)
    
    if type(cad) != str:
        cad = str(int(cad))
    if cad == '45':
        dcad = datetime.timedelta(seconds=35) # half HMI cadence (23) + margin
    elif cad == '720':
        dcad = datetime.timedelta(seconds=360+60) # half HMI cadence (23) + margin
    else:
        print('wrong HMI cadence, only 45 and 720 are accepted')
        return None
    
    dltt = datetime.timedelta(seconds=ht['EAR_TDEL']) # difference in light travel time S/C-Earth

    kwlist = ['T_REC','T_OBS','DATE-OBS','CADENCE','DSUN_OBS']
    
    client = drms.Client(email=jsoc_email, verbose=True) 

    lt = np.nan
    n = 0
    while np.isnan(lt):
        n += 2
        if ht['BTYPE'] == 'BLOS':
            keys = client.query('hmi.m_'+cad+'s['+(t_obs+dtai-dcad+dltt).strftime('%Y.%m.%d_%H:%M:%S')+'-'+
                            (t_obs+dtai+dcad+dltt).strftime('%Y.%m.%d_%H:%M:%S')+']',seg=None,key=kwlist,n=n)
        elif ht['BTYPE'] == 'VLOS':
            keys = client.query('hmi.v_'+cad+'s['+(t_obs+dtai-dcad+dltt).strftime('%Y.%m.%d_%H:%M:%S')+'-'+
                            (t_obs+dtai+dcad+dltt).strftime('%Y.%m.%d_%H:%M:%S')+']',seg=None,key=kwlist,n=n)
        else:
            keys = client.query('hmi.ic_'+cad+'s['+(t_obs+dtai-dcad+dltt).strftime('%Y.%m.%d_%H:%M:%S')+'-'+
                            (t_obs+dtai+dcad+dltt).strftime('%Y.%m.%d_%H:%M:%S')+']',seg=None,key=kwlist,n=n)
        keys = keys[keys['T_OBS'] != 'MISSING']
        if np.size(keys['T_OBS']) > 0:
            lt = (np.nanmean(keys['DSUN_OBS'])*u.m - ht['DSUN_OBS']*u.m)/c
        else:
            print('adding 60s margin')
            dcad += datetime.timedelta(seconds=60)
        
    dltt = datetime.timedelta(seconds=lt.value) # difference in light travel time S/C-SDO


    T_OBS = [(ind,np.abs((datetime.datetime.strptime(t,'%Y.%m.%d_%H:%M:%S_TAI') - dtai - dltt - t_obs).total_seconds())) for ind, t in zip(keys.index,keys['T_OBS'])]
    ind = T_OBS[np.argmin([t[1] for t in T_OBS])][0]

    if ht['BTYPE'] == 'BLOS':
        name_h = 'hmi.m_'+cad+'s['+keys['T_REC'][ind]+']{Magnetogram}'
    elif ht['BTYPE'] == 'VLOS':
        name_h = 'hmi.v_'+cad+'s['+keys['T_REC'][ind]+']{Dopplergram}'
    else:
        name_h = 'hmi.ic_'+cad+'s['+keys['T_REC'][ind]+']{Continuum}'

    if np.abs((datetime.datetime.strptime(keys['T_OBS'][ind],'%Y.%m.%d_%H:%M:%S_TAI') - dtai - dltt - t_obs).total_seconds()) > np.ceil(int(cad)/2):
        print('WARNING: Closer file exists but has not been found.')
        print(name_h)
        print('T_OBS:',datetime.datetime.strptime(keys['T_OBS'][ind],'%Y.%m.%d_%H:%M:%S_TAI') - dtai - dltt)
        print('DATE-AVG:',t_obs)
        print('')
    else:
        print('HMI T_OBS (corrected for TAI and Light travel time):',datetime.datetime.strptime(keys['T_OBS'][ind],'%Y.%m.%d_%H:%M:%S_TAI') - dtai - dltt)
        print('PHI DATE-AVG:',t_obs)
    s45 = client.export(name_h,protocol='fits')
    hmi_map = sunpy.map.Map(s45.urls.url[0],cache=False)
    cache_dir = sunpy.data.CACHE_DIR+'/'
    hmi_name = cache_dir + s45.urls.url[0].split("/")[-1]

    if verbose:
        hmi_map.peek()
    if path:
        return hmi_map, cache_dir, hmi_name
    else:
        return hmi_map


def WCS_correction(file_name,jsoc_email,dir_out='./',remapping = 'remap',undistortion = False, logpol=False, allDID=False,verbose=False, deriv = True, values_only = False, subregion = None, crota_manual_correction = 0.15):
    """This function saves new version of the fits file with updated WCS.
    It works by correlating HRT data on remapped HMI data. 
    This function exports the nearest HMI data from JSOC. [Not downloaded to out_dir]
    Not validated on limb data. 
    Not tested on data with different viewing angle.
    icnt, stokes or ilam files are expected as input.
    
    Version: 1.0.0 (July 2022)
             1.1.0 (August 2023)
                Updates on data handling and correlation. New remapping function based on coordinates reprojecting functions.

    Parameters
    ----------
    file_name: str
        path to the fits file
    jsoc_email: str
        email address to be used for JSOC connection
    dir_out: str
        path to the output directory, DEFAULT: './', if None no file will be saved
    remapping: str
        type of remapping procedure. 'remap' uses the reprojection algorithm by DeForest, 'ccd' uses a coordinate translation from HMI to HRT based on function in this file (not working yet). DEFAULT: 'remap'
    undistortion: bool
        if True, HRT will be undistorted (DEFAULT: False).
    logpol: bool
        (DEPRECATED) if True, log-polar transform applied until agnle smaller than a threshold (DEFAULT: False).
    allDID: bool
        if True, all the fits file with the same DID in the directory of the input file will be saved with the new WCS.
    verbose: bool
        if True, plot of the maps will be shown (DEFAULT: False)
    deriv: bool
        if True, correlation is computed using the derivative of the image (DEFAULT: True)
    values_only: bool
        if True, new fits will not be saved (DEFAULT: False).
    subregion: tuple, None
        if None, automatic subregion. Accepted values are only tuples of slices (sly,slx)
    crota_manual_correction: float
        manual change to HRT CROTA value (deg). The value is added to the original one (DEFAULT: 0.15)
    Returns
    -------
    ht: astropy.io.fits.header.Header
        new header for hrt
    """
    import sunpy, imreg_dft
    import sunpy.map
    # from reproject import reproject_interp, reproject_adaptive
    # from sunpy.coordinates import get_body_heliographic_stonyhurst
    
    from sunpy.coordinates import frames
    import warnings, sunpy
    warnings.filterwarnings("ignore", category=sunpy.util.SunpyMetadataWarning)

    
    # print('This is a preliminary procedure')
    # print('It has been optimized on raw, continuum and blos data')
    # print('This script is based on sunpy routines and examples')
    
    hdr_phi = fits.open(file_name)
    phi = hdr_phi[0].data; h_phi = hdr_phi[0].header
    start_row = int(h_phi['PXBEG2']-1)
    start_col = int(h_phi['PXBEG1']-1)
    _,_,_,cpos = fits_get_sampling(file_name)
    
    h_phi = rotate_header(h_phi.copy(),-crota_manual_correction, center=center_coord(h_phi))

    if phi.ndim == 3:
        phi = phi[cpos*4]
    elif phi.ndim == 4:
        if phi.shape[0] == 6:
            phi = phi[cpos,0]            
        else:
            phi = phi[:,:,0,cpos]
    original_shape = phi.shape
    
    if phi.shape[0] == 2048:
        if undistortion:
            und_phi = und(phi)
            h_phi['CRPIX1'],h_phi['CRPIX2'] = Inv2(1016,982,h_phi['CRPIX1'],h_phi['CRPIX2'],8e-9)
        else:
            und_phi = phi
        phi_map = sunpy.map.Map((und_phi,h_phi))
    else:
        phi = np.pad(phi,[(start_row,2048-(start_row+phi.shape[0])),(start_col,2048-(start_col+phi.shape[1]))])
        h_phi['NAXIS1'] = 2048; h_phi['NAXIS2'] = 2048
        h_phi['PXBEG1'] = 1; h_phi['PXBEG2'] = 1; h_phi['PXEND1'] = 2048; h_phi['PXEND2'] = 2048; 
        h_phi['CRPIX1'] += start_col; h_phi['CRPIX2'] += start_row
        if undistortion:
            und_phi = und(phi)
            h_phi['CRPIX1'],h_phi['CRPIX2'] = Inv2(1016,982,h_phi['CRPIX1'],h_phi['CRPIX2'],8e-9)
        else:
            und_phi = phi
        phi_map = sunpy.map.Map((und_phi,h_phi))
    
    if verbose:
        phi_map.peek()
    
    ht = phi_map.fits_header
    t0 = hdr_phi[10].data['EXP_START_TIME']
    if t0.size > 24:
        t0 = t0[int(round(t0.size//24/2,0))::t0.size//24]
    #             t0 = np.asarray([DT.datetime.fromisoformat(t0[i]) for i in range(len(t0))])
    t0 = [t0[i] for i in range(len(t0))]
    if cpos == 5:
        t0 = t0[20]
    else:
        t0 = t0[0]
        
    t_obs = datetime.datetime.fromisoformat(t0)
    if ht['BTYPE'] == 'BLOS':
        t0 = ht['DATE-AVG']
        t_obs = datetime.datetime.fromisoformat(ht['DATE-AVG'])
    
    try:
        hmi_map, cache_dir, hmi_name = downloadClosestHMI(ht,t_obs,jsoc_email,verbose,True)
    except Exception as e:
        print("Issue with downloading HMI. The code stops here. Restults obtained so far will be saved. This was the error:")
        print(e)
        return h_phi['CROTA'], h_phi['CRPIX1'] - start_col, h_phi['CRPIX2'] - start_row, h_phi['CRVAL1'], h_phi['CRVAL2'], t0, False, None, None
        
    if verbose:
        hmi_map.peek()
    
    sly = slice(128*4,128*12)
    slx = slice(128*4,128*12)

    ht = h_phi.copy()
    ht['DATE-BEG'] = ht['DATE-AVG']; ht['DATE-OBS'] = ht['DATE-AVG']
    # ht['DATE-BEG'] = datetime.datetime.isoformat(datetime.datetime.fromisoformat(ht['DATE-BEG']) + dltt)
    # ht['DATE-OBS'] = datetime.datetime.isoformat(datetime.datetime.fromisoformat(ht['DATE-OBS']) + dltt)
    shift = [1,1]
    i = 0
    angle = 1
    match = True
    
    # hgsPHI = ccd2HGS(phi_map.fits_header)
    # hgsHMI = ccd2HGS(hmi_map.fits_header)
    # hmi_remap = hmi2phi(hmi_map,phi_map)

    try:
        while np.any(np.abs(shift)>5e-2):
            
            if subregion is not None:
                sly,slx = subregion
            else:
                sly, slx = subregion_selection(ht,start_row,start_col,original_shape,dsmax = 512)
                print('Subregion size:',sly.stop-sly.start)

            if remapping == 'remap':
                phi_map = sunpy.map.Map((und_phi,ht))

                bl = phi_map.pixel_to_world(slx.start*u.pix, sly.start*u.pix)
                tr = phi_map.pixel_to_world((slx.stop-1)*u.pix, (sly.stop-1)*u.pix)
                phi_submap = phi_map.submap(np.asarray([slx.start, sly.start])*u.pix,
                                    top_right=np.asarray([slx.stop-1, sly.stop-1])*u.pix)

                hmi_remap = remap(phi_map, hmi_map, out_shape = (2048,2048), verbose=False)

                # necessary when the FoV is close to the HMI limb
                temp0 = hmi_remap.data.copy(); temp0[np.isinf(temp0)] = 0; temp0[np.isnan(temp0)] = 0
                hmi_remap = sunpy.map.Map((temp0, hmi_remap.fits_header))

                top_right = hmi_remap.world_to_pixel(tr)
                bottom_left = hmi_remap.world_to_pixel(bl)
                tr_hmi_map = np.array([top_right.x.value,top_right.y.value])
                bl_hmi_map = np.array([bottom_left.x.value,bottom_left.y.value])
                slyhmi = slice(int(round(bl_hmi_map[1])),int(round(tr_hmi_map[1]))+1)
                slxhmi = slice(int(round(bl_hmi_map[0])),int(round(tr_hmi_map[0]))+1)
                hmi_map_wcs = hmi_remap.submap(bl_hmi_map*u.pix,top_right=tr_hmi_map*u.pix)

            
            elif remapping == 'ccd':
                phi_map = sunpy.map.Map((und_phi,ht))
                # hgsPHI = ccd2HGS(phi_map.fits_header)
                
                hmi_remap = sunpy.map.Map((hmi2phi(hmi_map,phi_map),ht))
                
                phi_submap = phi_map.submap(np.asarray([slx.start, sly.start])*u.pix,
                                    top_right=np.asarray([slx.stop-1, sly.stop-1])*u.pix)
                slyhmi = sly
                slxhmi = slx
                hmi_map_wcs = hmi_remap.submap(np.asarray([slx.start, sly.start])*u.pix,
                                    top_right=np.asarray([slx.stop-1, sly.stop-1])*u.pix)

            ref = phi_submap.data.copy()
            temp = hmi_map_wcs.data.copy(); temp[np.isinf(temp)] = 0; temp[np.isnan(temp)] = 0
            s = [1,1]
            shift = [0,0]
            it = 0

            if abs(angle)>1e-2 and logpol:
                r = imreg_dft.similarity(ref.copy(),temp.copy(),numiter=3,constraints=dict(scale=(1,0)))
                shift = r['tvec']; angle = r['angle']
                hmi_map_shift = imreg_dft.transform_img(hmi_map_wcs.data,scale=1,angle=angle,tvec=shift)
                hmi_map_shift = sunpy.map.Map((hmi_map_shift,hmi_map_wcs.fits_header))
                print('logpol transform shift (x,y):',round(shift[1],2),round(shift[0],2),'angle (deg):',round(angle,3))

                ht = translate_header(rotate_header(ht.copy(),-angle),shift,mode='crval')

            else:
                while np.any(np.abs(s)>1e-2) and it<10:
                    if it == 0 and ~logpol:
                        _,s = image_register(ref,temp,False,deriv)
                        if np.any(np.abs(s)==0):
                            _,s = image_register(ref,temp,True,deriv)
                    else:
                        _,s = image_register(ref,temp,True,deriv)
                        # sr, sc, _ = SPG_shifts_FFT(np.asarray([ref,temp])); s = [sr[1],sc[1]]
                    shift = [shift[0]+s[0],shift[1]+s[1]]
                    # temp = fft_shift(hmi_map_wcs.data.copy(), shift); temp[np.isinf(temp)] = 0; temp[np.isnan(temp)] = 0
                    temp = fft_shift(hmi_remap.data.copy(), shift)[slyhmi,slxhmi]; temp[np.isinf(temp)] = 0; temp[np.isnan(temp)] = 0
                    it += 1
                    
                hmi_map_shift = sunpy.map.Map((temp,hmi_map_wcs.fits_header))

                ht = translate_header(ht.copy(),np.asarray(shift),mode='crval')
                print(it,'iterations shift (x,y):',round(shift[1],2),round(shift[0],2))

            i+=1
            if i == 10:
                print('Maximum iterations reached:',i)
                match = False
                break
    except Exception as e:
        printc("Issue with co-alignment. The code stops here. Restults obtained so far will be saved. This was the error:",bcolors.FAIL)
        printc(e,bcolors.FAIL)
        print(f'{it} iterations shift (x,y): {round(shift[1],2),round(shift[0],2)}',bcolors.FAIL)
        return ht['CROTA'], ht['CRPIX1'] - start_col, ht['CRPIX2'] - start_row, ht['CRVAL1'], ht['CRVAL2'], t0, False, phi_map, hmi_remap
    
    if remapping == 'remap':
        phi_map = sunpy.map.Map((und_phi,ht))

        # bl = phi_map.pixel_to_world(slx.start*u.pix, sly.start*u.pix)
        # tr = phi_map.pixel_to_world((slx.stop-1)*u.pix, (sly.stop-1)*u.pix)
        # phi_submap = phi_map.submap(np.asarray([slx.start, sly.start])*u.pix,
        #                     top_right=np.asarray([slx.stop-1, sly.stop-1])*u.pix)

        hmi_remap = remap(phi_map, hmi_map, out_shape = (2048,2048), verbose=False)
        
        ht['DATE-BEG'] = h_phi['DATE-BEG']
        ht['DATE-OBS'] = h_phi['DATE-OBS']
        phi_map = sunpy.map.Map((und_phi,ht))

        # top_right = hmi_remap.world_to_pixel(tr)
        # bottom_left = hmi_remap.world_to_pixel(bl)
        # tr_hmi_map = np.array([top_right.x.value,top_right.y.value])
        # bl_hmi_map = np.array([bottom_left.x.value,bottom_left.y.value])
        # hmi_map_wcs = hmi_remap.submap(bl_hmi_map*u.pix,top_right=tr_hmi_map*u.pix)

    elif remapping == 'ccd':
        phi_map = sunpy.map.Map((und_phi,ht))
        hmi_remap = sunpy.map.Map((hmi2phi(hmi_map,phi_map),ht))
        ht['DATE-BEG'] = h_phi['DATE-BEG']
        ht['DATE-OBS'] = h_phi['DATE-OBS']
        phi_map = sunpy.map.Map((und_phi,ht))

        phi_submap = phi_map.submap(np.asarray([slx.start, sly.start])*u.pix,
                            top_right=np.asarray([slx.stop-1, sly.stop-1])*u.pix)
        hmi_map_wcs = hmi_remap.submap(np.asarray([slx.start, sly.start])*u.pix,
                            top_right=np.asarray([slx.stop-1, sly.stop-1])*u.pix)
    
    
    if verbose:
        if remapping == 'remap':
            bl = phi_map.pixel_to_world(slx.start*u.pix, sly.start*u.pix)
            tr = phi_map.pixel_to_world((slx.stop-1)*u.pix, (sly.stop-1)*u.pix)
            top_right = hmi_remap.world_to_pixel(tr)
            bottom_left = hmi_remap.world_to_pixel(bl)
            tr_hmi_map = np.array([top_right.x.value,top_right.y.value])
            bl_hmi_map = np.array([bottom_left.x.value,bottom_left.y.value])
            hmi_map_wcs = hmi_remap.submap(bl_hmi_map*u.pix,top_right=tr_hmi_map*u.pix)
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
    
    if os.path.isfile(hmi_name):
        os.remove(hmi_name)
        import sqlite3
        # creating file path
        dbfile = cache_dir+'cache.db'
        if os.path.isfile(dbfile):
            # Create a SQL connection to our SQLite database
            con = sqlite3.connect(dbfile)
            # creating cursor
            cur = con.cursor()
            removeItem = "DELETE FROM cache_storage WHERE file_path = \'"+hmi_name+"\'"
            cur.execute(removeItem)
            con.commit()
        print(hmi_name.split("/")[-1]+' deleted')
    
    if values_only:
        return ht['CROTA'], ht['CRPIX1'] - start_col, ht['CRPIX2'] - start_row, ht['CRVAL1'], ht['CRVAL2'], t0, match, phi_map, hmi_remap
    else:
        if dir_out is not None:
            if allDID:
                did = h_phi['PHIDATID']
                name = file_name.split('/')[-1]
                new_name = name.replace("_phi-",".WCS_phi-")
    
                directory = file_name[:-len(name)]
                file_n = os.listdir(directory)
                if type(did) != str:
                    did = str(did)
                did_n = [directory+i for i in file_n if did in i]
                l2_n = ['stokes','icnt','bmag','binc','bazi','vlos','blos']
                for n in l2_n:
                    f = [i for i in did_n if n in i][0]
                    name = f.split('/')[-1]
                    new_name = name.replace("_phi-",".WCS_phi-")
                    with fits.open(f) as h:
                        h[0].header['CROTA'] = ht['CROTA']
                        h[0].header['CRPIX1'] = ht['CRPIX1'] - start_col
                        h[0].header['CRPIX2'] = ht['CRPIX2'] - start_row
                        h[0].header['CRVAL1'] = ht['CRVAL1']
                        h[0].header['CRVAL2'] = ht['CRVAL2']
                        h[0].header['PC1_1'] = ht['PC1_1']
                        h[0].header['PC1_2'] = ht['PC1_2']
                        h[0].header['PC2_1'] = ht['PC2_1']
                        h[0].header['PC2_2'] = ht['PC2_2']
                        h[0].header['HISTORY'] = 'WCS corrected via HRT - HMI cross correlation'
                        h.writeto(dir_out+new_name, overwrite=True)        
            else:
                with fits.open(file_name) as h:
                    h[0].header['CROTA'] = ht['CROTA']
                    h[0].header['CRPIX1'] = ht['CRPIX1'] - start_col
                    h[0].header['CRPIX2'] = ht['CRPIX2'] - start_row
                    h[0].header['CRVAL1'] = ht['CRVAL1']
                    h[0].header['CRVAL2'] = ht['CRVAL2']
                    h[0].header['PC1_1'] = ht['PC1_1']
                    h[0].header['PC1_2'] = ht['PC1_2']
                    h[0].header['PC2_1'] = ht['PC2_1']
                    h[0].header['PC2_2'] = ht['PC2_2']
                    h[0].header['HISTORY'] = 'WCS corrected via HRT - HMI cross correlation '
                    h.writeto(dir_out+new_name, overwrite=True)
        return ht['CROTA'], ht['CRPIX1'] - start_col, ht['CRPIX2'] - start_row, ht['CRVAL1'], ht['CRVAL2'], t0, match, phi_map, hmi_remap
###############################################

def cavity_shifts(cavity_f, wave_axis,rows,cols,returnWL = True):
    """applies cavity shifts to the wave axis for use in RTE

    Parameters
    ----------
    cavity_f : str or array
        path to cavity map fits file or cavity array (already cropped)
    wave_axis : array
        wavelength axis
    rows : array
        rows of the pixels in the image, where the respective wavelength is shifted
    cols : array
        columns of the pixels in the image, where the respective wavelength is shifted

    Returns
    -------
    new_wave_axis[rows, cols]: array
        wavelength axis with the cavity shifts applied to the respective pixels
    """
    if isinstance(cavity_f,str):
        cavityMap, _ = load_fits(cavity_f) # cavity maps
        if cavityMap.ndim == 3:
            cavityWave = cavityMap[:,rows,cols].mean(axis=0)
        else:
            cavityWave = cavityMap[rows,cols]
    else:
        cavityMap = cavity_f
        if cavityMap.ndim == 3:
            cavityWave = cavityMap.mean(axis=0)
        else:
            cavityWave = cavityMap
        
    if returnWL:
        new_wave_axis = wave_axis[np.newaxis,np.newaxis] - cavityWave[...,np.newaxis]
        return new_wave_axis
    else:
        return cavityWave

def load_l2_stk(directory,did,version=None):
    import glob
    file_n = os.listdir(directory)
    key = 'stokes'
    if version is None:
        version = '*'
    datfile = glob.glob(os.path.join(directory, f'solo_L2_phi-hrt-{key}_*_{version}_{did}.fits.gz'))
    if not(datfile):
        print('No data found')
        return np.empty([]), np.empty([])
    else:
        if len(datfile) == 1:
            return fits.getdata(datfile[0],header=True)
        else:
            print('More than one file found:')
            print(datfile)
            print('Please specify the Version')
            return np.empty([]), np.empty([])

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
    
    if coords is not None: # coords must be (N,3), N is the number of pixels, the second axis is (x,y,1)
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
    sun_dist_m=hdr['DSUN_OBS'] #Earth
    rsun = hdr['RSUN_REF'] # m
    if 'PXBEG1' in hdr:
        pxbeg1 = hdr['PXBEG1']
    else:
        pxbeg1 = 1
    if 'PXBEG2' in hdr:
        pxbeg2 = hdr['PXBEG2']
    else:
        pxbeg2 = 1
    if 'PXEND1' in hdr:
        pxend1 = hdr['PXEND1']
    else:
        pxend1 = hdr['NAXIS1']
    if 'PXEND2' in hdr:
        pxend2 = hdr['PXEND2']
    else:
        pxend2 = hdr['NAXIS2']
    crval1 = hdr['CRVAL1']
    crval2 = hdr['CRVAL2']
    crpix1 = hdr['CRPIX1']
    crpix2 = hdr['CRPIX2']
    if 'PC1_1' in hdr:
        PC1_1 = hdr['PC1_1']
        PC1_2 = hdr['PC1_2']
        PC2_1 = hdr['PC2_1']
        PC2_2 = hdr['PC2_2']
    else:
        crota = hdr['CROTA*'][0]
        PC1_1 = np.cos(crota*np.pi/180)
        PC1_2 = -np.sin(crota*np.pi/180)
        PC2_1 = np.sin(crota*np.pi/180)
        PC2_2 = np.cos(crota*np.pi/180)
    
    if coords is None:
        X,Y = np.meshgrid(np.arange(1,pxend1-pxbeg1+2),np.arange(1,pxend2-pxbeg2+2))
    else:
        X = coords[:,0]+1
        Y = coords[:,1]+1
    
    HPC1 = crval1 + pxsc*(PC1_1*(X-crpix1)+PC1_2*(Y-crpix2))
    HPC2 = crval2 + pxsc*(PC2_1*(X-crpix1)+PC2_2*(Y-crpix2))

    th = np.arctan(np.sqrt(np.cos(HPC2/3600*np.pi/180)**2*np.sin(HPC1/3600*np.pi/180)**2+np.sin(HPC2/3600*np.pi/180)**2/
                          (np.cos(HPC2/3600*np.pi/180)*np.cos(HPC1/3600*np.pi/180))))
    b = np.arcsin(sun_dist_m/rsun*np.sin(th)) - th
    d = (sun_dist_m-rsun*np.cos(b))/np.cos(th)
    
    return HPC1, HPC2, d

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
    
    sun_dist_m=hdr['DSUN_OBS']
    
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
    
    if 'HGLT_OBS' in hdr:
        B0 = hdr['HGLT_OBS']*np.pi/180
    else:
        B0 = hdr['CRLT_OBS']*np.pi/180
    if 'HGLN_OBS' in hdr:
        PHI0 = hdr['HGLN_OBS']*np.pi/180
    else:
        L0time = datetime.datetime.fromisoformat(hdr['DATE-OBS'])
        PHI0 = hdr['CRLN_OBS'] - sunpy.coordinates.sun.L0(L0time).value
    
    HCCx, HCCy, HCCz = ccd2HCC(file,coords)
        
    r = np.sqrt(HCCx**2 + HCCy**2 + HCCz**2)
    THETA = np.arcsin((HCCy*np.cos(B0) + HCCz*np.sin(B0))/r)*180/np.pi
    PHI = PHI0*180/np.pi + np.arctan(HCCx/(HCCz*np.cos(B0) - HCCy*np.sin(B0)))*180/np.pi
    
    # THETA == LAT; PHI == LON
    return r, THETA, PHI

def HPC2ccd(file, coords):
    """
    from Helioprojective Cartesian to CCD frame
    
    Input
    file: file_name, sunpy map or header
    coords: Helioprojective Cartesian coordinates (sse output of ccd2HPC)
            
    Output
    x,y (in pixels)
    
    """
    import sunpy.map
    if type(file) == str:
        hdr = fits.getheader(file)
    elif type(file) == sunpy.map.mapbase.GenericMap or type(file) == sunpy.map.sources.sdo.HMIMap:
        hdr = file.fits_header
    elif type(file) == fits.header.Header:
        hdr = file
            
    try:
        HPC1, HPC2 = coords
    except:
        HPC1, HPC2, d = coords
    assert (np.shape(HPC1) == np.shape(HPC2))

    if type(HPC1) == list or type(HPC2) == tuple:
        HPC1 = np.asarray(HPC1)
        HPC2 = np.asarray(HPC2)
        
    pxsc = hdr['CDELT1']
    crval1 = hdr['CRVAL1']
    crval2 = hdr['CRVAL2']
    crpix1 = hdr['CRPIX1']
    crpix2 = hdr['CRPIX2']
    if 'PC1_1' in hdr:
        PC1_1 = hdr['PC1_1']
        PC1_2 = hdr['PC1_2']
        PC2_1 = hdr['PC2_1']
        PC2_2 = hdr['PC2_2']
    else:
        crota = hdr['CROTA*'][0]
        PC1_1 = np.cos(crota*np.pi/180)
        PC1_2 = -np.sin(crota*np.pi/180)
        PC2_1 = np.sin(crota*np.pi/180)
        PC2_2 = np.cos(crota*np.pi/180)
    
    x = crpix1 + 1/pxsc * (PC1_1*(HPC1-crval1) + PC2_1*(HPC2-crval2)) - 1
    y = crpix2 + 1/pxsc * (PC1_2*(HPC1-crval1) + PC2_2*(HPC2-crval2)) - 1
        
    return x,y

def HCC2ccd(file,coords):
    """
    from Heliocentric Cartesian to CCD frame
    
    Input
    file: file_name, sunpy map or header
    coords: Heliocentric Cartesian coordinates (sse output of ccd2HCC)
            
    Output
    x,y (in pixels)
    
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
    
    HCCx,HCCy,HCCz = coords
    assert (np.shape(HCCx) == np.shape(HCCy) and np.shape(HCCx) == np.shape(HCCz))

    if type(HCCx) == list or type(HCCx) == tuple:
        HCCx = np.asarray(HCCx)
        HCCy = np.asarray(HCCy)
        HCCz = np.asarray(HCCz)

    sun_dist_m=hdr['DSUN_OBS']
    
    HPCd = np.sqrt(HCCx**2+HCCy**2+(sun_dist_m-HCCz)**2)
    HPCx = np.arctan(HCCx/(sun_dist_m-HCCz))*180/np.pi*3600
    HPCy = np.arcsin(HCCy/HPCd)*180/np.pi*3600
    
    
    x,y = HPC2ccd(hdr,(HPCx,HPCy,HPCd))
    
    return x,y

def HGS2ccd(file, coords):
    """
    from Heliographic Stonyhurst to CCD frame
    
    Input
    file: file_name, sunpy map or header
    coords: Heliographic Stonyhurst coordinates (sse output of ccd2HGS)
            
    Output
    x,y (in pixels)
    
    """
    
    import sunpy.map
    if type(file) == str:
        hdr = fits.getheader(file)
    elif type(file) == sunpy.map.mapbase.GenericMap or type(file) == sunpy.map.sources.sdo.HMIMap:
        hdr = file.fits_header
    elif type(file) == fits.header.Header:
        hdr = file
    
    r, THETA, PHI = coords
    assert (np.shape(r) == np.shape(THETA) and np.shape(r) == np.shape(PHI))

    if type(r) == list or type(r) == tuple:
        r = np.asarray(r)
        THETA = np.asarray(THETA)
        PHI = np.asarray(PHI)

    if 'HGLT_OBS' in hdr:
        B0 = hdr['HGLT_OBS']*np.pi/180
    else:
        B0 = hdr['CRLT_OBS']*np.pi/180
    if 'HGLN_OBS' in hdr:
        PHI0 = hdr['HGLN_OBS']*np.pi/180
    else:
        L0time = datetime.datetime.fromisoformat(hdr['DATE-OBS'])
        PHI0 = hdr['CRLN_OBS'] - sunpy.coordinates.sun.L0(L0time).value
    
    THETA = THETA * np.pi/180
    PHI = PHI * np.pi/180
    
    HCCx = r * np.cos(THETA) * np.sin(PHI-PHI0)
    HCCy = r * (np.sin(THETA)*np.cos(B0) - np.cos(THETA)*np.cos(PHI-PHI0)*np.sin(B0))
    HCCz = r * (np.sin(THETA)*np.sin(B0) + np.cos(THETA)*np.cos(PHI-PHI0)*np.cos(B0))
    
    
    x, y = HCC2ccd(hdr,(HCCx,HCCy,HCCz))
    
    return x,y

def hmi2phi(hmi_map, phi_map,  order=1):
    
    from scipy.ndimage import map_coordinates

    hgsPHI = ccd2HGS(phi_map.fits_header)
    new_coord = HGS2ccd(hmi_map.fits_header,hgsPHI)
    hmi_remap = map_coordinates(hmi_map.data,[new_coord[1],new_coord[0]],order=order)
    
    return hmi_remap

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

def dataset_colorbar(ax,im,location="top",label=None,xy=None,fontsize=9):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(location, size="5%", pad=0.05)
    if location == 'top' or location == 'bottom':
        orientation = 'horizontal'
    else:
        orientation = 'vertical'
    if xy is None:
        cb = plt.colorbar(im, orientation=orientation, cax=cax, label=label)
    else:
        cb = plt.colorbar(im, orientation=orientation, cax=cax)
        cax.set_title(label,fontsize=fontsize,x=xy[0],y=xy[1])
        
    if location == 'top' or location == 'bottom':
        cax.xaxis.set_ticks_position(location)    
    else:
        cax.yaxis.set_ticks_position(location)
    
    cax.tick_params(labelsize=8)

    return cax

def show_image_array(arr, hdr, grayscales, row_labels=None, 
                     column_labels=None, titles=None,
                     fig_title=None, ax_order=None):
    """Show array images of shape (rows, columns, X, Y).

    Parameters
    ----------
    
    """

    import itertools

    panel_sz = 3.3
    rows, columns = arr.shape[0:2]
    _, _, _, sly, slx = limb_side_finder(arr[0,0],hdr,False)
    # fig_width, fig_height = plt.gcf().get_size_inches()
    # print(fig_width, fig_height)    

    fig, axs = plt.subplots(
        rows, columns,
        sharex=True, sharey=True,
        subplot_kw=dict(aspect=1),
        figsize=(columns * panel_sz, rows * panel_sz),
        layout='constrained',
        # gridspec_kw={'hspace': 0, 'wspace': 0},
        # **kwargs
    )

    # Sort plots
    if ax_order is not None:
        axs = axs.flatten()
        axs = [axs[i] for i in ax_order]

    axs = np.reshape(axs, (rows, columns))

    # plt.subplots_adjust(top=0.92)

    for i, j in itertools.product(range(rows), range(columns)):
        im = arr[i, j, :, :]

        mean = im[sly,slx].mean()

        ax = axs[i, j]
        
        # Print color scale range
        im = axs[i, j].imshow(im, cmap='gray', clim=grayscales[i],interpolation=None)
        if i == 0:
            ax.text(0.05, 0.94, f'{grayscales[i][0]:.1f} - {grayscales[i][1]:.1f}', transform=ax.transAxes, color='white')
        else:
            ax.text(0.05, 0.94, f'{grayscales[i][0]:.3f} - {grayscales[i][1]:.3f}', transform=ax.transAxes, color='white')
        # ax.text(0.05, 0.94, f'{grayscales[i][0]:.1f} - {grayscales[i][1]:.1f}', transform=ax.transAxes, color='white')
        # else:
        #     im = axs[i, j].imshow(im, cmap='gray', vmin=mean+grayscales[i][0], vmax=mean+grayscales[i][1],interpolation=None)
        #     ax.text(0.05, 0.94, f'{mean:.4f} $\pm$ {grayscales[i][1]:.3f}', transform=ax.transAxes, color='white')

    # Set row labels
    if row_labels is not None:
        for row_label, ax in zip(row_labels, axs[:, 0]):
            ax.set_ylabel(row_label)

    # Set column labels
    if column_labels is not None:
        for column_label, ax in zip(column_labels, axs[0, :]):
            ax.set_title(column_label)

    # Set panel titles
    if titles is not None:
        for title, ax in zip(titles, axs.flatten()):
            ax.set_title(title, fontsize=12)

    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=14)

    return fig

def plot_l2_pdf(path,did,version=None):
    """
    Generate standard plots for pipeline results
    """

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import glob
    from matplotlib.colors import LinearSegmentedColormap
    # import os
    # import re
    # from argparse import ArgumentParser
    # import numpy as np
    
    # import sys

    import matplotlib as mpl
    mpl.rc_file_defaults()
    mpl.rcParams['image.origin'] = 'lower'
    # plt.rcParams['figure.dpi'] = 300
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['image.cmap'] = 'gist_heat'
    plt.rcParams['image.interpolation'] = 'none'

    import cmasher as cmr

    pipe_dir = os.path.realpath(__file__)
    pipe_dir = pipe_dir.split('src/')[0]
    hmimag = LinearSegmentedColormap.from_list('hmimag', np.loadtxt(pipe_dir+'csv/hmimag.csv',delimiter=','), N=256)

    file_n = os.listdir(path)
    if type(did) != str:
        did = str(did).rjust(10,'0')
    if version is None:
        version = '*'

    # pdf_name = re.search('.*/(.*)$', args.path).groups()[0]
    
    # # -----------------------------------------------------------------------------
    # # Loading data
    # # -----------------------------------------------------------------------------

    dat = {}
    keys = ['icnt', 'vlos', 'blos', 'binc', 'bmag', 'bazi', 'chi2']
    missingKeys = []
    for key in keys:
        try:
            datfile = glob.glob(os.path.join(path, f'solo_L2_phi-hrt-{key}_*_{version}_{did}.fits*'))[0]
            dat[key], h = load_fits(datfile)
        except:
            print('Missing '+key)
            missingKeys += [key]
    
    for key in missingKeys:
        k = next((e for e in keys if e!=key), None)
        dat[key] = np.zeros(dat[k].shape)
    
    _, _, _, sly, slx = limb_side_finder(dat['icnt'], h, False)
    

    datfile = glob.glob(os.path.join(path, f'solo_L2_phi-hrt-stokes_*_{version}_{did}.fits*'))[0]
    stk, h = load_fits(datfile)
    if stk.shape[0] != 6 or stk.shape[1] != 4:
        stk = np.einsum('yxpl->lpyx',stk)
    
    wavelengths,_,_,cpos = fits_get_sampling(datfile)

    if version == '*':
        version = 'V'+h['VERSION']
    save_file = os.path.join(path, f'{did}_{version}.pdf')
    p = PdfPages(save_file)

    # # -----------------------------------------------------------------------------
    # # Plot inversion results
    # # -----------------------------------------------------------------------------

    # Plot parameters
    panel_sz = 4
    dpi = 300
    rows = 2
    columns = 3

    fig, axs = plt.subplots(
        rows, columns,
        sharey=True,
        subplot_kw={'aspect': 1},
        figsize=(columns * panel_sz + 2, rows * panel_sz), dpi=dpi,
        layout='constrained')

    # Continuum intensity
    ax = axs[0, 0]
    im = ax.imshow(dat['icnt'], cmap='gist_heat', vmin=0.2, vmax=1.2,interpolation='none')
    dataset_colorbar(ax,im,"right")
    ax.set_title('Continuum intensity')

    # vLOS
    ax = axs[0, 1]
    shape = dat['vlos'].shape
    avg = dat['vlos'][int(shape[0]//4):-int(shape[0]//4),int(shape[1]//4):-int(shape[1]//4)].mean()
    im = ax.imshow(dat['vlos'], cmap=cmr.fusion.reversed(), vmin=-2+avg, vmax=2+avg,interpolation='none')
    dataset_colorbar(ax,im,"right", label='km/s')
    ax.set_title('LoS velocity')

    # BLOS
    ax = axs[0, 2]
    im = ax.imshow(dat['blos'], cmap=hmimag, vmin=-1500, vmax=1500,interpolation='none')
    dataset_colorbar(ax,im,"right", label='G')
    ax.set_title('LoS magnetic field')

    # B inclination
    ax = axs[1, 0]
    im = ax.imshow(dat['binc'], cmap=cmr.fusion, vmin=0, vmax=180,interpolation='none')
    dataset_colorbar(ax,im,"right", label='°')
    ax.set_title('Magn. field inclination')

    # B
    ax = axs[1, 1]
    im = ax.imshow(dat['bmag'], cmap='gnuplot_r', vmin=0, vmax=1000,interpolation='none')
    dataset_colorbar(ax,im,"right", label='G')
    ax.set_title('Magn. field strength')

    # B azimuth
    ax = axs[1, 2]
    im = ax.imshow(dat['bazi'], cmap='hsv', vmin=0, vmax=180,interpolation='none')
    dataset_colorbar(ax,im,"right", label='°')
    ax.set_title('Magn. field azimuth')

    # Figure title
    timestp = h['FILENAME'].split('_')[-3]
    fig.suptitle(os.path.join(path, f'solo_L2_phi-hrt-*_{timestp}_{version}_{did}.fits'), fontsize=12)

    fig.savefig(p, format='pdf')
    plt.close(fig)

    panel_sz = 4
    dpi = 300
    rows = 2
    columns = 3

    fig, axs = plt.subplots(
        rows, columns,
        subplot_kw={'aspect': 1},
        figsize=(columns * panel_sz + 2, rows * panel_sz), dpi=dpi,
        layout='constrained')

    # Chisq
    ax = axs[0,0]
    im = ax.imshow(dat['chi2'], cmap='turbo', vmin=0, vmax=100,interpolation='none')
    dataset_colorbar(ax,im,"right")
    ax.set_title('$\chi^2$')


    # Blos Noise
    values = dat['blos'][sly,slx]

    ax = axs[0,1]
    hi = ax.hist(values.flatten(), bins=np.linspace(-2e2,2e2,200),)
    tmp = [0,0]
    tmp[0] = hi[0].astype('float64')
    tmp[1] = hi[1].astype('float64')

    #guassian fit + label
    pp = gaussian_fit(tmp, show = False)    
    xx=hi[1][:-1] + (hi[1][1]-hi[1][0])/2
    lbl = f'{pp[1]:.2e} $\pm$ {pp[2]:.2e} G'


    ax.plot(xx,gaus(xx,*pp),'r--', label=lbl)
    try:
        p_iter, hi_iter = iter_noise(values,[1.,0.,10.],eps=1e-4); p_iter[0] = pp[0]
        ax.plot(xx,gaus(xx,*p_iter),'g-.', label= f"Iter Fit: {p_iter[1]:.2e} $\pm$ {p_iter[2]:.2e} G")
        # ax[1].scatter(0,0, color = 'white', s = 0, label = lbl) #also display the original fit in legend
    except:
        print("Iterative Gauss Fit failed")
    ax.set_aspect('auto')
    ax.legend()
    ax.set_title(f"LoS magnetic field NSR")

    # Blos Transverse
    values = (dat['bmag']*np.sin(dat['binc']*np.pi/180))[sly,slx]

    ax = axs[0,2]
    hi = ax.hist(values.flatten(), bins=np.linspace(0,10e2,200))
    tmp = [0,0]
    tmp[0] = hi[0].astype('float64')
    tmp[1] = hi[1].astype('float64')

    #guassian fit + label
    pp = gaussian_fit(tmp, show = False)    
    xx=hi[1][:-1] + (hi[1][1]-hi[1][0])/2
    lbl = f'{pp[1]:.2e} $\pm$ {pp[2]:.2e} G'


    ax.plot(xx,gaus(xx,*pp),'r--', label=lbl)
    try:
        p_iter, hi_iter = iter_noise(values,[1.,0.,1000.],eps=1e-4); p_iter[0] = pp[0]
        ax.plot(xx,gaus(xx,*p_iter),'g-.', label= f"Iter Fit: {p_iter[1]:.2e} $\pm$ {p_iter[2]:.2e} G")
        # ax[1].scatter(0,0, color = 'white', s = 0, label = lbl) #also display the original fit in legend
    except:
        print("Iterative Gauss Fit failed")
    ax.set_aspect('auto')
    ax.legend()
    ax.set_title(f"Transverse magnetic field NSR")

    # Stokes Q Noise
    values = stk[cpos,1,sly,slx]

    ax = axs[1,0]
    hi = ax.hist(values.flatten(), bins=np.linspace(-1e-2,1e-2,200),)
    tmp = [0,0]
    tmp[0] = hi[0].astype('float64')
    tmp[1] = hi[1].astype('float64')

    #guassian fit + label
    pp = gaussian_fit(tmp, show = False)    
    xx=hi[1][:-1] + (hi[1][1]-hi[1][0])/2
    lbl = f'{pp[1]:.2e} $\pm$ {pp[2]:.2e}'


    ax.plot(xx,gaus(xx,*pp),'r--', label=lbl)
    try:
        p_iter, hi_iter = iter_noise(values,[1.,0.,.1],eps=1e-6); p_iter[0] = pp[0]
        ax.plot(xx,gaus(xx,*p_iter),'g-.', label= f"Iter Fit: {p_iter[1]:.2e} $\pm$ {p_iter[2]:.2e}")
        # ax[1].scatter(0,0, color = 'white', s = 0, label = lbl) #also display the original fit in legend
    except:
        print("Iterative Gauss Fit failed")
    ax.set_aspect('auto')
    ax.legend()
    ax.set_title(f"Stokes Q NSR")

    # Stokes U Noise
    values = stk[cpos,2,sly,slx]

    ax = axs[1,1]
    hi = ax.hist(values.flatten(), bins=np.linspace(-1e-2,1e-2,200),)
    tmp = [0,0]
    tmp[0] = hi[0].astype('float64')
    tmp[1] = hi[1].astype('float64')

    #guassian fit + label
    pp = gaussian_fit(tmp, show = False)    
    xx=hi[1][:-1] + (hi[1][1]-hi[1][0])/2
    lbl = f'{pp[1]:.2e} $\pm$ {pp[2]:.2e}'


    ax.plot(xx,gaus(xx,*pp),'r--', label=lbl)
    try:
        p_iter, hi_iter = iter_noise(values,[1.,0.,.1],eps=1e-6); p_iter[0] = pp[0]
        ax.plot(xx,gaus(xx,*p_iter),'g-.', label= f"Iter Fit: {p_iter[1]:.2e} $\pm$ {p_iter[2]:.2e}")
        # ax[1].scatter(0,0, color = 'white', s = 0, label = lbl) #also display the original fit in legend
    except:
        print("Iterative Gauss Fit failed")
    ax.set_aspect('auto')
    ax.legend()
    ax.set_title(f"Stokes U NSR")

    # Stokes V Noise
    values = stk[cpos,3,sly,slx]

    ax = axs[1,2]
    hi = ax.hist(values.flatten(), bins=np.linspace(-1e-2,1e-2,200),)
    tmp = [0,0]
    tmp[0] = hi[0].astype('float64')
    tmp[1] = hi[1].astype('float64')

    #guassian fit + label
    pp = gaussian_fit(tmp, show = False)    
    xx=hi[1][:-1] + (hi[1][1]-hi[1][0])/2
    lbl = f'{pp[1]:.2e} $\pm$ {pp[2]:.2e}'


    ax.plot(xx,gaus(xx,*pp),'r--', label=lbl)
    try:
        p_iter, hi_iter = iter_noise(values,[1.,0.,.1],eps=1e-6); p_iter[0] = pp[0]
        ax.plot(xx,gaus(xx,*p_iter),'g-.', label= f"Iter Fit: {p_iter[1]:.2e} $\pm$ {p_iter[2]:.2e}")
        # ax[1].scatter(0,0, color = 'white', s = 0, label = lbl) #also display the original fit in legend
    except:
        print("Iterative Gauss Fit failed")
    ax.set_aspect('auto')
    ax.legend()
    ax.set_title(f"Stokes V NSR")

    fig.savefig(p, format='pdf')
    plt.close(fig)

    # # -----------------------------------------------------------------------------
    # # Plot Stokes images
    # # -----------------------------------------------------------------------------

    dat = np.transpose(stk, (1, 0, 2, 3))  # re-arrange Stokes and wavelength axes 

    grayscales = [(.3,1.2)] + [(-3e-3,3e-3)]*3 # [1] + [0.01] * 3  # I, Q, U, V
    row_labels = ['I', 'Q', 'U', 'V']
    column_labels = ['{:.3f} nm'.format(wave) for wave in wavelengths]
    title = os.path.basename(datfile) 

    fig = show_image_array(
        dat, h, grayscales, row_labels=row_labels,
        column_labels=column_labels, fig_title=title)

    fig.savefig(p, format='pdf')
    plt.close(fig)
    p.close()

