import numpy as np
from scipy.ndimage import gaussian_filter
from sophi_hrt_pipe.utils import *
import os
import time
import cv2

def setup_header(hdr_arr):
    """Add calibration keywords to header

    Parameters
    ----------
    hdr_arr: header
        Array containing header of each file to be written

    Returns
    -------
    hdr_arr
        Updated header array
    """
    k = ['CAL_FLAT','CAL_FNUM','CAL_USH','SIGM_USH','CAL_TEMP',
    'CAL_PRE','CAL_GHST','CAL_PREG','CAL_REAL',
    'CAL_CRT0','CAL_CRT1','CAL_CRT2','CAL_CRT3','CAL_CRT4','CAL_CRT5',
    'CAL_CRT6','CAL_CRT7','CAL_CRT8','CAL_CRT9',
    'CAL_WREG','CAL_NORM','CAL_FRIN','CAL_PSF','CAL_ZER','CAL_IPOL',
    'CAL_CAVM','CAL_SCIP','RTE_MOD','RTE_SW','RTE_ITER','VERS_CAL']

    v = [0,24,' ',' ','False',
    ' ','None ','None','NA',
    0,0,0,0,0,0,
    0,0,0,0,
    'None',' ','NA','NA','NA',' ',
    'None','None',' ',' ',4294967295, hdr_arr[0]['VERS_SW'][1:4]]

    c = ['Onboard calibrated for gain table','Unsharp masking correction','Number of flat field frames used','Sigma for unsharp masking [px]','Wavelengths correction for FG temperature',
    'Prefilter correction (DID/file)','Ghost correction (name + version of module)',
         'Polarimetric registration','Prealigment of images before demodulation',
    'cross-talk from I to Q (slope)','cross-talk from I to Q (offset)','cross-talk from I to U (slope)','cross-talk from I to U (offset)','cross-talk from I to V (slope)','cross-talk from I to V (offset)',
    'cross-talk from V to Q (slope)','cross-talk from V to Q (offset)','cross-talk from V to U (slope)','cross-talk from V to U (offset)','Wavelength Registration',
    'Normalization (normalization constant PROC_Ic)','Fringe correction (name + version of module)','PSF deconvolution','Zernike coefficients (rad)','Onboard calibrated for instrumental polarizatio',
    'Cavity map used during inversion','Onboard scientific data analysis','Inversion mode','Inversion software','Number RTE inversion iterations', 'Version of calibration pack']

    for h in hdr_arr:
        for i in range(len(k)):
            if k[i] in h:  # Check for existence
                pass # changed to avoid change of parameters after partial processing
                # h[k[i]] = v[i]
            else:
                if i==0:
                    h.set(k[i], v[i], c[i], after='CAL_DARK')
                else:
                    h.set(k[i], v[i], c[i], after=k[i-1])
    return hdr_arr


def data_hdr_kw(hdr, data):
    """Add data description keywords

    Parameters
    ----------
    hdr: header
        file header

    Returns
    -------
    hdr
        Updated file header
    """
    hdr['DATAMEDN'] = float(f"{np.median(data):.8g}")
    hdr['DATAMEAN'] = float(f"{np.mean(data):.8g}")
    #DATARMS
    #DATASKEW
    #DATAKURT
    return hdr


def load_and_process_flat(flat_f, accum_scaling, bit_conversion, scale_data, header_imgdirx_exists, imgdirx_flipped, cpos_arr,pmp_temp) -> np.ndarray:
    """Load, properly scale, flip in X if needed, and make any necessary corrections for particular flat fields

    Parameters
    ----------
    flat_f: string
        PATH of the flat field
    accum_scaling: bool
        if True apply scaling to account for the accumulation
    bit_conversion: bool
        if True apply scaling to account for the bit conversion
    scale_data: bool
        if True apply scaling (dependent on if IP5 flat or not)
    header_imgdirx_exits: bool
        if True, the header keyword exists in the science data - if does not exist, runs to fall back option in `compare_IMGDIRX` func
    imgdirx_flipped: str or bool
        set to True if the science data is flipped, function will flip the flat to match, OPTIONS: 'YES' or 'NO', or False
    cpos_arr: np.ndarray
        array containing the continuum positions of the science scans - to make sure that the flat cpos matches the science flat

    Returns
    -------
    flat
        (2k,2k,4,6) shaped numpy array of the flat field
    """
    print(" ")
    printc('-->>>>>>> Reading Flats',color=bcolors.OKGREEN)

    start_time = time.perf_counter()
    
    # flat from IP-5
    if '0024151020000' in flat_f or '0024150020000' in flat_f:
        flat, header_flat = get_data(flat_f, scaling = accum_scaling,  bit_convert_scale=bit_conversion,
                                    scale_data=False)
    else:
        flat, header_flat = get_data(flat_f, scaling = accum_scaling,  bit_convert_scale=bit_conversion,
                                    scale_data=scale_data)
                
    if 'IMGDIRX' in header_flat:
        header_fltdirx_exists = True
        fltdirx_flipped = str(header_flat['IMGDIRX'])
    else:
        header_fltdirx_exists = False
        fltdirx_flipped = 'NO'
    
    print(f"Flat field shape is {flat.shape}")
    # correction based on science data - see if flat and science are both flipped or not
    flat = compare_IMGDIRX(flat,header_imgdirx_exists,imgdirx_flipped,header_fltdirx_exists,fltdirx_flipped)
    
    flat = np.moveaxis(flat, 0,-1) #so that it is [y,x,24]
    flat = flat.reshape(2048,2048,6,4) #separate 24 images, into 6 wavelengths, with each 4 pol states
    flat = np.moveaxis(flat, 2,-1)
    
    print(flat.shape)

    _, _, _, cpos_f = fits_get_sampling(flat_f,verbose = True) #get flat continuum position

    print(f"The continuum position of the flat field is at {cpos_f} index position")
    
    #--------
    # test if the science and flat have continuum at same position
    #--------

    flat = compare_cpos(flat,cpos_f,cpos_arr[0]) 

    flat_pmp_temp = str(int(header_flat['HPMPTSP1']))

    print(f"Flat PMP Temperature Set Point: {flat_pmp_temp}")

    if flat_pmp_temp != pmp_temp:
        printc('-->>>>>>> WARNING: Flat and dataset have different PMP temperatures',color=bcolors.WARNING)
        printc('                   Flat will be demodulated and then modulated back to the dataset PMP temperature',color=bcolors.WARNING)
        
        flat, _ = demod_hrt(flat,flat_pmp_temp)
        flat, _ = demod_hrt(flat,pmp_temp,modulate=True)
        flat_pmp_temp = pmp_temp

    #--------
    # correct for missing line in particular flat field
    #--------

    if flat_f[-15:] == '0162201100.fits':  # flat_f[-62:] == 'solo_L0_phi-hrt-flat_0667134081_V202103221851C_0162201100.fits'
        print("This flat has a missing line - filling in with neighbouring pixels")
        flat_copy = flat.copy()
        flat[:,:,1,1] = filling_data(flat_copy[:,:,1,1], 0, mode = {'exact rows':[1345,1346]}, axis=1)

        del flat_copy
        
    printc('--------------------------------------------------------------',bcolors.OKGREEN)
    printc(f"------------ Load flats time: {np.round(time.perf_counter() - start_time,3)} seconds",bcolors.OKGREEN)
    printc('--------------------------------------------------------------',bcolors.OKGREEN)

    return flat, flat_pmp_temp, header_flat


def load_dark(dark_f) -> np.ndarray:
    """Load dark field - for use in notebooks

    Parameters
    ----------
    dark_f: string
        PATH of the flat field

    Returns
    -------
    dark
        (2k,2k) numpy array of the flat field
    """
    print(" ")
    printc('-->>>>>>> Reading Darks',color=bcolors.OKGREEN)

    start_time = time.perf_counter()

    try:
        dark,_ = get_data(dark_f)

        dark_shape = dark.shape

        if dark_shape != (2048,2048):

            if dark.ndim > 2:
                printc("Dark Field Input File has more dimensions than the expected 2048,2048 format: {}",dark_f,color=bcolors.WARNING)
                raise ValueError
            
            printc("Dark Field Input File not in 2048,2048 format: {}",dark_f,color=bcolors.WARNING)
            printc("Attempting to correct ",color=bcolors.WARNING)

            
            try:
                if dark_shape[0] > 2048:
                    dark = dark[dark_shape[0]-2048:,:]
            
            except Exception:
                printc("ERROR, Unable to correct shape of dark field data: {}",dark_f,color=bcolors.FAIL)
                raise ValueError

        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        printc(f"------------ Load darks time: {np.round(time.perf_counter() - start_time,3)} seconds",bcolors.OKGREEN)
        printc('--------------------------------------------------------------',bcolors.OKGREEN)

        return dark

    except Exception:
        printc("ERROR, Unable to open and process darks file: {}",dark_f,color=bcolors.FAIL)


def apply_dark_correction(data, dark, rows, cols) -> np.ndarray:
    """Apply dark field correction to the input data

    Parameters
    ----------
    data: ndarray
        data to be dark fielded
    dark: ndarray
        dark field
    rows: slice object
        rows to be used from dark - used in case data.shape does not agree with dark, or for testing
    cols: slice object
        columns to tbe used from dark - used in case data.shape does not agree with dark, or for testing

    Returns
    -------
    data
        dark fielded data
    """
    print(" ")
    print("-->>>>>>> Subtracting dark field")
    
    start_time = time.perf_counter()

    data -= dark[rows,cols, np.newaxis, np.newaxis, np.newaxis] 
    #flat -= dark[..., np.newaxis, np.newaxis] #- # all processed flat fields should already be dark corrected

    printc('--------------------------------------------------------------',bcolors.OKGREEN)
    printc(f"------------- Dark Field correction time: {np.round(time.perf_counter() - start_time,3)} seconds",bcolors.OKGREEN)
    printc('--------------------------------------------------------------',bcolors.OKGREEN)

    return data


def normalise_flat(flat, ceny, cenx) -> np.ndarray:
    """Normalise flat field at each separate filtergram

    Parameters
    ----------
    flat: ndarray
        flat field
    ceny: slice object
        rows (y positions) to be used for the region over which the mean is taken
    cenx: slice object
        columns (x positions) to be used for the region over which the mean is taken

    Returns
    -------
    flat
        normalised flat field
    """
    print(" ")
    printc('-->>>>>>> Normalising Flats',color=bcolors.OKGREEN)

    start_time = time.perf_counter()

    try:
        flat[np.isinf(flat)] = 1
        flat[np.isnan(flat)] = 1
        flat[flat == 0] = 1
        norm_fac = np.mean(flat[ceny,cenx, :, :], axis = (0,1))[np.newaxis, np.newaxis, ...]  #mean of the central 1k x 1k
        flat /= norm_fac

        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        printc(f"------------- Normalising flat time: {np.round(time.perf_counter() - start_time,3)} seconds",bcolors.OKGREEN)
        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        
        return flat

    except Exception:
        printc("ERROR, Unable to normalise the flat field", color=bcolors.FAIL)


def demod_hrt(data, pmp_temp, verbose = True, modulate = False) -> np.ndarray:
    """Use constant demodulation matrices to demodulate input data

    Parameters
    ----------
    data: ndarray
        input data
    pmp_temp: str
        PMP temperature of data to be demodulated, options are '45' or '50
    verbose: bool
        if True, more console prints info, DEFAULT = True

    Returns
    -------
    data
        demodulated data
    demod
        demodulation matrix used
    """
    def _rotation_matrix(angle_rot):
        c, s = np.cos(2*angle_rot*np.pi/180), np.sin(2*angle_rot*np.pi/180)
        return np.array([[1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0], [0, 0, 0, 1]])
    def _rotate_m(angle,matrix):
        rot = _rotation_matrix(angle)
        return np.matmul(matrix,rot)
    
    HRT_MOD_ROTATION_ANGLE=0.2
    
    if pmp_temp == '50':
        # 'original (pre May 2022/RSW1 2022 matrices, that don't account for azimuth angle etc in PMP)
        # demod_data = np.array([[ 0.28037298,  0.18741922,  0.25307596,  0.28119895],
        #              [ 0.40408596,  0.10412157, -0.7225681,   0.20825675],
        #              [-0.19126636, -0.5348939,   0.08181918,  0.64422774],
        #              [-0.56897295,  0.58620095, -0.2579202,   0.2414017 ]])
        #Alberto 30/04/22
#         printc(f'Using Alberto demodulation matrix for temp=50',color = bcolors.OKGREEN)
        mod_matrix = np.array([[ 1.0014 ,  0.56715  , 0.3234 , -0.74743  ],
                               [ 1.0007 ,  0.0037942, 0.69968,  0.71423  ],
                               [ 1.0002 , -0.98937  , 0.04716, -0.20392  ],
                               [ 0.99769,  0.27904  ,-0.86715,  0.39908  ]])
        mod_matrix = _rotate_m(HRT_MOD_ROTATION_ANGLE,mod_matrix)
        demod_data = np.linalg.inv(mod_matrix)
        
    elif pmp_temp == '40':
        # 'original (pre May 2022/RSW1 2022 matrices, that don't account for azimuth angle etc in PMP)
        # demod_data = np.array([[ 0.26450154,  0.2839626,   0.12642948,  0.3216773 ],
        #              [ 0.59873885,  0.11278069, -0.74991184,  0.03091451],
        #              [ 0.10833212, -0.5317737,  -0.1677862,   0.5923593 ],
        #              [-0.46916953,  0.47738808, -0.43824592,  0.42579797]])
        #Alberto 14/04/22
#         printc(f'Using Alberto demodulation matrix for temp=40',color = bcolors.OKGREEN)
        mod_matrix = np.array([[ 0.99816  ,0.61485 , 0.010613 ,-0.77563 ], 
                               [ 0.99192 , 0.08382 , 0.86254 , 0.46818],
                               [ 1.0042 , -0.84437 , 0.12872 ,-0.53972],
                               [ 1.0057 , -0.30576 ,-0.87969 , 0.40134]])
        mod_matrix = _rotate_m(HRT_MOD_ROTATION_ANGLE,mod_matrix)
        demod_data = np.linalg.inv(mod_matrix)
        
    else:
        printc(f"Demodulation Matrix for PMP TEMP of {pmp_temp} deg is not available", color = bcolors.FAIL)
    if verbose:
        printc(f'Using a constant demodulation matrix for a PMP TEMP of {pmp_temp} deg, rotated by {HRT_MOD_ROTATION_ANGLE} deg',color = bcolors.OKGREEN)
    
    if modulate:
        demod_data = mod_matrix

    
    # demod_data = demod_data.reshape((4,4))
    # shape = data.shape
    # demod = np.tile(demod_data, (shape[0],shape[1],1,1))

    if data.ndim == 5:
        data = np.einsum('ij,abjcs->abics', demod_data, data)
        # # if data array has more than one scan
        # data = np.moveaxis(data,-1,0) #moving number of scans to first dimension

        # data = np.matmul(demod,data)
        # data = np.moveaxis(data,0,-1) #move scans back to the end
    
    elif data.ndim == 4:
        data = np.einsum('ij,abjc->abic', demod_data, data)
        # # if data has one scan
        # data = np.matmul(demod,data)
    
    return data, demod_data


def unsharp_masking(flat,sigma,flat_pmp_temp,cpos_arr,clean_mode,clean_f,pol_end=4,verbose=True):
    """Apply unsharp masking to the flat fields to remove polarimetric structures due to solar rotation

    Parameters
    ----------
    flat: ndarray
        input flat field
    sigma: float
        sigma of the gaussian filter
    flat_pmp_temp: str
        PMP temperature of flat to be demodulated, options are '45' or '50'
    cpos_arr: ndarray
        array of continuum positions
    clean_mode: str
        options are any combination of 'IQUV'
    clean_f: str
        options are 'blurring' or 'fft'
    pol_end: int
        last pol state to be cleaned, DEFAULT = 4
    verbose: bool
        if True, more console prints info, DEFAULT = True

    Returns
    -------
    flat_cleaned: ndarray
        cleaned flat field
    """
    flat_demod, demodM = demod_hrt(flat, flat_pmp_temp,verbose)

    norm_factor = np.mean(flat_demod[512:1536,512:1536,0,cpos_arr[0]])

    flat_demod /= norm_factor

    new_demod_flats = np.copy(flat_demod)

    if cpos_arr[0] == 0:
        wv_range = range(1,6)

    elif cpos_arr[0] == 5:
        wv_range = range(5)

    clean_pol = []
    if "I" in clean_mode:
        clean_pol += [0]
    if "Q" in clean_mode:
        clean_pol += [1]
    if "U" in clean_mode:
        clean_pol += [2]
    if "V" in clean_mode:
        clean_pol += [3]
    
    # add possibility to blur continuum
    if "cont" in clean_mode:
        wv_range = range(6)
        if "only" in clean_mode:
            wv_range = [cpos_arr[0]]
    
    print("Unsharp Masking",clean_mode, wv_range)

        
    if clean_f == 'blurring':
        blur = lambda a: gaussian_filter(a,sigma)
    elif clean_f == 'fft':
        x = np.fft.fftfreq(2048,1)
        fftgaus2d = np.exp(-2*np.pi**2*(x-0)**2*sigma**2)[:,np.newaxis] * np.exp(-2*np.pi**2*(x-0)**2*sigma**2)[np.newaxis]
        blur = lambda a : (np.fft.ifftn(fftgaus2d*np.fft.fftn(a.copy()))).real
    
    for pol in clean_pol:

        for wv in wv_range: #not the continuum

            a = np.copy(np.clip(flat_demod[:,:,pol,wv], -0.02, 0.02))
            b = a - blur(a)
            c = a - b

            new_demod_flats[:,:,pol,wv] = c

    flat_cleaned, _ = demod_hrt(new_demod_flats*norm_factor,flat_pmp_temp,verbose,modulate=True)
    
    return flat_cleaned


def flat_correction(data,flat,flat_states,cpos_arr,flat_pmp_temp=50,rows=slice(0,2048),cols=slice(0,2048)) -> np.ndarray:
    """Apply flat field correction to input data

    Parameters
    ----------
    data: ndarray
        input data
    flat: ndarray
        input flat field
    flat_states: int
        number of flat fields to use for flat fielding, options are 4, 6, 9 or 24
    cpos_arr: ndarray
        array of continuum positions
    flat_pmp_temp: str
        PMP temperature of flat to be demodulated, options are '45' or '50'
    rows: slice
        rows to be used for flat fielding, DEFAULT = slice(0,2048)
    cols: slice
        cols to be used for flat fielding, DEFAULT = slice(0,2048)

    Returns
    -------
    data: ndarray
        flat fielded data
    """
    print(" ")
    printc('-->>>>>>> Correcting Flatfield',color=bcolors.OKGREEN)

    start_time = time.perf_counter()

    try: 
        if flat_states == 6:
            
            printc("Dividing by 6 flats, one for each wavelength",color=bcolors.OKGREEN)
                
            tmp = np.mean(flat,axis=-2) #avg over pol states for the wavelength

            return data / tmp[rows,cols, np.newaxis, :, np.newaxis]


        elif flat_states == 24:

            printc("Dividing by 24 flats, one for each image",color=bcolors.OKGREEN)

            return data / flat[rows,cols, :, :, np.newaxis] #only one new axis for the scans
                
        elif flat_states == 4:

            printc("Dividing by 4 flats, one for each pol state",color=bcolors.OKGREEN)

            # tmp = np.mean(flat,axis=-1) #avg over wavelength
            tmp = flat[:,:,:,cpos_arr[0]] # continuum only

            return data / tmp[rows,cols, :, np.newaxis, np.newaxis]

        if flat_states == 9:
            
            printc("Dividing by 9 flats, one for each wavelength in Stokes I, only continuum in Stokes Q, U and V",color=bcolors.OKGREEN)
            
            tmp = np.zeros(flat.shape)
            demod_flat, demodM = demod_hrt(flat.copy(), flat_pmp_temp, False)
            tmp[:,:,0] = demod_flat[:,:,0]
            tmp[:,:,1:] = demod_flat[:,:,1:,cpos_arr[0],np.newaxis]
            del demod_flat
            invM = np.linalg.inv(demodM)
            tmp = np.matmul(invM, tmp)    
            
            return data / tmp[rows,cols, :, :, np.newaxis]

        else:
            print(" ")
            printc('-->>>>>>> Unable to apply flat correction. Please insert valid flat_states',color=bcolors.WARNING)

            
        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        printc(f"------------- Flat Field correction time: {np.round(time.perf_counter() - start_time,3)} seconds ",bcolors.OKGREEN)
        printc('--------------------------------------------------------------',bcolors.OKGREEN)

        return data

    except Exception as exc:
        printc(exc,color=bcolors.FAIL) 
        printc("ERROR, Unable to apply flat fields",color=bcolors.FAIL)


def prefilter_correctionNew(data,wave_axis_arr,rows,cols,Tetalon=66,imgdirx_flipped = 'YES'):
    """
    New prefilter correction based on JH email on 2023-08-17
    Based on on-ground measurements at Meudon

    Parameters
    ----------
    data: ndarray
        input data
    wave_axis_arr: ndarray
        array containing wavelengths
    rows: slice
        rows to be considered because of data cropping
    cols: slice
        columns to be considered because of data cropping
    imgdirx_flipped: str
        check if data have been flipped (all the hrt-L1 data are flipped), DEFAULT = 'YES'

    Returns
    -------
    data: ndarray
        prefilter corrected data
    """
    X,Y = np.meshgrid(np.arange(cols.start,cols.stop),np.arange(rows.start,rows.stop))
    r = np.sqrt((X-934)**2 + (Y-1148)**2)

    CWL = -0.332253 - 4.81875e-05*r - 3.24533e-07*r**2 #  in AA, 0 is the line core of the Fe617 line (6173.343 AA)
    FWHM = 2.59385 - 1.28984e-06*r - 7.38763e-09*r**2 # in AA
    EXP = 3.65789 - 5.76046e-05*r - 3.52776e-08*r**2

    xx = lambda wl: np.abs((wl[:,np.newaxis,np.newaxis]-CWL)*2/FWHM)  # in AA, lambda=0 is the Fe line core
    profile = lambda wl:  1/(1+xx(wl)**(2*EXP)) # max. transmission set to be 1. everywhere

    wlref = 6173.341

    for scan in range(data.shape[-1]):
        # prefilter = profile(wave_axis_arr[scan]-wlref) # [wl,y,x]
        # DC 20240612
        prefilter = profile(wave_axis_arr[scan]-wlref-(Tetalon-66)*34.25e-3) # [wl,y,x] # Temperature shift of the prefilter by TO
        prefilter = np.moveaxis(prefilter[...,np.newaxis],0,-1) # [y,x,1,wl]
        if imgdirx_flipped == 'YES':
            printc('Flipping prefilter on the Y axis')
            prefilter = prefilter[:,::-1]
        data[...,scan] /= prefilter
    return data

def prefilter_correction(data,wave_axis_arr,prefilter,Tetalon=0,prefilter_voltages = None, TemperatureCorrection=True, TemperatureConstant = 40.1225e-3, shift = None):
    """Apply prefilter correction to input data

    Parameters
    ----------
    data: ndarray
        input data
    wave_axis_arr: ndarray
        array containing wavelengths
    prefilter: ndarray
        prefilter data
    prefilter_voltages: ndarray
        prefilter voltages, DEFAULT = None - uses latest prefilter voltages from on ground calibration
    TemperatureCorrection: bool
        apply temperature correction to prefilter data, DEFAULT = False
    TemperatureConstant: float
        value of the temperature tuning constant to be used when TemperatureConstant is True, DEFAULT = 36.46e-3 mA/K

    Returns
    -------
    data: ndarray
        prefilter corrected data

    adapted from SPGPylibs
    """
    def _get_v1_index1(x):
        # index1, v1 = min(enumerate([abs(i) for i in x]), key=itemgetter(1))
        index1, v1 = min(enumerate(x), key = lambda i: abs(i[1]))
        # return  x[index1], index1
        return  v1, index1
    
    if prefilter_voltages is None:
        # OLD prefilter voltages
        # prefilter_voltages = np.asarray([-1300.00,-1234.53,-1169.06,-1103.59,-1038.12,-972.644,-907.173,-841.702,-776.231,-710.760,-645.289,
        #                                 -579.818,-514.347,-448.876,-383.404,-317.933,-252.462,-186.991,-121.520,-56.0490,9.42212,74.8932,
        #                                 140.364,205.835,271.307, 336.778,402.249,467.720,533.191,598.662,664.133,729.604,795.075,860.547,
        #                                 926.018,991.489,1056.96,1122.43,1187.90,1253.37, 1318.84,1384.32,1449.79,1515.26,1580.73,1646.20,
        #                                 1711.67,1777.14,1842.61])
        prefilter_voltages = np.asarray([-1277.   , -1210.75 , -1145.875, -1080.25 , -1015.25 ,  -950.25 ,
                                        -885.75 ,  -820.125,  -754.875,  -691.   ,  -625.5  ,  -559.75 ,
                                        -494.125,  -428.25 ,  -364.   ,  -298.875,  -233.875,  -169.   ,
                                        -104.625,   -40.875,    21.125,    86.25 ,   152.25 ,   217.5  ,
                                         282.625,   346.25 ,   411.   ,   476.125,   542.   ,   607.75 ,
                                         672.125,   738.   ,   803.75 ,   869.625,   932.   ,   996.625,
                                        1062.125,  1128.   ,  1192.   ,  1258.125,  1323.625,  1387.25 ,
                                        1451.875,  1516.875,  1582.125,  1647.75 ,  1713.875,  1778.375,
                                        1844.   ])
    if TemperatureCorrection:
        # temperature_constant_old = 40.323e-3 # old temperature constant, still used by Johann
        # temperature_constant_new = 37.625e-3 # new and more accurate temperature constant
        # temperature_constant_new = 36.46e-3 # value from HS
        Tfg = 66 # FG was at 66 deg during e2e calibration
        tunning_constant = 0.0003513 # this shouldn't change
        
        ref_wavelength = 6173.341 # this shouldn't change
        prefilter_wave = prefilter_voltages * tunning_constant + ref_wavelength + TemperatureConstant*(Tfg-61) - 0.002 # JH ref
        # DC 20240612
        prefilter_wave += (Tetalon-66)*34.25e-3 # Temperature shift of the prefilter by TO
        
        # ref_wavelength = round(6173.072 - (-1300*tunning_constant),3) # 6173.529. 0 level was different during e2e test
        # prefilter_wave = prefilter_voltages * tunning_constant + ref_wavelength # + temperature_constant_new*(Tfg-61)
       
    else:
        tunning_constant = 0.0003513
        ref_wavelength = 6173.341 # this shouldn't change
        prefilter_wave = prefilter_voltages * tunning_constant + ref_wavelength
    
    data_shape = data.shape
    
    for scan in range(data_shape[-1]):

        wave_list = wave_axis_arr[scan]
        
        if shift is None:
            for wv in range(len(wave_list)):

                v = wave_list[wv]

                vdif = [v - pf for pf in prefilter_wave]

                v1, index1 = _get_v1_index1(vdif)
                if v < prefilter_wave[-1] and v > prefilter_wave[0]:

                    if vdif[index1] >= 0:
                        v2 = vdif[index1 + 1]
                        index2 = index1 + 1

                    else:
                        v2 = vdif[index1-1]
                        index2 = index1 - 1

                    # imprefilter = (prefilter[:,:, index1]*(0-v1) + prefilter[:,:, index2]*(v2-0))/(v2-v1) #interpolation between nearest voltages

                elif v >= prefilter_wave[-1]:
                    index2 = index1 - 1
                    v2 = vdif[index2]

                elif v <= prefilter_wave[0]:
                    index2 = index1 + 1
                    v2 = vdif[index2]

                imprefilter = (prefilter[:,:, index1]*v2 + prefilter[:,:, index2]*(-v1))/(v2-v1) #interpolation between nearest voltages

                # imprefilter = (prefilter[:,:, index1]*v1 + prefilter[:,:, index2]*v2)/(v1+v2) #interpolation between nearest voltages

                data[:,:,:,wv,scan] /= imprefilter[...,np.newaxis]
        else:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    data[i,j,:,:,scan] /= np.interp(wave_list - shift[i,j],prefilter_wave,prefilter[i,j])
  
    return data

def apply_field_stop(data, rows, cols, header_imgdirx_exists, imgdirx_flipped) -> np.ndarray:
    """Apply field stop to input data

    Parameters
    ----------
    data: ndarray
        input data
    rows: slice
        rows to use
    cols: slice
        cols to use
    header_imgdirx_exists: bool
        if imgdirx exists in header
    imgdirx_flipped: str or bool
        if input data is flipped, OPTIONS: 'YES', 'NO', or False

    Returns
    -------
    data: ndarray
        data with field stop applied
    field_stop: ndarray
        field stop array
    """
    print(" ")
    printc("-->>>>>>> Applying field stop",color=bcolors.OKGREEN)

    start_time = time.perf_counter()

    field_stop_loc = os.path.realpath(__file__)

    field_stop_loc = field_stop_loc.split('src/')[0] + 'field_stop/'

    field_stop,_ = load_fits(field_stop_loc + 'HRT_field_stop_new.fits')

    field_stop = np.where(field_stop > 0,1,0)

    if header_imgdirx_exists:
        if imgdirx_flipped == 'YES': #should be YES for any L1 data, but mistake in processing software
            field_stop = field_stop[:,::-1] #also need to flip the flat data after dark correction

    data *= field_stop[rows,cols,np.newaxis, np.newaxis, np.newaxis]

    printc('--------------------------------------------------------------',bcolors.OKGREEN)
    printc(f"------------- Field stop time: {np.round(time.perf_counter() - start_time,3)} seconds",bcolors.OKGREEN)
    printc('--------------------------------------------------------------',bcolors.OKGREEN)

    return data, field_stop


def load_ghost_field_stop(header_imgdirx_exists, imgdirx_flipped) -> np.ndarray:
    """Load field stop with specific ghost correction

    Parameters
    ----------
    header_imgdirx_exists: bool
        if imgdirx exists in header
    imgdirx_flipped: str or bool
        if input data is flipped, OPTIONS: 'YES', 'NO', or False

    Returns
    -------
    field_stop_ghost: ndarray
        field stop array with some regions masked for ghost correction
    """
    print(" ")
    printc("-->>>>>>> Loading ghost field stop",color=bcolors.OKGREEN)

    start_time = time.perf_counter()

    field_stop_loc = os.path.realpath(__file__)
    field_stop_loc = field_stop_loc.split('src/')[0] + 'field_stop/'

    field_stop_ghost,_ = load_fits(field_stop_loc + 'HRT_field_stop_ghost_new.fits')
    field_stop_ghost = np.where(field_stop_ghost > 0,1,0)

    if header_imgdirx_exists:
        if imgdirx_flipped == 'YES': #should be YES for any L1 data, but mistake in processing software
            field_stop_ghost = field_stop_ghost[:,::-1]

    printc('--------------------------------------------------------------',bcolors.OKGREEN)
    printc(f"------------- Load Ghost Field Stop time: {np.round(time.perf_counter() - start_time,3)} seconds",bcolors.OKGREEN)
    printc('--------------------------------------------------------------',bcolors.OKGREEN)
    return field_stop_ghost


def crosstalk_2D_ItoQUV(data: np.ndarray,
                     verbose: bool = False,
                     mask: np.ndarray = np.empty([], dtype=float),
                     threshold: float = 0.5,
                     lower_threshold: float = 40.0,
                     norma: float = 1.0,
                     mode: str = 'standard',
                     divisions: int = 16,
                     ind_wave: bool = False,
                     continuum_pos: int = 0):
    """
    crosstalk_ItoQUV calculates the cross-talk from Stokes $I$ to Stokes $Q$, $U$, and $V$.

    The procedure works as follow: (see Sanchez Almeida, J. \& Lites, B.~W.\ 1992, \apj, 398, 359. doi:10.1086/171861)


    :param input_data: input data. Dimensions should be `[Stokes, wavelength, ydim,xdim]`
    :type input_data: np.ndarray
    :param verbose: activate verbosity, defaults to False
    :type verbose: bool, optional
    :param mask: mask for selecting the image area for cross-talk correction dimension = `[ydim,xdim]´, defaults to 0
    :type mask: np.ndarray, optional
    :param threshold: threshold for considering signals in the cross-talk calculation. :math:`p = \sqrt(Q^2 + U^2 = V^2) < threshold`. Given in percent, defaults to 0.5 %
    :type threshold: float, optional
    :param lower_threshold: lower threshold for considering signals in the cross-talk calculation. :math:`I(\\lambda) > lower_threshold`. Given in percent, defaults to 40 %
    :type lower_threshold: float, optional
    :param norma: Data normalization value, defaults to 1.0
    :type norma: float, optional
    :param mode: crosstalk mode, defaults to 'standard'

        there are two different modes for calculating the cross-talk.

        1. ´mode = 'standard'´. It is done using the whole Sun, i.e., one cross-talk coefficient for the full image
        2. ´mode = 'surface'´. In this case, the Sun is divided in different squares (`N = divisions`) along the two dimensions and the crosstalk is calculated for each of the squares.
           Then, a surface is fit to the :math:`N**2` coefficients and applied to the data. In this case, lower_trheshold is ignored and if a mask is not privided, the code generated a mask coinciding with the solar disk.
        2. ´mode = 'jaeggli'´. In this case, the cross talk correction follows the work by Jaeggli et al., 2022, \apj, https://doi.org/10.3847/1538-4357/ac6506
            The goal is to determine the diattenuator and retarder Mueller matrices that minimize certain criteria based on physical assumptions about the polarized signals from the Sun. The application of the combination of this matrices gives the recovered Stokes vector.

    :type mode: str, optional
    :param divisions: number of square divisions alon gone axis for `surface`mode. defaults to 6.0
    :type divisions: int, optional
    :param cntr_rad: Center and radius of the solar disk `[cx,cy,rad]`. Needed for `mode='surface'`. defaults to []
    :type cntr_rad: list, optional
    :param ind_wave: Use just the continuum wavelength for the crosstalk or the whole line. defaults to False
    :type ind_wave: bool, optional
    :param continuum_pos: If ind_wave, this keyword is mandatory and contains the position of the continuum. defaults to 0.
    :type continuum_pos: int, optional
    
    :return: cross-talk parameters
    :rtype: List of np.ndarray
    """

    from scipy.optimize import minimize

    def __fit_crosstalk(input,masked,axis = 2,full_verbose = False):
        # multiply the data by the mask and flatten everything. Flatenning makes things easier
        xI = (input.take(0,axis=axis) * masked).flatten() #Stokes I
        yQ = (input.take(1,axis=axis) * masked).flatten() #Stokes Q
        yU = (input.take(2,axis=axis) * masked).flatten() #Stokes U
        yV = (input.take(3,axis=axis) * masked).flatten() #Stokes V
        
        # check two conditions:
        # 1) mask should be > 0 and intensity above lower_threshold

        # if mask_set:
        #     idx = (xI != 0)
        # else:
        idx = (xI != 0) & (xI > (lower_threshold/100. * norma))

        xI = xI[idx]
        yQ = yQ[idx]
        yU = yU[idx]
        yV = yV[idx]
        
        # 2) Stokes Q,U, and V has to be below a limit (for not inclusing polarization signals)

        yP = np.sqrt(yQ**2 + yU**2 + yV**2)
        idx = yP < (threshold/100. * norma)
        # plt.figure(); plt.hist(yP,100); plt.axvline((threshold/100. * norma),color='r'); plt.show()
        xI = xI[idx]
        yQ = yQ[idx]
        yU = yU[idx]
        yV = yV[idx]
        
        # Now we perform a cross-talk fit.

        cQ = np.polyfit(xI, yQ, 1)
        cU = np.polyfit(xI, yU, 1)
        cV = np.polyfit(xI, yV, 1)

        if full_verbose:
            st = 0
            w = 5
            plt.figure(layout='tight')
            plt.imshow(masked)
            plt.show()
            plt.imshow(input[w,st,:,:])
            plt.show()

            xp = np.linspace(xI.min(), xI.max(), 100)
            ynew = np.polyval(cQ, xp)
            plt.plot(xI,yQ,'.')
            plt.plot(xp,ynew,'o')
            plt.show()
            ynew = np.polyval(cU, xp)
            plt.plot(xI,yU,'.')
            plt.plot(xp,ynew,'o')
            plt.show()
            ynew = np.polyval(cV, xp)
            plt.plot(xI,yV,'.')
            plt.plot(xp,ynew,'o')
            plt.show()

        return cQ, cU, cV

    def __generate_squares(radius,divisions: int = 6):
        square_centers = np.linspace(-radius,radius,divisions+1,endpoint=True)[1:] - radius / divisions
        X,Y = np.meshgrid(square_centers, square_centers)
        return np.vstack([X.ravel(), Y.ravel()])

    def __fit_plane(data, mask=None):

        """
        Fit 2D plane to data. Will be replazed by a global one (comming from had-hoc branch) at some point.
        """
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

        A = np.vstack([X_masked.flatten(), Y_masked.flatten(), np.full(X_masked.size,1)]).T
        a, b, c = np.linalg.lstsq(A, Z_masked.flatten(), rcond=None)[0]
        P = a * X + b * Y + c

        return (P, a, b, c)


    if data.ndim == 3:
        # no wavelength has been provided
        data = data[...,np.newaxis]

    if data.ndim != 4:
        printc('Input data shall have 3 or 4 dimensions but it is of ',data.ndim,' dimensions',color=bcolors.FAIL)
        ValueError("Check dimensions of input data into crosstalk_ItoQUV")

    yd,xd,sd,wd = data.shape

    wave_index = np.arange(wd,dtype=int)
    printc('                     wave_range:',wave_index, color=bcolors.OKBLUE )

    if ind_wave:
        wave_index[:] = continuum_pos
        printc('          Computing the cross-talk using just the continuum....', color=bcolors.OKBLUE)
        printc('                     wave_range has change to:',wave_index, color=bcolors.OKBLUE )

    # First check also if mask has been provided. If np.array in empty with dim (), shape is False.
    if mask.shape:
        printc('mask inside `crosstalk_ItoQUV`, has been provided',color=bcolors.OKGREEN)
        if mask.ndim != 2:
            printc('Input mask shall have 2 dimensions but it is of ',mask.ndim,' dimensions',color=bcolors.FAIL)
            ValueError("Check dimensions of input mask into crosstalk_ItoQUV")
        # mask_set = True
    else:
        mask = np.ones((yd,xd),dtype=bool)
        # mask[int(yd//2-yd//4):int(yd//2+yd//4),int(xd//2-xd//4):int(xd//2+xd//4)]
        # mask_set = False

    # threshold = 0.5
    # lower_threshold = 40.
    # norma = 1.

    if mode == 'standard':

        cQ, cU, cV = __fit_crosstalk(data[:,:,:,wave_index],mask[...,np.newaxis],full_verbose=verbose)

        print('Cross-talk from I to Q: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cQ[0], cQ[1],width=8,prec=4))
        print('Cross-talk from I to U: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cU[0], cU[1],width=8,prec=4))
        print('Cross-talk from I to V: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cV[0], cV[1],width=8,prec=4))

        # corrects the data:
        corrected_data = np.copy(data)
        
        corrected_data[:, :, 1] = data[:, :, 1] - cU[0] * data[:, :, 0] - cU[1]
        corrected_data[:, :, 2] = data[:, :, 2] - cQ[0] * data[:, :, 0] - cQ[1]
        corrected_data[:, :, 3] = data[:, :, 3] - cV[0] * data[:, :, 0] - cV[1]
        
        return cQ, cU, cV, 0 , 0 , 0, corrected_data

    elif mode == 'surface':
        sz = data.shape # y,x,p,l
        rad = sz[0]//2
        size2 = rad//divisions
        area = (size2*2)**2
        ndiv = __generate_squares(rad,divisions)
        cx = rad; cy = rad

        if verbose:
            fig, ax = plt.subplots(figsize=(8,8))
            ax.imshow(data[:,:,0,0],cmap='gray_r')

            for i in range(divisions**2):
                square = plt.Rectangle((ndiv[0,i] + cx - size2,ndiv[1,i] + cy - size2), size2 * 2, size2 * 2 , color='r', fill=False)
                ax.add_patch(square)
            plt.show()

        cQ = np.zeros((2,divisions**2))
        cU = np.zeros((2,divisions**2))
        cV = np.zeros((2,divisions**2))
        
        for i,loop in enumerate(range(divisions**2)):
            from_x, to_x = np.round(ndiv[0,i] + cx - size2).astype(int) , np.round(ndiv[0,i] + cx + size2).astype(int)
            from_y, to_y = np.round(ndiv[1,i] + cy - size2).astype(int) , np.round(ndiv[1,i] + cy + size2).astype(int)
            if np.sum(mask[from_y:to_y,from_x:to_x])  > area * 0.1:
                cQ[:,loop], cU[:,loop], cV[:,loop]  = \
                    __fit_crosstalk(data[from_y:to_y,from_x:to_x,:,wave_index],mask[from_y:to_y,from_x:to_x,np.newaxis],full_verbose=verbose)
        cQQ = np.reshape(cQ,(2,divisions,divisions))
        cUU = np.reshape(cU,(2,divisions,divisions))
        cVV = np.reshape(cV,(2,divisions,divisions))

        if verbose:
            plt.figure()
            plt.imshow(cQQ[0,:,:]*100,clim=[-0.5,0.5])
            plt.colorbar()
            plt.show()
            plt.figure()
            plt.imshow(cUU[0,:,:]*100,clim=[-0.5,0.5])
            plt.colorbar()
            plt.show()
            plt.figure()
            plt.imshow(cVV[0,:,:]*100,clim=[-0.5,0.5])
            plt.colorbar()
            plt.show()
            plt.figure()
            plt.imshow(cQQ[1,:,:]*100,clim=[-0.5,0.5])
            plt.colorbar()
            plt.show()
            plt.figure()
            plt.imshow(cUU[1,:,:]*100,clim=[-0.5,0.5])
            plt.colorbar()
            plt.show()
            plt.figure()
            plt.imshow(cVV[1,:,:]*100,clim=[-0.5,0.5])
            plt.colorbar()
            plt.show()
        
        dummy_s = np.zeros((yd,xd))
        dummy_c = np.zeros((yd,xd))
        for i,loop in enumerate(range(divisions**2)):
            dummy_s[np.round(ndiv[1,i] + cy).astype(int),np.round(ndiv[0,i] + cx).astype(int)] = cQ[0,loop]
            dummy_c[np.round(ndiv[1,i] + cy).astype(int),np.round(ndiv[0,i] + cx).astype(int)] = cQ[1,loop]
        sfitQ = (__fit_plane(dummy_s, mask = (dummy_s != 0)) , __fit_plane(dummy_c, mask = (dummy_c != 0)) )
        if verbose:
            plt.figure()
            plt.imshow(sfitQ[0][0]*100,clim=[-0.5,0.5])
            plt.colorbar()
            plt.show()
            plt.figure()
            plt.imshow(sfitQ[1][0]*100,clim=[-0.5,0.5])
            plt.colorbar()
            plt.show()
        
        for i,loop in enumerate(range(divisions**2)):
            dummy_s[np.round(ndiv[1,i] + cy).astype(int),np.round(ndiv[0,i] + cx).astype(int)] = cU[0,loop]
            dummy_c[np.round(ndiv[1,i] + cy).astype(int),np.round(ndiv[0,i] + cx).astype(int)] = cU[1,loop]
        sfitU = (__fit_plane(dummy_s, mask = (dummy_s != 0)) , __fit_plane(dummy_c, mask = (dummy_c != 0)) )
        if verbose:
            plt.figure()
            plt.imshow(sfitU[0][0]*100,clim=[-0.5,0.5])
            plt.colorbar()
            plt.show()
            plt.figure()
            plt.imshow(sfitU[1][0]*100,clim=[-0.5,0.5])
            plt.colorbar()
            plt.show()
        
        for i,loop in enumerate(range(divisions**2)):
            dummy_s[np.round(ndiv[1,i] + cy).astype(int),np.round(ndiv[0,i] + cx).astype(int)] = cV[0,loop]
            dummy_c[np.round(ndiv[1,i] + cy).astype(int),np.round(ndiv[0,i] + cx).astype(int)] = cV[1,loop]
        sfitV = (__fit_plane(dummy_s, mask = (dummy_s != 0)) , __fit_plane(dummy_c, mask = (dummy_c != 0)) )
        if verbose:
            plt.figure()
            plt.imshow(sfitV[0][0]*100,clim=[-0.5,0.5])
            plt.colorbar()
            plt.show()
            plt.figure()
            plt.imshow(sfitV[1][0]*100,clim=[-0.5,0.5])
            plt.colorbar()
            plt.show()
        
        # correction
        corrected_data = np.copy(data)
        corrected_data[:, :, 1] = data[:, :, 1] - sfitQ[0][0][...,np.newaxis] * data[:, :, 0] - sfitQ[1][0][...,np.newaxis]
        corrected_data[:, :, 2] = data[:, :, 2] - sfitU[0][0][...,np.newaxis] * data[:, :, 0] - sfitU[1][0][...,np.newaxis]
        corrected_data[:, :, 3] = data[:, :, 3] - sfitV[0][0][...,np.newaxis] * data[:, :, 0] - sfitV[1][0][...,np.newaxis]

        return cQ, cU, cV, sfitQ, sfitU, sfitV, corrected_data
    
        def _polmodel1(D,theta,chi):
            dH = D*np.cos(chi)*np.sin(theta)
            d45 = D*np.sin(chi)*np.sin(theta)
            dR = D*np.cos(theta)
            A = np.sqrt(1. - dH**2 - d45**2 - dR**2)

            mat1 = np.array([
                [ 1., dH, d45, dR], 
                [ dH,  A,  0., 0.], 
                [d45, 0.,   A, 0.],
                [ dR, 0.,  0.,  A]], dtype='double')

            mat2 = np.array([
                [0.,     0.,     0.,     0.],
                [0.,  dH**2, d45*dH,  dH*dR],
                [0., d45*dH, d45**2, d45*dR],
                [0.,  dH*dR, d45*dR,  dR**2]], dtype='double')

            return( mat1 + (1-A)/D**2*mat2 )

        def _fitfunc1(param, stokesin):
            D = param[0]
            theta = param[1]
            chi = param[2]

            # Keep diattenuation value in range
            if D>=1:
                D=0.999999

            if D<=-1:
                D = -0.999999

            MM = _polmodel1(D, theta, chi)
            iMM = np.linalg.inv(MM)

            out = _minimize_for_model1(iMM,stokesin)

            return(out)

        def _minimize_for_model1(iMM,bs):
            # apply a mueller matrix (rotation) to a 2D stokes vector (slit_Y,wavelength_X,4)
            new_stokes = np.einsum('ij,abj->abi',iMM, np.squeeze(bs))

            # Minimization criteria
            out = np.abs(np.sum(new_stokes[:,:,0]*new_stokes[:,:,3],axis=1)) + \
                np.abs(np.sum(new_stokes[:,:,0]*new_stokes[:,:,2],axis=1)) + \
                np.abs(np.sum(new_stokes[:,:,0]*new_stokes[:,:,1],axis=1))

            # sum over spatial positions
            out = np.sum(out)

            return(out)


        # Functions for the retarder modeling
        def _polmodel2(theta, delta):
            St = np.sin(theta)
            Ct = np.cos(theta)
            Sd = np.sin(delta)
            Cd = np.cos(delta)

            MM1 = np.array([
                [1.,  0., 0., 0.],
                [0.,  Ct, St, 0.],
                [0., -St, Ct, 0.],
                [0.,  0., 0., 1.]
            ], dtype='double')

            MM2 = np.array([
                [1., 0.,  0., 0.],
                [0., 1.,  0., 0.],
                [0., 0.,  Cd, Sd],
                [0., 0., -Sd, Cd]
            ], dtype='double')

            MM = np.einsum('ij,jk', MM1, MM2)
            return(MM)

        def _fitfunc2(fitangles, stokesin):
            theta = fitangles[0]
            delta = fitangles[1]

            MM = _polmodel2(theta, delta)
            iMM = np.linalg.inv(MM)

            out = _minimize_for_model2(iMM, stokesin)

            return(out)

        def _minimize_for_model2(iMM,bs):
            new_stokes = np.einsum('ij,abj->abi',iMM, np.squeeze(bs))

            # Minimization criteria
            out = np.sum(new_stokes[:,:,3],axis=1)**2 +\
                np.abs(np.sum(new_stokes[:,:,3]*new_stokes[:,:,2],axis=1)) +\
                np.abs(np.sum(new_stokes[:,:,3]*new_stokes[:,:,1],axis=1))

            # sum over spatial positions
            out = np.sum(out)

            return(out)

        
        # Make the continuum intensity map
        imap = data[:,:,0,continuum_pos]
        imap = imap.transpose()
        
        line_wv = [i for i in range(wd)]
        line_wv.remove(continuum_pos)
        
        # Make the polarization fraction map
        pmap = np.max( np.sqrt(np.sum(data[:,:,1:,line_wv]**2, axis=2))/np.mean(data[mask>0,0],axis=0)[line_wv], axis=2)
        pmap = pmap.transpose()

        # Apply thresholds to define different regions
        ithresh = (lower_threshold/100. * norma) # 0.5 # continuum intensity threshold for the sunspot umbra
        pthreshlow = (threshold/100. * norma) # polarization threshold for weak/strong polarization regions 
        pthreshhigh = (threshold/100. * norma * 4) # polarization threshold for weak/strong polarization regions 

        isumbra = np.argwhere(np.logical_and(imap < ithresh, mask > 0))
        uyidx = isumbra[:,0]
        uxidx = isumbra[:,1]

        ispolar = np.argwhere(np.logical_and(pmap > pthreshhigh, mask > 0))
        pyidx = ispolar[:,0]
        pxidx = ispolar[:,1]

        notpolar = np.argwhere(np.logical_and(pmap < pthreshlow, mask > 0))
        nyidx = notpolar[:,0]
        nxidx = notpolar[:,1]
        
        # Choose initial guess parameters for the diattenuation minimization
        D = 0.5
        theta = 0.
        chi = 0.
        initial_guess = (D, theta, chi)

        # Use just the region with weak polarization
        baddata = np.moveaxis(data[nxidx,nyidx,:][:,:,wave_index],1,2) #do selection for only strong polarization signals
        result = minimize(_fitfunc1, initial_guess, args=baddata)

        # Apply correction for I<->QUV cross-talk
        printc('Fitting Diattenuation Matrix',color=bcolors.OKGREEN)
        MM1a = _polmodel1(result.x[0],result.x[1], result.x[2])
        iMM1a = np.linalg.inv(MM1a)
        data1a =  np.einsum('ij,abjc->abic', iMM1a, data)

        # Destreaking correction to data might happen here

        # Choose an initial guess of those cross-talk parameters for the minimization algorithm
        theta = 0.*np.pi/180.
        delta = 0.*np.pi/180.
        initial_guess = (theta,delta)

        # Find the elliptical retardance parameters that minimize V*Q, V*U, and V*V
        if VtoQU:
            printc('Fitting Retardance Matrix',color=bcolors.OKGREEN)
            baddata = np.moveaxis(data1a[pxidx, pyidx,:][:,:,wave_index],1,2)
            result = minimize(_fitfunc2, initial_guess, args=baddata)

            # Apply correction for QU<->V cross-talk
            MM2a = _polmodel2(result.x[0],result.x[1])
            iMM2a = np.linalg.inv(MM2a)
            data2a =  np.einsum('ij,abjc->abic', iMM2a, data1a)
            
            # rotate back the Stokes Q and U signals (TBD)
            # Apply final sign correction to match original spectra
            theta = result.x[0]
            MM3a = np.array( [[1., 0.             , 0.             , 0.],
                               [0., np.cos(-theta) , np.sin(-theta) , 0.],
                               [0., -np.sin(-theta), np.cos(-theta) , 0.],
                               [0., 0.             , 0.             , 1.]], dtype='double')
            iMM3a = np.linalg.inv(MM3a)
            MMa=MM1a@MM2a@MM3a
            data3a = np.einsum('ij,abjc->abic', iMM3a, data2a)

        else:
            iMM2a = MM2a = np.array( [[1., 0., 0., 0.],
                                    [0., 1., 0., 0.],
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]], dtype='double')
            data2a =  np.einsum('ij,abjc->abic', iMM2a, data1a)
        
            # Apply final sign correction to match original spectra
            iMM3a = np.array( [[1., 0., 0., 0.],
                            [0., 1., 0., 0.],
                            [0., 0., 1., 0.],
                            [0., 0., 0., 1.]], dtype='double')
            MM3a = np.linalg.inv(iMM3a)
            MMa=MM1a@MM2a@MM3a
            data3a = np.einsum('ij,abjc->abic', iMM3a, data2a)
        
        return MM1a, MM2a, MM3a, None, None, None, data3a

def crosstalk_auto_ItoQUV(data_demod,cpos,wl,roi=np.ones((2048,2048)),verbose=0,npoints=5000,limit=0.2):
    """Get crosstalk coefficients for I to Q,U,V

    Parameters
    ----------
    data_demod: ndarray
        input data that has been demodulated
    cpos: int
        continuum position
    wl: int
        wavelength position
    roi: ndarray
        region of interest
    verbose: bool/int
        if True, plot results
    npoints: int
        number of points to use for fitting
    limit: float
        limit for Stokes I to be considered for fitting

    Returns
    -------
    ct: ndarray
        crosstalk coefficients for I to Q,U,V

    adapted from SPGPylibs
    """
    import random, statistics
    from scipy.optimize import curve_fit

    def linear(x,a,b):
        return a*x + b

    my = []
    sy = []
    
    x = data_demod[roi>0,0,cpos].flatten()
    ids = np.logical_and(x > limit, x < 1.5)
    x = x[ids].flatten()

    N = x.size
    idx = random.sample(range(N),npoints)
    mx = x[idx].mean() 
    sx = x[idx].std() 
    xp = np.linspace(x.min(), x.max(), 100)

    A = np.vstack([x, np.ones(len(x))]).T

    # I to Q
    yQ = data_demod[roi>0,1,wl].flatten()
    yQ = yQ[ids].flatten()
    my.append(yQ[idx].mean())
    sy.append(yQ[idx].std())
    cQ = curve_fit(linear,x,yQ,p0=[0,0])[0]
    pQ = np.poly1d(cQ)

    # I to U
    yU = data_demod[roi>0,2,wl].flatten()
    yU = yU[ids].flatten()
    my.append(yU[idx].mean())
    sy.append(yU[idx].std())
    cU = curve_fit(linear,x,yU,p0=[0,0])[0]
    pU = np.poly1d(cU)

    # I to V
    yV = data_demod[roi>0,3,wl].flatten()
    yV = yV[ids].flatten()
    my.append(yV[idx].mean())
    sy.append(yV[idx].std())
    cV = curve_fit(linear,x,yV,p0=[0,0])[0]
    pV = np.poly1d(cV)

    if verbose:
        
        PLT_RNG = 3
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(x[idx],yQ[idx],color='red',alpha=0.6,s=10)
        ax.plot(xp, pQ(xp), color='red', linestyle='dashed',linewidth=3.0)

        ax.scatter(x[idx],yU[idx],color='blue',alpha=0.6,s=10)
        ax.plot(xp, pU(xp), color='blue', linestyle='dashed',linewidth=3.0)

        ax.scatter(x[idx],yV[idx],color='green',alpha=0.6,s=10)
        ax.plot(xp, pV(xp), color='green', linestyle='dashed',linewidth=3.0)

        ax.set_xlim([mx - PLT_RNG * sx,mx + PLT_RNG * sx])
        ax.set_ylim([min(my) - 1.8*PLT_RNG * statistics.mean(sy),max(my) + PLT_RNG * statistics.mean(sy)])
        ax.set_xlabel('Stokes I')
        ax.set_ylabel('Stokes Q/U/V')
        ax.text(mx - 0.9*PLT_RNG * sx, min(my) - 1.4*PLT_RNG * statistics.mean(sy), 'Cross-talk from I to Q: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cQ[0],cQ[1],width=8,prec=4), style='italic',bbox={'facecolor': 'red', 'alpha': 0.1, 'pad': 1}, fontsize=15)
        ax.text(mx - 0.9*PLT_RNG * sx, min(my) - 1.55*PLT_RNG * statistics.mean(sy), 'Cross-talk from I to U: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cU[0],cU[1],width=8,prec=4), style='italic',bbox={'facecolor': 'blue', 'alpha': 0.1, 'pad': 1}, fontsize=15)
        ax.text(mx - 0.9*PLT_RNG * sx, min(my) - 1.7*PLT_RNG * statistics.mean(sy), 'Cross-talk from I to V: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cV[0],cV[1],width=8,prec=4), style='italic',bbox={'facecolor': 'green', 'alpha': 0.1, 'pad': 1}, fontsize=15)
#         fig.show()

        print('Cross-talk from I to Q: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cQ[0],cQ[1],width=8,prec=4))
        print('Cross-talk from I to U: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cU[0],cU[1],width=8,prec=4))
        print('Cross-talk from I to V: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cV[0],cV[1],width=8,prec=4))
    
#         return cQ,cU,cV, (idx,x,xp,yQ,yU,yV,pQ,pU,pV,mx,sx,my,sy)
    else:
        printc('Cross-talk from I to Q: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cQ[0],cQ[1],width=8,prec=4),color=bcolors.OKGREEN)
        printc('Cross-talk from I to U: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cU[0],cU[1],width=8,prec=4),color=bcolors.OKGREEN)
        printc('Cross-talk from I to V: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cV[0],cV[1],width=8,prec=4),color=bcolors.OKGREEN)
        ct = np.asarray((cQ,cU,cV)).T
        return ct

def CT_ItoQUV(data, ctalk_params, norm_stokes, cpos_arr, Ic_mask):
    """Apply cross talk correction from I to Q, U and V

    Parameters
    ----------
    data: ndarray
        input data to be corrected
    ctalk_params: ndarray
        cross talk parameters
    norm_stokes: bool
        if True, apply normalised offset to normed stokes
    cpos_arr: array
        array containing continuum positions
    Ic_mask: ndarray
        mask for Stokes I continuum to be used as reference

    Returns
    -------
    data: ndarray
        data with cross talk correction applied
    """
    before_ctalk_data = np.copy(data)
    data_shape = data.shape

    cont_stokes = np.ones(data_shape[-1])
    
    for scan in range(data_shape[-1]):
        cont_stokes[scan] = np.mean(data[Ic_mask[...,scan],0,cpos_arr[0],scan])
    
    for i in range(6):
                
#         stokes_i_wv_avg = np.mean(data[ceny,cenx,0,i,:], axis = (0,1))
        stokes_i_wv_avg = np.ones(data_shape[-1])
        for scan in range(data_shape[-1]):
            stokes_i_wv_avg[scan] = np.mean(data[Ic_mask[...,scan],0,i,scan])
            
        if norm_stokes:
            #if normed, applies normalised offset to normed stokes

            tmp_param = ctalk_params*np.divide(stokes_i_wv_avg,cont_stokes)

            q_slope = tmp_param[0,0,:]
            u_slope = tmp_param[0,1,:]
            v_slope = tmp_param[0,2,:]

            q_int = tmp_param[1,0,:]
            u_int = tmp_param[1,1,:]
            v_int = tmp_param[1,2,:]

            data[:,:,1,i,:] = before_ctalk_data[:,:,1,i,:] - before_ctalk_data[:,:,0,i,:]*q_slope - q_int

            data[:,:,2,i,:] = before_ctalk_data[:,:,2,i,:] - before_ctalk_data[:,:,0,i,:]*u_slope - u_int

            data[:,:,3,i,:] = before_ctalk_data[:,:,3,i,:] - before_ctalk_data[:,:,0,i,:]*v_slope - v_int
            
        else:
            #if not normed, applies raw offset cross talk correction to raw stokes counts

            tmp_param = ctalk_params[0,:,:]*np.divide(stokes_i_wv_avg,cont_stokes)

            q_slope = tmp_param[0,:]
            u_slope = tmp_param[1,:]
            v_slope = tmp_param[2,:]

            q_int = ctalk_params[1,0,:]
            u_int = ctalk_params[1,1,:]
            v_int = ctalk_params[1,2,:]

            data[:,:,1,i,:] = before_ctalk_data[:,:,1,i,:] - before_ctalk_data[:,:,0,i,:]*q_slope - q_int*stokes_i_wv_avg 

            data[:,:,2,i,:] = before_ctalk_data[:,:,2,i,:] - before_ctalk_data[:,:,0,i,:]*u_slope - u_int*stokes_i_wv_avg 

            data[:,:,3,i,:] = before_ctalk_data[:,:,3,i,:] - before_ctalk_data[:,:,0,i,:]*v_slope - v_int*stokes_i_wv_avg
    
    return data


def hot_pixel_mask(data, rows, cols, mode='median'):
    """
    Apply hot pixel mask to the data, just after cross talk to remove pixels that diverge
    
    Parameters
    ----------
    data: ndarray
        input data to be corrected
    rows: slice
        rows of the data to be corrected
    cols: slice
        columns of the data to be corrected
    mode: str
        'median' or 'mean' to apply to the data

    Returns
    -------
    data: ndarray
        data with hot pixels masked
    """
    file_loc = os.path.realpath(__file__)
    field_stop_fol = file_loc.split('src/')[0] + 'field_stop/'
    hot_pix_mask,_ = load_fits(field_stop_fol + 'bad_pixels.fits')
    hot_pix_cont,_ = load_fits(field_stop_fol + 'bad_pixels_contour.fits')
    
    s = data.shape # [y,x,p,l,s]
    
    if mode == 'median':
        func = lambda a: np.median(a,axis=0)
    elif mode == 'mean':
        func = lambda a: np.mean(a,axis=0)
    else:
        print('mode not found, input dataset not corrected')
        return data
    
    l = int(np.max(hot_pix_mask))
    
    for i in range(1,l+1):
        bad = (hot_pix_mask[rows,cols] == i)
        if np.sum(bad) > 0:
            med = (hot_pix_cont[rows,cols] == i)
            data[bad] = func(data[med])
    
    return data

    
def crosstalk_auto_VtoQU(data_demod,cpos,wl,roi=np.ones((2048,2048)),verbose=0,npoints=5000,nlevel=0.3):
    """Get crosstalk coefficients for V to Q,

    Parameters
    ----------
    data_demod: ndarray
        input data that has been demodulated
    cpos: int
        continuum position
    wl: int
        wavelength position
    roi: ndarray
        region of interest
    verbose: bool/int
        if True, plot results
    npoints: int
        number of points to use for fitting
    nlevel: float
        limit for Stokes V to be considered for fitting

    Returns
    -------
    ct: ndarray
        crosstalk coefficients for V to Q and U

    adapted from SPGPylibs
    """
    import random, statistics
    from scipy.optimize import curve_fit
    def linear(x,a,b):
        return a*x + b
    my = []
    sy = []
    
    x = data_demod[roi>0,3,cpos].flatten()
    lx = data_demod[roi>0,0,cpos].flatten()
    lv = np.abs(data_demod[roi>0,3,cpos]).flatten()
    
    ids = (lv > nlevel/100.)
    x = x[ids].flatten()

    N = x.size
    idx = random.sample(range(N),npoints)
    mx = x[idx].mean() 
    sx = x[idx].std() 
    xp = np.linspace(x.min(), x.max(), 100)

    A = np.vstack([x, np.ones(len(x))]).T

    # V to Q
    yQ = data_demod[roi>0,1,wl].flatten()
    yQ = yQ[ids].flatten()
    my.append(yQ[idx].mean())
    sy.append(yQ[idx].std())
    cQ = curve_fit(linear,x,yQ,p0=[0,0])[0]
    pQ = np.poly1d(cQ)

    # V to U
    yU = data_demod[roi>0,2,wl].flatten()
    yU = yU[ids].flatten()
    my.append(yU[idx].mean())
    sy.append(yU[idx].std())
    cU = curve_fit(linear,x,yU,p0=[0,0])[0]
    pU = np.poly1d(cU)

    if verbose:
        
        PLT_RNG = 2
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(x[idx],yQ[idx],color='red',alpha=0.6,s=10)
        ax.plot(xp, pQ(xp), color='red', linestyle='dashed',linewidth=3.0)

        ax.scatter(x[idx],yU[idx],color='blue',alpha=0.6,s=10)
        ax.plot(xp, pU(xp), color='blue', linestyle='dashed',linewidth=3.0)

        ax.set_xlim([mx - PLT_RNG * sx,mx + PLT_RNG * sx])
        ax.set_ylim([min(my) - 1.8*PLT_RNG * statistics.mean(sy),max(my) + PLT_RNG * statistics.mean(sy)])
        ax.set_xlabel('Stokes V')
        ax.set_ylabel('Stokes Q/U')
        ax.text(mx - 0.9*PLT_RNG * sx, min(my) - 1.4*PLT_RNG * statistics.mean(sy), 'Cross-talk from V to Q: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cQ[0],cQ[1],width=8,prec=4), style='italic',bbox={'facecolor': 'red', 'alpha': 0.1, 'pad': 1}, fontsize=15)
        ax.text(mx - 0.9*PLT_RNG * sx, min(my) - 1.55*PLT_RNG * statistics.mean(sy), 'Cross-talk from V to U: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cU[0],cU[1],width=8,prec=4), style='italic',bbox={'facecolor': 'blue', 'alpha': 0.1, 'pad': 1}, fontsize=15)
#         fig.show()

        print('Cross-talk from V to Q: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cQ[0],cQ[1],width=8,prec=4))
        print('Cross-talk from V to U: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cU[0],cU[1],width=8,prec=4))
    
#         return cQ,cU,cV, (idx,x,xp,yQ,yU,yV,pQ,pU,pV,mx,sx,my,sy)
    else:
        printc('Cross-talk from V to Q: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cQ[0],cQ[1],width=8,prec=4),color=bcolors.OKGREEN)
        printc('Cross-talk from V to U: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cU[0],cU[1],width=8,prec=4),color=bcolors.OKGREEN)

    ct = np.asarray((cQ,cU)).T
    return ct


def CT_VtoQU(data, ctalk_params):
    """Apply cross talk correction from V to Q, U

    Parameters
    ----------
    data: ndarray
        input data to be corrected
    ctalk_params: ndarray
        cross talk parameters

    Returns
    -------
    data: ndarray
        data with cross talk correction applied
    """
    before_ctalk_data = np.copy(data)

    for i in range(6):
        tmp_param = ctalk_params#*stokes_i_wv_avg/cont_stokes

        q_slope = tmp_param[0,0]
        u_slope = tmp_param[0,1]
        
        q_int = tmp_param[1,0]
        u_int = tmp_param[1,1]
        
        data[:,:,1,i,:] = before_ctalk_data[:,:,1,i,:] - before_ctalk_data[:,:,3,i,:]*q_slope - q_int

        data[:,:,2,i,:] = before_ctalk_data[:,:,2,i,:] - before_ctalk_data[:,:,3,i,:]*u_slope - u_int

    return data


def polarimetric_registration(data, sly, slx, hdr_arr):
    """Align the mod (pol) states 2,3,4 with state 1 for a given wavelength
    loop through all wavelengths

    Parameters
    ----------
    data: ndarray
        input data to be aligned polarimetrically
    sly: slice
        slice in y direction
    slx: slice
        slice in x direction
    hdr_arr: ndarray
        header array
    
    Returns
    -------
    data: ndarray
        data with polarimetric registration applied
    hdr_arr: ndarray
        header array with updated CAL_PREG keyword
    """

    pn = 4 
    wln = 6 
    # iterations = 3
    
    data_shape = data.shape
    data_size = data_shape[:2]

    old_data = data.copy()

    for scan in range(data_shape[-1]):
        
        shift_raw = np.zeros((2,pn*wln))
        for j in range(shift_raw.shape[1]):
            if j%pn == 0:
                pass
            else:
                ref = image_derivative(old_data[:,:,0,j//pn,scan])[sly,slx]
                temp = image_derivative(old_data[:,:,j%pn,j//pn,scan])[sly,slx]
                it = 0
                s = [1,1]
                
                while np.any(np.abs(s)>.5e-2):#for it in range(iterations):
                    sr, sc, r = SPG_shifts_FFT(np.asarray([ref,temp]))
                    s = [sr[1],sc[1]]
                    shift_raw[:,j] = [shift_raw[0,j]+s[0],shift_raw[1,j]+s[1]]
                    
                    temp = image_derivative(fft_shift(old_data[:,:,j%pn,j//pn,scan], shift_raw[:,j]))[sly,slx]

                    it += 1
                    if it ==10:
                        break
                
                print(it,'iterations shift (x,y):',round(shift_raw[1,j],3),round(shift_raw[0,j],3))
                Mtrans = np.float32([[1,0,shift_raw[1,j]],[0,1,shift_raw[0,j]]])
                data[:,:,j%pn,j//pn,scan]  = cv2.warpAffine(old_data[:,:,j%pn,j//pn,scan].astype(np.float32), Mtrans, data_size[::-1], flags=cv2.INTER_LANCZOS4)
    
        hdr_arr[scan]['CAL_PREG'] = 'y: '+str([round(shift_raw[0,i],3) for i in range(pn*wln)]) + ', x: '+str([round(shift_raw[1,i],3) for i in range(pn*wln)])
    
    del old_data

    return data, hdr_arr
    

def wavelength_registration(data, cpos_arr, sly, slx, hdr_arr, derivative = True, deconv=False):
    """Align the wavelengths, from the Stokes I image, (after demodulation), using cv2.warpAffine

    Parameters
    ----------
    data: ndarray
        input data to be aligned in wavelength
    cpos_arr: ndarray
        array of continuum positions
    sly: slice
        slice in y direction
    slx: slice
        slice in x direction
    hdr_arr: ndarray
        header array
    derivative: bool
        if True, the spatial derivative of the images is used in the correlation (Default: True)
    deconv: bool
        if True, array is deconvolved before the correlation (Default: False)

    Returns
    -------
    data: ndarray
        data with wavelength registration applied
    hdr_arr: ndarray
        header array with updated CAL_WREG keyword
    """

    pn = 4
    wln = 6
    
    if cpos_arr[0] == 5:
        l_i =    [0,1,3,4,2] # shift wl
        refl_i = [5,0,1,5,3]
        cwl = 2
    else:
        l_i =    [1,2,4,5,3] # shift wl
        refl_i = [0,1,2,0,4]
        cwl = 3
    
    new_data = data.copy()
        
    data_shape = data.shape
    data_size = data_shape[:2]
    if derivative:
        im_der = lambda x: image_derivative(x)
    else:
        im_der = lambda x: x

    for scan in range(data_shape[-1]):
        shift_stk = np.zeros((2,wln-1))
        if deconv != False:
            from sophi_hrt_pipe.PSF import fran_restore
            dat = data[sly.start-5:sly.stop+5,slx.start-5:slx.stop+5,:,:,scan].copy()
            # old_data, _ = fran_restore(dat, datetime.datetime.fromisoformat(hdr_arr[scan]['DATE-OBS']),
            #                             mask=np.ones((dat.shape[0],dat.shape[1])), gamma2=0.02, low_f=0.8, aberr_cor=False)
            old_data, _ = fran_restore(dat, datetime.datetime.fromisoformat(hdr_arr[scan]['DATE-OBS']), sly=slice(0,dat.shape[0]), slx=slice(0,dat.shape[1]),
                                        mask=np.ones((dat.shape[0],dat.shape[1])), gamma2=0, low_f=0.1, aberr_cor=False, PD_f=deconv['PD_f'], straylight_corr=deconv['straylight_correction'])
            sly, slx = slice(5,sly.stop-sly.start+5), slice(5,slx.stop-slx.start+5)
        else:
            old_data = data[...,scan].copy()

        for i,l in enumerate(l_i):

            ref = im_der(old_data[:,:,0,refl_i[i]].copy())[sly,slx]
            temp = im_der(old_data[:,:,0,l])[sly,slx]
            it = 0
            s = [1,1]
            # if l == cwl:
            #     temp = im_der(np.abs(old_data[:,:,0,l,scan]))[sly,slx]
            #     ref = im_der(np.abs((data[:,:,0,l-1,scan] + data[:,:,0,l+1,scan]) / 2))[sly,slx]
            
            while np.any(np.abs(s)>.5e-2):#for it in range(iterations):
                sr, sc, r = SPG_shifts_FFT(np.asarray([ref,temp]))
                s = [sr[1],sc[1]]
                shift_stk[:,i] = [shift_stk[0,i]+s[0],shift_stk[1,i]+s[1]]
                temp = im_der(fft_shift(old_data[:,:,0,l].copy(), shift_stk[:,i]))[sly,slx]

                it += 1
                if it == 10:
                    break
            print(it,'iterations shift (x,y):',round(shift_stk[1,i],3),round(shift_stk[0,i],3))
            
            for ss in range(pn):
                Mtrans = np.float32([[1,0,shift_stk[1,i]],[0,1,shift_stk[0,i]]])
                new_data[:,:,ss,l,scan]  = cv2.warpAffine(data[:,:,ss,l,scan].copy().astype(np.float32), Mtrans, data_size[::-1], flags=cv2.INTER_LANCZOS4)
            
            old_data[:,:,ss,l]  = cv2.warpAffine(old_data[:,:,0,l].copy().astype(np.float32), Mtrans, (old_data.shape[0],old_data.shape[1]), flags=cv2.INTER_LANCZOS4)

            # if l == cwl:
            #     ref = image_derivative(old_data[:,:,0,cpos_arr[0],scan])[sly,slx]
        
        hdr_arr[scan]['CAL_WREG'] = 'y: '+str([round(shift_stk[0,i],3) for i in range(wln-1)]) + ', x: '+str([round(shift_stk[1,i],3) for i in range(wln-1)])
    
    del old_data

    return new_data, hdr_arr    

def create_intermediate_hdr(data, hdr_interm, history_str, file_name, **kwargs):
    """add basic keywords to the intermediate file header

    Parameters
    ----------
    data: ndarray
        data array
    hdr_interm: fits header
        intermediate header from the input file
    history_str: str
        history string to be added to the header
    file_name: str
        name of the output file
    **kwargs: dict
        optional arguments: bunit, btype, DEFAULTS: bunit = DN, btype = Intensity

    Returns
    -------
    hdr: fits header
        header with updated keywords
    """
    hdr = hdr_interm.copy()

    hdr['FILENAME'] = file_name #scan_name_list[count]
    #overwrite the stokes history entry
    hdr['HISTORY'] = history_str
    #need to define in case kwargs not passed through
    b_unit = None
    b_type = None

    for arg, value in kwargs.items():
        if arg == 'bunit':
            b_unit = value
        if arg == 'btype':
            b_type = value

    #need separate, as if no kwargs, the top won't show
    if b_type is None:
        hdr['BTYPE'] = 'Intensity'
    else:
        hdr['BTYPE'] = b_type
    if b_unit is None:
        hdr['BUNIT'] = 'DN'
    else:
        hdr['BUNIT'] = b_unit

    hdr['DATAMIN'] = int(np.min(data))
    hdr['DATAMAX'] = int(np.max(data))
    hdr = data_hdr_kw(hdr, data)#add datamedn, datamean etc

    return hdr


def write_out_intermediate(data_int, hdr_interm, history_str, scan, root_scan_name, suffix, version, out_dir, **kwargs):
    """Write out intermediate files to output directory

    Parameters
    ----------
    data_int: ndarray
        data array of intermediate step to be written out
    hdr_interm: fits header
        intermediate header from the input file
    history_str: str
        history string to be added to the header
    scan: int
        scan number
    root_scan_name: str
        root file name of the intermediate file to be written
    suffix: str
        suffix to be added to the intermediate file name
    version: str
        version of the file
    out_dir: str
        output directory
    **kwargs: dict
        optional arguments: bunit, btype, DEFAULTS: bunit = DN, btype = Intensity

    Returns
    -------
    None
    """
    hdr_int = create_intermediate_hdr(data_int, hdr_interm, history_str, f'{suffix}_V{version}_{root_scan_name}.fits', **kwargs)

    with fits.open(scan) as hdu_list:
        print(f"Writing intermediate file as: {suffix}_V{version}_{root_scan_name}.fits")
        hdu_list[0].data = data_int.astype(np.float32)
        hdu_list[0].header = hdr_int #update the calibration keywords
        hdu_list.writeto(out_dir + f'{suffix}_V{version}_{root_scan_name}.fits', overwrite=True)

        
def PDProcessing(data_f, flat_f, dark_f, norm_f = True, prefilter_f = None, TemperatureCorrection = True, TemperatureConstant = 40.1225e-3, level = 'CAL2', version = 'V01', out_dir = None):   
    # from sophi_hrt_pipe.processes import apply_field_stop, hot_pixel_mask
    PD, h = get_data(data_f,True,True,True)
    
    start_row = int(h['PXBEG2']-1)
    start_col = int(h['PXBEG1']-1)
    data_size = PD.shape[1:]
    nfocus = PD.shape[0]

    rows = slice(start_row,start_row + data_size[0])
    cols = slice(start_col,start_col + data_size[1])

    if 'IMGDIRX' in h:
        header_PDdirx_exists = True
        PDdirx_flipped = str(h['IMGDIRX'])
    else:
        header_PDdirx_exists = False
        PDdirx_flipped = 'NO'
        
    PD = compare_IMGDIRX(PD,True,'YES',header_PDdirx_exists,PDdirx_flipped)

    if dark_f is not None:
        D, hD = get_data(dark_f,True,True,True)
        if 'IMGDIRX' in hD:
            header_drkdirx_exists = True
            drkdirx_flipped = str(hD['IMGDIRX'])
        else:
            header_drkdirx_exists = False
            drkdirx_flipped = 'NO'
            
        D = compare_IMGDIRX(D[np.newaxis],True,'YES',header_drkdirx_exists,drkdirx_flipped)[0]
    
        PD = (PD - D[np.newaxis,rows,cols])# / F[np.newaxis,:,:,0,5]

    if prefilter_f is not None:
        prefilter, _ = load_fits(prefilter_f)
        prefilter = prefilter[:,::-1]
    
        tunning_constant = 0.0003513 # this shouldn't change
        # temperature_constant_new = 37.625e-3 # new and more accurate temperature constant
        # temperature_constant_new = 36.46e-3 # value from HSref_wavelength = 6173.341 # this shouldn't change
        ref_wavelength = 6173.341
        Tfg = h['FGOV1PT1'] # ['FGH_TSP1']
        Volt = fits.open(data_f)[3].data['PHI_FG_voltage'][0]
        if TemperatureCorrection:
            wl = Volt * tunning_constant + ref_wavelength + TemperatureConstant*(Tfg-61)
        else:
            wl = Volt * tunning_constant + ref_wavelength
        fakePD = np.zeros((data_size[0],data_size[1],4,nfocus,1)); fakePD[:,:,0,:,0] = np.moveaxis(PD.copy(),0,-1);
        # voltagesData_arr = [np.asarray([Volt,Volt,Volt,Volt,Volt,Volt])]
        wlData_arr = [np.ones(nfocus)*wl]
        fakePD = prefilter_correction(fakePD,wlData_arr,prefilter[rows,cols],Tetalon=Tfg,TemperatureCorrection=TemperatureCorrection,TemperatureConstant=TemperatureConstant,shift=None)
        PD = np.squeeze(np.moveaxis(fakePD[:,:,0,:,0],2,0))
        
    
    # Voltage is 850, same as the continuum in the observations on the same day
    # PMP voltages are 2093 2036 as for the first polarization state in each cycle
    if flat_f is not None:
        F, hF = get_data(flat_f,True, True, True)
        if 'IMGDIRX' in hF:
            header_flatdirx_exists = True
            flatdirx_flipped = str(hF['IMGDIRX'])
        else:
            header_flatdirx_exists = False
            flatdirx_flipped = 'NO'
        
        F = compare_IMGDIRX(F,True,'YES',header_flatdirx_exists,flatdirx_flipped)
        F = stokes_reshape(F)
        wave_flat, voltagesData_flat, _, cpos_f = fits_get_sampling(flat_f,verbose = True,TemperatureCorrection=TemperatureCorrection,TemperatureConstant=TemperatureConstant)

        if norm_f:
            F = F/F[slice(1024-256,1024+256),slice(1024-256,1024+256)].mean(axis=(0,1))[np.newaxis,np.newaxis]
        else:
            F = F/F[slice(0,2048),slice(0,2048)].mean(axis=(0,1))[np.newaxis,np.newaxis]

        if prefilter_f is not None:
            TFfg = hF['FGOV1PT1']
            F = prefilter_correction(F[...,np.newaxis],[wave_flat],prefilter,Tetalon=TFfg,TemperatureCorrection=TemperatureCorrection,TemperatureConstant=TemperatureConstant,shift=None)[...,0]
        PD = PD / F[np.newaxis,rows,cols,0,cpos_f]
    
    field_stop_loc = os.path.realpath(__file__)
    field_stop_loc = field_stop_loc.split('src/')[0] + 'field_stop/'
    field_stop,_ = load_fits(field_stop_loc + 'HRT_field_stop_new.fits')
    field_stop = np.where(field_stop > 0,1,0)

    if header_PDdirx_exists:
        if PDdirx_flipped == 'YES': #should be YES for any L1 data, but mistake in processing software
            field_stop = field_stop[:,::-1] #also need to flip the flat data after dark correction


    PD *= field_stop[np.newaxis,rows,cols]

    PD = np.moveaxis(hot_pixel_mask(np.moveaxis(PD,0,-1),rows,cols),-1,0)
    
    if out_dir is not None:
        if 'CAL1' in data_f:
            temp = data_f.split('/')[-1].split('CAL1')
        else:
            temp = data_f.split('/')[-1].split('L1')
        temp[1] = temp[1].split('V')
        temp[1][1][13:]
        name = temp[0]+level+temp[1][0]+version+temp[1][1][13:]

        # add wavelength keywords
        previousKey = 'WAVEMAX'
        for i in range(1):
            newKey = f'WAVE{i+1}'
            h.set(newKey, round(wl,3), f'[Angstrom] {i+1}. wavelength of observation', after=previousKey)
            previousKey = newKey
        # add voltage keywords
        for i in range(1):
            newKey = f'VOLTAGE{i+1}'
            h.set(newKey, int(Volt), f'[Volt] {i+1}. voltage of observation', after=previousKey)
            previousKey = newKey
        # add continuum position keywords
        # newKey = 'CONTPOS'
        # h.set(newKey, int(cpos+1), 'continuum position (1: blue, 6: red)', after=previousKey)
        # previousKey = newKey
        # add voltage tuning constant keywords
        newKey = 'TUNCONS'
        h.set(newKey, tunning_constant, f'[Angstrom / Volt] voltage tuning constant', after=previousKey)
        previousKey = newKey
        # add temperature tuning constant keywords
        newKey = 'TEMPCONS'
        if TemperatureCorrection:
            h.set(newKey, TemperatureConstant, '[Angstrom / Kelvin] temperature constant', after=previousKey)
        else:
            h.set(newKey, 0, '[Angstrom / Kelvin] temperature constant', after=previousKey)
        previousKey = newKey

        # dark file
        h['CAL_DARK'] = dark_f
        # flat file
        h['CAL_FLAT'] = flat_f
        if prefilter_f is not None:
            h['CAL_PRE'] = prefilter_f
        # change NAXIS1, 2
        h.comments['NAXIS1'] = 'number of pixels on the x axis'
        h.comments['NAXIS2'] = 'number of pixels on the y axis'
        
        with fits.open(data_f) as hdr:
            hdr[0].data = PD.astype('float32')
            hdr[0].header = h
            
            hdr.writeto(out_dir+name, overwrite=True)
    
    return PD

def solarRotation(hdr):
    # vrot from hathaway et al., 2011, values in deg/day
    # proper vlos projection without thetarho ~ 0 approximation from Schuck et al., 2016
    X = ccd2HGS(hdr)
    HPCx, HPCy, HPCd = ccd2HPC(hdr)
    thetarho = np.arctan(np.sqrt(np.cos(HPCy*u.arcsec)**2*np.sin(HPCx*u.arcsec)**2+np.sin(HPCy*u.arcsec)**2) / 
                      (np.cos(HPCy*u.arcsec)*np.cos(HPCx*u.arcsec)))
    psi = np.arctan(-(np.cos(HPCy*u.arcsec)*np.sin(HPCx*u.arcsec)) / np.sin(HPCy*u.arcsec))
    psi[np.logical_and(HPCy>=0,HPCx<0)] += 0 * u.rad
    psi[np.logical_and(HPCy<0,HPCx<0)] += np.pi * u.rad
    psi[np.logical_and(HPCy<0,HPCx>=0)] += np.pi * u.rad
    psi[np.logical_and(HPCy>=0,HPCx>=0)] += 2*np.pi * u.rad
    
    a = (14.437 * u.deg/u.day).to(u.rad/u.s); 
    b = (-1.48 * u.deg/u.day).to(u.rad/u.s); 
    c = (-2.99 * u.deg/u.day).to(u.rad/u.s); 
    vrot = (a + b*np.sin(X[1]*u.deg)**2 + c*np.sin(X[1]*u.deg)**4)*np.cos(X[1]*u.deg)* hdr['RSUN_REF'] * u.m/u.rad
    B0 = hdr['HGLT_OBS']*u.deg
    THETA = (X[1])*u.deg # lat
    PHI = (X[2]-hdr['HGLN_OBS'])*u.deg # lon
    It = -np.cos(B0)*np.sin(PHI)*np.cos(thetarho) + \
         (np.cos(PHI)*np.sin(psi)-np.sin(B0)*np.sin(PHI)*np.cos(psi))*np.sin(thetarho)
    vlos = -(vrot) * It
    
    return vlos.value

def SCVelocityResidual(hdr,wlcore):
    # s/c velocity signal (considering line shift compensation)
#     X = ccd2HGS(hdr)
    HPCx, HPCy, HPCd = ccd2HPC(hdr)
    thetarho = np.arctan(np.sqrt(np.cos(HPCy*u.arcsec)**2*np.sin(HPCx*u.arcsec)**2+np.sin(HPCy*u.arcsec)**2) / 
                      (np.cos(HPCy*u.arcsec)*np.cos(HPCx*u.arcsec)))
    psi = np.arctan(-(np.cos(HPCy*u.arcsec)*np.sin(HPCx*u.arcsec)) / np.sin(HPCy*u.arcsec))
    psi[np.logical_and(HPCy>=0,HPCx<0)] += 0 * u.rad
    psi[np.logical_and(HPCy<0,HPCx<0)] += np.pi * u.rad
    psi[np.logical_and(HPCy<0,HPCx>=0)] += np.pi * u.rad
    psi[np.logical_and(HPCy>=0,HPCx>=0)] += 2*np.pi * u.rad

    vsc = hdr['OBS_VW']*np.sin(thetarho)*np.sin(psi) - hdr['OBS_VN']*np.sin(thetarho)*np.cos(psi) + hdr['OBS_VR']*np.cos(thetarho)
    c = 299792.458
    wlref = 6173.341
    vsc_compensation = (wlcore-wlref)/wlref*c*1e3
    
    return vsc.value - vsc_compensation

def meridionalFlow(hdr):
    # from hathaway et al., 2011, values in m/s
    # proper vlos projection without thetarho ~ 0 approximation from Schuck et al., 2016
    X = ccd2HGS(hdr)
    HPCx, HPCy, HPCd = ccd2HPC(hdr)
    thetarho = np.arctan(np.sqrt(np.cos(HPCy*u.arcsec)**2*np.sin(HPCx*u.arcsec)**2+np.sin(HPCy*u.arcsec)**2) / 
                      (np.cos(HPCy*u.arcsec)*np.cos(HPCx*u.arcsec)))
    psi = np.arctan(-(np.cos(HPCy*u.arcsec)*np.sin(HPCx*u.arcsec)) / np.sin(HPCy*u.arcsec))
    psi[np.logical_and(HPCy>=0,HPCx<0)] += 0 * u.rad
    psi[np.logical_and(HPCy<0,HPCx<0)] += np.pi * u.rad
    psi[np.logical_and(HPCy<0,HPCx>=0)] += np.pi * u.rad
    psi[np.logical_and(HPCy>=0,HPCx>=0)] += 2*np.pi * u.rad
    
    d = 29.7 * u.m/u.s; e = -17.7 * u.m/u.s; 
    vmer = (d*np.sin(X[1]*u.deg) + e*np.sin(X[1]*u.deg)**3)*np.cos(X[1]*u.deg)
    B0 = hdr['HGLT_OBS']*u.deg
    THETA = (X[1])*u.deg
    PHI = (X[2]-hdr['HGLN_OBS'])*u.deg
    It = (np.sin(B0)*np.cos(THETA) - np.cos(B0)*np.cos(PHI)*np.sin(THETA))*np.cos(thetarho) - \
        (np.sin(PHI)*np.sin(THETA)*np.sin(psi) + \
        (np.sin(B0)*np.cos(PHI)*np.sin(THETA) + np.cos(B0)*np.cos(THETA))*np.cos(psi))*np.sin(thetarho)
    
    vlos = (-vmer) * It
    
    return vlos.value

def SCGravitationalRedshift(hdr):
    # ok
    # gravitational redshift (theoretical) from a distance dsun from the sun
    dsun = hdr['DSUN_OBS'] # m
    c = 299792.458e3 # m/s
    Rsun = hdr['RSUN_REF'] # m
    Msun = 1.9884099e30 # kg
    G = 6.6743e-11 # m3/kg/s2
    vg = G*Msun/c * (1/Rsun - 1/dsun)
    
    return vg

def CavityMapComputation(filen,out_name=None,nc=32,TemperatureCorrection=True, TemperatureConstant = 40.1225e-3,prefilter_f=None,solar_rotation=True):
    """
    Cavity Map computation from flat field.
    This function returns the Cavity errors in \AA at each polarimetric modulation.
    It requires multiprocess package
    
    INPUT
    filen (str): file name of the flat field
    out_name: name of the output file. If None, no output file is saved. Header from the parent flat field + some changes (Default: None)
    nc (int): number of cores to be used for parallel computing (Default: 32)
    TemperatureCorrection (bool): if True, wavelengths are corrected for the etalon temperature (Default: True)
    TemperatureConstant (float): value of the temperature tuning constant to be used when TemperatureConstant is True (Default: 36.46e-3 mA/K)
    prefilter_f: file name of the prefilter. If None, no prefilter correction is applied (Default: None)
    solar_rotation (bool): if True, Doppler shift due to solar rotation is removed from the cavity (Default: True)
    
    OUTPUT
    CM (array): Cavity Map array. Units are \AA. Shape: (4,2048,2048)
    """
    
    def gausfit_1profile(profile, x, center=False, out_value=0, show=False, weight=True):
        def _gaus(x,a,x0,sigma):
            return a*np.exp(-(x-x0)**2/(2*sigma**2))

        def _gaussian_fit(a,xx=None,weight=True, show=True):
            from scipy import optimize
            if xx is None:
                print('x not given')
                xx=np.arange(np.size(a))
            y=a
            p0=[max(y),xx[np.argmax(y)],np.sqrt(sum(y * (xx - xx[np.argmax(y)])**2) / sum(y))]#,np.min(y)]
            if weight == True:
                sigma = np.abs(np.linspace(-np.size(xx)//2,+np.size(xx)//2,np.size(xx)))
            elif weight == False:
                sigma = None
            else:
                sigma = weight

            p,cov=optimize.curve_fit(_gaus,xx,y,p0=p0,
                                     sigma=sigma,
                                     absolute_sigma=True)
            if show:
                plt.figure()
                plt.plot(xx,a,'k.',alpha=.2)
                plt.plot(xx,_gaus(xx,*p),'r.')
            return p

        y = -profile; y -= y.min(); xx = x
        dd = 2
        if center:
            ind = y.argmax(); y = y[ind-dd:ind+dd+1]; xx = np.asarray(x[ind-dd:ind+dd+1].copy())
        try:
            p = _gaussian_fit(y,xx,weight=weight,show=show)
            v = p[1]
        except:
            v = out_value
        return v

    def Iprofile_gaus_parallel(cube,x, nc = 32, out_value = 0, center = False):
        import multiprocess as mp
        import time

        def f(row,x,center,out_value):
            c = []
            for r in row:
                c += [gausfit_1profile(r,x,center,out_value,False,True)]
            return c

    #     print("Number of processors: ", mp.cpu_count())
        if nc > mp.cpu_count():
            print('WARNING: Number of processor greater than the maximum: st to half maximum')
            nc = int(mp.cpu_count()/2)

        ny = cube.shape[0]
        nx = cube.shape[1]

        dd = 2

        N = nx*ny
        pool = mp.Pool(nc)

        t0 = time.time()
        CM = pool.starmap(f, [(row,x,center,out_value) for row in cube])
        t1 = time.time()
        print('CM computation time:',np.round(t1-t0,1),'s')

        CM = np.asarray(CM,dtype=np.float32)

        return CM

    def CMvlos(hdr,wlcore):
        # updated on 6/3/2023
#         X = ccd2HGS(hdr)
#         a = 2.894e-6 * u.rad/u.s; b = -0.428e-6 * u.rad/u.s; c = -0.370e-6 * u.rad/u.s; 
#         vrot = (a + b*np.sin(X[1]*u.deg)**2 + c*np.sin(X[1]*u.deg)**4)*np.cos(X[1]*u.deg)* 695700000. * u.m/u.rad
#         vlos = (vrot)*np.sin((X[2]-hdr['HGLN_OBS'])*u.deg)*np.cos(hdr['CRLT_OBS']*u.deg)
        
        vrot = solarRotation(hdr)
        vmer = meridionalFlow(hdr)
        vgr = SCGravitationalRedshift(hdr)
        vsc = SCVelocityResidual(hdr,wlcore)

        c = 299792.458
        wlref = 6173.341
        wlvlos = ((vrot+vmer+vgr+vsc)*1e-3*wlref/c)

        return wlvlos

    import warnings, datetime
    from scipy.optimize import OptimizeWarning
    warnings.filterwarnings("ignore", category=OptimizeWarning)
    hh = fits.open(filen)
    idx = np.where(hh[8].data['PHI_PROC_operation']=='PROC_MEAN')[0]+1
    values = hh[8].data['PHI_PROC_scalar1'][idx]*0.125
    wl, v, _, cpos = fits_get_sampling(filen,TemperatureCorrection = TemperatureCorrection, TemperatureConstant=TemperatureConstant, verbose=False)
    if cpos == 0:
        values /= values[:4].mean()
    else:
        values /= values[-4:].mean()
    flat = hh[0].data
    flat *= values[:,np.newaxis,np.newaxis]
    flat = stokes_reshape(flat)
    
    if prefilter_f is not None:
        print("Prefilter correction")
        prefilter = fits.getdata(prefilter_f)[:,::-1]
        flat = prefilter_correction(flat.copy()[...,np.newaxis],[wl],prefilter,TemperatureCorrection=TemperatureCorrection,TemperatureConstant=TemperatureConstant)[...,0]
    
    CM = np.zeros((4,flat.shape[0],flat.shape[1]))
    for p in range(4):
        print(f"Cavity Map computation on polarization modulation {p+1}/4")
        cube = flat[:,:,p,:] / flat[:,:,p,cpos].mean()
        x = wl
        CM[p] = Iprofile_gaus_parallel(cube,x,nc = nc, out_value = 0, center = False)
    CM -= x[cpos-3]
    
    if solar_rotation:
        print("Removing signal of the solar rotation according to WCS Keywords in the flat header")
        rotation = CMvlos(hh[0].header,wl[cpos-3])
        CM -= rotation
    
    print("Saving Cavity Maps")
    ntime = datetime.datetime.now()
    
    if out_name is not None:
        with fits.open(filen) as hdr:
            hdr[0].data = CM.astype(np.float32)
            hdr[0].header['SUBJECT'] = 'CAVITY MAP'
            hdr[0].header['LEVEL'] = 'CAL'
            hdr[0].header['BTYPE'] = 'Wavelength Shift'
            hdr[0].header['BUNIT'] = '\AA'
            hdr[0].header['DATE'] = ntime.strftime("%Y-%m-%dT%H:%M:%S")
            hdr[0].header['FILENAME'] = out_name.split('/')[-1]

            hdr[0].header['HISTORY'] = 'Cavity Map computed from flat field '+filen.split('/')[-1]
            if solar_rotation:
                hdr[0].header['HISTORY'] = 'Solar rotation removed from the cavity'
                hdr[0].header['HISTORY'] = 'Parameters: a = 2.894e-6 * u.rad/u.s; b = -0.428e-6 * u.rad/u.s; c = -0.370e-6 * u.rad/u.s; '
                hdr[0].header['HISTORY'] = 'Parameters from Hathaway et al. 2011, LoS reprojection from Schuck et al. 2016, Solar Rotation + Merdional Flow + Gravitational Redshift + S/C velocity offset '
            hdr.writeto(out_name,overwrite=True)

    return CM

