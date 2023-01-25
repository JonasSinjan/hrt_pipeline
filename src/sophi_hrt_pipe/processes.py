import numpy as np
from scipy.ndimage import gaussian_filter
from operator import itemgetter
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
    k = ['CAL_FLAT','CAL_USH','SIGM_USH',
    'CAL_PRE','CAL_GHST','CAL_PREG','CAL_REAL',
    'CAL_CRT0','CAL_CRT1','CAL_CRT2','CAL_CRT3','CAL_CRT4','CAL_CRT5',
    'CAL_CRT6','CAL_CRT7','CAL_CRT8','CAL_CRT9',
    'CAL_WREG','CAL_NORM','CAL_FRIN','CAL_PSF','CAL_IPOL',
    'CAL_SCIP','RTE_MOD','RTE_SW','RTE_ITER','VERS_CAL']

    v = [0,' ',' ',
    ' ','None ','None','NA',
    0,0,0,0,0,0,
    0,0,0,0,
    'None',' ','NA','NA',' ',
    'None',' ',' ',4294967295, hdr_arr[0]['VERS_SW'][1:4]]

    c = ['Onboard calibrated for gain table','Unsharp masking correction','Sigma for unsharp masking [px]',
    'Prefilter correction (DID/file)','Ghost correction (name + version of module)',
         'Polarimetric registration','Prealigment of images before demodulation',
    'cross-talk from I to Q (slope)','cross-talk from I to Q (offset)','cross-talk from I to U (slope)','cross-talk from I to U (offset)','cross-talk from I to V (slope)','cross-talk from I to V (offset)',
    'cross-talk from V to Q (slope)','cross-talk from V to Q (offset)','cross-talk from V to U (slope)','cross-talk from V to U (offset)','Wavelength Registration',
    'Normalization (normalization constant PROC_Ic)','Fringe correction (name + version of module)','PSF deconvolution','Onboard calibrated for instrumental polarization',
    'Onboard scientific data analysis','Inversion mode','Inversion software','Number RTE inversion iterations', 'Version of calibration pack']

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


def load_and_process_flat(flat_f, accum_scaling, bit_conversion, scale_data, header_imgdirx_exists, imgdirx_flipped, cpos_arr) -> np.ndarray:
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

    flat_pmp_temp = str(header_flat['HPMPTSP1'])

    print(f"Flat PMP Temperature Set Point: {flat_pmp_temp}")

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

    return flat


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
        norm_fac = np.mean(flat[ceny,cenx, :, :], axis = (0,1))[np.newaxis, np.newaxis, ...]  #mean of the central 1k x 1k
        flat /= norm_fac

        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        printc(f"------------- Normalising flat time: {np.round(time.perf_counter() - start_time,3)} seconds",bcolors.OKGREEN)
        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        
        return flat

    except Exception:
        printc("ERROR, Unable to normalise the flat field", color=bcolors.FAIL)


def demod_hrt(data, pmp_temp, verbose = True) -> np.ndarray:
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
        printc("Demodulation Matrix for PMP TEMP of {pmp_temp} deg is not available", color = bcolors.FAIL)
    if verbose:
        printc(f'Using a constant demodulation matrix for a PMP TEMP of {pmp_temp} deg, rotated by {HRT_MOD_ROTATION_ANGLE} deg',color = bcolors.OKGREEN)
    
    demod_data = demod_data.reshape((4,4))
    shape = data.shape
    demod = np.tile(demod_data, (shape[0],shape[1],1,1))

    if data.ndim == 5:
        # if data array has more than one scan
        data = np.moveaxis(data,-1,0) #moving number of scans to first dimension

        data = np.matmul(demod,data)
        data = np.moveaxis(data,0,-1) #move scans back to the end
    
    elif data.ndim == 4:
        # if data has one scan
        data = np.matmul(demod,data)
    
    return data, demod


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
        options are 'QUV', 'UV', 'V'
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

    if clean_mode == "QUV":
        start_clean_pol = 1
        if verbose:
            print("Unsharp Masking Q,U,V")
        
    elif clean_mode == "UV":
        start_clean_pol = 2
        if verbose:
            print("Unsharp Masking U,V")
        
    elif clean_mode == "V":
        start_clean_pol = 3
        if verbose:
            print("Unsharp Masking V")
        
    if clean_f == 'blurring':
        blur = lambda a: gaussian_filter(a,sigma)
    elif clean_f == 'fft':
        x = np.fft.fftfreq(2048,1)
        fftgaus2d = np.exp(-2*np.pi**2*(x-0)**2*sigma**2)[:,np.newaxis] * np.exp(-2*np.pi**2*(x-0)**2*sigma**2)[np.newaxis]
        blur = lambda a : (np.fft.ifftn(fftgaus2d*np.fft.fftn(a.copy()))).real
    
    for pol in range(start_clean_pol,pol_end):

        for wv in wv_range: #not the continuum

            a = np.copy(np.clip(flat_demod[:,:,pol,wv], -0.02, 0.02))
            b = a - blur(a)
            c = a - b

            new_demod_flats[:,:,pol,wv] = c

    invM = np.linalg.inv(demodM)

    flat_cleaned = np.matmul(invM, new_demod_flats*norm_factor)

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


def prefilter_correction(data,wave_axis_arr,prefilter,prefilter_voltages = None, TemperatureCorrection=False):
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
    temperatureCorrection: bool
        apply temperature correction to prefilter data, DEFAULT = False

    Returns
    -------
    data: ndarray
        prefilter corrected data

    adapted from SPGPylibs
    """
    def _get_v1_index1(x):
        index1, v1 = min(enumerate([abs(i) for i in x]), key=itemgetter(1))
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
        temperature_constant_old = 40.323e-3 # old temperature constant, still used by Johann
        temperature_constant_new = 37.625e-3 # new and more accurate temperature constant
        Tfg = 66 # FG was at 66 deg during e2e calibration
        tunning_constant = 0.0003513 # this shouldn't change
        
        ref_wavelength = 6173.341 # this shouldn't change
        prefilter_wave = prefilter_voltages * tunning_constant + ref_wavelength + temperature_constant_new*(Tfg-61) - 0.002 # JH ref
        
        # ref_wavelength = round(6173.072 - (-1300*tunning_constant),3) # 6173.529. 0 level was different during e2e test
        # prefilter_wave = prefilter_voltages * tunning_constant + ref_wavelength # + temperature_constant_new*(Tfg-61)
       
    else:
        tunning_constant = 0.0003513
        ref_wavelength = 6173.341 # this shouldn't change
        prefilter_wave = prefilter_voltages * tunning_constant + ref_wavelength
    
    data_shape = data.shape
    
    for scan in range(data_shape[-1]):

        wave_list = wave_axis_arr[scan]
        
        for wv in range(6):

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
    limit: float
        limit for Stokes I to be considered for fitting

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
    

def wavelength_registration(data, cpos_arr, sly, slx, hdr_arr):
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
        l_i = [0,1,3,4,2] # shift wl
        cwl = 2
    else:
        l_i = [1,2,4,5,3] # shift wl
        cwl = 3
    
    old_data = data.copy()

    data_shape = data.shape
    data_size = data_shape[:2]
    
    for scan in range(data_shape[-1]):
        shift_stk = np.zeros((2,wln-1))
        ref = image_derivative(old_data[:,:,0,cpos_arr[0],scan])[sly,slx]
        
        for i,l in enumerate(l_i):
            temp = image_derivative(old_data[:,:,0,l,scan])[sly,slx]
            it = 0
            s = [1,1]
            if l == cwl:
                temp = image_derivative(np.abs(old_data[:,:,0,l,scan]))[sly,slx]
                ref = image_derivative(np.abs((data[:,:,0,l-1,scan] + data[:,:,0,l+1,scan]) / 2))[sly,slx]
            
            while np.any(np.abs(s)>.5e-2):#for it in range(iterations):
                sr, sc, r = SPG_shifts_FFT(np.asarray([ref,temp]))
                s = [sr[1],sc[1]]
                shift_stk[:,i] = [shift_stk[0,i]+s[0],shift_stk[1,i]+s[1]]
                temp = image_derivative(fft_shift(old_data[:,:,0,l,scan].copy(), shift_stk[:,i]))[sly,slx]

                it += 1
                if it == 10:
                    break
            print(it,'iterations shift (x,y):',round(shift_stk[1,i],3),round(shift_stk[0,i],3))
            
            for ss in range(pn):
                Mtrans = np.float32([[1,0,shift_stk[1,i]],[0,1,shift_stk[0,i]]])
                data[:,:,ss,l,scan]  = cv2.warpAffine(old_data[:,:,ss,l,scan].copy().astype(np.float32), Mtrans, data_size[::-1], flags=cv2.INTER_LANCZOS4)

            if l == cwl:
                ref = image_derivative(old_data[:,:,0,cpos_arr[0],scan])[sly,slx]
        
        hdr_arr[scan]['CAL_WREG'] = 'y: '+str([round(shift_stk[0,i],3) for i in range(wln-1)]) + ', x: '+str([round(shift_stk[1,i],3) for i in range(wln-1)])
    
    del old_data

    return data, hdr_arr
    

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
        if arg is 'btype':
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


def write_out_intermediate(data_int, hdr_interm, history_str, scan, root_scan_name, suffix, out_dir, **kwargs):
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
    out_dir: str
        output directory
    **kwargs: dict
        optional arguments: bunit, btype, DEFAULTS: bunit = DN, btype = Intensity

    Returns
    -------
    None
    """
    hdr_int = create_intermediate_hdr(data_int, hdr_interm, history_str, f'{root_scan_name}_{suffix}.fits', **kwargs)

    with fits.open(scan) as hdu_list:
        print(f"Writing intermediate file as: {root_scan_name}_{suffix}.fits")
        hdu_list[0].data = data_int.astype(np.float32)
        hdu_list[0].header = hdr_int #update the calibration keywords
        hdu_list.writeto(out_dir + root_scan_name + f'_{suffix}.fits', overwrite=True)