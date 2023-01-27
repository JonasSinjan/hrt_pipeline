from posix import listdir
import numpy as np 
import os.path
from astropy.io import fits
import time
import datetime
from operator import itemgetter
import json
import matplotlib.pyplot as plt
from numpy.core.numeric import True_
from scipy.ndimage import binary_dilation, generate_binary_structure
import cv2

from sophi_hrt_pipe.utils import *
from sophi_hrt_pipe.processes import *
from sophi_hrt_pipe.inversions import *
from sophi_hrt_pipe.PSF import *

def phihrt_pipe(input_json_file):

    '''
    PHI-HRT data reduction pipeline
    1. read in science data (+scaling) (one or multiple files)
    2. read in flat field (+scaling) - accepts only one flat field fits file
    3. read in dark field (+scaling)
    4. prefilter correction
    5. apply dark field (to only science - assumes flat is dark fielded)
    6. clean flat field with unsharp masking (Stokes QUV, UV or V)
    7. normalise flat field
    8. apply flat field
    9. apply field stop
    10. apply hot pixels mask
    11. polarimetric registration
    12. demodulate with const demod matrix <br>
            a) option to output demod to fits file <br>
    13. normalise to quiet sun
    14. calibration <br>
            a) ItoQUV cross talk correction <br>
            b) VtoQU cross talk correction <br>
    15. wavelengths registration
    16. PSF deconvolution
    17. rte inversion with cmilos <br>
            a) output rte data products to fits files <br>

    Parameters
    ----------
        Input:
    data_f : list or string
        list containing paths to fits files of the raw HRT data OR string of path to one file  - must have last 10 characters before .fits as the DID - for naming purposes of output files
    dark_f : string, DEFAULT ''
        Fits file of a dark file (ONLY ONE FILE)
    flat_f : string, DEFAULT ''
        Fits file of a HRT flatfield (ONLY ONE FILE)

    ** Options:
    L1_input: bool, DEFAULT True
        ovverides scale_data, bit_conversion, and accum_scaling, so that correct scaling for L1 data applied
    scale_data: bool, DEFAULT True
        performs the accumulation scaling + conversion for flat and science (only FALSE for commissioning data)
    bit_conversion: bool, DEFAULT True
        divides the scan + flat by 256 to convert from 24.8bit to 32bits
    norm_f: bool, DEFAULT: True
        to normalise the flat fields before applying
    clean_f: str, DEFAULT: None
        clean the flat field with unsharp masking, accepted values = ['blurring','fft']
    sigma: int, DEFAULT: 59
        sigma of the gaussian convolution used for unsharp masking if clean_f == 'blurring', 'fft'
    clean_mode: str, DEFAULT: "V"
        The polarisation states of the flat field to be unsharp masked, options are "V", "UV" and "QUV"
    flat_states: int, DEFAULT: 24
        Number of flat fields to be applied, options are 4 (one for each pol state), 6 (one for each wavelength), 24 (one for each image)
    prefilter_f: str, DEFAULT None
        file path location to prefilter fits file, apply prefilter correction
    flat_c: bool, DEFAULT: True
        apply flat field correction
    dark_c: bool, DEFAULT: True
        apply dark field correction
    fs_c: bool, DEFAULT True
        apply HRT field stop
    ghost_c: bool, DEFAULT True
        apply HRT field stop and avoid ghost region in CrossTalk parameters computation
    iss_off: bool, DEFAULT False
        if True, registration between frames and V to Q,U correction is applied
    demod: bool, DEFAULT: True
        apply demodulate to the stokes
    norm_stokes: bool, DEFAULT: True
        normalise the stokes vector to the quiet sun (I_continuum)
    out_dir : string, DEFUALT: './'
        directory for the output files
    out_stokes_file: bool, DEFAULT: False
        output file with the stokes vectors to fits file
    out_stokes_filename: str, DEFAULT = None
        if None, takes last 10 characters of input scan filename (assumes its a DID), change if want other name
    ItoQUV: bool, DEFAULT: False 
        apply I -> Q,U,V correction
    VtoQU: bool, DEFAULT: False 
        apply V -> Q,U correction
    rte: str, DEFAULT: False 
        invert using cmilos, options: 'RTE' for Milne Eddington Inversion, 'CE' for Classical Estimates, 'CE+RTE' for combined
    out_rte_filename: str, DEFAULT = ''
        if '', takes last 10 characters of input scan filename (assumes its a DID), change if want other name
    out_intermediate: bool, DEFAULT = False
        if True, dark corrected, flat corrected, prefilter corrected and demodulated data will be saved
    p_milos: bool, DEFAULT = True
        if True, will execute the RTE inversion using the parallel version of the CMILOS code on 16 processors
    Returns
    -------
    data: numpy array
        stokes vector
    flat: numpy array
        flat field 

    References
    ----------
    SPGYlib

    '''
    version = 'V1.6 January 6th 2023'

    printc('--------------------------------------------------------------',bcolors.OKGREEN)
    printc('PHI HRT data reduction software  ',bcolors.OKGREEN)
    printc('Version: '+version,bcolors.OKGREEN)
    printc('--------------------------------------------------------------',bcolors.OKGREEN)

    #-----------------
    # READ INPUT JSON
    #-----------------
    
    input_dict = json.load(open(input_json_file))

    try:
        #input data
        data_f = input_dict['data_f']
        flat_f = input_dict['flat_f']
        dark_f = input_dict['dark_f']

        #input/output type + scaling
        L1_input = input_dict['L1_input']
        scale_data = input_dict['scale_data']
        accum_scaling = input_dict['accum_scaling']
        bit_conversion = input_dict['bit_conversion']

        #reduction
        dark_c = input_dict['dark_c']
        flat_c = input_dict['flat_c']
        if 'TemperatureCorrection' not in input_dict: #if FG != 61 deg - will correct wavelengths
            TemperatureCorrection = False
        else:
            TemperatureCorrection = input_dict['TemperatureCorrection']
        norm_f = input_dict['norm_f']
        clean_f = input_dict['clean_f']
        sigma = input_dict['sigma']
        clean_mode = input_dict['clean_mode']
        flat_states = input_dict['flat_states']
        prefilter_f = input_dict['prefilter_f']
        fs_c = input_dict['fs_c']
        iss_off = input_dict['iss_off']
        demod = input_dict['demod']
        norm_stokes = input_dict['norm_stokes']
        ItoQUV = input_dict['ItoQUV']
        VtoQU = input_dict['VtoQU']
        PSForbit = input_dict['PSForbit']  
        PSFaberr = input_dict['PSFaberr']  

        #is PSForbit is 'perihelion' or '0.5'
        if PSForbit is not False:
            PSFstokes = True
        else:
            PSFstokes = False

        ghost_c = input_dict['ghost_c']  #20211116
        cavity_f = input_dict['cavity_f']
        rte = input_dict['rte']
        out_intermediate = input_dict['out_intermediate']  #20211116
        pymilos_opt = input_dict['pymilos']
        
        #output dir/filenames
        out_dir = input_dict['out_dir']
        out_stokes_file = input_dict['out_stokes_file']
        out_stokes_filename = input_dict['out_stokes_filename']
        out_rte_filename = input_dict['out_rte_filename']

        if 'config' not in input_dict:
            config = True
        else:
            config = input_dict['config']

        #standard harcoded options/backup - do not change
        hot_px_mask = True
        avg_stokes_before_rte = False
        if 'vers' not in input_dict:
            # vrs = '01'
            vrs = time.strftime('%Y%m%d%H%M')
        else:
            vrs = input_dict['vers']
            if len(vrs) != 2:
                printc("WARNING: Version string is larger than 2 digits",color=bcolors.WARNING)
        #behaviour if clean mode is set to None (null in json)
        if clean_mode is None:
            clean_mode = "V" 
            
    except Exception as e:
        print(f"Missing key(s) in the input config file: {e}")
        raise KeyError
    
    overall_time = time.perf_counter()

    if L1_input:
        #print("L1_input param set to True - Assuming L1 science data")
        accum_scaling = True 
        bit_conversion = True
        scale_data = True

    #-----------------
    # READ DATA
    #-----------------

    print(" ")
    printc('-->>>>>>> Reading Data',color=bcolors.OKGREEN) 

    start_time = time.perf_counter()

    if isinstance(data_f, str):
        data_f = [data_f]

    if isinstance(data_f, list):
        #if the data_f contains several scans
        printc(f'Input contains {len(data_f)} scan(s)',color=bcolors.OKGREEN)
        
        number_of_scans = len(data_f)

        data_arr = [0]*number_of_scans
        hdr_arr = [0]*number_of_scans

        wave_axis_arr = [0]*number_of_scans
        cpos_arr = [0]*number_of_scans
        voltagesData_arr = [0]*number_of_scans
        tuning_constant_arr = [0]*number_of_scans

        for scan in range(number_of_scans):
            data_arr[scan], hdr_arr[scan] = get_data(data_f[scan], scaling = accum_scaling, bit_convert_scale = bit_conversion, scale_data = scale_data)

            wave_axis_arr[scan], voltagesData_arr[scan], tuning_constant_arr[scan], cpos_arr[scan] = fits_get_sampling(data_f[scan], TemperatureCorrection = TemperatureCorrection, verbose = True)

            if 'IMGDIRX' in hdr_arr[scan] and hdr_arr[scan]['IMGDIRX'] == 'YES':
                print(f"This scan has been flipped in the Y axis to conform to orientation standards. \n File: {data_f[scan]}")

        #--------
        # test if the scans have different sizes
        #--------

        check_size(data_arr)

        #--------
        # test if the scans have different continuum wavelength_positions
        #--------

        check_cpos(cpos_arr)

        #--------
        # test if the scans have different pmp temperatures
        #--------

        pmp_temp = check_pmp_temp(hdr_arr)

        #so that data is [24,y,x,scans]
        data = np.stack(data_arr, axis = -1)

        print(f"Data shape is {data.shape}")

        #--------
        # test if the scans have same IMGDIRX keyword
        #--------
    
        header_imgdirx_exists, imgdirx_flipped = check_IMGDIRX(hdr_arr)
    
    else:
        printc("ERROR, data_f argument is neither a string nor list containing strings: {} \n Ending Process",data_f,color=bcolors.FAIL)
        exit()

    data_shape = data.shape

    #converting to [y,x,pol,wv,scans]

    data = stokes_reshape(data)
    
    data_size = data.shape[:2]
    
    #enabling cropped datasets, so that the correct regions of the dark field and flat field are applied
    print("Data reshaped to: ", data.shape)

    diff = 2048-data_size[0] #handling 0/2 errors
    
    if np.abs(diff) > 0:
        
        printc("WARNING: Dataset is cropped. Cropping will be considered the same for all the data", color=bcolors.WARNING)
        start_row = int(hdr_arr[0]['PXBEG2']-1)
        start_col = int(hdr_arr[0]['PXBEG1']-1)
        
    else:
        start_row, start_col = 0, 0
    
    rows = slice(start_row,start_row + data_size[0])
    cols = slice(start_col,start_col + data_size[1])
    ceny = slice(data_size[0]//2 - data_size[0]//4, data_size[0]//2 + data_size[0]//4)
    cenx = slice(data_size[1]//2 - data_size[1]//4, data_size[1]//2 + data_size[1]//4)

    for hdr in hdr_arr:
        hdr['VERS_SW'] = version #version of pipeline
        hdr['VERSION'] = vrs #version of the file V01
        
    hdr_arr = setup_header(hdr_arr)
    
    printc('--------------------------------------------------------------',bcolors.OKGREEN)
    printc(f"------------ Load science data time: {np.round(time.perf_counter() - start_time,3)} seconds",bcolors.OKGREEN)
    printc('--------------------------------------------------------------',bcolors.OKGREEN)

    #-----------------
    # READ FLAT FIELDS
    #-----------------

    if flat_c:
        
        flat = load_and_process_flat(flat_f,accum_scaling,bit_conversion,scale_data,header_imgdirx_exists,imgdirx_flipped,cpos_arr)

    else:
        print(" ")
        printc('-->>>>>>> No flats mode',color=bcolors.WARNING)

    #-----------------
    # READ AND CORRECT DARK FIELD
    #-----------------

    if dark_c:
        print(" ")
        printc('-->>>>>>> Reading Darks                   ',color=bcolors.OKGREEN)

        start_time = time.perf_counter()

        try:

            if dark_f[-18:] == '0022210004.fits.gz':
                dark,h = get_data(dark_f,scaling = accum_scaling, bit_convert_scale = False, scale_data = False)
            elif dark_f[-19:] == '0022210004_000.fits':
                dark,h = get_data(dark_f,scaling = accum_scaling, bit_convert_scale = bit_conversion,scale_data = False)
            else:
                dark,h = get_data(dark_f, scaling = accum_scaling, bit_convert_scale = bit_conversion, scale_data = scale_data)
            
            dark_shape = dark.shape
            if dark_shape != (2048,2048):
                
                printc("Dark Field Input File not in 2048,2048 format: {}",dark_f,color=bcolors.WARNING)
                printc("Attempting to correct ",color=bcolors.WARNING)
          
                try:
                    if dark_shape[0] > 2048:
                        dark = dark[dark_shape[0]-2048:,:]
                
                except Exception:
                    printc("ERROR, Unable to correct shape of dark field data: {}",dark_f,color=bcolors.FAIL)
            # DC change 20211018
            if 'IMGDIRX' in h:
                header_drkdirx_exists = True
                drkdirx_flipped = str(h['IMGDIRX'])
            else:
                header_drkdirx_exists = False
                drkdirx_flipped = 'NO'
            
            dark = compare_IMGDIRX(dark[np.newaxis],header_imgdirx_exists,imgdirx_flipped,header_drkdirx_exists,drkdirx_flipped)[0]
            
            printc('--------------------------------------------------------------',bcolors.OKGREEN)
            printc(f"------------ Load darks time: {np.round(time.perf_counter() - start_time,3)} seconds",bcolors.OKGREEN)
            printc('--------------------------------------------------------------',bcolors.OKGREEN)

        except Exception:
            printc("ERROR, Unable to open darks file: {}",dark_f,color=bcolors.FAIL)
            raise ValueError() 

        #-----------------
        # APPLY DARK CORRECTION 
        #-----------------  

        if flat_c == False:
            flat = np.empty((2048,2048,4,6))

        data = apply_dark_correction(data, dark, rows, cols)  
        
        if flat_c == False:
            flat = np.empty((2048,2048,4,6))

        if out_intermediate:
            data_darkc = data.copy()

        DID_dark = h['FILENAME']

        for hdr in hdr_arr:
            hdr['CAL_DARK'] = DID_dark

    else:
        print(" ")
        printc('-->>>>>>> No dark mode',color=bcolors.WARNING)
    
    #-----------------
    # PREFILTER CORRECTION  
    #-----------------

    if prefilter_f is not None:
        print(" ")
        printc('-->>>>>>> Prefilter Correction',color=bcolors.OKGREEN)
        prefilter_c = True
        start_time = time.perf_counter()

        prefilter, _ = load_fits(prefilter_f)
        if imgdirx_flipped == 'YES':
            print('Flipping prefilter on the Y axis')
            prefilter = prefilter[:,::-1]
        # prefilter = prefilter[rows,cols]
        
        data = prefilter_correction(data,wave_axis_arr,prefilter[rows,cols],TemperatureCorrection=TemperatureCorrection)
        # DC 20221109 test for Smitha. PF already removed from the flat
        flat = prefilter_correction(flat[...,np.newaxis],[wave_flat],prefilter,TemperatureCorrection=TemperatureCorrection)[...,0]
        
        for hdr in hdr_arr:
            hdr['CAL_PRE'] = prefilter_f
        
        if out_intermediate:
            data_PFc = data.copy()  # DC 20211116

        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        printc(f"------------- Prefilter correction time: {np.round(time.perf_counter() - start_time,3)} seconds",bcolors.OKGREEN)
        printc('--------------------------------------------------------------',bcolors.OKGREEN)

    else:
        print(" ")
        printc('-->>>>>>> No prefilter mode',color=bcolors.WARNING)
        prefilter_c = False

    #-----------------
    # OPTIONAL Unsharp Masking clean the flat field stokes Q, U or V images
    #-----------------

    if clean_f and flat_c:
        flat_copy = flat.copy()
    
        print(" ")
        printc('-->>>>>>> Cleaning flats with Unsharp Masking',color=bcolors.OKGREEN)

        start_time = time.perf_counter()

        flat = unsharp_masking(flat,sigma,flat_pmp_temp,cpos_arr,clean_mode, clean_f = "blurring")

        for hdr in hdr_arr:
            hdr['CAL_USH'] = clean_mode
            hdr['SIGM_USH'] = sigma
        
        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        printc(f"------------- Cleaning flat time: {np.round(time.perf_counter() - start_time,3)} seconds",bcolors.OKGREEN)
        printc('--------------------------------------------------------------',bcolors.OKGREEN)

    else:
        print(" ")
        printc('-->>>>>>> No clean flats mode',color=bcolors.WARNING)

    #-----------------
    # NORM FLAT FIELDS
    #-----------------

    if norm_f and flat_c:
        flat = normalise_flat(flat, ceny, cenx)

        print(" ")
        printc('-->>>>>>> Normalising flats over central region',color=bcolors.WARNING)

    else:
        flat = normalise_flat(flat, slice(0,2048), slice(0,2048))

        print(" ")
        printc('-->>>>>>> Normalising flats over whole FOV',color=bcolors.WARNING)
        
    #-----------------
    # APPLY FLAT CORRECTION 
    #-----------------

    if flat_c:
        try:
            data = flat_correction(data,flat,flat_states,cpos_arr,flat_pmp_temp,rows,cols)
            
            DID_flat = header_flat['PHIDATID']
            
            for hdr in hdr_arr:
                hdr['CAL_FLAT'] = DID_flat
            
            if out_intermediate:
                data_flatc = data.copy()
            
            if '/' in flat_f:
                filename = flat_f.split('/')[-1]
            else:
                filename = flat_f
            
            for hdr in hdr_arr:
                hdr['CAL_FLAT'] = filename#DID_flat  - not the FILENAME keyword, in case we are trying with extra cleaned flats

            printc('--------------------------------------------------------------',bcolors.OKGREEN)
            printc(f"------------- Flat Field correction time: {np.round(time.perf_counter() - start_time,3)} seconds ",bcolors.OKGREEN)
            printc('--------------------------------------------------------------',bcolors.OKGREEN)
        except: 
          printc("ERROR, Unable to apply flat fields",color=bcolors.FAIL)

    else:
        print(" ")
        printc('-->>>>>>> No flat field correction mode',color=bcolors.WARNING)

    #-----------------
    # FIELD STOP 
    #-----------------

    if fs_c:
        _, field_stop = apply_field_stop(data, rows, cols, header_imgdirx_exists, imgdirx_flipped)
        if ghost_c:
            field_stop_ghost = load_ghost_field_stop(header_imgdirx_exists, imgdirx_flipped)
        
    else:
        print(" ")
        printc('-->>>>>>> No field stop mode',color=bcolors.WARNING)

    #-----------------
    # HOT PIXEL MASK 
    #-----------------
    
    # Hot pixels make problem for the interpolation
    # Procedure is masking the hot pixel with a dilated mask
    # New values are the median of the contour of each pixel
    
    if hot_px_mask:
        data = hot_pixel_mask(data, rows, cols)
        print(" ")
        printc('-->>>>>>> Hot Pixel Mask',color=bcolors.OKGREEN)
        
    else:
        printc('-->>>>>>> No hot pixel mask',color=bcolors.WARNING)

    #-----------------
    # NO-ISS POLARIMETRIC REGISTRATION
    #-----------------
    
    # new procedure to improve calibration when the ISS is off
    # loops trough wavelengths, and shifts mod states 2,3,4 to line up with mod state 1
    # use SPG_shift_FFT pre built func
    
    if iss_off:
        print(" ")
        printc('-->>>>>>> Polarimetric Frames Registration (--> ISS OFF)',color=bcolors.OKGREEN)
        #find central region, on disc, for the registration region
        limb_side, _, _, sly, slx = limb_side_finder(data[:,:,0,cpos_arr[0],int(scan)], hdr_arr[int(scan)])
        
        if fs_c:
            field_stop = ~binary_dilation(field_stop==0,generate_binary_structure(2,2), iterations=3)
            field_stop = np.where(field_stop > 0,1,0)
            if ghost_c:
                field_stop_ghost = ~binary_dilation(field_stop_ghost==0,generate_binary_structure(2,2), iterations=3)
                field_stop_ghost = np.where(field_stop_ghost > 0,1,0)
                
        
        start_time = time.perf_counter()

        data, hdr_arr = polarimetric_registration(data, sly, slx, hdr_arr)
        
        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        printc(f"------------- Registration time: {np.round(time.perf_counter() - start_time,3)} seconds ",bcolors.OKGREEN)
        printc('--------------------------------------------------------------',bcolors.OKGREEN)

    else:
        print(" ")
        printc('-->>>>>>> No frame registration (--> ISS ON)',color=bcolors.WARNING)

    #-----------------
    # PSF DECONVOLUTION ON MODULATED DATA
    #-----------------
    
    # was a considered option - now DEPRECATED
    # decided not the ideal case
    # applied on demodulated data

    #-----------------
    # APPLY DEMODULATION 
    #-----------------

    if demod:

        print(" ")
        printc('-->>>>>>> Demodulating data',color=bcolors.OKGREEN)

        start_time = time.perf_counter()

        data,_ = demod_hrt(data, pmp_temp)

        for hdr in hdr_arr:
            hdr['CAL_IPOL'] = 'HRT'+pmp_temp
        
        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        printc(f"------------- Demodulation time: {np.round(time.perf_counter() - start_time,3)} seconds ",bcolors.OKGREEN)
        printc('--------------------------------------------------------------',bcolors.OKGREEN)

    else:
        print(" ")
        printc('-->>>>>>> No demod mode',color=bcolors.WARNING)

    #-----------------
    # APPLY NORMALIZATION 
    #-----------------

    if norm_stokes:
        
        print(" ")
        printc('-->>>>>>> Normalising Stokes to Quiet Sun',color=bcolors.OKGREEN)
        
        start_time = time.perf_counter()

        Ic_mask = np.zeros((data_size[0],data_size[1],data_shape[-1]),dtype=bool)
        AR_mask = np.zeros((data_size[0],data_size[1],data_shape[-1]),dtype=bool)
        I_c = np.ones(data_shape[-1])
        limb_mask = np.ones((data_size[0],data_size[1],data_shape[-1]))
        limb = False
        
        for scan in range(data_shape[-1]):
           
            try:
                limb_temp, sly, slx, side = limb_fitting(data[:,:,0,cpos_arr[0],int(scan)], hdr_arr[int(scan)],field_stop[rows,cols])
                
                if limb_temp is not None:
                    #get region of pixels for norm, which are on for certain on disc
                    Ic_temp = np.zeros((data_size[0],data_size[1]))
                    Ic_temp[sly,slx] = 1
                    Ic_temp *= field_stop[rows,cols]
                    Ic_temp = np.where(Ic_temp>0,1,0) #final making sure

                    #for use later in the RTE
                    limb_temp = np.where(limb_temp>0,1,0)
                    limb_mask[...,scan] = limb_temp 
                    limb = True
                   
                else:
                    Ic_temp = np.zeros(data_size)
                    Ic_temp[ceny,cenx] = 1
                    Ic_temp = np.where(Ic_temp>0,1,0)

            except Exception as e:
                print(f"Error during limb fitting: {e}")
                #revert to central region
                Ic_temp = np.zeros(data_size)
                Ic_temp[ceny,cenx] = 1
                Ic_temp = np.where(Ic_temp>0,1,0)
             
            if fs_c:
                Ic_temp *= field_stop[rows,cols]
            
            Ic_temp = np.array(Ic_temp, dtype=bool)
            
            ##################################################################
            """new Icont normalization removing high magnetic field regions"""
            AR_temp = np.ones(Ic_temp.shape,dtype=bool)
                    
            for p in range(1,4):
                hi = np.histogram(data[Ic_temp,p,:,scan].flatten(),bins=np.linspace(-20,20,150))
                gval = gaussian_fit(hi, show = False)
                AR_temp *= np.max(np.abs(data[:,:,p,:,scan] - gval[1]),axis=-1) < 5*gval[2]

            AR_temp = np.asarray(AR_temp, dtype=bool)
            ##################################################################
            
            I_c[scan] = np.mean(data[Ic_temp*AR_temp,0,cpos_arr[0],int(scan)])
            data[:,:,:,:,scan] = data[:,:,:,:,scan]/I_c[scan]
            Ic_mask[...,scan] = Ic_temp
            AR_mask[...,scan] = AR_temp
            hdr_arr[scan]['CAL_NORM'] = round(I_c[scan],4) # DC 20211116

        if out_intermediate:
            data_demod_normed = data.copy()

        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        printc(f"------------- Stokes Normalising time: {np.round(time.perf_counter() - start_time,3)} seconds ",bcolors.OKGREEN)
        printc('--------------------------------------------------------------',bcolors.OKGREEN)

    else:
        print(" ")
        printc('-->>>>>>> No normalising Stokes mode',color=bcolors.WARNING)

    #-----------------
    # CROSS-TALK CALCULATION 
    #-----------------

    if ItoQUV:
        
        print(" ")
        printc('-->>>>>>> Cross-talk correction I to Q,U,V ',color=bcolors.OKGREEN)

        start_time = time.perf_counter()

        slope, offset = 0, 1
        q, u, v = 0, 1, 2
        CTparams = np.zeros((2,3,number_of_scans))
        
        #get ctalk parameters
        for scan, scan_hdr in enumerate(hdr_arr):
            printc(f'  ---- >>>>> CT parameters computation of data scan number: {scan} .... ',color=bcolors.OKGREEN)
            if ghost_c: #20211116
                ctalk_params = crosstalk_auto_ItoQUV(data[...,scan],cpos_arr[scan],cpos_arr[scan],roi=np.asarray(Ic_mask[...,scan]*field_stop_ghost[rows,cols],dtype=bool)) #20211116
            else: #20211116
                ctalk_params = crosstalk_auto_ItoQUV(data[...,scan],cpos_arr[scan],cpos_arr[scan],roi=Ic_mask[...,scan]) #20211116
            
            CTparams[...,scan] = ctalk_params
            
            scan_hdr['CAL_CRT0'] = round(ctalk_params[slope,q],4) #I-Q slope
            scan_hdr['CAL_CRT2'] = round(ctalk_params[slope,u],4) #I-U slope
            scan_hdr['CAL_CRT4'] = round(ctalk_params[slope,v],4) #I-V slope
            scan_hdr['CAL_CRT1'] = round(ctalk_params[offset,q],4) #I-Q offset
            scan_hdr['CAL_CRT3'] = round(ctalk_params[offset,u],4) #I-U offset
            scan_hdr['CAL_CRT5'] = round(ctalk_params[offset,v],4) #I-V offset
            
            scan_hdr['CAL_CRT6'] = 0 #V-Q slope
            scan_hdr['CAL_CRT8'] = 0 #V-U slope
            scan_hdr['CAL_CRT7'] = 0 #V-Q offset
            scan_hdr['CAL_CRT9'] = 0 #V-U offset
                
        #apply crosstalk correction - with error handling        
        try:    
            data = CT_ItoQUV(data, CTparams, norm_stokes, cpos_arr, Ic_mask*AR_mask) # new continuum normalization

        except Exception:
            print("There was an issue applying the I -> Q,U,V cross talk correction")
            if 'Ic_mask' not in vars():
                print("This could be because 'Ic_mask' was not initialised")
                if data.shape[:2] == (2048,2048):
                    response = input("The input data is 2k x 2k \n Are all the input data files disk centre pointing? [y/n]")
                    if response == 'y' or response == 'Y':
                        try:
                            Ic_mask = np.zeros(data_size)
                            Ic_mask[ceny,cenx] = 1
                            Ic_mask = np.where(Ic_mask>0,1,0)
                            Ic_mask = np.array(Ic_mask, dtype = bool)
                            data = CT_ItoQUV(data, ctalk_params, norm_stokes, cpos_arr, Ic_mask[rows,cols])
                        except Exception:
                            print("Unable to handle error\n Please check the input config file\n Aborting")
                            exit()
                    else:
                        raise KeyError("Response was not 'y' or 'Y'\n 'norm_f' keyword in input config file not set to True\n Aborting")
                else:
                    raise KeyError("The issue could not be overcome as the Input data is not 2k x 2k\n 'norm_f' keyword in input config file not set to True\n Aborting")
            else:
                raise KeyError("Unable to handle error \n Aborting")

        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        printc(f"------------- I -> Q,U,V cross talk correction time: {np.round(time.perf_counter() - start_time,3)} seconds ",bcolors.OKGREEN)
        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        
        if not iss_off or not PSFstokes:
            data *= field_stop[rows,cols, np.newaxis, np.newaxis, np.newaxis]

    else:
        print(" ")
        printc('-->>>>>>> No ItoQUV mode',color=bcolors.WARNING)

    if VtoQU:
        
        print(" ")
        printc('-->>>>>>> Cross-talk correction V to Q,U ',color=bcolors.OKGREEN)

        start_time = time.perf_counter()

        slope, offset = 0, 1
        q, u = 0, 1
        CTparams = np.zeros((2,2,number_of_scans))
        
        for scan, scan_hdr in enumerate(hdr_arr):
            printc(f'  ---- >>>>> CT parameters computation of data scan number: {scan} .... ',color=bcolors.OKGREEN)
            if ghost_c: #20211116
                ctalk_params = crosstalk_auto_VtoQU(data[...,scan],slice(0,6),slice(0,6),roi=np.asarray(Ic_mask[...,scan]*field_stop_ghost[rows,cols],dtype=bool),nlevel=0.3) #20211116
            else: #20211116
                ctalk_params = crosstalk_auto_VtoQU(data[...,scan],slice(0,6),slice(0,6),roi=Ic_mask[...,scan],nlevel=0.3) #20211116
            
            CTparams[...,scan] = ctalk_params
            #wrong keywords for CT parameters: fixed on 2022-10-07
            scan_hdr['CAL_CRT6'] = round(ctalk_params[slope,q],4) #V-Q slope
            scan_hdr['CAL_CRT8'] = round(ctalk_params[slope,u],4) #V-U slope
            scan_hdr['CAL_CRT7'] = round(ctalk_params[offset,q],4) #V-Q offset
            scan_hdr['CAL_CRT9'] = round(ctalk_params[offset,u],4) #V-U offset
                
        data = CT_VtoQU(data, CTparams)
        
        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        printc(f"------------- V -> Q,U cross talk correction time: {np.round(time.perf_counter() - start_time,3)} seconds ",bcolors.OKGREEN)
        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        
        if not iss_off or not PSFstokes:
            data *= field_stop[rows,cols, np.newaxis, np.newaxis, np.newaxis]

    else:
        print(" ")
        printc('-->>>>>>> No VtoQU mode',color=bcolors.WARNING)

    #-----------------
    # NO-ISS WAVELENGTH REGISTRATION
    #-----------------
    
    # new procedure to improve calibration when the ISS is off
    # align the wavelengths, from the Stokes I image, (after demodulation), using cv2

    if iss_off:
        print(" ")
        printc('-->>>>>>> Wavelength Frames Registration (--> ISS OFF)',color=bcolors.OKGREEN)
        
        start_time = time.perf_counter()
        
        data, hdr_arr = wavelength_registration(data, cpos_arr, sly, slx, hdr_arr)
        
        if not PSFstokes:
            data *= field_stop[rows,cols, np.newaxis, np.newaxis, np.newaxis]

        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        printc(f"------------- Registration time: {np.round(time.perf_counter() - start_time,3)} seconds ",bcolors.OKGREEN)
        printc('--------------------------------------------------------------',bcolors.OKGREEN)

    else:
        print(" ")
        printc('-->>>>>>> No frame registration (--> ISS ON)',color=bcolors.WARNING)

    #-----------------
    # PSF DECONVOLUTION ON STOKES
    #-----------------

    if PSFstokes:
        start_time = time.perf_counter()
        
        print(" ")
        printc('-->>>>>>> PSF deconvolution on Stokes vectors',color=bcolors.OKGREEN)
        for scan in range(data_shape[-1]):
            data[...,scan] = restore_stokes_cube(data[...,scan], hdr_arr[scan],orbit=PSForbit,aberr_cor=PSFaberr)
            hdr_arr[scan]['CAL_PSF'] = PSForbit+'; aberration: '+str(PSFaberr)

        data *= field_stop[rows,cols, np.newaxis, np.newaxis, np.newaxis]

        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        printc(f"------------- PSF deconvolution time: {np.round(time.perf_counter() - start_time,3)} seconds ",bcolors.OKGREEN)
        printc('--------------------------------------------------------------',bcolors.OKGREEN)
    else:
        print(" ")
        printc('-->>>>>>> No PSF deconvolution on Stokes vectors',color=bcolors.WARNING)

    #-----------------
    # CHECK FOR INFs
    #-----------------

    data[np.isinf(data)] = 0
    data[np.isnan(data)] = 0

    #-----------------
    # WRITE OUT STOKES VECTOR
    #-----------------

    #rewrite the PARENT keyword to the original FILENAME
    for count, scan in enumerate(data_f):
        hdr_arr[count]['PARENT'] = hdr_arr[count]['FILENAME'] 

    #these two ifs need to be outside out_stokes_file if statement - needed for inversion
    if out_dir[-1] != "/":
        print("Desired Output directory missing / character, will be added")
        out_dir = out_dir + "/"

    #check if the output directory exists, if not, create it
    if not os.path.exists(out_dir): 
        print(f"{out_dir} does not exist -->>>>>>> Creating it")
        os.makedirs(out_dir)

    #take out of ifs, so that if both are true, they have the exact same time in the header
    ntime = datetime.datetime.now()

    if out_stokes_file:
        
        print(" ")
        printc('Saving demodulated data to one \'stokes\' file per scan')

        #check if user set specific output filenames, check for duplicates, otherwise use the DID
        if out_stokes_filename is not None:

            if isinstance(out_stokes_filename,str):
                out_stokes_filename = [out_stokes_filename]

            if int(len(out_stokes_filename)) == int(data_shape[-1]):
                scan_name_list = out_stokes_filename
                scan_name_defined = True
            else:
                print("Input stokes filenames do not match the number of input arrays, reverting to default naming")
                scan_name_defined = False
        else:
            scan_name_defined = False

        if not scan_name_defined: #check if already defined by user - if not, use auto name generation function
            scan_name_list = check_filenames(data_f) #extract the DIDs and check no duplicates
        
        for count, scan in enumerate(data_f):

            if ".gz" in scan:
                gzip = True
            else:
                gzip = False
            stokes_file = create_output_filenames(scan, scan_name_list[count], version = vrs, gzip = gzip)[0]
            hdr_arr[count]['DATE'] = ntime.strftime("%Y-%m-%dT%H:%M:%S")
            hdr_arr[count]['FILENAME'] = stokes_file
            hdr_arr[count]['LEVEL'] = 'L2'
            hdr_arr[count]['BTYPE'] = 'STOKES'
            hdr_arr[count]['BUNIT'] = 'I_CONT'
            hdr_arr[count]['DATAMIN'] = int(np.min(data[:,:,:,:,count]))
            hdr_arr[count]['DATAMAX'] = int(np.max(data[:,:,:,:,count]))
            hdr_arr[count] = data_hdr_kw(hdr_arr[count], data[:,:,:,:,count]) #add datamedn, datamean etc
            hdr_interm = hdr_arr[count].copy()
            hdr_arr[count]['HISTORY'] = f"Version: {version}. Dark: {dark_c}. Prefilter: {prefilter_c}. Flat: {flat_c}, Unsharp: {clean_f}. Flat norm: {norm_f}. I->QUV ctalk: {ItoQUV}. PSF deconvolution: {hdr_arr[count]['CAL_PSF']}"
            
            with fits.open(scan) as hdu_list:
                print(f"Writing out stokes file as: {stokes_file}")
                tmp = data[:,:,:,:,count].astype(np.float32)
                hdu_list[0].data = np.moveaxis(tmp, [-1,-2], [0,1]) #want, 6,4,y,x to be consistent with FDT
                hdu_list[0].header = hdr_arr[count] #update the calibration keywords
                hdu_list.writeto(out_dir + stokes_file, overwrite=True)  
        
    else:
        print(" ")
        #check if already defined by input, otherwise generate
        scan_name_list = check_filenames(data_f)
        printc('-->>>>>>> No output Stokes file mode',color=bcolors.WARNING)

    #-----------------
    # WRITE OUT INTERMEDIATE
    #-----------------

    if out_intermediate: 
        #intermediate output with their own header, so they can be easily used as input for the pipeline
        for count, scan in enumerate(data_f):
            root_scan_name = scan_name_list[count]
            hdr_arr[count]['DATE'] = ntime.strftime("%Y-%m-%dT%H:%M:%S")
            hdr_interm = hdr_arr[count].copy()

            if dark_c: 
                history_str = f"Intermediate. Version: {version}. Dark: {dark_c}. Prefilter: {False}. Flat: {False}, Unsharp: {False}. Flat norm: {False}. I->QUV ctalk: {False}. PSF deconvolution: {False}"
                file_suffix = 'dark_corrected'
                write_out_intermediate(data_darkc[:,:,:,:,count], hdr_interm, history_str, scan, root_scan_name, file_suffix, out_dir)
            
            if prefilter_c:
                history_str =  f"Intermediate. Version: {version}. Dark: {dark_c}. Prefilter: {prefilter_c}. Flat: {False}, Unsharp: {False}. Flat norm: {False}. I->QUV ctalk: {False}. PSF deconvolution: {False}"
                file_suffix = 'prefilter_corrected'
                write_out_intermediate(data_PFc[:,:,:,:,count], hdr_interm, history_str, scan, root_scan_name, file_suffix, out_dir)

            if flat_c: 
                history_str = f"Intermediate. Version: {version}. Dark: {dark_c}. Prefilter: {prefilter_c}. Flat: {flat_c}, Unsharp: {clean_f}. Flat norm: {norm_f}. I->QUV ctalk: {False}. PSF deconvolution: {False}"
                #with US
                file_suffix =  'flat_corrected'
                write_out_intermediate(data_flatc[:,:,:,:,count], hdr_interm, history_str, scan, root_scan_name, file_suffix, out_dir)
                #without US
                root_scan_name_before_US = f"copy_{flat_f.split('/')[-1]}"
                file_suffix = ''
                write_out_intermediate(flat_copy, hdr_interm, history_str, scan, root_scan_name_before_US, file_suffix, out_dir)

            if demod:
                history_str = f"Intermediate. Version: {version}. Dark: {dark_c}. Prefilter: {prefilter_c}. Flat: {flat_c}, Unsharp: {clean_f}. Flat norm: {norm_f}. I->QUV ctalk: {False}. PSF deconvolution: {False}"
                file_suffix = 'demodulated'
                write_out_intermediate(data_demod_normed[:,:,:,:,count], hdr_interm, history_str, scan, root_scan_name, file_suffix, out_dir, bunit = 'I_CONT', btype = 'STOKES')

    else:
        print(" ")
        printc('-->>>>>>> No intermediate files requested',color=bcolors.WARNING)

    #-----------------
    # INVERSION OF DATA WITH CMILOS
    #-----------------

    if rte == 'RTE' or rte == 'CE' or rte == 'CE+RTE' or rte == 'RTE_seq':
        #check out_dir has "/" character
        if out_dir[-1] != "/":
            print("Desired Output directory missing / character, will be added")
            out_dir = out_dir + "/"

        if limb:
            mask = limb_mask*field_stop[rows,cols,np.newaxis]
        else:
            mask = np.ones((data_size[0],data_size[1],data_shape[-1]))*field_stop[rows,cols,np.newaxis]
            
        if avg_stokes_before_rte:
            data = np.mean(data, axis = (-1))
            data_shape = (data_size[0], data_size[1], 1)

        if pymilos_opt:
            #weight = np.asarray([1.,4.,5.4,4.1]); initial_model = np.asarray([400,30,120,1,0.05,1.5,.01,.22,.85])
            #py_cmilos(data_f, hdr_arr, wave_axis_arr, data_shape, cpos_arr, data, rte, mask, imgdirx_flipped, out_rte_filename, out_dir, weight = weight, initial_model = initial_model, vers = vrs)
            py_cmilos(data_f, hdr_arr, wave_axis_arr, data_shape, cpos_arr, data, rte, mask, imgdirx_flipped, out_rte_filename, out_dir, vers = vrs)
        else:
            cmilos(data_f, hdr_arr, wave_axis_arr, data_shape, cpos_arr, data, rte, mask, imgdirx_flipped, out_rte_filename, out_dir, cavity_f, rows, cols, vers = vrs)

    else:
        print(" ")
        printc('-->>>>>>> No RTE Inversion mode',color=bcolors.WARNING)

    #-----------------
    # SAVING CONFIG FILE
    #-----------------
   
    if config:
        print(" ")
        printc('-->>>>>>> Saving copy of input config file ',color=bcolors.OKGREEN)

        dt = datetime.datetime.fromtimestamp(overall_time)
        runtime = dt.strftime("%d_%m_%YT%H_%M_%S")

        json.dump(input_dict, open(out_dir + f"config_file_{runtime}.json", "w"))
        
    print(" ")
    printc('--------------------------------------------------------------',color=bcolors.OKGREEN)
    printc(f'------------ Reduction Complete: {np.round(time.perf_counter() - overall_time,3)} seconds',color=bcolors.OKGREEN)
    printc('--------------------------------------------------------------',color=bcolors.OKGREEN)
   
    return data

