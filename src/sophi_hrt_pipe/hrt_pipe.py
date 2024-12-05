import numpy as np 
import os.path
from astropy.io import fits
import time
import datetime
import json
from numpy.core.numeric import True_
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

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
    version = 'V1.9.0 November 22nd 2024'

    printc('--------------------------------------------------------------',bcolors.OKGREEN)
    printc('PHI HRT data reduction software  ',bcolors.OKGREEN)
    printc('Version: '+version,bcolors.OKGREEN)
    printc('--------------------------------------------------------------',bcolors.OKGREEN)

    #-----------------
    # READ INPUT JSON
    #-----------------
    
    start_proc = time.strftime('%Y%m%d%H%M')
    input_dict = json.load(open(input_json_file))

    try:
        #input data
        data_f = input_dict['data_f']
        flat_f = input_dict['flat_f']
        dark_f = input_dict['dark_f']
        prefilter_f = input_dict['prefilter_f']
        cavity_f = input_dict['cavity_f']

        #input/output type + scaling
        L1_input = input_dict['L1_input']
        if L1_input:
            accum_scaling = True 
            bit_conversion = True
            scale_data = True
        else:
            scale_data = input_dict['scale_data']
            accum_scaling = input_dict['accum_scaling']
            bit_conversion = input_dict['bit_conversion']

        #reduction
        dark_c = input_dict['dark_c']
        flat_c = input_dict['flat_c']
        prefilter_c = input_dict['prefilter_c']
        if 'TemperatureCorrection' not in input_dict: #if FG != 61 deg - will correct wavelengths
            TemperatureCorrection = False
        else:
            TemperatureCorrection = input_dict['TemperatureCorrection']
        if 'TemperatureConstant' not in input_dict:
            TemperatureConstant = 36.46e-3
        else:
            TemperatureConstant = input_dict['TemperatureConstant']
        norm_f = input_dict['norm_f']
        clean_f = input_dict['clean_f']
        if clean_f:
            sigma = input_dict['sigma']
            clean_mode = input_dict['clean_mode']
        flat_states = input_dict['flat_states']
        if 'fs_c' in input_dict:
            fs_c = input_dict['fs_c']
        else:
            fs_c = True
        if 'iss_off' in input_dict:
            iss_off = input_dict['iss_off']
        demod = input_dict['demod']
        norm_stokes = input_dict['norm_stokes']
        ItoQUV = input_dict['ItoQUV']
        CTmode = input_dict['CTmode']
        VtoQU = input_dict['VtoQU']
        
        if isinstance(input_dict['PSFstokes'],bool) and isinstance(input_dict['PSFaberr'],bool) and isinstance(input_dict['PSFstraylight'],bool):
            PSFstokes = {'PD_f': "/data/slam/home/calchetti/hrt_pipeline/csv/PD_result.csv",
                         'deconvolution':input_dict['PSFstokes'],
                         'aberration_correction':input_dict['PSFaberr'],
                         'straylight_correction':input_dict['PSFstraylight'],
                         'gamma2':0.02,
                         'low_f':0.8,
                         'roi':False,
                         'method':'lofdahl'}
        else:
            PSFstokes = input_dict['PSFstokes']
            for k,v in zip(['PD_f','low_f','gamma2','aberration_correction','straylight_correction','roi','method'],["/data/slam/home/calchetti/hrt_pipeline/csv/PD_result.csv",0.8,0.02,True,True,False,'lofdahl']):
                if k not in PSFstokes.keys():
                    PSFstokes[k] = v
        # PSFaberr = input_dict['PSFaberr']  

        if 'ghost_c' in input_dict:
            ghost_c = input_dict['ghost_c']  #20211116
        else:
            ghost_c = False
        cavity_c = input_dict['cavity_c']
        if cavity_c:
            cavity_f = input_dict['cavity_f']
        else:
            cavity_f = None
        
        # rte = input_dict['rte']
        out_intermediate = input_dict['out_intermediate']  #20211116
        if 'out_synthesis' in input_dict:
            out_synthesis = input_dict['out_synthesis']
        else:
            out_synthesis = False
        # pymilos_opt = input_dict['pymilos']
        
        RTE_options = input_dict["RTE"]
        # inputs for RTE inversions. Last update of the values: 2023-09-04
        # if RTE_options['weight'] is not None:
        #     RTE_options['weight'] = np.asarray(input_dict['weight'])
        # else:
        #     if 'PSF' in RTE_options['rte']:
        #         RTE_options['weight'] = np.asarray([1.,3.8,4.1,3.6]) # with spectral PSF
        #     else:
        #         RTE_options['weight'] = np.asarray([1.,3.5,4.,3.5]) # OK without spectral PSF
        #     # if iss_off:
        #     #     weight = np.asarray([1.,4.,5.4,4.1]) # until RSW 6

        # if RTE_options['initial_model'] is not None:
        #     RTE_options['initial_model'] = np.asarray(input_dict['initial_model'])
        # else:
        #     if 'PSF' in RTE_options['rte']:
        #         RTE_options['initial_model'] = np.asarray([400,30,120,1.,0.03,0.05,.01,.2,.8]) # with spectral PSF
        #     else:
        #         RTE_options['initial_model'] = np.asarray([400,30,120,14.,0.06,0.05,.5,.25,.75]) # OK without spectral PSF
            # if iss_off:
            #     initial_model = np.asarray([400,30,120,1,0.05,1.5,.01,.22,.85]) # until RSW 6
    

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
            vrs = start_proc
        else:
            vrs = input_dict['vers']
            if len(vrs) != 2:
                printc("WARNING: Version string is larger than 2 digits",color=bcolors.WARNING)
        #behaviour if clean mode is set to None (null in json)
        if 'clean_mode' in locals():
            if clean_mode is None:
                clean_mode = "V" 
            
    except Exception as e:
        print(f"Missing key(s) in the input config file: {e}")
        raise KeyError
    
    overall_time = time.perf_counter()

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

            wave_axis_arr[scan], voltagesData_arr[scan], tuning_constant_arr[scan], cpos_arr[scan] = fits_get_sampling(data_f[scan], TemperatureCorrection = TemperatureCorrection, TemperatureConstant = TemperatureConstant, verbose = True)
            
            if hdr_arr[scan]['PHIDATID'] == '0250180115':
                printc(f'Manual shift of the wavelength due to Etalon dots',color=bcolors.WARNING)
                wave_axis_arr[scan] -= 0.015 # Angstrom

            if 'IMGDIRX' in hdr_arr[scan] and hdr_arr[scan]['IMGDIRX'] == 'YES':
                print(f"This scan has been flipped in the Y axis to conform to orientation standards. \n File: {data_f[scan]}")

            # add wavelength keywords
            hdr_arr[scan]['WAVEMIN'] = round(wave_axis_arr[scan][0],3)
            hdr_arr[scan]['WAVEMAX'] = round(wave_axis_arr[scan][-1],3)
            previousKey = 'WAVEMAX'
            # WAVEBAND entry
            newKey = 'WAVEBAND' # new implementation
            hdr_arr[scan].set(newKey, 'FE6173', 'Bandpass description', after=previousKey)
            previousKey = newKey
            # WAVELNTH entry
            newKey = 'WAVELNTH' # new implementation
            hdr_arr[scan].set(newKey, 6173.341, '[Angstrom] Characteristic wavelength', after=previousKey)
            previousKey = newKey
            newKey = 'WAVEUNIT' # new implementation
            hdr_arr[scan].set(newKey, 'angstrom', 'Physical units of the wavelengths', after=previousKey)
            previousKey = newKey
            for i in range(6):
                newKey = 'WAVELN'+str(int(i)+1).rjust(2,'0')
                hdr_arr[scan].set(newKey, round(wave_axis_arr[scan][i],3), '[Angstrom] Wavelength '+str(int(i)+1).rjust(2,'0'), after=previousKey)
                previousKey = newKey
            # add voltage keywords
            # for i in range(6):
                # newKey = f'VOLTAGE{i+1}'
                # hdr_arr[scan].set(newKey, int(voltagesData_arr[scan][i]), f'[Volt] {i+1}. voltage of observation', after=previousKey)
                # previousKey = newKey
            # add continuum position keywords
            # newKey = 'CONTPOS' # as implemented by DG
            # hdr_arr[scan].set(newKey, int(cpos_arr[scan]//5), 'Continuum pos 0 = blue,1 = red,-1 = undef', after=previousKey)
            # previousKey = newKey
            newKey = 'CONTPOS' # new implementation
            hdr_arr[scan].set(newKey, int(cpos_arr[scan])+1, 'Index to WAVELNnn (-1 = undef)', after=previousKey)
            previousKey = newKey
            # add voltage tuning constant keywords
            newKey = 'TUNCONS'
            hdr_arr[scan].set(newKey, tuning_constant_arr[scan], f'[Angstrom / Volt] voltage tuning constant', after=previousKey)
            previousKey = newKey
            # add temperature tuning constant keywords
            newKey = 'TEMPCONS'
            if TemperatureCorrection:
                hdr_arr[scan].set(newKey, TemperatureConstant, '[Angstrom / Kelvin] temperature constant', after=previousKey)
            else:
                hdr_arr[scan].set(newKey, 0, '[Angstrom / Kelvin] temperature constant', after=previousKey)
            previousKey = newKey

            # change NAXIS1, 2, WAVEMIN, and MAX comments
            hdr_arr[scan].comments['NAXIS1'] = 'number of pixels on the x axis'
            hdr_arr[scan].comments['NAXIS2'] = 'number of pixels on the y axis'
            # hdr_arr[scan].comments['WAVEMIN'] = '[nm] min wavelength of observation'
            # hdr_arr[scan].comments['WAVEMAX'] = '[nm] max wavelength of observation'

        #--------
        # check if ISS is ON or OFF
        #--------

        if 'iss_off' not in locals():
            if hdr_arr[0]['ISSMODE1'] == 'ISS_IDLE':
                iss_off = True
                printc('-->>>>>>> ISS is OFF',color=bcolors.OKGREEN) 
            else:
                iss_off = False
                printc('-->>>>>>> ISS is ON',color=bcolors.OKGREEN) 
        # change RTE parameters if ISS is off and if they were chosen automatically
        if RTE_options['weight'] is None:
            if iss_off:
                if 'PSF' in RTE_options['rte']:
                    RTE_options['weight'] = [1.,4.7,5.6,4.]
                else:
                    RTE_options['weight'] = [1.,4.,5.4,4.1] # until RSW 6
            else:
                if 'PSF' in RTE_options['rte']:
                    RTE_options['weight'] = [1.,3.8,4.1,3.6] # with spectral PSF
                else:
                    RTE_options['weight'] = [1.,3.5,4.,3.5] # OK without spectral PSF
        
        if RTE_options['initial_model'] is None:
            if iss_off:
                if 'PSF' in RTE_options['rte']:
                    RTE_options['initial_model'] = [400,30,120,2.5,0.05,.5,.01,.22,.85]
                else:
                    RTE_options['initial_model'] = [400,30,120,1,0.05,1.5,.01,.22,.85] # until RSW 6
            else:
                if 'PSF' in RTE_options['rte']:
                    RTE_options['initial_model'] = [400,30,120,1.,0.03,0.05,.01,.2,.8] # with spectral PSF
                else:
                    RTE_options['initial_model'] = [400,30,120,14.,0.06,0.05,.5,.25,.75] # OK without spectral PSF

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
        # pxbeg1 and pxend1 do not take into account the inverted direction of the X axis in L1 data (FIXED)
        # start_col = int((2048 - hdr_arr[0]['PXEND1'] + 1) - 1)
        
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
        
        flat, flat_pmp_temp, header_flat = load_and_process_flat(flat_f,accum_scaling,bit_conversion,scale_data,header_imgdirx_exists,imgdirx_flipped,cpos_arr,pmp_temp)

    else:
        print(" ")
        printc('-->>>>>>> No flats mode',color=bcolors.WARNING)

    #-----------------
    # READ CAVITY MAPS
    #-----------------

    if cavity_c:
        
        cavity = cavity_shifts(cavity_f,wave_axis_arr[0],slice(0,flat.shape[0]),slice(0,flat.shape[1]),False)

    else:
        cavity = None
        print(" ")
        printc('-->>>>>>> No cavity compensation in Prefilter Correction',color=bcolors.WARNING)

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

    if prefilter_c:
        print(" ")
        printc('-->>>>>>> Prefilter Correction ON FLAT FIELD ONLY',color=bcolors.OKGREEN)
        prefilter_c = True
        start_time = time.perf_counter()

        prefilter, _ = load_fits(prefilter_f)
        if imgdirx_flipped == 'YES':
            print('Flipping prefilter on the Y axis')
            prefilter = prefilter[:,::-1]
        # prefilter = prefilter[rows,cols]
        
        if flat_c:
            wave_flat, voltagesData_flat, _, cpos_f = fits_get_sampling(flat_f,verbose = True,TemperatureCorrection=TemperatureCorrection,TemperatureConstant=TemperatureConstant)
            wave_flat = compare_cpos(wave_flat,cpos_f,cpos_arr[0])
            Tetalon_flat = header_flat['FGOV1PT1'] # ['FGH_TSP1']
            flat = prefilter_correction(flat[...,np.newaxis],[wave_flat],prefilter,Tetalon=Tetalon_flat,TemperatureCorrection=TemperatureCorrection,TemperatureConstant=TemperatureConstant,shift=cavity)[...,0]
            # flat = prefilter_correctionNew(flat[...,np.newaxis],[wave_flat],slice(0,2048),slice(0,2048),Tetalon=Tetalon_flat,imgdirx_flipped = 'YES')[...,0]
        
        # for hdr in hdr_arr:
        #     hdr['CAL_PRE'] = prefilter_f
        
        # if out_intermediate:
        #     data_PFc = data.copy()  # DC 20211116

        # printc('--------------------------------------------------------------',bcolors.OKGREEN)
        # printc(f"------------- Prefilter correction time: {np.round(time.perf_counter() - start_time,3)} seconds",bcolors.OKGREEN)
        # printc('--------------------------------------------------------------',bcolors.OKGREEN)

    else:
        print(" ")
        # printc('-->>>>>>> No prefilter mode',color=bcolors.WARNING)
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

    if flat_c:
        if norm_f:
            flat = normalise_flat(flat, ceny, cenx)

            print(" ")
            printc('-->>>>>>> Normalising flats over central region',color=bcolors.WARNING)

        else:
            flat = normalise_flat(flat, slice(0,2048), slice(0,2048))
            # Test for temporary flat
            # flat = normalise_flat(flat, rows, cols)
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
                hdr['CAL_FNUM'] = flat_states
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

    if prefilter_c:
        print(" ")
        printc('-->>>>>>> Prefilter Correction On Data AFTER FLAT FIELDING',color=bcolors.OKGREEN)
        # prefilter_c = True
        start_time = time.perf_counter()
        Tetalon = hdr_arr[0]['FGOV1PT1'] # ['FGH_TSP1']
            
        # data = prefilter_correction(data,wave_axis_arr,rows,cols,imgdirx_flipped)

        # prefilter, _ = load_fits(prefilter_f)
        # if imgdirx_flipped == 'YES':
        #     print('Flipping prefilter on the Y axis')
        #     prefilter = prefilter[:,::-1]
        # prefilter = prefilter[rows,cols]
        
        if cavity is not None:
            data = prefilter_correction(data,wave_axis_arr,prefilter[rows,cols],Tetalon=Tetalon,TemperatureCorrection=TemperatureCorrection,TemperatureConstant=TemperatureConstant,shift=cavity[rows,cols])
        else:
            data = prefilter_correction(data,wave_axis_arr,prefilter[rows,cols],Tetalon=Tetalon,TemperatureCorrection=TemperatureCorrection,TemperatureConstant=TemperatureConstant,shift=None)
            # data = prefilter_correctionNew(data,wave_axis_arr,rows,cols,Tetalon=Tetalon,imgdirx_flipped = 'YES')
        # DC 20221109 test for Smitha. PF already removed from the flat
        # wave_flat, voltagesData_flat, _, cpos_f = fits_get_sampling(flat_f,verbose = True)
        # wave_flat = compare_cpos(wave_flat,cpos_f,cpos_arr[0]) 
        # flat = prefilter_correction(flat[...,np.newaxis],[wave_flat],prefilter,TemperatureCorrection=TemperatureCorrection)[...,0]
        # flat = prefilter_correction(flat[...,np.newaxis],[wave_flat],slice(0,2048),slice(0,2048),imgdirx_flipped)[...,0]
        
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
        # limb_side, _, _, sly, slx = limb_side_finder(data[:,:,0,cpos_arr[0],int(scan)], hdr_arr[int(scan)])
        
        # added here to have the high contrast slices
        AR_temp = ARmasking(data[...,0], field_stop[rows,cols], cpos = cpos_arr[0]) # for ellipse limb fit
        _, sly, slx, _ = limb_ellipse(data[:,:,0,cpos_arr[0],0], hdr_arr[0],field_stop[rows,cols],AR_temp,high_contrast=True)
        
        del AR_temp
        ####
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
        limb_percent_mask = np.zeros((data_size[0],data_size[1],data_shape[-1]))
        limb_percent_mask[32:-32,32:-32] = 1
        limb = False
        
        for scan in range(data_shape[-1]):
           
            try:
                AR_temp = ARmasking(data[...,scan], field_stop[rows,cols], cpos = cpos_arr[scan]) # for ellipse limb fit
                limb_temp, sly, slx, side, limb_percent_temp = limb_ellipse(data[:,:,0,cpos_arr[0],int(scan)], hdr_arr[int(scan)],field_stop[rows,cols],AR_temp,percent=True,high_contrast=True)
                
                # limb_temp, sly, slx, side, limb_percent_temp = limb_fitting(data[:,:,0,cpos_arr[0],int(scan)], hdr_arr[int(scan)],field_stop[rows,cols],percent=True)
                
                if limb_temp is not None:
                    #get region of pixels for norm, which are for certain on disc
                    Ic_temp = np.zeros((data_size[0],data_size[1]))
                    Ic_temp[sly,slx] = 1
                    Ic_temp *= field_stop[rows,cols]
                    Ic_temp = np.where(Ic_temp>0,1,0) #final making sure

                    #for use later in the CT correction
                    limb_temp = np.where(limb_temp>0,1,0)
                    limb_mask[...,scan] = limb_temp 
                    limb_percent_temp = np.where(limb_percent_temp>0,1,0)
                    limb_percent_mask[...,scan] = limb_percent_temp 
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
                limb_mask *= field_stop[rows,cols,np.newaxis]
                limb_percent_mask *= field_stop[rows,cols,np.newaxis]
            
            Ic_temp = np.array(Ic_temp, dtype=bool)
            limb_mask = np.array(limb_mask, dtype=bool)
            limb_percent_mask = np.array(limb_percent_mask, dtype=bool)
            
            ##################################################################
            """new Icont normalization removing high magnetic field regions"""
            # AR_temp = np.ones(Ic_temp.shape,dtype=bool)
            # # automatic bins looking at max std of the continuum polarization
            # lim = np.max((data[Ic_temp,1:,cpos_arr[0],scan]).std(axis=(0,1)))*7
            # bins = np.linspace(-lim,lim,150)

            # for p in range(1,4):
            #     hi = np.histogram(data[Ic_temp,p,:,scan].flatten(),bins=bins)
            #     gval = gaussian_fit(hi, show = False)
            #     AR_temp *= np.max(np.abs(data[:,:,p,:,scan] - gval[1]),axis=-1) < 5*gval[2]

            # AR_temp = np.asarray(AR_temp, dtype=bool)
            
            # # erosion and dilation to remove small scale masked elements
            # AR_temp = ~binary_dilation(binary_erosion(~AR_temp.copy(),generate_binary_structure(2,2), iterations=3),generate_binary_structure(2,2), iterations=3)

            AR_temp = ARmasking(data[...,scan], limb_percent_mask[:,:,scan], cpos = cpos_arr[scan])
            ##################################################################
            
            I_c[scan] = np.nanmean(data[Ic_temp*AR_temp,0,cpos_arr[0],int(scan)])
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
            # if ghost_c: #20211116
            #     ctalk_params = crosstalk_auto_ItoQUV(data[...,scan],cpos_arr[scan],cpos_arr[scan],roi=np.asarray(Ic_mask[...,scan]*field_stop_ghost[rows,cols],dtype=bool)) #20211116
            # else: #20211116
            #     ctalk_params = crosstalk_auto_ItoQUV(data[...,scan],cpos_arr[scan],cpos_arr[scan],roi=Ic_mask[...,scan]) #20211116
            
            cQ, cU, cV, sfitQ, sfitU, sfitV, data[...,scan] = crosstalk_2D_ItoQUV(data[...,scan],
                                                                                  False,
                                                                                  limb_percent_mask[...,scan],
                                                                                  mode=CTmode,
                                                                                  threshold = .5,
                                                                                  divisions = 16,
                                                                                  norma = 2,
                                                                                  VtoQU = VtoQU)
            # CTparams[...,scan] = ctalk_params
            
            # scan_hdr['CAL_CRT0'] = round(ctalk_params[slope,q],4) #I-Q slope
            # scan_hdr['CAL_CRT2'] = round(ctalk_params[slope,u],4) #I-U slope
            # scan_hdr['CAL_CRT4'] = round(ctalk_params[slope,v],4) #I-V slope
            # scan_hdr['CAL_CRT1'] = round(ctalk_params[offset,q],4) #I-Q offset
            # scan_hdr['CAL_CRT3'] = round(ctalk_params[offset,u],4) #I-U offset
            # scan_hdr['CAL_CRT5'] = round(ctalk_params[offset,v],4) #I-V offset
            
            # handling of the header is still missing
            scan_hdr['CAL_CRT0'] = round(np.mean(cQ[0]),4) #I-Q slope
            scan_hdr['CAL_CRT2'] = round(np.mean(cU[0]),4) #I-U slope
            scan_hdr['CAL_CRT4'] = round(np.mean(cV[0]),4) #I-V slope
            scan_hdr['CAL_CRT3'] = round(np.mean(cQ[1]),4) #I-U offset
            scan_hdr['CAL_CRT1'] = round(np.mean(cU[1]),4) #I-Q offset
            scan_hdr['CAL_CRT5'] = round(np.mean(cV[1]),4) #I-V offset
            
            scan_hdr['CAL_CRT6'] = 0 #V-Q slope
            scan_hdr['CAL_CRT8'] = 0 #V-U slope
            scan_hdr['CAL_CRT7'] = 0 #V-Q offset
            scan_hdr['CAL_CRT9'] = 0 #V-U offset
                
        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        printc(f"------------- I -> Q,U,V cross talk correction time: {np.round(time.perf_counter() - start_time,3)} seconds ",bcolors.OKGREEN)
        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        
        if (not iss_off or not PSFstokes['deconvolution']) and fs_c:
            data *= field_stop[rows,cols, np.newaxis, np.newaxis, np.newaxis]

    else:
        print(" ")
        printc('-->>>>>>> No ItoQUV mode',color=bcolors.WARNING)

    if VtoQU:
        if CTmode == 'jaeggli':
            printc('-->>>>>>> Cross-talk correction V to Q,U already applied',color=bcolors.OKGREEN)
        else:        
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
        
        if (not iss_off or not PSFstokes['deconvolution']) and fs_c:
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
        
        data, hdr_arr = wavelength_registration(data, cpos_arr, sly, slx, hdr_arr, deconv=PSFstokes['deconvolution'])
        
        if not PSFstokes['deconvolution']:
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

    if PSFstokes['deconvolution']:

        if out_intermediate:
            data_not_deconvolved = data.copy()
        
        start_time = time.perf_counter()
        
        print(" ")
        printc('-->>>>>>> PSF deconvolution on Stokes vectors',color=bcolors.OKGREEN)
        for scan in range(data_shape[-1]):
            ## Fatima's code
            # res_stokes, coefs = restore_stokes_cube(data[...,scan], hdr_arr[scan],aberr_cor=PSFaberr)
            # data[...,scan] = res_stokes
            # hdr_arr[scan]['CAL_PSF'] = 'Interpolated PSF; aberration: '+str(PSFaberr)
            ##

            ## Fran's code
            mask = np.ones((data_size[0],data_size[1]))
            if norm_stokes:
                if limb and ~PSFstokes['roi']:
                    mask = limb_mask[...,scan]
            if fs_c:
                mask = mask*field_stop[rows,cols]

            if iss_off:
                mask = binary_erosion(mask>0,generate_binary_structure(2,2), iterations=3)
            
            if np.sum(mask==0) == 0:
                mask = None
            
            if PSFstokes['roi']:
                psfy, psfx = sly, slx
            else:
                psfy, psfx = slice(0,data.shape[0]), slice(0,data.shape[1])
            # deconvolution on modulated data
            # dat, _ = demod_hrt(data[...,scan],pmp_temp,modulate=True)
            if cpos_arr[scan] == 5: # set continuum in the first wavelength for the deconvolution
                data[...,scan] = np.roll(data[...,scan], 1, axis = -1)
            
            if cavity_c:
                restore_results = fran_restore(data[...,scan], datetime.datetime.fromisoformat(hdr_arr[scan]['DATE-OBS']), rest=PSFstokes['method'], mask=mask, sly=psfy, slx=psfx,
                                             gamma2=PSFstokes['gamma2'], low_f=PSFstokes['low_f'], aberr_cor=PSFstokes['aberration_correction'], straylight_corr=PSFstokes['straylight_correction'],
                                             cavity=cavity[rows,cols], PD_f = PSFstokes['PD_f'])
                res_stokes, coefs, cavity = restore_results
            else:
                restore_results = fran_restore(data[...,scan], datetime.datetime.fromisoformat(hdr_arr[scan]['DATE-OBS']), rest=PSFstokes['method'], mask=mask, sly=psfy, slx=psfx,
                                             gamma2=PSFstokes['gamma2'], low_f=PSFstokes['low_f'], aberr_cor=PSFstokes['aberration_correction'],
                                             straylight_corr=PSFstokes['straylight_correction'], 
                                             cavity=None, PD_f = PSFstokes['PD_f'])
                res_stokes, coefs = restore_results
                cavity = None

            if cpos_arr[scan] == 5: # set continuum back in its position
                res_stokes = np.roll(res_stokes, -1, axis = -1)
            # res_stokes, _ = demod_hrt(res_stokes,pmp_temp)

            data[...,scan] = res_stokes
            hdr_arr[scan]['CAL_PSF'] = '{0:s} PSF deconv; gamma2={1:f}; low_f={2:f}; aberration: {3:}'.format(PSFstokes['method'],PSFstokes['gamma2'],PSFstokes['low_f'],PSFstokes['aberration_correction'],PSFstokes['straylight_correction'])
            ##
            
            hdr_arr[scan]['CAL_ZER'] = str(list(np.round(coefs,5)))

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
    # SET MEDIAN TO ZERO
    #-----------------

    if norm_stokes:
        print(" ")
        printc('-->>>>>>> Set Median to 0',color=bcolors.OKGREEN)
        for scan in range(data_shape[-1]):
            maski = limb_mask[...,scan] * AR_mask[...,scan]
            for l in range(data.shape[3]):
                    PQm = np.median(data[maski>0,1,l,scan])
                    PUm = np.median(data[maski>0,2,l,scan])
                    PVm = np.median(data[maski>0,3,l,scan])
                    
                    data[:,:,1,l,scan] -= PQm
                    data[:,:,2,l,scan] -= PUm
                    data[:,:,3,l,scan] -= PVm

                    # print('Median (wl,p) = ({:d},1): {:.2e}. After correction: {:.2e}'.format(l,PQm,np.median(data[maski>0,1,l,scan])))
                    # print('Median (wl,p) = ({:d},2): {:.2e}. After correction: {:.2e}'.format(l,PUm,np.median(data[maski>0,2,l,scan])))
                    # print('Median (wl,p) = ({:d},3): {:.2e}. After correction: {:.2e}'.format(l,PVm,np.median(data[maski>0,3,l,scan])))
                    # print('')
                    
    #-----------------
    # WRITE OUT STOKES VECTOR
    #-----------------

    #rewrite the PARENT keyword to the original FILENAME
    for count, scan in enumerate(data_f):
        hdr_arr[count]['PARENT'] = hdr_arr[count]['FILENAME'] 

    #write if the TemperatureCorrection is on or not
    for count, scan in enumerate(data_f):
        hdr_arr[count]['CAL_TEMP'] = str(TemperatureCorrection) 

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

    #-----------------
    # WRITE OUT INTERMEDIATE [wl,pol,y,x]
    #-----------------

    if out_intermediate: 
        #intermediate output with their own header, so they can be easily used as input for the pipeline
        for count, scan in enumerate(data_f):
            # root_scan_name = scan_name_list[count]
            root_scan_name = hdr_arr[count]['PHIDATID']
            hdr_arr[count]['DATE'] = ntime.strftime("%Y-%m-%dT%H:%M:%S")
            hdr_interm = hdr_arr[count].copy()

            if dark_c: 
                history_str = f"Intermediate. Version: {version}. Dark: {dark_c}. Prefilter: {False}. Flat: {False}, Unsharp: {False}. Flat norm: {False}. I->QUV ctalk: {False}. PSF deconvolution: {False}"
                file_suffix = 'dark_corrected'
                tmp = data_darkc[:,:,:,:,count]
                tmp = np.moveaxis(tmp, [-1,-2], [0,1])
                write_out_intermediate(tmp, hdr_interm, history_str, scan, root_scan_name, file_suffix, vrs, out_dir)
            
            if prefilter_c:
                history_str =  f"Intermediate. Version: {version}. Dark: {dark_c}. Prefilter: {prefilter_c}. Flat: {False}, Unsharp: {False}. Flat norm: {False}. I->QUV ctalk: {False}. PSF deconvolution: {False}"
                file_suffix = 'prefilter_corrected'
                tmp = data_PFc[:,:,:,:,count]
                tmp = np.moveaxis(tmp, [-1,-2], [0,1])
                write_out_intermediate(tmp, hdr_interm, history_str, scan, root_scan_name, file_suffix, vrs, out_dir)

            if flat_c: 
                history_str = f"Intermediate. Version: {version}. Dark: {dark_c}. Prefilter: {prefilter_c}. Flat: {flat_c}, Unsharp: {clean_f}. Flat norm: {norm_f}. I->QUV ctalk: {False}. PSF deconvolution: {False}"
                #with US
                file_suffix =  'flat_corrected'
                tmp = data_flatc[:,:,:,:,count]
                tmp = np.moveaxis(tmp, [-1,-2], [0,1])
                write_out_intermediate(tmp, hdr_interm, history_str, scan, root_scan_name, file_suffix, vrs, out_dir)
                #without US
                # root_scan_name_before_US = f"copy_{flat_f.split('/')[-1]}"
                # file_suffix = ''
                # write_out_intermediate(flat_copy, hdr_interm, history_str, scan, root_scan_name_before_US, file_suffix, out_dir)

            if demod:
                history_str = f"Intermediate. Version: {version}. Dark: {dark_c}. Prefilter: {prefilter_c}. Flat: {flat_c}, Unsharp: {clean_f}. Flat norm: {norm_f}. I->QUV ctalk: {False}. PSF deconvolution: {False}"
                file_suffix = 'demodulated'
                tmp = data_demod_normed[:,:,:,:,count]
                tmp = np.moveaxis(tmp, [-1,-2], [0,1])
                write_out_intermediate(tmp, hdr_interm, history_str, scan, root_scan_name, file_suffix, vrs, out_dir, bunit = 'I_CONT', btype = 'STOKES')

            if PSFstokes:
                history_str = f"Intermediate. Version: {version}. Dark: {dark_c}. Prefilter: {prefilter_c}. Flat: {flat_c}, Unsharp: {clean_f}. Flat norm: {norm_f}. I->QUV ctalk: {ItoQUV}. PSF deconvolution: {True}"
                file_suffix = 'stokes_noPSF'
                tmp = data_not_deconvolved[:,:,:,:,count]
                tmp = np.moveaxis(tmp, [-1,-2], [0,1])
                write_out_intermediate(tmp, hdr_interm, history_str, scan, root_scan_name, file_suffix, vrs, out_dir, bunit = 'I_CONT', btype = 'STOKES')
                
                if cavity_c:
                    new_cavity_f = out_dir + cavity_f.split('/')[-1].replace('cavity','cavityPSF').replace('V02','V'+hdr_interm['PHIDATID'])
                    print('Writing deconvolved cavity file')
                    with fits.open(cavity_f) as hdr_cavity:
                        hdr_cavity[0].data = cavity
                        hdr_cavity[0].header['PXBEG1'] = hdr_interm['PXBEG1']
                        hdr_cavity[0].header['PXBEG2'] = hdr_interm['PXBEG2']
                        hdr_cavity[0].header['PXEND1'] = hdr_interm['PXEND1']
                        hdr_cavity[0].header['PXEND2'] = hdr_interm['PXEND2']
                        hdr_cavity[0].header['HISTORY'] = 'Cavity deconvolved with PSF associated to '+scan
                        hdr_cavity.writeto(new_cavity_f,overwrite=True)
                        
    else:
        print(" ")
        printc('-->>>>>>> No intermediate files requested',color=bcolors.WARNING)

    #-----------------
    # WRITE OUT STOKES [wl,pol,y,x]
    #-----------------


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
            if cavity_c:
                hdr_arr[count]['CAL_CAVM'] = cavity_f
            hdr_arr[count] = data_hdr_kw(hdr_arr[count], data[:,:,:,:,count]) #add datamedn, datamean etc
            hdr_interm = hdr_arr[count].copy()
            hdr_arr[count]['HISTORY'] = f"Version: {version}. Dark: {dark_c}. Prefilter: {prefilter_c}. Flat: {flat_c}, Unsharp: {clean_f}. Flat norm: {norm_f}. I->QUV ctalk: {ItoQUV}. PSF deconvolution: {hdr_arr[count]['CAL_PSF']}. Cavity correction: {cavity_c}"
            
            with fits.open(scan) as hdu_list:
                print(f"Writing out stokes file as: {stokes_file}")
                tmp = data[:,:,:,:,count].astype(np.float32)
                hdu_list[0].data = np.moveaxis(tmp, [-1,-2], [0,1]) #want, 6,4,y,x to be consistent with FDT
                hdu_list[0].header = hdr_arr[count] #update the calibration keywords
                hdu_list[0].header.comments['NAXIS3'] = 'number of Stokes parameters (I, Q, U, V)'
                hdu_list[0].header.comments['NAXIS4'] = 'number of sampled wavelengths'
                hdu_list.writeto(out_dir + stokes_file, overwrite=True)  
            # hdu_list[0].header.set('NAXIS3',data.shape[2],'number of Stokes parameters (I, Q, U, V)',after='NAXIS2')
            # hdu_list[0].header.set('NAXIS4',data.shape[3],'number of sampled wavelengths','NAXIS3')
    else:
        print(" ")
        #check if already defined by input, otherwise generate
        scan_name_list = check_filenames(data_f)
        printc('-->>>>>>> No output Stokes file mode',color=bcolors.WARNING)

    #-----------------
    # INVERSION OF DATA WITH CMILOS
    #-----------------

    if RTE_options['rte'] in {"RTE","CE","CE+RTE","RTE+PSF","CE+RTE+PSF"}:
        #check out_dir has "/" character
        if out_dir[-1] != "/":
            print("Desired Output directory missing / character, will be added")
            out_dir = out_dir + "/"

        mask = np.ones((data_size[0],data_size[1],data_shape[-1]))
        if norm_stokes:
            if limb:
                mask = limb_mask
        if fs_c:
            mask = mask*field_stop[rows,cols,np.newaxis]
           
        if avg_stokes_before_rte:
            data = np.mean(data, axis = (-1))
            data_shape = (data_size[0], data_size[1], 1)

        # if pymilos_opt:
        #     RTE_code = 'pymilos' # it will become an input in the json (DEFAULT: 'pymilos')
        # else:
        #     RTE_code = 'cmilos'
        # options = []

        generate_l2(data_f, hdr_arr, wave_axis_arr, cpos_arr, 
                    data, mask, imgdirx_flipped, out_rte_filename, out_dir, 
                    cavity, rows, cols, vrs, out_synthesis,
                    **RTE_options)
                    
        # if pymilos_opt:
        #     #weight = np.asarray([1.,4.,5.4,4.1]); initial_model = np.asarray([400,30,120,1,0.05,1.5,.01,.22,.85])
        #     #py_cmilos(data_f, hdr_arr, wave_axis_arr, data_shape, cpos_arr, data, rte, mask, imgdirx_flipped, out_rte_filename, out_dir, weight = weight, initial_model = initial_model, vers = vrs)
        #     py_cmilos(data_f, hdr_arr, wave_axis_arr, data_shape, cpos_arr, data, rte, mask, imgdirx_flipped, out_rte_filename, out_dir, cavity_f, vers = vrs)
        # else:
        #     cmilos(data_f, hdr_arr, wave_axis_arr, data_shape, cpos_arr, data, rte, mask, imgdirx_flipped, out_rte_filename, out_dir, cavity_f, rows, cols, vers = vrs)

    else:
        print(" ")
        printc('-->>>>>>> No RTE Inversion mode',color=bcolors.WARNING)

    #-----------------
    # SAVING CONFIG FILE
    #-----------------
   
    if config:
        print(" ")
        printc('-->>>>>>> Saving copy of input config file ',color=bcolors.OKGREEN)

        # dt = datetime.datetime.fromtimestamp(overall_time)
        # runtime = dt.strftime("%d_%m_%YT%H_%M_%S")

        json.dump(input_dict, open(out_dir + f"config_file_{start_proc}.json", "w"))
        
    print(" ")
    printc('--------------------------------------------------------------',color=bcolors.OKGREEN)
    printc(f'------------ Reduction Complete: {np.round(time.perf_counter() - overall_time,3)} seconds',color=bcolors.OKGREEN)
    printc('--------------------------------------------------------------',color=bcolors.OKGREEN)
   
    return data

