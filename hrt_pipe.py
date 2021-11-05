from posix import listdir
import numpy as np 
import os.path
from astropy.io import fits
# from scipy.ndimage import gaussian_filter
import time
from operator import itemgetter
import json
import matplotlib.pyplot as plt
from numpy.core.numeric import True_

from utils import *
from hrt_pipe_sub import *


def phihrt_pipe(input_json_file):

    '''
    PHI-HRT data reduction pipeline
    1. read in science data (+ OPTION: scaling) open path option + open for several scans at once
    2. read in flat field (+scaling)- just accepts one flat field fits file
    3. read in dark field (+scaling)
    4. apply dark field
    5. option to clean flat field with unsharp masking
    6. normalise flat field
    7. apply flat field
    8. apply prefilter
    9. read in field stop
    10. apply field stop
    11. demodulate with const demod matrix
        a) option to output demod to fits file
    12. normalise to quiet sun
    13. calibration
        a) ghost correction - not implemented yet
        b) cross talk correction
    14. rte inversion with pmilos/cmilos (CE, RTE or CE+RTE)
        a) output rte data products to fits file

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
    L1_8_generate: bool, DEFAULT False
        if True, assumes L1 input, and generates RTE output with the calibration header information
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
    limb: str, DEFAULT None
        specify if it is a limb observation, options are 'N', 'S', 'W', 'E'
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
    ctalk_params: numpy arr, DEFAULT: None 
        cross talk parameters for ItoQUV, (2,3) numpy array required: first axis: Slope, Offset (Normalised to I_c) - second axis:  Q,U,V
    rte: str, DEFAULT: False 
        invert using cmilos, options: 'RTE' for Milne Eddington Inversion, 'CE' for Classical Estimates, 'CE+RTE' for combined
    out_rte_filename: str, DEFAULT = ''
        if '', takes last 10 characters of input scan filename (assumes its a DID), change if want other name
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

    printc('--------------------------------------------------------------',bcolors.OKGREEN)
    printc('PHI HRT data reduction software  ',bcolors.OKGREEN)
    printc('--------------------------------------------------------------',bcolors.OKGREEN)

    #-----------------
    # READ INPUT JSON
    #-----------------
    
    input_dict = json.load(open(input_json_file))

    data_f = input_dict['data_f']
    flat_f = input_dict['flat_f']
    dark_f = input_dict['dark_f']

    L1_input = input_dict['L1_input']
    L1_8_generate = input_dict['L1_8_generate']
    scale_data = input_dict['scale_data']
    accum_scaling = input_dict['accum_scaling']
    bit_conversion = input_dict['bit_conversion']

    dark_c = input_dict['dark_c']
    flat_c = input_dict['flat_c']
    norm_f = input_dict['norm_f']
    clean_f = input_dict['clean_f']
    sigma = input_dict['sigma']
    clean_mode = input_dict['clean_mode']
    flat_states = input_dict['flat_states']
    prefilter_f = input_dict['prefilter_f']
    fs_c = input_dict['fs_c']
    limb = input_dict['limb']
    demod = input_dict['demod']
    norm_stokes = input_dict['norm_stokes']
    CT_ItoQUV = input_dict['ItoQUV']
    ctalk_params = input_dict['ctalk_params']
    rte = input_dict['rte']
    p_milos = input_dict['p_milos']
    cmilos_fits_opt = input_dict['cmilos_fits_opt']

    out_dir = input_dict['out_dir']
    out_stokes_file = input_dict['out_stokes_file']
    out_stokes_filename = input_dict['out_stokes_filename']
    out_rte_filename = input_dict['out_rte_filename']
    
    overall_time = time.time()

    if L1_input:
        print("L1_input param set to True - Assuming L1 science data")
        accum_scaling = True 
        bit_conversion = True
        scale_data = True

    #-----------------
    # READ DATA
    #-----------------

    print(" ")
    printc('-->>>>>>> Reading Data',color=bcolors.OKGREEN) 

    start_time = time.time()

    if isinstance(data_f, str):
        data_f = [data_f]

    if isinstance(data_f, list):
        #if the data_f contains several scans
        printc(f'Input contains {len(data_f)} scan(s)',color=bcolors.OKGREEN)
        
        number_of_scans = len(data_f)

        data_arr = [0]*number_of_scans
        hdr_arr = [0]*number_of_scans

        wve_axis_arr = [0]*number_of_scans
        cpos_arr = [0]*number_of_scans
        voltagesData_arr = [0]*number_of_scans
        tuning_constant_arr = [0]*number_of_scans

        for scan in range(number_of_scans):
            data_arr[scan], hdr_arr[scan] = get_data(data_f[scan], scaling = accum_scaling, bit_convert_scale = bit_conversion, scale_data = scale_data)

            wve_axis_arr[scan], voltagesData_arr[scan], tuning_constant_arr[scan], cpos_arr[scan] = fits_get_sampling(data_f[scan],verbose = True)

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

        #so that data is [y,x,24,scans]
        data = np.stack(data_arr, axis = -1)
        data = np.moveaxis(data, 0,-2) 

        print(f"Data shape is {data.shape}")

        #--------
        # test if the scans have same IMGDIRX keyword
        #--------
    
        header_imgdirx_exists, imgdirx_flipped = check_IMGDIRX(hdr_arr)
    
    else:
        printc("ERROR, data_f argument is neither a string nor list containing strings: {} \n Ending Process",data_f,color=bcolors.FAIL)
        exit()

    data_shape = data.shape

    data_size = data_shape[:2]
    
    #converting to [y,x,pol,wv,scans]

    data = stokes_reshape(data)

    #enabling cropped datasets, so that the correct regions of the dark field and flat field are applied
    print("Data reshaped to: ", data.shape)

    diff = 2048-data_size[0] #handling 0/2 errors
    
    if np.abs(diff) > 0:
    
        start_row = int((2048-data_size[0])/2)
        start_col = int((2048-data_size[1])/2)
        
    else:
        start_row, start_col = 0, 0
    
    rows = slice(start_row,start_row + data_size[0])
    cols = slice(start_col,start_col + data_size[1])
    ceny = slice(data_size[0]//2 - data_size[0]//4, data_size[0]//2 + data_size[0]//4)
    cenx = slice(data_size[1]//2 - data_size[1]//4, data_size[1]//2 + data_size[1]//4)
    
    printc('--------------------------------------------------------------',bcolors.OKGREEN)
    printc(f"------------ Load science data time: {np.round(time.time() - start_time,3)} seconds",bcolors.OKGREEN)
    printc('--------------------------------------------------------------',bcolors.OKGREEN)


    #-----------------
    # READ FLAT FIELDS
    #-----------------

    if flat_c:
        print(" ")
        printc('-->>>>>>> Reading Flats',color=bcolors.OKGREEN)

        start_time = time.time()
        
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

#             flat[1345, 296:, 1, 1] = flat_copy[1344, 296:, 1, 1]
#             flat[1346, :291, 1, 1] = flat_copy[1345, :291, 1, 1]
            del flat_copy
            
        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        printc(f"------------ Load flats time: {np.round(time.time() - start_time,3)} seconds",bcolors.OKGREEN)
        printc('--------------------------------------------------------------',bcolors.OKGREEN)


    else:
        print(" ")
        printc('-->>>>>>> No flats mode',color=bcolors.WARNING)


    #-----------------
    # READ AND CORRECT DARK FIELD
    #-----------------

    if dark_c:
        print(" ")
        printc('-->>>>>>> Reading Darks                   ',color=bcolors.OKGREEN)

        start_time = time.time()

        try:

            if dark_f[-19:] != '0022210004_000.fits':
                dark,h = get_data(dark_f,scaling = accum_scaling, bit_convert_scale = bit_conversion,scale_data = scale_data) 
            else:
                dark,h = get_data(dark_f, scaling = accum_scaling, bit_convert_scale = bit_conversion, scale_data = False)
            
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
            printc(f"------------ Load darks time: {np.round(time.time() - start_time,3)} seconds",bcolors.OKGREEN)
            printc('--------------------------------------------------------------',bcolors.OKGREEN)

        except Exception:
            printc("ERROR, Unable to open darks file: {}",dark_f,color=bcolors.FAIL)
        
        #-----------------
        # LOAD DARK
        #-----------------  


        #-----------------
        # APPLY DARK CORRECTION 
        #-----------------  

        if flat_c == False:
            flat = np.empty((2048,2048,4,6))

        data, flat = apply_dark_correction(data, flat, dark, header_imgdirx_exists, imgdirx_flipped, rows, cols)  
        
        if flat_c == False:
            flat = np.empty((2048,2048,4,6))

        # if out_intermediate:
        #     data_darkc = data.copy()
            
        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        printc(f"------------- Dark Field correction time: {np.round(time.time() - start_time,3)} seconds",bcolors.OKGREEN)
        printc('--------------------------------------------------------------',bcolors.OKGREEN)

        #data, flat = apply_dark_correction(data, flat, dark, header_imgdirx_exists, imgdirx_flipped, rows, cols)  
        


    else:
        print(" ")
        printc('-->>>>>>> No dark mode',color=bcolors.WARNING)


    #-----------------
    # OPTIONAL Unsharp Masking clean the flat field stokes V images
    #-----------------

    if clean_f is not None and flat_c:
        print(" ")
        printc('-->>>>>>> Cleaning flats with Unsharp Masking',color=bcolors.OKGREEN)

        start_time = time.time()

        flat = unsharp_masking(flat,sigma,flat_pmp_temp,cpos_arr,clean_mode, clean_f = "blurring")
        
        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        printc(f"------------- Cleaning flat time: {np.round(time.time() - start_time,3)} seconds",bcolors.OKGREEN)
        printc('--------------------------------------------------------------',bcolors.OKGREEN)

    else:
        print(" ")
        printc('-->>>>>>> No clean flats mode',color=bcolors.WARNING)


    #-----------------
    # NORM FLAT FIELDS
    #-----------------

    if norm_f and flat_c:
        
        flat = normalise_flat(flat, flat_f, ceny, cenx)

    else:
        print(" ")
        printc('-->>>>>>> No normalising flats mode',color=bcolors.WARNING)


    #-----------------
    # APPLY FLAT CORRECTION 
    #-----------------

    if flat_c:
        try:
            data = flat_correction(data,flat,flat_states,rows,cols)
            
            # if out_intermediate:
            #     data_flatc = data.copy()
            
            printc('--------------------------------------------------------------',bcolors.OKGREEN)
            printc(f"------------- Flat Field correction time: {np.round(time.time() - start_time,3)} seconds ",bcolors.OKGREEN)
            printc('--------------------------------------------------------------',bcolors.OKGREEN)
        except: 
          printc("ERROR, Unable to apply flat fields",color=bcolors.FAIL)

    else:
        print(" ")
        printc('-->>>>>>> No flat field correction mode',color=bcolors.WARNING)


    #-----------------
    # PREFILTER CORRECTION  
    #-----------------

    if prefilter_f is not None:
        print(" ")
        printc('-->>>>>>> Prefilter Correction',color=bcolors.OKGREEN)

        start_time = time.time()

        prefilter_voltages = [-1300.00,-1234.53,-1169.06,-1103.59,-1038.12,-972.644,-907.173,-841.702,-776.231,-710.760,-645.289,
                            -579.818,-514.347,-448.876,-383.404,-317.933,-252.462,-186.991,-121.520,-56.0490,9.42212,74.8932,
                            140.364,205.835,271.307, 336.778,402.249,467.720,533.191,598.662,664.133,729.604,795.075,860.547,
                            926.018,991.489,1056.96,1122.43,1187.90,1253.37, 1318.84,1384.32,1449.79,1515.26,1580.73,1646.20,
                            1711.67,1777.14,1842.61]

        prefilter, _ = load_fits(prefilter_f)

        #prefilter = prefilter[:,652:1419,613:1380] #crop the helioseismology data

        data = prefilter_correction(data,voltagesData_arr,prefilter,prefilter_voltages)


        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        printc(f"------------- Prefilter correction time: {np.round(time.time() - start_time,3)} seconds",bcolors.OKGREEN)
        printc('--------------------------------------------------------------',bcolors.OKGREEN)

    else:
        print(" ")
        printc('-->>>>>>> No prefilter mode',color=bcolors.WARNING)


    #-----------------
    # FIELD STOP 
    #-----------------

    if fs_c:
        data, field_stop = apply_field_stop(data, rows, cols, header_imgdirx_exists, imgdirx_flipped)


    else:
        print(" ")
        printc('-->>>>>>> No field stop mode',color=bcolors.WARNING)


    #-----------------
    # APPLY DEMODULATION 
    #-----------------

    if demod:

        print(" ")
        printc('-->>>>>>> Demodulating data',color=bcolors.OKGREEN)

        start_time = time.time()

        data,_ = demod_hrt(data, pmp_temp)
        
        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        printc(f"------------- Demodulation time: {np.round(time.time() - start_time,3)} seconds ",bcolors.OKGREEN)
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
        
        start_time = time.time()

        for scan in range(data_shape[-1]):
            
            #I_c = np.mean(data[ceny,cenx,0,cpos_arr[0],int(scan)]) #mean of central 1k x 1k of continuum stokes I
            #I_c = np.mean(data[50:500,700:1700,0,cpos_arr[0],int(scan)]) # mean in the not-out-of the Sun north limb
            #I_c = np.mean(data[1500:2000,800:1300,0,cpos_arr[0],int(scan)]) # mean in the not-out-of the Sun south limb
            #I_c = np.mean(data[350:1700,200:900,0,cpos_arr[0],int(scan)]) # West limb
            limb_copy = np.copy(data)
            #from Daniele Calchetti
            if limb is not None:
                if limb == 'N':
                    limb_mask, Ic_mask = limb_fitting(data[:,:,0,cpos_arr[0],int(scan)], mode = 'columns', switch = True)
                if limb == 'S':
                    limb_mask, Ic_mask = limb_fitting(data[:,:,0,cpos_arr[0],int(scan)], mode = 'columns', switch = False)
                if limb == 'W':
                    limb_mask, Ic_mask = limb_fitting(data[:,:,0,cpos_arr[0],int(scan)], mode = 'rows', switch = True)
                if limb == 'E':
                    limb_mask, Ic_mask = limb_fitting(data[:,:,0,cpos_arr[0],int(scan)], mode = 'rows', switch = False)
                
                limb_mask = np.where(limb_mask>0,1,0)
                Ic_mask = np.where(Ic_mask>0,1,0)
                
                data[:,:,:,:,scan] = data[:,:,:,:,scan] * limb_mask[:,:,np.newaxis,np.newaxis]
            else:
                Ic_mask = np.zeros(data_size)
                Ic_mask[ceny,cenx] = 1
                Ic_mask = np.where(Ic_mask>0,1,0)
             
            if fs_c:
                Ic_mask *= field_stop
            
            Ic_mask = np.array(Ic_mask, dtype=bool)
            I_c = np.mean(data[Ic_mask,0,cpos_arr[0],int(scan)])
            data[:,:,:,:,scan] = data[:,:,:,:,scan]/I_c 
            
        # if out_intermediate:
        #     data_demod = data.copy()
        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        printc(f"------------- Stokes Normalising time: {np.round(time.time() - start_time,3)} seconds ",bcolors.OKGREEN)
        printc('--------------------------------------------------------------',bcolors.OKGREEN)

    else:
        print(" ")
        printc('-->>>>>>> No normalising Stokes mode',color=bcolors.WARNING)


    #-----------------
    # CROSS-TALK CALCULATION 
    #-----------------

    if CT_ItoQUV:
        
        print(" ")
        printc('-->>>>>>> Cross-talk correction I to Q,U,V ',color=bcolors.OKGREEN)

        start_time = time.time()

        num_of_scans = data_shape[-1]

        try:
            assert ctalk_params.shape == (2,3)
        except AssertionError:
            print("ctalk_params is not in the required (2,3) shape, please reconcile")


        slope, offset = 0, 1
        q, u, v = 0, 1, 2

        for scan_hdr in hdr_arr:
            if 'CAL_CRT0' in scan_hdr: #check to make sure the keywords exist
                scan_hdr['CAL_CRT0'] = ctalk_params[slope,q] #I-Q slope
                scan_hdr['CAL_CRT2'] = ctalk_params[slope,u] #I-U slope
                scan_hdr['CAL_CRT4'] = ctalk_params[slope,v] #I-V slope

                scan_hdr['CAL_CRT1'] = ctalk_params[offset,q] #I-Q offset
                scan_hdr['CAL_CRT3'] = ctalk_params[offset,u] #I-U offset
                scan_hdr['CAL_CRT5'] = ctalk_params[offset,v] #I-V offset

        ctalk_params = np.repeat(ctalk_params[:,:,np.newaxis], num_of_scans, axis = 2)

        data = CT_ItoQUV(data, ctalk_params, norm_stokes, cpos_arr, Ic_mask)


        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        printc(f"------------- I -> Q,U,V cross talk correction time: {np.round(time.time() - start_time,3)} seconds ",bcolors.OKGREEN)
        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        
        data *= field_stop[rows,cols, np.newaxis, np.newaxis, np.newaxis]
        # DC change 20211019 only for limb
        if limb is not None:
            data *= limb_mask[rows,cols, np.newaxis, np.newaxis, np.newaxis]

    else:
        print(" ")
        printc('-->>>>>>> No ItoQUV mode',color=bcolors.WARNING)


    #-----------------
    #CHECK FOR INFs
    #-----------------

    data[np.isinf(data)] = 0
    data[np.isnan(data)] = 0

 
    #-----------------
    #WRITE OUT STOKES VECTOR
    #-----------------

    #these two ifs need to be outside out_stokes_file if statement - needed for inversion
    if out_dir[-1] != "/":
        print("Desired Output directory missing / character, will be added")
        out_dir = out_dir + "/"

        #check if the output directory exists, if not, create it
    if not os.path.exists(out_dir): 
        print(f"{out_dir} does not exist -->>>>>>> Creating it")
        os.makedirs(out_dir)


    if out_stokes_file:
        
        print(" ")
        printc('Saving demodulated data to one _reduced.fits file per scan')

        if out_stokes_filename is not None:

            if isinstance(out_stokes_filename,str):
                out_stokes_filename = [out_stokes_filename]

            if int(len(out_stokes_filename)) == int(data_shape[-1]):
                scan_name_list = out_stokes_filename
                scan_name_defined = True
            else:
                print("Input demod filenames do not match the number of input arrays, reverting to default naming")
                scan_name_defined = False
        else:
            scan_name_defined = False

        if not scan_name_defined: #check if already defined by input, otherwise generate
            scan_name_list = check_filenames(data_f)
        

        for count, scan in enumerate(data_f):

            with fits.open(scan) as hdu_list:
                print(f"Writing out demod file as: {scan_name_list[count]}_reduced.fits")
                hdu_list[0].data = data[:,:,:,:,count]
                hdu_list[0].header = hdr_arr[count] #update the calibration keywords
                hdu_list.writeto(out_dir + scan_name_list[count] + '_reduced.fits', overwrite=True)
            
            # DC change 20211014
            """
            if out_intermediate:
                with fits.open(scan) as hdu_list:
                    print(f"Writing out demod file as: {scan_name_list[count]}_dark_corrected.fits")
                    hdu_list[0].data = data_darkc[:,:,:,:,count]
                    hdu_list[0].header = hdr_arr[count] #update the calibration keywords
                    hdu_list.writeto(out_dir + scan_name_list[count] + '_dark_corrected.fits', overwrite=True)

                with fits.open(scan) as hdu_list:
                    print(f"Writing out demod file as: {scan_name_list[count]}_flat_corrected.fits")
                    hdu_list[0].data = data_flatc[:,:,:,:,count]
                    hdu_list[0].header = hdr_arr[count] #update the calibration keywords
                    hdu_list.writeto(out_dir + scan_name_list[count] + '_flat_corrected.fits', overwrite=True)
                    
                with fits.open(scan) as hdu_list:
                    print(f"Writing out demod file as: {scan_name_list[count]}_demodulated.fits")
                    hdu_list[0].data = data_demod[:,:,:,:,count]
                    hdu_list[0].header = hdr_arr[count] #update the calibration keywords
                    hdu_list.writeto(out_dir + scan_name_list[count] + '_demodulated.fits', overwrite=True)
            """

            # with fits.open(scan) as hdu_list:
            #     hdu_list[0].data = limb_copy
            #     hdu_list.writeto(out_dir+scan_name_list[count]+'_limb_fit_input.fits', overwrite=True)

    else:
        print(" ")
        #check if already defined by input, otherwise generate
        scan_name_list = check_filenames(data_f)
        printc('-->>>>>>> No output demod file mode',color=bcolors.WARNING)

    #-----------------
    # INVERSION OF DATA WITH CMILOS
    #-----------------

    if rte == 'RTE' or rte == 'CE' or rte == 'CE+RTE' or rte == 'RTE_seq':

        #check out_dir has "/" character
        if out_dir[-1] != "/":
            print("Desired Output directory missing / character, will be added")
            out_dir = out_dir + "/"

        
        if limb is not None:
            mask = limb_mask*field_stop
        
        if p_milos:

            try:
                pmilos(data_f, wve_axis_arr, data_shape, cpos_arr, data, rte, mask, start_row, start_col, out_rte_filename, out_dir)
                    
            except ValueError:
                print("Running CMILOS instead!")
                cmilos(data_f, hdr_arr, wve_axis_arr, data_shape, cpos_arr, data, rte, mask, start_row, start_col, out_rte_filename, out_dir)

        else:
            if cmilos_fits_opt:

                 cmilos_fits(data_f, hdr_arr, wve_axis_arr, data_shape, cpos_arr, data, rte, mask, start_row, start_col, out_rte_filename, out_dir)
            else:
                cmilos(data_f, hdr_arr, wve_axis_arr, data_shape, cpos_arr, data, rte, mask, start_row, start_col, out_rte_filename, out_dir)

    else:
        print(" ")
        printc('-->>>>>>> No RTE Inversion mode',color=bcolors.WARNING)

    print(" ")
    printc('--------------------------------------------------------------',color=bcolors.OKGREEN)
    printc(f'------------ Reduction Complete: {np.round(time.time() - overall_time,3)} seconds',color=bcolors.OKGREEN)
    printc('--------------------------------------------------------------',color=bcolors.OKGREEN)

    if flat_c:
        return data, flat
    else:
        return data
