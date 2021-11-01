from hrt_pipe import phihrt_pipe
import numpy as np
import json

input_json_file = './input_jsons/nov_2020_L1.txt' # feb_2k_2021_L1.txt' # center_limb.txt' # sep11_500_2021.txt' # nov_2020_L1_feb_flats.txt' # 

dataset = 'nov2020'

input_dict = json.load(open(input_json_file))

data_f = input_dict['data_f']#[1:]
flat_f = input_dict['flat_f']
dark_f = input_dict['dark_f']

prefilter_f = '../fits_files/fitted_prefilter.fits'

#######################################################################################################
#
# The fitted prefilter can be downloaded from: 
# http://www2.mps.mpg.de/data/outgoing/hirzberger/solo/RSCW3_helioseismology_test/fitted_prefilter.fits
# OR from BOB: /www/docs/data/outgoing/hirzberger/solo/RSCW3_helioseismology_test/fitted_prefilter.fits
#
#######################################################################################################

c_talk_params = np.zeros((2,3))

if dataset == 'nov2020':
    q_slope = -0.0263#0.0038#-0.0140##-0.0098#
    u_slope = 0.0023#-0.0077#-0.0008##-0.0003#
    v_slope = -0.0116#-0.0009#-0.0073##-0.0070#

    q_int = 0.0138#-0.0056#0.0016#-0.0056#-0.0015# #the offset, normalised to I_c
    u_int = -0.0016#0.0031#0.0016##0.0007#
    v_int = 0.0057#-0.0002#0.0007##0.0006# 
    sigma = 49

if dataset == 'feb2021':
    q_slope = 0.0038#-0.0140##-0.0098#-0.0263#
    u_slope = -0.0077#-0.0008##-0.0003#0.0023#
    v_slope = -0.0009#-0.0073##-0.0070#-0.0116#

    q_int =-0.0056#0.0016#-0.0056#-0.0015# 0.0138# #the offset, normalised to I_c
    u_int = 0.0031#0.0016##0.0007#-0.0016#
    v_int = -0.0002#0.0007##0.0006# 0.0057#
    sigma = 59

if dataset == 'sep2021':
    q_slope = 0.0038#-0.0140##-0.0098#-0.0263#
    u_slope = -0.0077#-0.0008##-0.0003#0.0023#
    v_slope = -0.0009#-0.0073##-0.0070#-0.0116#

    q_int =-0.0056#0.0016#-0.0056#-0.0015# 0.0138# #the offset, normalised to I_c
    u_int = 0.0031#0.0016##0.0007#-0.0016#
    v_int = -0.0002#0.0007##0.0006# 0.0057#
    sigma = 59

if dataset == 'limb':
    q_slope = 0.0038#-0.0140##-0.0098#-0.0263#
    u_slope = -0.0077#-0.0008##-0.0003#0.0023#
    v_slope = -0.0009#-0.0073##-0.0070#-0.0116#

    q_int =-0.0056#0.0016#-0.0056#-0.0015# 0.0138# #the offset, normalised to I_c
    u_int = 0.0031#0.0016##0.0007#-0.0016#
    v_int = -0.0002#0.0007##0.0006# 0.0057#
    sigma = 59

c_talk_params[0,0] = q_slope
c_talk_params[0,1] = u_slope
c_talk_params[0,2] = v_slope

c_talk_params[1,0] = q_int
c_talk_params[1,1] = u_int
c_talk_params[1,2] = v_int

phihrt_pipe(data_f, flat_f = flat_f, dark_f = dark_f, 
            L1_input = True, scale_data = True, bit_conversion = True, accum_scaling = True, norm_f = True, 
            clean_f = 'blurring', sigma = sigma, clean_mode = 'V', flat_states = 24, norm_stokes = True, prefilter_f = None, 
            dark_c = True, flat_c = True, fs_c = True, limb = None, demod = True, ctalk_params = c_talk_params, 
            ItoQUV = True, out_demod_file = True, out_demod_filename = None, 
            out_dir = '/data/slam/home/calchetti/hrt_pipe_results/'+dataset+'/', out_intermediate = True, 
            rte = 'RTE', out_rte_filename=None, p_milos = False, cmilos_fits_opt = False, config_file = True) 
"""
 Input Parameters:
----------
data_f : list or string
    list containing paths to fits files of the raw HRT data OR string of path to one file  

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

accum_scaling: bool, DEFAULT True
    applies the scaling for the accumulation, (extracted from header)
    
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

out_demod_file: bool, DEFAULT: False
    output file with the stokes vectors to fits file

out_demod_filename: str, DEFAULT = None
    if None, takes last 10 characters of input scan filename (assumes its a DID), change if want other name

ItoQUV: bool, DEFAULT: False 
    apply I -> Q,U,V correction

ctalk_params: numpy arr, DEFAULT: None 
    cross talk parameters for ItoQUV, (2,3) numpy array required: first axis: Slope, Offset (Normalised to I_c) - second axis:  Q,U,V

rte: str, DEFAULT: False 
    invert using cmilos, options: 'RTE' for Milne Eddington Inversion, 'CE' for Classical Estimates, 'CE+RTE' for combined

out_rte_filename: str or list, DEFAULT = ''
    if '', takes last 10 characters of input scan filename (assumes its a DID), change if want other name(s) for each scan

out_intermediate: bool, DEFAULT = False
    if True, dark corrected and flat corrected data will be saved

config_file: bool, DEFAULT = True
    if True, will generate config.txt file that writes the reduction process steps done
"""
