from hrt_pipe import phihrt_pipe
import numpy as np
#import json

input_json_file = './input_jsons/sep_2021_L1_west_flat.json'

#prefilter_f = '../fits_files/fitted_prefilter.fits'

#######################################################################################################
#
# The fitted prefilter can be downloaded from: 
# http://www2.mps.mpg.de/data/outgoing/hirzberger/solo/RSCW3_helioseismology_test/fitted_prefilter.fits
# OR from BOB: /www/docs/data/outgoing/hirzberger/solo/RSCW3_helioseismology_test/fitted_prefilter.fits
#
#######################################################################################################


#out_names = ['nov_17_0051170001']#, '0024160032000_noflat', '0024160033000_noflat', '0024160034000_noflat', '0024160035000_noflat', '0024160036000_noflat', '0024160037000_noflat', '0024160038000_noflat', '0024160039000_noflat']

phihrt_pipe(input_json_file) 
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

clean_f: bool, DEFAULT: False
    clean the flat field with unsharp masking

sigma: int, DEFAULT: 59
    sigma of the gaussian convolution used for unsharp masking if clean_f == True 

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

field_stop: bool, DEFAULT True
    apply HRT field stop

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

config_file: bool, DEFAULT = True
    if True, will generate config.txt file that writes the reduction process steps done
"""