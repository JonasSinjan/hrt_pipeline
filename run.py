from hrt_pipe import phihrt_pipe
import numpy as np
import json

#sciencedata_fits_filenames = ['solo_L0_phi-hrt-ilam_20201117T170209_V202106300922C_0051170001.fits']##'/data/slam/home/sinjan/fits_files/solo_L0_phi-hrt-ilam_20200420T141752_V202004221450C_0024160030000.fits'#solo_L1_phi-hrt-ilam_20201117T170209_V202107090920C_0051170001.fits'#solo_L1_phi-hrt-ilam_20201117T170209_V202107060746C_0051170001.fits'#['solo_L1_phi-hrt-ilam_20200528T171109_V202106111600C_0045140102.fits']

# ['solo_L0_phi-hrt-ilam_20210421T120003_V202106080929C_0144210101.fits', 'solo_L0_phi-hrt-ilam_20210424T120003_V202106141014C_0144240101.fits',
#   'solo_L0_phi-hrt-ilam_20210425T120002_V202106141020C_0144250101.fits', 'solo_L0_phi-hrt-ilam_20210426T120002_V202106162118C_0144260101.fits',
#   'solo_L0_phi-hrt-ilam_20210427T120002_V202106162052C_0144270101.fits', 'solo_L0_phi-hrt-ilam_20210427T120002_V202106171444C_0144270101.fits', 
#   'solo_L0_phi-hrt-ilam_20210427T120002_V202106171517C_0144270101.fits'] ##['solo_L1_phi-hrt-ilam_20210223T170002_V202106111612C_0142230201.fits']
  
# #['solo_L0_phi-hrt-ilam_0667414748_V202103221851C_0142230201.fits']#['solo_L0_phi-hrt-ilam_0667414905_V202103221851C_0142230602.fits', 'solo_L0_phi-hrt-ilam_0667415054_V202103221851C_0142230603.fits', 'solo_L0_phi-hrt-ilam_0667415205_V202103221851C_0142230604.fits', 'solo_L0_phi-hrt-ilam_0667415354_V202103221851C_0142230605.fits', 'solo_L0_phi-hrt-ilam_0667415505_V202103221851C_0142230606.fits', 'solo_L0_phi-hrt-ilam_0667415654_V202103221851C_0142230607.fits', 'solo_L0_phi-hrt-ilam_0667415805_V202103221851C_0142230608.fits']#['../fits_files/solo_L0_phi-hrt-ilam_0667414748_V202103221851C_0142230201.fits']
# sciencedata_fits_filenames = ['solo_L0_phi-hrt-ilam_0667414905_V202103221851C_0142230602.fits']#['solo_L0_phi-hrt-ilam_0667414748_V202103221851C_0142230201.fits']
# #sciencedata_fits_filenames = ['solo_L0_phi-hrt-ilam_0667414905_V202103221851C_0142230602.fits', 'solo_L0_phi-hrt-ilam_0667415054_V202103221851C_0142230603.fits', 'solo_L0_phi-hrt-ilam_0667415205_V202103221851C_0142230604.fits', 'solo_L0_phi-hrt-ilam_0667415354_V202103221851C_0142230605.fits', 'solo_L0_phi-hrt-ilam_0667415505_V202103221851C_0142230606.fits', 'solo_L0_phi-hrt-ilam_0667415654_V202103221851C_0142230607.fits', 'solo_L0_phi-hrt-ilam_0667415805_V202103221851C_0142230608.fits']

# flatfield_fits_filename = '/data/slam/home/sinjan/fits_files/solo_L0_phi-hrt-flat_0667134081_V202103221851C_0162201100.fits'#solo_L0_phi-hrt-flat_0667134081_V202103221851C_0162201100.fits'#solo_L0_phi-hrt-ilam_20200417T174529_V202004241516C_0024150020000.fits' #solo_L0_phi-hrt-ilam_20200417T174529_V202004241516C_0024150020000.fits'#solo_L1_phi-hrt-ilam_20200417T174538_V202106111549C_0024150020.fits'#solo_L0_phi-hrt-flat_0667134081_V202103221851C_0162201100.fits'

input_json_file = './input_jsons/nov_2020_L1.txt'

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

q_slope = -0.0263#0.0038#-0.0140##-0.0098#
u_slope = 0.0023#-0.0077#-0.0008##-0.0003#
v_slope = -0.0116#-0.0009#-0.0073##-0.0070#

q_int = 0.0138#-0.0056#0.0016#-0.0056#-0.0015# #the offset, normalised to I_c
u_int = -0.0016#0.0031#0.0016##0.0007#
v_int = 0.0057#-0.0002#0.0007##0.0006# 

c_talk_params[0,0] = q_slope
c_talk_params[0,1] = u_slope
c_talk_params[0,2] = v_slope

c_talk_params[1,0] = q_int
c_talk_params[1,1] = u_int
c_talk_params[1,2] = v_int

#out_names = ['nov_17_0051170001']#, '0024160032000_noflat', '0024160033000_noflat', '0024160034000_noflat', '0024160035000_noflat', '0024160036000_noflat', '0024160037000_noflat', '0024160038000_noflat', '0024160039000_noflat']

phihrt_pipe(data_f, flat_f = flat_f, dark_f = dark_f, scale_data = False, bit_conversion = False, accum_scaling = True, norm_f = True, 
            clean_f = True, sigma = 49, flat_states = 24, norm_stokes = True, prefilter_f = None, 
            dark_c = True, flat_c = True, fs_c = True, demod = True, ctalk_params = c_talk_params, 
            ItoQUV = True, out_demod_file = True, out_demod_filename = None, 
            out_dir = '/data/slam/home/sinjan/hrt_pipe_results/nov_17_april_flats_uv_clean/', rte = 'False', 
            out_rte_filename=None, p_milos = False, config_file = True) 
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