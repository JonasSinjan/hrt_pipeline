from hrt_pipe import phihrt_pipe
import numpy as np

sciencedata_fits_filenames = ['solo_L0_phi-hrt-ilam_20200420T142022_V202004221451C_0024160031000.fits','solo_L0_phi-hrt-ilam_20200420T142252_V202004221452C_0024160032000.fits','solo_L0_phi-hrt-ilam_20200420T142522_V202004221457C_0024160033000.fits','solo_L0_phi-hrt-ilam_20200420T142752_V202004221511C_0024160034000.fits','solo_L0_phi-hrt-ilam_20200420T143023_V202004221517C_0024160035000.fits', 'solo_L0_phi-hrt-ilam_20200420T143253_V202004221518C_0024160036000.fits','solo_L0_phi-hrt-ilam_20200420T143523_V202004221522C_0024160037000.fits', 'solo_L0_phi-hrt-ilam_20200420T143753_V202004231605C_0024160038000.fits','solo_L0_phi-hrt-ilam_20200420T144023_V202004231605C_0024160039000.fits']#'/data/slam/home/sinjan/fits_files/solo_L0_phi-hrt-ilam_20200420T141752_V202004221450C_0024160030000.fits'#solo_L1_phi-hrt-ilam_20201117T170209_V202107090920C_0051170001.fits'#solo_L1_phi-hrt-ilam_20201117T170209_V202107060746C_0051170001.fits'#['solo_L1_phi-hrt-ilam_20200528T171109_V202106111600C_0045140102.fits']

# ['solo_L0_phi-hrt-ilam_20210421T120003_V202106080929C_0144210101.fits', 'solo_L0_phi-hrt-ilam_20210424T120003_V202106141014C_0144240101.fits',
#   'solo_L0_phi-hrt-ilam_20210425T120002_V202106141020C_0144250101.fits', 'solo_L0_phi-hrt-ilam_20210426T120002_V202106162118C_0144260101.fits',
#   'solo_L0_phi-hrt-ilam_20210427T120002_V202106162052C_0144270101.fits', 'solo_L0_phi-hrt-ilam_20210427T120002_V202106171444C_0144270101.fits', 
#   'solo_L0_phi-hrt-ilam_20210427T120002_V202106171517C_0144270101.fits'] ##['solo_L1_phi-hrt-ilam_20210223T170002_V202106111612C_0142230201.fits']
  
#['solo_L0_phi-hrt-ilam_0667414748_V202103221851C_0142230201.fits']#['solo_L0_phi-hrt-ilam_0667414905_V202103221851C_0142230602.fits', 'solo_L0_phi-hrt-ilam_0667415054_V202103221851C_0142230603.fits', 'solo_L0_phi-hrt-ilam_0667415205_V202103221851C_0142230604.fits', 'solo_L0_phi-hrt-ilam_0667415354_V202103221851C_0142230605.fits', 'solo_L0_phi-hrt-ilam_0667415505_V202103221851C_0142230606.fits', 'solo_L0_phi-hrt-ilam_0667415654_V202103221851C_0142230607.fits', 'solo_L0_phi-hrt-ilam_0667415805_V202103221851C_0142230608.fits']#['../fits_files/solo_L0_phi-hrt-ilam_0667414748_V202103221851C_0142230201.fits']
#sciencedata_fits_filenames = ['solo_L0_phi-hrt-ilam_0667414748_V202103221851C_0142230201.fits']
#sciencedata_fits_filenames = ['solo_L0_phi-hrt-ilam_0667414905_V202103221851C_0142230602.fits', 'solo_L0_phi-hrt-ilam_0667415054_V202103221851C_0142230603.fits', 'solo_L0_phi-hrt-ilam_0667415205_V202103221851C_0142230604.fits', 'solo_L0_phi-hrt-ilam_0667415354_V202103221851C_0142230605.fits', 'solo_L0_phi-hrt-ilam_0667415505_V202103221851C_0142230606.fits', 'solo_L0_phi-hrt-ilam_0667415654_V202103221851C_0142230607.fits', 'solo_L0_phi-hrt-ilam_0667415805_V202103221851C_0142230608.fits']

flatfield_fits_filename = '/data/slam/home/sinjan/fits_files/solo_L0_phi-hrt-ilam_20200417T174529_V202004241516C_0024150020000.fits'#solo_L0_phi-hrt-flat_0667134081_V202103221851C_0162201100.fits' #solo_L0_phi-hrt-ilam_20200417T174529_V202004241516C_0024150020000.fits'#solo_L1_phi-hrt-ilam_20200417T174538_V202106111549C_0024150020.fits'#solo_L0_phi-hrt-flat_0667134081_V202103221851C_0162201100.fits'
darkfield_fits_filename = '../fits_files/solo_L0_phi-fdt-ilam_20200228T155100_V202002281636_0022210004_000.fits'

sciencedata_fits_filenames = ['/data/slam/home/sinjan/fits_files/' + i for i in sciencedata_fits_filenames]

prefilter_f = '../fits_files/fitted_prefilter.fits'

#######################################################################################################
#
# The fitted prefilter can be downloaded from: 
# http://www2.mps.mpg.de/data/outgoing/hirzberger/solo/RSCW3_helioseismology_test/fitted_prefilter.fits
# OR from BOB: /www/docs/data/outgoing/hirzberger/solo/RSCW3_helioseismology_test/fitted_prefilter.fits
#
#######################################################################################################

c_talk_params = np.zeros((2,3))

q_slope = -0.0098#0.0038
u_slope = -0.0003#-0.0077
v_slope = -0.0070#-0.0009

q_int = -0.0015#-0.0056 #the offset, normalised to I_c
u_int = 0.0007#0.0031
v_int = 0.0006#-0.0002 

c_talk_params[0,0] = q_slope
c_talk_params[0,1] = u_slope
c_talk_params[0,2] = v_slope

c_talk_params[1,0] = q_int
c_talk_params[1,1] = u_int
c_talk_params[1,2] = v_int

out_names = ['0024160031000_noflat', '0024160032000_noflat', '0024160033000_noflat', '0024160034000_noflat', '0024160035000_noflat', '0024160036000_noflat', '0024160037000_noflat', '0024160038000_noflat', '0024160039000_noflat']

phihrt_pipe(sciencedata_fits_filenames, flat_f = flatfield_fits_filename, dark_f = darkfield_fits_filename, scale_data = False, 
            bit_flat = True, norm_f = True, clean_f = False, sigma = 59, flat_states = 24, norm_stokes = True, prefilter_f = None, 
            dark_c = False, flat_c = False, fs_c = True, demod = True, ctalk_params = c_talk_params, ItoQUV = False, out_demod_file = True, 
            out_demod_filename = out_names, out_dir = '/data/slam/home/sinjan/hrt_pipe_results/april_2020/', rte = 'False', out_rte_filename='') 
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
scale_data: bool, DEFAULT True
    performs the accumulation scaling + conversion for flat and science (only FALSE for commissioning data)
    
bit_flat: bool, DEFAULT True
    divides the scan + flat by 256 to convert from 24.8bit to 32bits

norm_f: bool, DEFAULT: True
    to normalise the flat fields before applying

clean_f: bool, DEFAULT: False
    clean the flat field with unsharp masking

sigma: int, DEFAULT: 59
    sigma of the gaussian convolution used for unsharp masking if clean_f == True 

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

out_rte_filename: str, DEFAULT = ''
    if '', takes last 10 characters of input scan filename (assumes its a DID), change if want other name
"""