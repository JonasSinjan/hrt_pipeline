
import json

#Nov 17 2020 L1 Exmaple

science_fits_filenames = ['solo_L1_phi-hrt-ilam_20201117T170209_V202108301639C_0051170001.fits.gz'] #note .gz here
flatfield_fits_filename = '/data/slam/home/sinjan/fits_files/april_avgd_2020_flat.fits'

darkfield_fits_filename = '../fits_files/solo_L0_phi-fdt-ilam_20200228T155100_V202002281636_0022210004_000.fits'

science_dir = '/data/solo/phi/data/fmdb/l1/2020-11-17/'

science = [science_dir + i for i in science_fits_filenames]

input_dict = {

    #input data
    'data_f' : science, #hrt pipeline allows multiple files at once to be processed (if same flat, dark, continuum position, pmp temp etc)
    'flat_f' : flatfield_fits_filename,
    'dark_f' : darkfield_fits_filename,
    
    #input/output type + scaling
    'L1_input' : False, 
    'L1_8_generate': False, #not developed yet
    'scale_data' : True,  #these 3 will be made redundant once L1 data scaling is normalised - needed mainly for comissioning (IP5) data
    'accum_scaling' : True, 
    'bit_conversion' : True, 
    
    #reduction
    'dark_c' : True,
    'flat_c' : True, 
    'TemperatureCorrection': None, #if True, wavelength corrected for the FG temperature
    'norm_f' : True, 
    'clean_f' : True, 
    'sigma' : 59, #unsharp masking gaussian width
    'clean_mode' : "V", #options 'QUV', 'UV', 'V' for the unsharp masking
    'flat_states' : 24, #options 4 (continuum only), 6 (one each wavelength), 9 (continuum + Stokes I in all the other wavelengths), 24
    'prefilter_f': None,
    'fs_c' : True, 
    'iss_off': True,
    'demod' : True, 
    'norm_stokes' : True, 
    'ItoQUV' : True,
    'VtoQU' : True,
    'PSForbit': False, #options: "perihelion" or "0.5"
    'PSFaberr': False, #if True, aberration-correction mode
    'ghost_c' : True,
    'cavity_f': None, #filename of the cavity maps used to shift the wavelengths before RTE inversion - #0263091100_flatPF-Tcorr-0_GAUS-FIT_header.fits'
    'rte' : False, #options: 'RTE', 'CE', 'CE+RTE'
    'pymilos' : False, #run python version of C-milos (~30% faster)
    
    #output dir/filenames
    'out_dir' : './',  
    'out_stokes_file' : False,  #if True, will save stokes array to fits, the array that is fed into the RTE inversions
    'out_stokes_filename' : None, #if specific and not default name
    'out_rte_filename' : None,  #if specific and not default name
    'config': True,
    'out_intermediate': False,
    # 'vers': 'V01', #if not given, version is yyyymmddhhMM
}

json.dump(input_dict, open(f"./input_jsons/nov_2020_L1.txt", "w"))
