import json
import numpy as np

"""

#April 2020

# CAN also do the L1 files - but must set bit_convert_scale to false, and scale data to False for the flat fields, and presumably same for the input data

science_april = ['solo_L0_phi-hrt-ilam_20200420T141752_V202004221450C_0024160030000.fits',
'solo_L0_phi-hrt-ilam_20200420T142022_V202004221451C_0024160031000.fits',
'solo_L0_phi-hrt-ilam_20200420T142252_V202004221452C_0024160032000.fits',
'solo_L0_phi-hrt-ilam_20200420T142522_V202004221457C_0024160033000.fits',
'solo_L0_phi-hrt-ilam_20200420T142752_V202004221511C_0024160034000.fits',
'solo_L0_phi-hrt-ilam_20200420T143023_V202004221517C_0024160035000.fits', 
'solo_L0_phi-hrt-ilam_20200420T143253_V202004221518C_0024160036000.fits',
'solo_L0_phi-hrt-ilam_20200420T143523_V202004221522C_0024160037000.fits', 
'solo_L0_phi-hrt-ilam_20200420T143753_V202004231605C_0024160038000.fits',
'solo_L0_phi-hrt-ilam_20200420T144023_V202004231605C_0024160039000.fits']

flatfield_fits_filename = '/data/slam/home/sinjan/fits_files/april_avgd_2020_flat.fits' #solo_L0_phi-hrt-flat_0667134081_V202103221851C_0162201100.fits'

darkfield_fits_filename = '../fits_files/solo_L0_phi-fdt-ilam_20200228T155100_V202002281636_0022210004_000.fits'

science_april = ['/data/slam/home/sinjan/fits_files/' + i for i in science_april]

input_dict = {
  'data_f': science_april,
  'flat_f' : flatfield_fits_filename,
  'dark_f' : darkfield_fits_filename
}

json.dump(input_dict, open(f"./input_jsons/april_2020.txt", "w"))

"""

"""
#April 2020 L1

# CAN also do the L1 files - but must set bit_convert_scale to false, and scale data to False for the flat fields, and presumably same for the input data

science_april = ['solo_L1_phi-hrt-ilam_20200420T141802_V202107221036C_0024160030.fits',
'solo_L1_phi-hrt-ilam_20200420T142032_V202107221036C_0024160031.fits',
'solo_L1_phi-hrt-ilam_20200420T142302_V202107221036C_0024160032.fits',
'solo_L1_phi-hrt-ilam_20200420T142532_V202107221037C_0024160033.fits',
'solo_L1_phi-hrt-ilam_20200420T142803_V202107221037C_0024160034.fits',
'solo_L1_phi-hrt-ilam_20200420T143033_V202107221037C_0024160035.fits',
'solo_L1_phi-hrt-ilam_20200420T143303_V202107221037C_0024160036.fits',
'solo_L1_phi-hrt-ilam_20200420T143533_V202107221037C_0024160037.fits',
'solo_L1_phi-hrt-ilam_20200420T143803_V202107221037C_0024160038.fits',
'solo_L1_phi-hrt-ilam_20200420T144033_V202107221038C_0024160039.fits']

flatfield_fits_filename = '/data/slam/home/sinjan/fits_files/april_avgd_2020_flat.fits' #solo_L0_phi-hrt-flat_0667134081_V202103221851C_0162201100.fits'

darkfield_fits_filename = '../fits_files/solo_L0_phi-fdt-ilam_20200228T155100_V202002281636_0022210004_000.fits'

science_april = ['/data/slam/home/sinjan/fits_files/' + i for i in science_april]

input_dict = {
  'data_f': science_april,
  'flat_f' : flatfield_fits_filename,
  'dark_f' : darkfield_fits_filename
}

json.dump(input_dict, open(f"./input_jsons/april_2020_L1.txt", "w"))
"""

"""
#Nov 17 2020 L1

# CAN also do the L1 files - but must set bit_convert_scale to false, and scale data to False for the flat fields, and presumably same for the input data

science_nov = ['solo_L1_phi-hrt-ilam_20201117T170209_V202108301639C_0051170001.fits']#['solo_L1_phi-hrt-ilam_20201117T170209_V202107060747C_0051170001.fits']

flatfield_fits_filename = '/data/slam/home/sinjan/fits_files/april_avgd_2020_flat.fits' #solo_L0_phi-hrt-flat_0667134081_V202103221851C_0162201100.fits'

darkfield_fits_filename = '../fits_files/solo_L0_phi-fdt-ilam_20200228T155100_V202002281636_0022210004_000.fits'

science_nov = ['/data/slam/home/sinjan/fits_files/' + i for i in science_nov]

input_dict = {
  'data_f': science_nov,
  'flat_f' : flatfield_fits_filename,
  'dark_f' : darkfield_fits_filename
}

json.dump(input_dict, open(f"./input_jsons/nov_2020_L1.txt", "w"))
"""


"""
#Nov 17 2020 L1 Feb Flats

# CAN also do the L1 files - but must set bit_convert_scale to false, and scale data to False for the flat fields, and presumably same for the input data

science_nov = ['solo_L1_phi-hrt-ilam_20201117T170209_V202107060747C_0051170001.fits']

flatfield_fits_filename = '/data/slam/home/sinjan/fits_files/solo_L0_phi-hrt-flat_0667134081_V202103221851C_0162201100.fits' #solo_L0_phi-hrt-flat_0667134081_V202103221851C_0162201100.fits'

darkfield_fits_filename = '../fits_files/solo_L0_phi-fdt-ilam_20200228T155100_V202002281636_0022210004_000.fits'

science_nov = ['/data/slam/home/sinjan/fits_files/' + i for i in science_nov]

input_dict = {
  'data_f': science_nov,
  'flat_f' : flatfield_fits_filename,
  'dark_f' : darkfield_fits_filename
}

json.dump(input_dict, open(f"./input_jsons/nov_2020_L1_feb_flats.txt", "w"))
"""

"""
#Nov 17 2020 L1 KLL flats

# CAN also do the L1 files - but must set bit_convert_scale to false, and scale data to False for the flat fields, and presumably same for the input data

science_nov = ['solo_L1_phi-hrt-ilam_20201117T170209_V202107060747C_0051170001.fits']

flatfield_fits_filename = '/data/slam/home/sinjan/fits_files/solo_L1_phi-hrt-flat_20210321T210847_V202108301617C_0163211100.fits' #solo_L0_phi-hrt-flat_0667134081_V202103221851C_0162201100.fits'

darkfield_fits_filename = '../fits_files/solo_L0_phi-fdt-ilam_20200228T155100_V202002281636_0022210004_000.fits'

science_nov = ['/data/slam/home/sinjan/fits_files/' + i for i in science_nov]

input_dict = {
  'data_f': science_nov,
  'flat_f' : flatfield_fits_filename,
  'dark_f' : darkfield_fits_filename
}

json.dump(input_dict, open(f"./input_jsons/nov_2020_L1_kll.txt", "w"))

"""


"""
#Feb 2021 L1


science_feb = ['solo_L1_phi-hrt-ilam_20210223T170002_V202107221048C_0142230201.fits']

flatfield_fits_filename = '/data/slam/home/sinjan/fits_files/solo_L0_phi-hrt-flat_0667134081_V202103221851C_0162201100.fits' #solo_L0_phi-hrt-flat_0667134081_V202103221851C_0162201100.fits'

darkfield_fits_filename = '../fits_files/solo_L0_phi-fdt-ilam_20200228T155100_V202002281636_0022210004_000.fits'

science_feb = ['/data/slam/home/sinjan/fits_files/' + i for i in science_feb]

input_dict = {
  'data_f': science_feb,
  'flat_f' : flatfield_fits_filename,
  'dark_f' : darkfield_fits_filename
}

json.dump(input_dict, open(f"./input_jsons/feb_2k_2021_L1.txt", "w"))

"""

"""
#Sep 2021 data

science_sep = ['solo_L1_phi-hrt-ilam_20210914T053015_V202110150939C_0149140301.fits']#['solo_L1_phi-hrt-ilam_20210914T034515_V202110211713C_0149140201.fits']

flat = '/data/slam/home/sinjan/fits_files/solo_L1_phi-hrt-ilam_20210911T120504_V202110200555C_0169110100.fits' #solo_L0_phi-hrt-flat_0667134081_V202103221851C_0162201100.fits'

dark = '/data/slam/home/sinjan/fits_files/solo_L1_phi-hrt-ilam_20210428T130238_V202109240900C_0164281001.fits'

science_nov = ['/data/slam/home/sinjan/fits_files/' + i for i in science_sep]

input_dict = {
  'data_f': science_nov,
  'flat_f' : flat,
  'dark_f' : dark
}

json.dump(input_dict, open(f"./input_jsons/sep_2021_L1_sl.txt", "w"))

"""

science = ['solo_L1_phi-hrt-ilam_20210914T071515_V202110260809C_0149140401.fits','solo_L1_phi-hrt-ilam_20210914T071945_V202110260809C_0149140402.fits','solo_L1_phi-hrt-ilam_20210914T072409_V202110260809C_0149140403.fits','solo_L1_phi-hrt-ilam_20210914T072833_V202110260809C_0149140404.fits']#['solo_L1_phi-hrt-ilam_20210914T034515_V202110211713C_0149140201.fits']

flat = '/data/slam/home/sinjan/fits_files/0169111100_DC_9data.fits'#solo_L1_phi-hrt-ilam_20210911T120504_V202110200555C_0169110100.fits' #solo_L0_phi-hrt-flat_0667134081_V202103221851C_0162201100.fits'

dark = '/data/slam/home/sinjan/fits_files/solo_L1_phi-hrt-ilam_20210428T130238_V202109240900C_0164281001.fits'

science_2 = ['/data/slam/home/sinjan/fits_files/' + i for i in science]

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


input_dict = {
  #input data
  'data_f': science_2,
  'flat_f' : flat,
  'dark_f' : dark,

  #input/output type + scaling
  'L1_input' : True, 
  'L1_8_generate': False, #not developed yet
  'scale_data' : True,  #these 3 will be made redundant once L1 data scaling is normalised - needed mainly for comissioning (IP5) data
  'accum_scaling' : True, 
  'bit_conversion' : True, 
  'scale_data': True,
  
  #reduction
  'dark_c' : True,
  'flat_c' : False, 
  'norm_f' : True, 
  'clean_f' : False, 
  'sigma' : 59, #unsharp masking gaussian width
  'clean_mode' : "V", #options 'QUV', 'UV', 'V' for the unsharp masking
  'flat_states' : 24, #options 4 (one each pol state), 6 (one each wavelength), 24
  'prefilter_f': None,
  'fs_c' : True, 
  'demod' : True, 
  'norm_stokes' : True, 
  'ItoQUV' : False, #missing VtoQU - not developed yet
  'ctalk_params' : None, #VtoQU parameters will be required in this argument once ready
  'rte' : 'none', #options: ''RTE', 'CE', 'CE+RTE'
  'p_milos' : False, #attempted, ran into problems - on hold
  'cmilos_fits_opt': False, #whether to use cmilos-fits
  
  #output dir/filenames
  'out_dir' : '/data/slam/home/sinjan/hrt_pipe_results/sep_2021_no_flat/',  
  'out_demod_file' : True,  #if True, will save stokes array to fits, the array that is fed into the RTE inversions
  'out_demod_filename' : None, #if specific and not default name
  'out_rte_filename' : None,  #if specific and not default name
  'config_file' : False #now redudant if json input files used
}

json.dump(input_dict, open(f"./input_jsons/sep_2021_L1_west_noflat.json", "w"))