import json

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

science_nov = ['solo_L1_phi-hrt-ilam_20201117T170209_V202107060747C_0051170001.fits']

flatfield_fits_filename = '/data/slam/home/sinjan/fits_files/april_avgd_2020_flat.fits' #solo_L0_phi-hrt-flat_0667134081_V202103221851C_0162201100.fits'

darkfield_fits_filename = '../fits_files/solo_L0_phi-fdt-ilam_20200228T155100_V202002281636_0022210004_000.fits'

science_april = ['/data/slam/home/sinjan/fits_files/' + i for i in science_nov]

input_dict = {
  'data_f': science_nov,
  'flat_f' : flatfield_fits_filename,
  'dark_f' : darkfield_fits_filename
}

json.dump(input_dict, open(f"./input_jsons/nov_2020_L1.txt", "w"))
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