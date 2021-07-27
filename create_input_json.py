import json

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