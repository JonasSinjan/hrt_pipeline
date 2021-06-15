from hrt_pipe import phihrt_pipe

sciencedata_fits_filenames = ['solo_L0_phi-hrt-ilam_0667414905_V202103221851C_0142230602.fits', 'solo_L0_phi-hrt-ilam_0667415054_V202103221851C_0142230603.fits', 'solo_L0_phi-hrt-ilam_0667415205_V202103221851C_0142230604.fits', 'solo_L0_phi-hrt-ilam_0667415354_V202103221851C_0142230605.fits', 'solo_L0_phi-hrt-ilam_0667415505_V202103221851C_0142230606.fits', 'solo_L0_phi-hrt-ilam_0667415654_V202103221851C_0142230607.fits', 'solo_L0_phi-hrt-ilam_0667415805_V202103221851C_0142230608.fits']#['../fits_files/solo_L0_phi-hrt-ilam_0667414748_V202103221851C_0142230201.fits']
flatfield_fits_filename = '../fits_files/solo_L0_phi-hrt-flat_0667134081_V202103221851C_0162201100.fits'
darkfield_fits_filename = '../fits_files/solo_L0_phi-fdt-ilam_20200228T155100_V202002281636_0022210004_000.fits'

sciencedata_fits_filenames = ['../fits_files/' + i for i in sciencedata_fits_filenames]

data = phihrt_pipe(sciencedata_fits_filenames, darkfield_fits_filename, flatfield_fits_filename, norm_stokes = True, 
                    clean_f = True, out_demod_file = True, out_dir = '/data/slam/home/sinjan/hrt_pipe_results/stp-136/', rte = 'RTE')