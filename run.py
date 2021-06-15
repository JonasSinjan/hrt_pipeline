from hrt_pipe import phihrt_pipe

sciencedata_fits_filenames = ['../fits_files/solo_L0_phi-hrt-ilam_0667414748_V202103221851C_0142230201.fits']
flatfield_fits_filename = '../fits_files/solo_L0_phi-hrt-flat_20210321T210847_V202106071514C_0163211100.fits'
darkfield_fits_filename = '../fits_files/solo_L0_phi-fdt-ilam_20200228T155100_V202002281636_0022210004_000.fits'

data = phihrt_pipe(sciencedata_fits_filenames, darkfield_fits_filename, flatfield_fits_filename, norm_stokes = True, 
                    clean_f = True, out_demod_file = True, out_dir = '../hrt_pipe_results/kll/', rte = 'False')