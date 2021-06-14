from hrt_pipe import phihrt_pipe

sciencedata_fits_filenames = ['solo_L0_phi-hrt-ilam_0667414748_V202103221851C_0142230201.fits']
flatfield_fits_filename = 'solo_L0_phi-hrt-flat_0667134081_V202103221851C_0162201100.fits'
darkfield_fits_filename = 'solo_L0_phi-fdt-ilam_20200228T155100_V202002281636_0022210004_000.fits'

folder = '../fits_files/'

if folder != None:

    def add_folder_to_file_path(folder, filename_list):
        return [folder + i for i in filename_list]

    sciencedata_fits_filenames = add_folder_to_file_path(folder, sciencedata_fits_filenames)
    flatfield_fits_filename = folder + flatfield_fits_filename
    darkfield_fits_filename = folder + darkfield_fits_filename

data = phihrt_pipe(sciencedata_fits_filenames, darkfield_fits_filename, flatfield_fits_filename, norm_stokes = True, clean_f = True, out_demod_file = True, out_dir = './', rte = 'RTE')