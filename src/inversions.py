import numpy as np
from astropy.io import fits
from utils import *
from processes import data_hdr_kw
import os
import time
import subprocess
import datetime

def create_output_filenames(filename, DID, version = '01'):
    """
    creating the L2 output filenames from the input, assuming L1
    """
    try:
        file_start = filename.split('solo_')[1]
        file_start = 'solo_' + file_start
        L2_str = file_start.replace('L1', 'L2')
        versioned = L2_str.split('V')[0] + 'V' + version + '_' + DID + '.fits.gz'
        stokes_file = versioned.replace('ilam', 'stokes')
        icnt_file = versioned.replace('ilam', 'icnt')
        bmag_file = versioned.replace('ilam', 'bmag')
        bazi_file = versioned.replace('ilam', 'bazi')
        binc_file = versioned.replace('ilam', 'binc')
        blos_file = versioned.replace('ilam', 'blos')
        vlos_file = versioned.replace('ilam', 'vlos')

        return stokes_file, icnt_file, bmag_file, bazi_file, binc_file, blos_file, vlos_file

    except Exception:
        print("The input file: {file_path} does not contain 'L1'")
        raise KeyError


def write_output_inversion(rte_data_products, file_path, scan, hdr_scan, imgdirx_flipped, out_dir, out_rte_filename, vers):
    """
    write out the L2 files + stokes
    taking care of the azimuth definition if the image is flipped
    """

    if imgdirx_flipped:
        print("Input image has been flipped as per convention - converting Azimuth to convention")
        azi = rte_data_products[3,:,:].copy()
        rte_data_products[3,:,:] = 180 - azi

    if out_rte_filename is None:
            filename_root = str(file_path.split('.fits')[0][-10:])
            _, icnt_file, bmag_file, bazi_file, binc_file, blos_file, vlos_file = create_output_filenames(file_path, filename_root, version = vers)

    else:
        if isinstance(out_rte_filename, list):
            filename_root = out_rte_filename[scan]

        elif isinstance(out_rte_filename, str):
            filename_root = out_rte_filename

        else:
            filename_root = str(file_path.split('.fits')[0][-10:])
            print(f"out_rte_filename neither string nor list, reverting to default: {filename_root}")

        blos_file, icnt_file, bmag_file, bazi_file, binc_file, vlos_file = 'blos_' + filename_root, 'icnt_' + filename_root, 'bmag_' + filename_root, 'bazi_' + filename_root, 'binc_' + filename_root, 'vlos_' + filename_root

    ntime = datetime.datetime.now()
    hdr_scan['DATE'] = ntime.strftime("%Y-%m-%dT%H:%M:%S")

    version_k = hdr_scan['VERS_SW']
    if '.fits' in hdr_scan['CAL_DARK']:
        dark_f_k = 'True'
    else:
        dark_f_k = 'False'
    if '.fits' in hdr_scan['CAL_FLAT']:
        flat_f_k = 'True'
    else:
        flat_f_k = 'False'
    clean_f_k = hdr_scan['CAL_USH']
    if hdr_scan['CAL_CRT1'] > 0:
        ItoQUV_k ='True'
    else:
        ItoQUV_k = 'False'
    rte_sw_k = hdr_scan['RTE_SW']
    rte_mod_k = hdr_scan['RTE_MOD']

    with fits.open(file_path) as hdu_list:
        hdu_list[0].header = hdr_scan
        hdu_list[0].data = rte_data_products.astype(np.float32)
        hdu_list.writeto(out_dir+filename_root+'_rte_data_products.fits.gz', overwrite=True)

    #blos
    with fits.open(file_path) as hdu_list:
        hdr_scan['FILENAME'] = blos_file
        hdr_scan['HISTORY'] = f"Version: {version_k}. Dark: {dark_f_k}. Flat : {flat_f_k}, Unsharp: {clean_f_k}. I->QUV ctalk: {ItoQUV_k}. RTE: {rte_sw_k}. RTEmode: {rte_mod_k}."
        hdr_scan['LEVEL'] = 'L2'
        hdr_scan['BTYPE'] = 'BLOS'
        hdr_scan['BUNIT'] = 'Gauss'
        hdr_scan['DATAMIN'] = int(np.min(rte_data_products[5,:,:]))
        hdr_scan['DATAMAX'] = int(np.max(rte_data_products[5,:,:]))
        hdr_scan = data_hdr_kw(hdr_scan, rte_data_products[5,:,:])

        hdu_list[0].header = hdr_scan
        hdu_list[0].data = rte_data_products[5,:,:]
        hdu_list.writeto(out_dir+blos_file, overwrite=True)

    #bazi
    with fits.open(file_path) as hdu_list:
        hdr_scan['FILENAME'] = bazi_file
        hdr_scan['HISTORY'] = f"Version: {version_k}. Dark: {dark_f_k}. Flat : {flat_f_k}, Unsharp: {clean_f_k}. I->QUV ctalk: {ItoQUV_k}. RTE: {rte_sw_k}. RTEmode: {rte_mod_k}."
        hdr_scan['LEVEL'] = 'L2'
        hdr_scan['BTYPE'] = 'BAZI'
        hdr_scan['BUNIT'] = 'Degrees'
        hdr_scan['DATAMIN'] = int(0)
        hdr_scan['DATAMAX'] = int(180)
        hdr_scan = data_hdr_kw(hdr_scan, rte_data_products[3,:,:])

        hdu_list[0].header = hdr_scan
        hdu_list[0].data = rte_data_products[3,:,:].astype(np.float32)
        hdu_list.writeto(out_dir+bazi_file, overwrite=True)

    #binc
    with fits.open(file_path) as hdu_list:
        hdr_scan['FILENAME'] = binc_file
        hdr_scan['HISTORY'] = f"Version: {version_k}. Dark: {dark_f_k}. Flat : {flat_f_k}, Unsharp: {clean_f_k}. I->QUV ctalk: {ItoQUV_k}. RTE: {rte_sw_k}. RTEmode: {rte_mod_k}."
        hdr_scan['LEVEL'] = 'L2'
        hdr_scan['BTYPE'] = 'BINC'
        hdr_scan['BUNIT'] = 'Degrees'
        hdr_scan['DATAMIN'] = int(0)
        hdr_scan['DATAMAX'] = int(180)
        hdr_scan = data_hdr_kw(hdr_scan, rte_data_products[2,:,:])

        hdu_list[0].header = hdr_scan
        hdu_list[0].data = rte_data_products[2,:,:].astype(np.float32)
        hdu_list.writeto(out_dir+binc_file, overwrite=True)

    #bmag
    with fits.open(file_path) as hdu_list:
        hdr_scan['FILENAME'] = bmag_file
        hdr_scan['HISTORY'] = f"Version: {version_k}. Dark: {dark_f_k}. Flat : {flat_f_k}, Unsharp: {clean_f_k}. I->QUV ctalk: {ItoQUV_k}. RTE: {rte_sw_k}. RTEmode: {rte_mod_k}."
        hdr_scan['LEVEL'] = 'L2'
        hdr_scan['BTYPE'] = 'BMAG'
        hdr_scan['BUNIT'] = 'Gauss'
        hdr_scan['DATAMIN'] = int(0)
        hdr_scan['DATAMAX'] = round(np.max(rte_data_products[1,:,:]),3)
        hdr_scan = data_hdr_kw(hdr_scan, rte_data_products[1,:,:])

        hdu_list[0].header = hdr_scan
        hdu_list[0].data = rte_data_products[1,:,:].astype(np.float32)
        hdu_list.writeto(out_dir+bmag_file, overwrite=True)

    #vlos
    with fits.open(file_path) as hdu_list:
        hdr_scan['FILENAME'] = vlos_file
        hdr_scan['HISTORY'] = f"Version: {version_k}. Dark: {dark_f_k}. Flat : {flat_f_k}, Unsharp: {clean_f_k}. I->QUV ctalk: {ItoQUV_k}. RTE: {rte_sw_k}. RTEmode: {rte_mod_k}."
        hdr_scan['LEVEL'] = 'L2'
        hdr_scan['BTYPE'] = 'VLOS'
        hdr_scan['BUNIT'] = 'km/s'
        hdr_scan['DATAMIN'] = round(np.min(rte_data_products[4,:,:]),6)
        hdr_scan['DATAMAX'] = round(np.max(rte_data_products[4,:,:]),6)
        hdr_scan = data_hdr_kw(hdr_scan, rte_data_products[4,:,:])

        hdu_list[0].header = hdr_scan
        hdu_list[0].data = rte_data_products[4,:,:].astype(np.float32)
        hdu_list.writeto(out_dir+vlos_file, overwrite=True)

    #Icnt
    with fits.open(file_path) as hdu_list:
        hdr_scan['FILENAME'] = icnt_file
        hdr_scan['HISTORY'] = f"Version: {version_k}. Dark: {dark_f_k}. Flat : {flat_f_k}, Unsharp: {clean_f_k}. I->QUV ctalk: {ItoQUV_k}. RTE: {rte_sw_k}. RTEmode: {rte_mod_k}."
        hdr_scan['LEVEL'] = 'L2'
        hdr_scan['BTYPE'] = 'ICNT'
        hdr_scan['BUNIT'] = 'Normalised Intensity'
        hdr_scan['DATAMIN'] = 0
        hdr_scan['DATAMAX'] = round(np.max(rte_data_products[0,:,:]),6)
        hdr_scan = data_hdr_kw(hdr_scan, rte_data_products[0,:,:])

        hdu_list[0].header = hdr_scan
        hdu_list[0].data = rte_data_products[0,:,:].astype(np.float32)
        hdu_list.writeto(out_dir+icnt_file, overwrite=True)


def cmilos(data_f, hdr_arr, wve_axis_arr, data_shape, cpos_arr, data, rte, mask, imgdirx_flipped, out_rte_filename, out_dir, vers = '01'):
    """
    RTE inversion using CMILOS
    """
    print(" ")
    printc('-->>>>>>> RUNNING CMILOS ',color=bcolors.OKGREEN)
    
    try:
        CMILOS_LOC = os.path.realpath(__file__)

        CMILOS_LOC = CMILOS_LOC.split('src/')[0] + 'cmilos/' #-11 as hrt_pipe.py is 11 characters

        if os.path.isfile(CMILOS_LOC+'milos'):
            printc("Cmilos executable located at:", CMILOS_LOC,color=bcolors.WARNING)

        else:
            raise ValueError('Cannot find cmilos:', CMILOS_LOC)

    except ValueError as err:
        printc(err.args[0],color=bcolors.FAIL)
        printc(err.args[1],color=bcolors.FAIL)
        return        

    wavelength = 6173.3354

    for scan in range(int(data_shape[-1])):

        start_time = time.perf_counter()

        file_path = data_f[scan]
        wave_axis = wve_axis_arr[scan]
        hdr_scan = hdr_arr[scan]

        #must invert each scan independently, as cmilos only takes in one dataset at a time

        #get wave_axis from the hdr information of the science scans - hard code first one, as must all have cpos in initial science load case
        if cpos_arr[0] == 0:
            shift_w =  wave_axis[3] - wavelength
        elif cpos_arr[0] == 5:
            shift_w =  wave_axis[2] - wavelength
        # DC TEST
        wave_axis = wave_axis - shift_w

        print('It is assumed the wavelength array is given by the hdr')
        #print(wave_axis,color = bcolors.WARNING)
        print("Wave axis is: ", (wave_axis - wavelength)*1000.)
        print('Saving data into dummy_in.txt for RTE input')

        if data.ndim == 5:
            sdata = data[:,:,:,:,scan]
        elif data.ndim > 5 or data.ndim < 4:
            print("Incorrect dimensions of 'data' array")
            exit()
        elif data.ndim == 4:
            sdata = data
        y,x,p,l = sdata.shape
        #print(y,x,p,l)

        filename = out_dir + 'dummy_in.txt'
        with open(filename,"w") as f:
            for i in range(x):
                for j in range(y):
                    for k in range(l):
                        f.write('%e %e %e %e %e \n' % (wave_axis[k],sdata[j,i,0,k],sdata[j,i,1,k],sdata[j,i,2,k],sdata[j,i,3,k])) #wv, I, Q, U, V

        printc(f'  ---- >>>>> Inverting data scan number: {scan} .... ',color=bcolors.OKGREEN)

        cmd = CMILOS_LOC+"./milos"
        cmd = fix_path(cmd)

        if rte == 'RTE':
            rte_on = subprocess.call(cmd+f" 6 15 0 0 {out_dir+'dummy_in.txt'}  >  {out_dir+'dummy_out.txt'}",shell=True)
        if rte == 'CE':
            rte_on = subprocess.call(cmd+f" 6 15 2 0 {out_dir+'dummy_in.txt'}  >  {out_dir+'dummy_out.txt'}",shell=True)
        if rte == 'CE+RTE':
            rte_on = subprocess.call(cmd+f" 6 15 1 0 {out_dir+'dummy_in.txt'}  >  {out_dir+'dummy_out.txt'}",shell=True)

        printc('  ---- >>>>> Reading results.... ',color=bcolors.OKGREEN)
        del_dummy = subprocess.call(f"rm {out_dir + 'dummy_in.txt'}",shell=True)

        res = np.loadtxt(out_dir+'dummy_out.txt')
        npixels = res.shape[0]/12.
        #print(npixels)
        #print(npixels/x)
        result = np.zeros((12,y*x)).astype(float)
        rte_invs = np.zeros((12,y,x)).astype(float)
        for i in range(y*x):
            result[:,i] = res[i*12:(i+1)*12]
        result = result.reshape(12,y,x)
        result = np.einsum('ijk->ikj', result)
        rte_invs = result
        del result
        rte_invs_noth = np.copy(rte_invs)

        """
        From 0 to 11
        Counter (PX Id)
        Iterations
        Strength
        Inclination
        Azimuth
        Eta0 parameter
        Doppler width
        Damping
        Los velocity
        Constant source function
        Slope source function
        Minimum chisqr value
        """

        noise_in_V =  np.mean(sdata[:,:,3,cpos_arr[0]]) #ellipsis in case data has 4 dimensions
        low_values_flags = np.max(np.abs(sdata[:,:,3,:]),axis=-1) < noise_in_V  # Where values are low
        
        del sdata

        rte_invs[2,low_values_flags] = 0
        rte_invs[3,low_values_flags] = 0
        rte_invs[4,low_values_flags] = 0

        #np.savez_compressed(out_dir+'_RTE', rte_invs=rte_invs, rte_invs_noth=rte_invs_noth)
        
        _ = subprocess.call(f"rm {out_dir+'dummy_out.txt'}",shell=True)

        rte_data_products = np.zeros((6,rte_invs_noth.shape[1],rte_invs_noth.shape[2]))

        rte_data_products[0,:,:] = rte_invs_noth[9,:,:] + rte_invs_noth[10,:,:] #continuum
        rte_data_products[1,:,:] = rte_invs_noth[2,:,:] #b mag strength
        rte_data_products[2,:,:] = rte_invs_noth[3,:,:] #inclination
        rte_data_products[3,:,:] = rte_invs_noth[4,:,:] #azimuth
        rte_data_products[4,:,:] = rte_invs_noth[8,:,:] #vlos
        rte_data_products[5,:,:] = rte_invs_noth[2,:,:]*np.cos(rte_invs_noth[3,:,:]*np.pi/180.) #blos

        rte_data_products *= mask[np.newaxis, :, :, 0] #field stop, set outside to 0

        hdr_scan['RTE_MOD'] = rte
        hdr_scan['RTE_SW'] = 'cmilos'
        hdr_scan['RTE_ITER'] = str(15)

        write_output_inversion(rte_data_products, file_path, scan, hdr_scan, imgdirx_flipped, out_dir, out_rte_filename, vers)
            
        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        printc(f"------------- CMILOS RTE Run Time: {np.round(time.perf_counter() - start_time,3)} seconds ",bcolors.OKGREEN)
        printc('--------------------------------------------------------------',bcolors.OKGREEN)


def cmilos_fits(data_f, hdr_arr, wve_axis_arr, data_shape, cpos_arr, data, rte, mask, imgdirx_flipped, out_rte_filename, out_dir, vers = '01'):
    """
    RTE inversion using CMILOS
    """
    print(" ")
    printc('-->>>>>>> RUNNING CMILOS ',color=bcolors.OKGREEN)
    
    try:
        CMILOS_LOC = os.path.realpath(__file__)

        CMILOS_LOC = CMILOS_LOC.split('src/')[0] + 'cmilos-fits/' #-11 as hrt_pipe.py is 11 characters

        if os.path.isfile(CMILOS_LOC+'milos'):
            printc("Cmilos-fits executable located at:", CMILOS_LOC,color=bcolors.WARNING)

        else:
            raise ValueError('Cannot find cmilos-fits:', CMILOS_LOC)

    except ValueError as err:
        printc(err.args[0],color=bcolors.FAIL)
        printc(err.args[1],color=bcolors.FAIL)
        return        

    wavelength = 6173.3354

    for scan in range(int(data_shape[-1])):

        start_time = time.perf_counter()

        file_path = data_f[scan]
        wave_axis = wve_axis_arr[scan]
        hdr_scan = hdr_arr[scan] # DC 20211117

        #must invert each scan independently, as cmilos only takes in one dataset at a time

        #get wave_axis from the hdr information of the science scans
        if cpos_arr[0] == 0:
            shift_w =  wave_axis[3] - wavelength
        elif cpos_arr[0] == 5:
            shift_w =  wave_axis[2] - wavelength

        wave_axis = wave_axis - shift_w

        print('It is assumed the wavelength array is given by the hdr')
        #print(wave_axis,color = bcolors.WARNING)
        print("Wave axis is: ", (wave_axis - wavelength)*1000.)
        print('Saving data into dummy_in.txt for RTE input')

        if data.ndim == 5:
            sdata = data[:,:,:,:,scan]
        elif data.ndim > 5 or data.ndim < 4:
            print("Incorrect dimensions of 'data' array")
            exit()
        elif data.ndim == 4:
            sdata = data
        y,x,p,l = sdata.shape

        #create hdr with wavelength positions
        hdr = fits.Header()
        print(wave_axis[0])
        hdr['LAMBDA0'] = wave_axis[0]#needs it in Angstrom 6173.1 etc
        hdr['LAMBDA1'] = wave_axis[1]
        hdr['LAMBDA2'] = wave_axis[2]
        hdr['LAMBDA3'] = wave_axis[3]
        hdr['LAMBDA4'] = wave_axis[4]
        hdr['LAMBDA5'] = wave_axis[5]
        
        #write out data to temp fits for cmilos-fits input
        input_arr = np.transpose(sdata, axes = (3,2,0,1)) #must transpose due to cfitsio (wl,pol,y,x) #3201 originally
        hdu1 = fits.PrimaryHDU(data=input_arr, header = hdr)

        #mask
        mask = np.ones((sdata.shape[0],sdata.shape[1])) #change this for fdt
        hdu2 = fits.ImageHDU(data=mask)

        #write out to temp fits
        hdul_tmp = fits.HDUList([hdu1, hdu2])
        hdul_tmp.writeto(out_dir+'temp_cmilos_io.fits', overwrite=True)

        printc(f'  ---- >>>>> Inverting data scan number: {scan} .... ',color=bcolors.OKGREEN)

        cmd = CMILOS_LOC+"milos"
        #cmd = fix
        #fix_path(cmd)
        print(cmd)

        if rte == 'RTE':
            rte_on = subprocess.call(cmd+f" 6 15 0 {out_dir+'temp_cmilos_io.fits'}",shell=True)
        if rte == 'CE':
            rte_on = subprocess.call(cmd+f" 6 15 2 {out_dir+'temp_cmilos_io.fits'}",shell=True)
        if rte == 'CE+RTE':
            rte_on = subprocess.call(cmd+f" 6 15 1 {out_dir+'temp_cmilos_io.fits'}",shell=True)

        print(rte_on)

        printc('  ---- >>>>> Reading results.... ',color=bcolors.OKGREEN)
        #print(del_dummy)

        with fits.open(out_dir+'temp_cmilos_io.fits') as hdu_list:
            rte_out = hdu_list[0].data
            #hdu_list.writeto(out_dir+'rte_out.fits', overwrite=True)
        
        del input_arr

        """
        From 0 to 11
        Iterations
        Strength
        Inclination
        Azimuth
        Eta0 parameter
        Doppler width
        Damping/aa
        Los velocity
        alfa? Counter PID?
        Constant source function
        Slope source function
        Minimum chisqr value

        Direct from cmilos-fits/milos.c
        inv->iter = malloc(npix*sizeof(int));
        inv->B    = malloc(npix*sizeof(double));
        inv->gm   = malloc(npix*sizeof(double));
        inv->az   = malloc(npix*sizeof(double));
        inv->eta0 = malloc(npix*sizeof(double));
        inv->dopp = malloc(npix*sizeof(double));
        inv->aa   = malloc(npix*sizeof(double));
        inv->vlos = malloc(npix*sizeof(double)); //km/s
        inv->alfa = malloc(npix*sizeof(double)); //stay light factor
        inv->S0   = malloc(npix*sizeof(double));
        inv->S1   = malloc(npix*sizeof(double));
        inv->nchisqrf = malloc(npix*sizeof(double));
        """
        noise_in_V =  np.mean(sdata[:,:,3,cpos_arr[0]])
        low_values_flags = np.max(np.abs(sdata[:,:,3,:]),axis=-1) < noise_in_V  # Where values are low
        
        rte_out[2,low_values_flags] = 0 #not sure about 2,3,4 indexing here
        rte_out[3,low_values_flags] = 0
        rte_out[4,low_values_flags] = 0
        
       
        rte_data_products = np.zeros((6,rte_out.shape[1],rte_out.shape[2]))

        rte_data_products[0,:,:] = rte_out[9,:,:] + rte_out[10,:,:] #continuum
        rte_data_products[1,:,:] = rte_out[1,:,:] #b mag strength
        rte_data_products[2,:,:] = rte_out[2,:,:] #inclination
        rte_data_products[3,:,:] = rte_out[3,:,:] #azimuth
        rte_data_products[4,:,:] = rte_out[7,:,:] #vlos
        rte_data_products[5,:,:] = rte_out[1,:,:]*np.cos(rte_out[2,:,:]*np.pi/180.) #blos

        rte_data_products *= mask[np.newaxis, :, :, 0] #field stop, set outside to 0

        hdr_scan['RTE_MOD'] = rte
        hdr_scan['RTE_SW'] = 'cmilos-fits'
        hdr_scan['RTE_ITER'] = str(15)

        write_output_inversion(rte_data_products, file_path, scan, hdr_scan, imgdirx_flipped, out_dir, out_rte_filename, vers)

        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        printc(f"------------- CMILOS-FITS RTE Run Time: {np.round(time.perf_counter() - start_time,3)} seconds ",bcolors.OKGREEN)
        printc('--------------------------------------------------------------',bcolors.OKGREEN)



def pmilos(data_f, hdr_arr, wve_axis_arr, data_shape, cpos_arr, data, rte, mask, imgdirx_flipped, out_rte_filename, out_dir, vers = '01'):
    """
    RTE inversion using PMILOS
    """
    print(" ")
    printc('-->>>>>>> RUNNING PMILOS ',color=bcolors.OKGREEN)
    
    try:
        PMILOS_LOC = os.path.realpath(__file__)

        PMILOS_LOC = PMILOS_LOC.split('src/')[0] + 'p-milos/' #11 as hrt_pipe.py is 11 characters -8 if in utils.py

        if os.path.isfile(PMILOS_LOC+'pmilos.x'):
            printc("Pmilos executable located at:", PMILOS_LOC,color=bcolors.WARNING)

        else:
            raise ValueError('Cannot find pmilos:', PMILOS_LOC)

    except ValueError as err:
        printc(err.args[0],color=bcolors.FAIL)
        printc(err.args[1],color=bcolors.FAIL)
        return  
    
    wavelength = 6173.3354

    for scan in range(int(data_shape[-1])):

        start_time = time.perf_counter()

        file_path = data_f[scan]
        wave_axis = wve_axis_arr[scan]
        hdr_scan = hdr_arr[scan]

        #must invert each scan independently, as cmilos only takes in one dataset at a time

        #get wave_axis from the hdr information of the science scans
        if cpos_arr[0] == 0:
            shift_w =  wave_axis[3] - wavelength
        elif cpos_arr[0] == 5:
            shift_w =  wave_axis[2] - wavelength

        wave_axis = wave_axis - shift_w

        print('It is assumed the wavelength array is given by the hdr')
        #print(wave_axis,color = bcolors.WARNING)
        print("Wave axis is: ", (wave_axis - wavelength)*1000.)
        print('Saving data into ./p-milos/run/data/input_tmp.fits for pmilos RTE input')

        #write wavelengths to wavelength.fits file for the settings

        wave_input = np.zeros((2,6)) #cfitsio reads dimensions in opposite order
        wave_input[0,:] = 1
        wave_input[1,:] = wave_axis

        print(wave_axis)

        if data.ndim == 5:
            sdata = data[:,:,:,:,scan]
        elif data.ndim > 5 or data.ndim < 4:
            print("Incorrect dimensions of 'data' array")
            exit()
        elif data.ndim == 4:
            sdata = data

        hdr = fits.Header()

        primary_hdu = fits.PrimaryHDU(wave_input, header = hdr)
        hdul = fits.HDUList([primary_hdu])
        hdul.writeto(f'./p-milos/run/wavelength_tmp.fits', overwrite=True)

        sdata = sdata.T
        sdata = sdata.astype(np.float32)
        #create input fits file for pmilos
        hdr = fits.Header() 
        
        hdr['CTYPE1'] = 'HPLT-TAN'
        hdr['CTYPE2'] = 'HPLN-TAN'
        hdr['CTYPE3'] = 'STOKES' #check order of stokes
        hdr['CTYPE4'] = 'WAVE-GRI' 
    
        primary_hdu = fits.PrimaryHDU(sdata, header = hdr)
        hdul = fits.HDUList([primary_hdu])
        hdul.writeto(f'./p-milos/run/data/input_tmp.fits', overwrite=True)

        if rte == 'RTE':
            cmd = "mpiexec -n 64 ../pmilos.x pmilos.minit" #../milos.x pmilos.mtrol" ##
        
        if rte == 'CE':
            cmd = "mpiexec -np 16 ../pmilos.x pmilos_ce.minit"

        if rte == 'CE+RTE':
            print("CE+RTE not possible on PMILOS, performing RTE instead")
            cmd = "mpiexec -np 16 ../pmilos.x pmilos.minit"

        if rte == 'RTE_seq':
            cmd = '../milos.x pmilos.mtrol'

        del sdata
        #need to change settings for CE or CE+RTE in the pmilos.minit file here
        
        printc(f'  ---- >>>>> Inverting data scan number: {scan} .... ',color=bcolors.OKGREEN)

        cwd = os.getcwd()
        os.chdir("./p-milos/run/")
        rte_on = subprocess.call(cmd,shell=True)
        os.chdir(cwd)
        
        if rte == 'CE':
            out_file = 'inv__mod_ce.fits' # not sure about this one

        else:
            out_file = 'inv__mod.fits' #only when one datacube and 16 processors

        with fits.open(f'./p-milos/run/results/{out_file}') as hdu_list:
            result = hdu_list[0].data

        #del_dummy = subprocess.call(f"rm ./p-milos/run/results/{out_file}.fits",shell=True) 
        del_dummy = subprocess.call(f"rm ./p-milos/run/results/{out_file}",shell=True) #must delete the output file
      
        #result has dimensions [rows,cols,13]
        result = np.moveaxis(result,0,2)
        print(result.shape)
        #printc(f'  ---- >>>>> You are HERE .... ',color=bcolors.WARNING)
        """
        PMILOS Output 13 columns
        0. eta0 = line-to-continuum absorption coefficient ratio 
        1. B = magnetic field strength [Gauss] 
        2. vlos = line-of-sight velocity [km/s] 
        3. dopp = Doppler width [Angstroms] 
        4. aa = damping parameter 
        5. gm = magnetic field inclination [deg] 
        6. az = magnetic field azimuth [deg] 
        7. S0 = source function constant 
        8. S1 = source function gradient 
        9. mac = macroturbulent velocity [km/s] 
        10. filling factor of the magnetic component [0-1]  
        11. Number of iterations performed 
        12. Chisqr value
        """

        rte_data_products = np.zeros((6,result.shape[0],result.shape[1]))

        rte_data_products[0,:,:] = result[:,:,7] + result[:,:,8] #continuum
        rte_data_products[1,:,:] = result[:,:,1] #b mag strength
        rte_data_products[2,:,:] = result[:,:,5] #inclination
        rte_data_products[3,:,:] = result[:,:,6] #azimuth
        rte_data_products[4,:,:] = result[:,:,2] #vlos
        rte_data_products[5,:,:] = result[:,:,1]*np.cos(result[:,:,5]*np.pi/180.) #blos

        rte_data_products *= mask[np.newaxis, :, :, 0] #field stop, set outside to 0

        hdr_scan['RTE_MOD'] = rte
        hdr_scan['RTE_SW'] = 'pmilos'
        hdr_scan['RTE_ITER'] = str(15)

    write_output_inversion(rte_data_products, file_path, scan, hdr_scan, imgdirx_flipped, out_dir, out_rte_filename, vers)

    printc('--------------------------------------------------------------',bcolors.OKGREEN)
    printc(f"------------- PMILOS RTE Run Time: {np.round(time.perf_counter() - start_time,3)} seconds ",bcolors.OKGREEN)
    printc('--------------------------------------------------------------',bcolors.OKGREEN)

