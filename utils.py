from astropy.io import fits
import numpy as np
import os
import time
import subprocess


class bcolors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'
  RESET = '\u001b[0m'

def printc(*args, color = bcolors.RESET, **kwargs):
  """My custom print() function."""
  print(u"\u001b"+f"{color}", end='\r')
  print(*args, **kwargs)
  print(u"\u001b"+f"{bcolors.RESET}", end='\r')
  return 

def load_fits(path):
  """
  load the fits file
  
  Parameters
  ----------
  path: string, location of the fits file
  
  Output
  ------
  data: numpy array, of stokes images in (row, col, wv, pol) 
  header: hdul header object, header of the fits file
  """

  hdul_tmp = fits.open(f'{path}')
  
  data = np.asarray(hdul_tmp[0].data, dtype = np.float32)

  header = hdul_tmp[0].header
  
  return data, header 


def fits_get_sampling(file,verbose = False):
    '''
    wave_axis,voltagesData,tunning_constant,cpos = fits_get_sampling(file)
    No S/C velocity corrected!!!
    cpos = 0 if continuum is at first wavelength and = 5 if continuum is at the end

    From SPGPylibs PHITools
    '''
    #print('-- Obtaining voltages......')
    fg_head = 3
    #try:
    with fits.open(file) as hdu_list:
        header = hdu_list[fg_head].data
        j = 0
        dummy = 0
        voltagesData = np.zeros((6))
        tunning_constant = 0.0
        ref_wavelength = 0.0
        for v in header:
            #print(v)
            if (j < 6):
                if tunning_constant == 0:
                    tunning_constant = float(v[4])/1e9
                if ref_wavelength == 0:
                    ref_wavelength = float(v[5])/1e3
                #print(dummy, v[2], type(dummy), type(v[2]))
                if np.abs(np.abs(float(v[2])) - np.abs(dummy)) > 5: #check that the next voltage is more than 5 from the previous, as voltages change slightly
                    #print(dummy, v[2])
                    voltagesData[j] = float(v[2])
                    dummy = voltagesData[j] 
                    j += 1

    #except Exception:
    #   print("Unable to open fits file: {}",file)     
    #print(voltagesData)
    d1 = voltagesData[0] - voltagesData[1]
    d2 = voltagesData[4] - voltagesData[5]
    if np.abs(d1) > np.abs(d2):
        cpos = 0
    else:
        cpos = 5
    if verbose:
        print('Continuum position at wave: ', cpos)
    wave_axis = voltagesData*tunning_constant + ref_wavelength  #6173.3356
    #print(wave_axis)
    return wave_axis,voltagesData,tunning_constant,cpos

def fix_path(path,dir='forward',verbose=False):
    path = repr(path)
    if dir == 'forward':
        path = path.replace(")", "\)")
        path = path.replace("(", "\(")
        path = path.replace(" ", "\ ")
        path = os.path.abspath(path).split("'")[1]
        if verbose == True:
            print('forward')
            print(path)
        return path
    elif dir == 'backward':
        path = path.replace("\\\\", "")
        path = path.split("'")[1]
        if verbose == True:
            print('backward')
            print(path)
        return path
    else:
        pass   

def cmilos(data_f, wve_axis_arr, data_shape, cpos_arr, data, rte, field_stop, start_row, start_col, out_rte_filename, out_dir):
    print(" ")
    printc('-->>>>>>> RUNNING CMILOS ',color=bcolors.OKGREEN)
    
    try:
        CMILOS_LOC = os.path.realpath(__file__)

        CMILOS_LOC = CMILOS_LOC[:-8] + 'cmilos/' #-11 as hrt_pipe.py is 11 characters

        if os.path.isfile(CMILOS_LOC+'milos'):
            printc("Cmilos executable located at:", CMILOS_LOC,color=bcolors.WARNING)

        else:
            raise ValueError('Cannot find cmilos:', CMILOS_LOC)

    except ValueError as err:
        printc(err.args[0],color=bcolors.FAIL)
        printc(err.args[1],color=bcolors.FAIL)
        return        

    wavelength = 6173.3356

    for scan in range(int(data_shape[-1])):

        start_time = time.time()

        file_path = data_f[scan]
        wave_axis = wve_axis_arr[scan]

        #must invert each scan independently, as cmilos only takes in one dataset at a time

        #get wave_axis from the header information of the science scans
        if cpos_arr[0] == 0:
            shift_w =  wave_axis[3] - wavelength
        elif cpos_arr[0] == 5:
            shift_w =  wave_axis[2] - wavelength

        wave_axis = wave_axis - shift_w

        print('It is assumed the wavelength array is given by the header')
        #print(wave_axis,color = bcolors.WARNING)
        print("Wave axis is: ", (wave_axis - wavelength)*1000.)
        print('Saving data into dummy_in.txt for RTE input')

        sdata = data[:,:,:,:,scan]
        y,x,p,l = sdata.shape
        #print(y,x,p,l)

        filename = 'dummy_in.txt'
        with open(filename,"w") as f:
            for i in range(x):
                for j in range(y):
                    for k in range(l):
                        f.write('%e %e %e %e %e \n' % (wave_axis[k],sdata[j,i,0,k],sdata[j,i,1,k],sdata[j,i,2,k],sdata[j,i,3,k])) #wv, I, Q, U, V
        del sdata

        printc(f'  ---- >>>>> Inverting data scan number: {scan} .... ',color=bcolors.OKGREEN)

        cmd = CMILOS_LOC+"./milos"
        cmd = fix_path(cmd)

        if rte == 'RTE':
            rte_on = subprocess.call(cmd+" 6 15 0 0 dummy_in.txt  >  dummy_out.txt",shell=True)
        if rte == 'CE':
            rte_on = subprocess.call(cmd+" 6 15 2 0 dummy_in.txt  >  dummy_out.txt",shell=True)
        if rte == 'CE+RTE':
            rte_on = subprocess.call(cmd+" 6 15 1 0 dummy_in.txt  >  dummy_out.txt",shell=True)

        #print(rte_on)

        printc('  ---- >>>>> Reading results.... ',color=bcolors.OKGREEN)
        del_dummy = subprocess.call("rm dummy_in.txt",shell=True)
        #print(del_dummy)

        res = np.loadtxt('dummy_out.txt')
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

        noise_in_V =  np.mean(data[:,:,3,cpos_arr[0],:])
        low_values_flags = np.max(np.abs(data[:,:,3,:,scan]),axis=-1) < noise_in_V  # Where values are low
        
        rte_invs[2,low_values_flags] = 0
        rte_invs[3,low_values_flags] = 0
        rte_invs[4,low_values_flags] = 0

        #np.savez_compressed(out_dir+'_RTE', rte_invs=rte_invs, rte_invs_noth=rte_invs_noth)
        
        del_dummy = subprocess.call("rm dummy_out.txt",shell=True)
        #print(del_dummy)

        rte_data_products = np.zeros((6,rte_invs_noth.shape[1],rte_invs_noth.shape[2]))

        rte_data_products[0,:,:] = rte_invs_noth[9,:,:] + rte_invs_noth[10,:,:] #continuum
        rte_data_products[1,:,:] = rte_invs_noth[2,:,:] #b mag strength
        rte_data_products[2,:,:] = rte_invs_noth[3,:,:] #inclination
        rte_data_products[3,:,:] = rte_invs_noth[4,:,:] #azimuth
        rte_data_products[4,:,:] = rte_invs_noth[8,:,:] #vlos
        rte_data_products[5,:,:] = rte_invs_noth[2,:,:]*np.cos(rte_invs_noth[3,:,:]*np.pi/180.) #blos

        rte_data_products *= field_stop[np.newaxis,start_row:start_row + data.shape[0],start_col:start_col + data.shape[1]] #field stop, set outside to 0

        if out_rte_filename is None:
            filename_root = str(file_path.split('.fits')[0][-10:])
        else:
            filename_root = out_rte_filename

        with fits.open(file_path) as hdu_list:
            hdu_list[0].data = rte_data_products
            hdu_list.writeto(out_dir+filename_root+'_rte_data_products.fits', overwrite=True)

        with fits.open(file_path) as hdu_list:
            hdu_list[0].data = rte_data_products[5,:,:]
            hdu_list.writeto(out_dir+filename_root+'_blos_rte.fits', overwrite=True)

        with fits.open(file_path) as hdu_list:
            hdu_list[0].data = rte_data_products[4,:,:]
            hdu_list.writeto(out_dir+filename_root+'_vlos_rte.fits', overwrite=True)

        with fits.open(file_path) as hdu_list:
            hdu_list[0].data = rte_data_products[0,:,:]
            hdu_list.writeto(out_dir+filename_root+'_Icont_rte.fits', overwrite=True)

        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        printc(f"------------- CMILOS RTE Run Time: {np.round(time.time() - start_time,3)} seconds ",bcolors.OKGREEN)
        printc('--------------------------------------------------------------',bcolors.OKGREEN)

def pmilos(data_f, wve_axis_arr, data_shape, cpos_arr, data, rte, field_stop, start_row, start_col, out_rte_filename, out_dir):
    print(" ")
    printc('-->>>>>>> RUNNING PMILOS ',color=bcolors.OKGREEN)
    
    try:
        PMILOS_LOC = os.path.realpath(__file__)

        PMILOS_LOC = PMILOS_LOC[:-8] + 'P-MILOS/' #11 as hrt_pipe.py is 11 characters -8 if in utils.py

        if os.path.isfile(PMILOS_LOC+'pmilos.x'):
            printc("Pmilos executable located at:", PMILOS_LOC,color=bcolors.WARNING)

        else:
            raise ValueError('Cannot find pmilos:', PMILOS_LOC)

    except ValueError as err:
        printc(err.args[0],color=bcolors.FAIL)
        printc(err.args[1],color=bcolors.FAIL)
        return  
    
    wavelength = 6173.3356

    for scan in range(int(data_shape[-1])):

        start_time = time.time()

        file_path = data_f[scan]
        wave_axis = wve_axis_arr[scan]

        #must invert each scan independently, as cmilos only takes in one dataset at a time

        #get wave_axis from the header information of the science scans
        if cpos_arr[0] == 0:
            shift_w =  wave_axis[3] - wavelength
        elif cpos_arr[0] == 5:
            shift_w =  wave_axis[2] - wavelength

        wave_axis = wave_axis - shift_w

        print('It is assumed the wavelength array is given by the header')
        #print(wave_axis,color = bcolors.WARNING)
        print("Wave axis is: ", (wave_axis - wavelength)*1000.)
        print('Saving data into ./P-MILOS/run/data/input_tmp.fits for pmilos RTE input')

        #write wavelengths to wavelength.fits file for the settings

        wave_input = np.zeros((6,2))
        wave_input[:,0] = int(1)
        wave_input[:,1] = wave_axis

        hdr = fits.Header()

        primary_hdu = fits.PrimaryH0DU(wave_input, header = hdr)
        hdul = fits.HDUList([primary_hdu])
        hdul.writeto(f'./P-MILOS/run/wavelength_tmp.fits', overwrite=True)

        #create input fits file for pmilos
        sdata = data[:,:,:,:,scan]
        
        hdr = fits.Header() 
        
        hdr['CTYPE1'] = 'HPLT-TAN'
        hdr['CTYPE2'] = 'HPLN-TAN'
        hdr['CTYPE3'] = 'STOKES' #check order of stokes
        hdr['CTYPE4'] = 'WAVE-GRI' 
    
        primary_hdu = fits.PrimaryHDU(sdata, header = hdr)
        hdul = fits.HDUList([primary_hdu])
        hdul.writeto(f'./P-MILOS/run/data/input_tmp.fits', overwrite=True)

        #need to change settings for CE or CE+RTE in the pmilos.minit file here
        
        printc(f'  ---- >>>>> Inverting data scan number: {scan} .... ',color=bcolors.OKGREEN)

        cwd = os.getcwd()
        os.chdir("./P-MILOS/run/")
        cmd = "mpiexec -np 16 ../pmilos.x pmilos.minit" #PMILOS_LOC+"./milos"

        if rte == 'RTE':
            cmd = "mpiexec -np 16 ../pmilos.x pmilos.minit"
            
        if rte == 'CE':
            cmd = "mpiexec -np 16 ../pmilos.x pmilos_ce.minit"

        if rte == 'CE+RTE':
            print("CE+RTE not possible on PMILOS, performing RTE instead")
            cmd = "mpiexec -np 16 ../pmilos.x pmilos.minit"

        rte_on = subprocess.call(cmd,shell=True)

        os.chdir(cwd)

        with fits.open('./P-MILOS/run/results/inv_input_tmp_mod.fits') as hdu_list:
            result = hdu_list[0].data

        del_dummy = subprocess.call("rm ./P-MILOS/run/results/inv_input_tmp_mod.fits",shell=True) #must delete the output file
      
        #result has dimensions [rows,cols,13]

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

        rte_data_products *= field_stop[np.newaxis,start_row:start_row + data.shape[0],start_col:start_col + data.shape[1]] #field stop, set outside to 0

        #flipping taken care of for the field stop in the hrt_pipe 

        if out_rte_filename is None:
            filename_root = str(file_path.split('.fits')[0][-10:])
        else:
            filename_root = out_rte_filename

        with fits.open(file_path) as hdu_list:
            hdu_list[0].data = rte_data_products
            hdu_list.writeto(out_dir+filename_root+'_rte_data_products.fits', overwrite=True)

        with fits.open(file_path) as hdu_list:
            hdu_list[0].data = rte_data_products[5,:,:]
            hdu_list.writeto(out_dir+filename_root+'_blos_rte.fits', overwrite=True)

        with fits.open(file_path) as hdu_list:
            hdu_list[0].data = rte_data_products[4,:,:]
            hdu_list.writeto(out_dir+filename_root+'_vlos_rte.fits', overwrite=True)

        with fits.open(file_path) as hdu_list:
            hdu_list[0].data = rte_data_products[0,:,:]
            hdu_list.writeto(out_dir+filename_root+'_Icont_rte.fits', overwrite=True)


    printc('--------------------------------------------------------------',bcolors.OKGREEN)
    printc(f"------------- PMILOS RTE Run Time: {np.round(time.time() - start_time,3)} seconds ",bcolors.OKGREEN)
    printc('--------------------------------------------------------------',bcolors.OKGREEN)