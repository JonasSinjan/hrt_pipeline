import numpy as np 
import os.path
from astropy.io import fits
import random, statistics
import subprocess
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from .utils import *
from plot_lib import plib

def get_data(path):
    """load science data from path"""
    try:
        data, header = load_fits(path)
      
        data /=  256. #conversion from 24.8bit to 32bit

        accu = header['ACCACCUM']*header['ACCROWIT']*header['ACCCOLIT'] #getting the number of accu from header

        data /= accu

        printc(f"Dividing by number of accumulations: {accu}",color=bcolors.OKGREEN)

    except Exception:
        printc("ERROR, Unable to open fits file: {}",path,color=bcolors.FAIL)

    return data, header


def demod(data,pmp_temp,const_demod=False):
    '''
    Use demodulation matrices to demodulate data
    '''

    try:
        demod_data = load_fits(f'./demod_matrices/demod_fitted_for_upload_HRT{pmp_temp}degC.fits')

    except:
        printc('No demod available',color = bcolors.FAIL)
        raise SystemError()

    printc('Demodulation matrix for ', pmp_temp,color = bcolors.WARNING)
    
    if const_demod:

        printc(f"Applying constant demodulation matrix for temp: {pmp_temp}",color=bcolors.OKGREEN)

        demod_data = demod_data.reshape((2048,2048,4,4)) #change into the required shape
        demod_avg = np.mean(demod_data[512:1536,512:1536,:,:], axis = (0,1))

        shape = data.shape
        demod = np.tile(demod_avg, (shape[0],shape[1],1,1))

    else:

        printc(f"Applying original (non constant) demodulation matrix for temp: {pmp_temp}",color=bcolors.OKGREEN)

        shape = data.shape
        diff = 2048-shape[0] 
    
        if np.abs(diff) > 0: #for cropped datasets

            start_row = int((2048-shape[0])/2) #assuming the crop is always based from the central section (ie with centre of fov in centre of cropped data)
            start_col = int((2048-shape[1])/2)

        else:
            start_row, start_col = 0, 0
        
        demod_data = demod_data.reshape((2048,2048,4,4)) #change into the required shape
        demod_data = demod_data[start_row:start_row + shape[0],start_col:start_col + shape[1],...]


    if data.ndim == 5:
        #if data array has more than one scan
        data = data.reshape(shape[0],shape[1],6,4,shape[-1]) #separate 24 images, into 6 wavelengths, with each 4 pol states
        data = np.moveaxis(data, 2,-2) #swap order
        data = np.moveaxis(data,-1,0) #moving number of scans to first dimension

        stokes_arr = np.matmul(demod,data)
        stokes_arr = np.moveaxis(stokes_arr,0,-1) #move scans back to the end
    
    elif data.ndim == 4:
        #for if data has just one scan
        data = data.reshape(shape[0],shape[1],6,4)
        data = np.moveaxis(data,-2,-1)

        stokes_arr = np.matmul(demod,data)
    
    return stokes_arr
    


def phihrt_pipe(data_f,dark_f,flat_f,norm_f = True, clean_f = False, flat_states = 24, 
                pmp_temp = '50',flat_c = True,dark_c = True, normalize = 1., 
                outdirectory = './',  outdemodfile=None,  correct_ghost = False, 
                ItoQUV = False, rte = False, outrtefile = None):

    '''
    PHI-HRT data reduction pipeline
    1. read in science data (+scaling) open path option + open for several scans at once
    2. read in flat field - just one, so any averaging must be done before
    3. option to clean flat field with unsharp masking
    4. read in dark field
    5. apply dark field
    6. normalise flat field
    7. apply flat field
    8. read in field stop
    9. apply field stop
    10. read in demod matrix
    11. demodulate
    12. normalise to quiet sun
    13. calibration
        a) ghost correction (still needs to be finalised)
        b) cross talk correction (including offset)
    14. rte inversion with sophism (atm still using sophism)

    Parameters
    ----------
        Input:
    data_f : list or string
        list containing paths to fits files of the raw HRT data OR string of path to one file  
    dark_f : string
        Fits file of a dark file (ONLY ONE FILE)
    flat_f : string
        Fits file of a HRT flatfield (ONLY ONE FILE)

    ** Options:
  
    flat_states = 24 : int
        Number of flat fields to be applied, options are 0,4,6,24
    correct_ghost = False

    directory = './' : string

    vqu = False : bool

    rte = False : bool
    
    debug = False : bool

    Returns
    -------
    None 

    Raises
    ------

    References
    ----------
    SPGYlib

    '''

    printc('--------------------------------------------------------------',bcolors.OKGREEN)
    printc('PHI HRT data reduction software  ',bcolors.OKGREEN)
    printc('--------------------------------------------------------------',bcolors.OKGREEN)

    #-----------------
    # READ DATA
    #-----------------
    printc('-->>>>>>> Reading Data              ',color=bcolors.OKGREEN)
    printc('          Data is divided by 256.   ',color=bcolors.OKGREEN)         

   

    if isinstance(data_f, list):
      printc(' -- Input contains several science scans -- ',color=bcolors.OKGREEN)
      
      number_of_scans = len(data_f)

      output_arr = [0]*number_of_scans
      hdr_arr = [0]*number_of_scans

      for scan in range(number_of_scans):
        output_arr[scan], hdr_arr[scan] = get_data(data_f[scan])

      data = np.stack(output_arr, axis = -1)
      data = np.moveaxis(data, 0,-2)

    elif isinstance(data_f, str):
      data, header = get_data(data_f)
      data = np.moveaxis(data, 0, -1)
  
    else:
      printc("ERROR, data_f argument is neither a string nor list containing strings: {}",data_f,color=bcolors.FAIL)
      

    #-----------------
    # TODO: Could check data dimensions? As an extra fail safe before progressing?
    #-----------------
    
    
    #-----------------
    # READ FLAT FIELDS
    #-----------------

    if flat_c:

        printc('-->>>>>>> Reading Flats                    ',color=bcolors.OKGREEN)
        printc('          Input should be [wave X Stokes,y-dim,x-dim].',color=bcolors.OKGREEN)

        try:
            flat,header_flat = load_fits(flat_f)
            fz,fy,fx = flat.shape
            flat = np.reshape(flat,(fz//4,4,fy,fx))
            printc('-->>>>>>> Reshaping Flat to [wave,Stokes,y-dim,x-dim] ',color=bcolors.OKGREEN)
        except Exception:
            printc("ERROR, Unable to open flats file: {}",flat_f,color=bcolors.FAIL)


    else:
        printc('-->>>>>>> No flats mode                    ',color=bcolors.WARNING)

    
    #-----------------
    # NORM FLAT FIELDS
    #-----------------

    if norm_f and flat_c:
        printc('-->>>>>>> Normalising Flats                    ',color=bcolors.OKGREEN)
        try:
            norm_fac = np.mean(flat[:,:,512:1536,512:1536], axis = (2,3))  #mean of the central 1k x 1k
            flat /= norm_fac[..., np.newaxis, np.newaxis]
        except Exception:
            printc("ERROR, Unable to normalise the flat fields: {}",flat_f,color=bcolors.FAIL)
   

    #-----------------
    # READ AND CORRECT DARK FIELD
    #-----------------
    if dark_c:

        printc('-->>>>>>> Reading Darks                   ',color=bcolors.OKGREEN)
        printc('          Input should be [y-dim,x-dim].',color=bcolors.OKGREEN)
        printc('          DARK IS DIVIDED by 256.   ',color=bcolors.OKGREEN)
        try:
            dark,h = get_data(dark_f)

            dark_shape = dark.shape

            if dark_shape != (2048,2048):
                
                printc("Dark Field Input File not in 2048,2048 format: {}",dark_f,color=bcolors.WARNING)
                printc("Attempting to correct ",color=bcolors.WARNING)

                
                try:
                    if dark_shape[0] > 2048:
                        dark = dark[dark_shape[0]-2048:,:]
                
                except:
                    printc("ERROR, Unable to correct shape of dark field data: {}",dark_f,color=bcolors.FAIL)

        except Exception:
            printc("ERROR, Unable to open darks file: {}",dark_f,color=bcolors.FAIL)



    #-----------------
    # APPLY FLAT CORRECTION 
    #-----------------

    if flat_c:
        printc('-->>>>>>> Correcting Flatfield',color=bcolors.OKGREEN)
        try:
          if flat_states == 6:
        
            printc("Dividing by six flats, one for each wavelength",color=bcolors.OKGREEN)
        
            for j in range(4):
                
                tmp = np.mean(flat[:,i,:,:],axis=-1)
                data[:,:,j:j+4,:] /= tmp[np.newaxis, np.newaxis, start_row:start_row + shape[0],start_col:start_col + shape[1]] #divide science data by normalised flats for each wavelength (avg over pol states)
          
          elif flat_states == 0:
              pass

          elif flat_states == 24:
              
              data[:,:,:,:] /= flatfield_avg[start_row:start_row + shape[0],start_col:start_col + shape[1], :, np.newaxis]
                #only one new axis for the scans
                  
          elif flat_states == 4:
              for i in range(4):
                  tmp = np.mean(flatfield_avg[:,:,i::4],axis=-1)
                  sciencedata[:,:,i::4,:] /= tmp[start_row:start_row + shape[0],start_col:start_col + shape[1], np.newaxis, np.newaxis]
        
          print(f"--- Divide by flats time: {np.round(time.time() - start_time,3)} seconds ---")

        except: 
          printc("ERROR, Unable to apply flat fields",color=bcolors.FAIL)
        

    rlabel='DN',save=None,cmap='gray')


    
    #-----------------
    # GHOST CORRECTION  
    #-----------------

    if correct_ghost:
        printc('-->>>>>>> Correcting ghost image ',color=bcolors.OKGREEN)

  
    #-----------------
    # APPLY DEMODULATION 
    #-----------------

    printc('-->>>>>>> Demodulating data...         ',color=bcolors.OKGREEN)

    data = demod_phi(data, pmp_temp, demod = True)

    #-----------------
    # APPLY NORMALIZATION 
    #-----------------

    printc('-->>>>>>> Applying normalization --',color=bcolors.OKGREEN)
   

    #-----------------
    # CROSS-TALK CALCULATION 
    #-----------------
    printc('-->>>>>>> Cross-talk correction from Stokes I to Stokes Q,U,V --',color=bcolors.OKGREEN)
   

    

    #-----------------
    #CHECK FOR INFs
    #-----------------

    data[np.isinf(data)] = 0
    data[np.isnan(data)] = 0

    #-----------------
    # SAVE DATA TODO: CMILOS FORMAT AND FITS
    #-----------------
    printc('---------------------------------------------------------',color=bcolors.OKGREEN)
    if outfile == None:
        outfile = data_f[:-4]
    printc(' Saving data to: ',directory+outfile+'_red.fits')

    # hdu = pyfits.PrimaryHDU(data)
    # hdul = pyfits.HDUList([hdu])
    # hdul.writeto(outfile, overwrite=True)

    with pyfits.open(data_f) as hdu_list:
        hdu_list[0].data = data
        hdu_list.writeto(directory+outfile+'_red.fits', clobber=True)

    #-----------------
    # INVERSION OF DATA WITH CMILOS
    #-----------------

    if rte == 'RTE' or rte == 'CE' or rte == 'CE+RTE':
        printc('---------------------RUNNING CMILOS --------------------------',color=bcolors.OKGREEN)

        try:
            CMILOS_LOC = os.path.realpath(__file__) 
            CMILOS_LOC = CMILOS_LOC[:-14] + 'cmilos/'
            if os.path.isfile(CMILOS_LOC+'milos'):
                printc("Cmilos executable located at:", CMILOS_LOC,color=bcolors.WARNING)
            else:
                raise ValueError('Cannot find cmilos:', CMILOS_LOC)
        except ValueError as err:
            printc(err.args[0],color=bcolors.FAIL)
            printc(err.args[1],color=bcolors.FAIL)
            return        

        wavelength = 6173.3356
        #OJO, REMOVE. NEED TO CHECK THE REF WAVE FROM S/C-PHI H/K
        shift_w =  wave_axis[3] - wavelength
        wave_axis = wave_axis - shift_w
        #wave_axis = np.array([-300,-160,-80,0,80,160])/1000.+wavelength
        # wave_axis = np.array([-300,-140,-70,0,70,140])
        printc('   It is assumed the wavelength is given by the header info ')
        printc(wave_axis,color = bcolors.WARNING)
        printc((wave_axis - wavelength)*1000.,color = bcolors.WARNING)
        printc('   saving data into dummy_in.txt for RTE input')

        sdata = data[:,:,ry[0]:ry[1],rx[0]:rx[1]]
        l,p,x,y = sdata.shape
        print(l,p,x,y)

        filename = 'dummy_in.txt'
        with open(filename,"w") as f:
            for i in range(x):
                for j in range(y):
                    for k in range(l):
                        f.write('%e %e %e %e %e \n' % (wave_axis[k],sdata[k,0,j,i],sdata[k,1,j,i],sdata[k,2,j,i],sdata[k,3,j,i]))
        del sdata

        printc('  ---- >>>>> Inverting data.... ',color=bcolors.OKGREEN)
        umbral = 3.

        cmd = CMILOS_LOC+"./milos"
        cmd = fix_path(cmd)
        if rte == 'RTE':
            rte_on = subprocess.call(cmd+" 6 15 0 0 dummy_in.txt  >  dummy_out.txt",shell=True)
        if rte == 'CE':
            rte_on = subprocess.call(cmd+" 6 15 2 0 dummy_in.txt  >  dummy_out.txt",shell=True)
        if rte == 'CE+RTE':
            rte_on = subprocess.call(cmd+" 6 15 1 0 dummy_in.txt  >  dummy_out.txt",shell=True)

        print(rte_on)
        printc('  ---- >>>>> Finishing.... ',color=bcolors.OKGREEN)
        printc('  ---- >>>>> Reading results.... ',color=bcolors.OKGREEN)
        del_dummy = subprocess.call("rm dummy_in.txt",shell=True)
        print(del_dummy)

        res = np.loadtxt('dummy_out.txt')
        npixels = res.shape[0]/12.
        print(npixels)
        print(npixels/x)
        result = np.zeros((12,y*x)).astype(float)
        rte_invs = np.zeros((12,yd,xd)).astype(float)
        for i in range(y*x):
            result[:,i] = res[i*12:(i+1)*12]
        result = result.reshape(12,y,x)
        result = np.einsum('ijk->ikj', result)
        rte_invs[:,ry[0]:ry[1],rx[0]:rx[1]] = result
        del result
        rte_invs_noth = np.copy(rte_invs)

        noise_in_V =  np.mean(data[0,3,rry[0]:rry[1],rrx[0]:rrx[1]])
        low_values_flags = np.max(np.abs(data[:,3,:,:]),axis=0) < noise_in_V*umbral  # Where values are low
        rte_invs[2,low_values_flags] = 0
        rte_invs[3,low_values_flags] = 0
        rte_invs[4,low_values_flags] = 0
        for i in range(12):
            rte_invs[i,:,:] = rte_invs[i,:,:] * mask
        #save plots!!!!
        if verbose:
            plib.show_four_row(rte_invs_noth[2,:,:],rte_invs_noth[3,:,:],rte_invs_noth[4,:,:],rte_invs_noth[8,:,:],svmin=[0,0,0,-6.],svmax=[1200,180,180,+6.],title=['Field strengh [Gauss]','Field inclination [degree]','Field azimuth [degree]','LoS velocity [km/s]'],xlabel='Pixel',ylabel='Pixel')#,save=outfile+'_VLoS.png')
            plib.show_four_row(rte_invs[2,:,:],rte_invs[3,:,:],rte_invs[4,:,:],rte_invs[8,:,:],svmin=[0,0,0,-6.],svmax=[1200,180,180,+6.],title=['Field strengh [Gauss]','Field inclination [degree]','Field azimuth [degree]','LoS velocity [km/s]'],xlabel='Pixel',ylabel='Pixel')#,save=outfile+'BLoS.png')
        rte_invs_noth[8,:,:] = rte_invs_noth[8,:,:] - np.mean(rte_invs_noth[8,rry[0]:rry[1],rrx[0]:rrx[1]])
        rte_invs[8,:,:] = rte_invs[8,:,:] - np.mean(rte_invs[8,rry[0]:rry[1],rrx[0]:rrx[1]])

        np.savez_compressed(directory+outfile+'_RTE', rte_invs=rte_invs, rte_invs_noth=rte_invs_noth,mask=mask)
        
        del_dummy = subprocess.call("rm dummy_out.txt",shell=True)
        print(del_dummy)

        b_los = rte_invs_noth[2,:,:]*np.cos(rte_invs_noth[3,:,:]*np.pi/180.)*mask
        # b_los = np.zeros((2048,2048))
        # b_los[PXBEG2:PXEND2+1,PXBEG1:PXEND1+1] = b_los_cropped

        v_los = rte_invs_noth[8,:,:] * mask
        # v_los = np.zeros((2048,2048))
        # v_los[PXBEG2:PXEND2+1,PXBEG1:PXEND1+1] = v_los_cropped
        if verbose:
            plib.show_one(v_los,vmin=-2.5,vmax=2.5,title='LoS velocity')
            plib.show_one(b_los,vmin=-30,vmax=30,title='LoS magnetic field')

        with pyfits.open(data_f) as hdu_list:
            hdu_list[0].data = b_los
            hdu_list.writeto(directory+outfile+'_blos_rte.fits', clobber=True)

        with pyfits.open(data_f) as hdu_list:
            hdu_list[0].data = v_los
            hdu_list.writeto(directory+outfile+'_vlos_rte.fits', clobber=True)

        with pyfits.open(data_f) as hdu_list:
            hdu_list[0].data = rte_invs[9,:,:]+rte_invs[10,:,:]
            hdu_list.writeto(directory+outfile+'_Icont_rte.fits', clobber=True)

        printc('  ---- >>>>> Saving plots.... ',color=bcolors.OKGREEN)

        #-----------------
        # PLOTS VLOS
        #-----------------
        Zm = np.ma.masked_where(mask == 1, mask)

        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        plt.title('PHI-FDT LoS velocity',size=20)

        # Hide grid lines
        ax.grid(False)
        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])

    #        im = ax.imshow(np.fliplr(rotate(v_los[PXBEG2:PXEND2+1,PXBEG1:PXEND1+1], 52, reshape=False)), cmap='bwr',vmin=-3.,vmax=3.)
        im = ax.imshow(v_los, cmap='bwr',vmin=-3.,vmax=3.)

        divider = plib.make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plib.plt.colorbar(im, cax=cax)
        cbar.set_label('[km/s]')
        cbar.ax.tick_params(labelsize=16)

        #ax.imshow(Zm, cmap='gray')
        plt.savefig('imagenes/velocity-map-'+outfile+'.png',dpi=300)
        plt.close()

        #-----------------
        # PLOTS BLOS
        #-----------------

        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        plt.title('PHI-FDT Magnetogram',size=20)

        # Hide grid lines
        ax.grid(False)
        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
                    
        #im = ax.imshow(np.fliplr(rotate(b_los[PXBEG2:PXEND2+1,PXBEG1:PXEND1+1], 52, reshape=False)), cmap='gray',vmin=-100,vmax=100) 
        im = ax.imshow(b_los, cmap='gray',vmin=-100,vmax=100)

        divider = plib.make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plib.plt.colorbar(im, cax=cax)
        # cbar.set_label('Stokes V amplitude [%]')
        cbar.set_label('LoS magnetic field [Mx/cm$^2$]')
        cbar.ax.tick_params(labelsize=16)
        #ax.imshow(Zm, cmap='gray')

        plt.savefig('imagenes/Blos-map-'+outfile+'.png',dpi=300)
        plt.close()

        printc('--------------------- END  ----------------------------',color=bcolors.FAIL)



    return

    #-----------------
    # CORRECT CAVITY
    #-----------------

    