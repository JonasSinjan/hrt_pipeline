import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from operator import itemgetter
from utils import *
import os
import time
import subprocess

def load_flat(flat_f, accum_scaling, bit_conversion, scale_data, header_imgdirx_exists, imgdirx_flipped, cpos_arr) -> np.ndarray:
    """
    load, scale, flip and correct flat
    """
    print(" ")
    printc('-->>>>>>> Reading Flats',color=bcolors.OKGREEN)

    start_time = time.time()
    
    # flat from IP-5
    if '0024151020000' in flat_f or '0024150020000' in flat_f:
        flat, header_flat = get_data(flat_f, scaling = accum_scaling,  bit_convert_scale=bit_conversion,
                                    scale_data=False)
    else:
        flat, header_flat = get_data(flat_f, scaling = accum_scaling,  bit_convert_scale=bit_conversion,
                                    scale_data=scale_data)
                
    if 'IMGDIRX' in header_flat:
        header_fltdirx_exists = True
        fltdirx_flipped = str(header_flat['IMGDIRX'])
    else:
        header_fltdirx_exists = False
        fltdirx_flipped = 'NO'
    
    print(f"Flat field shape is {flat.shape}")
    # correction based on science data - see if flat and science are both flipped or not
    flat = compare_IMGDIRX(flat,header_imgdirx_exists,imgdirx_flipped,header_fltdirx_exists,fltdirx_flipped)
    
    flat = np.moveaxis(flat, 0,-1) #so that it is [y,x,24]
    flat = flat.reshape(2048,2048,6,4) #separate 24 images, into 6 wavelengths, with each 4 pol states
    flat = np.moveaxis(flat, 2,-1)
    
    print(flat.shape)

    _, _, _, cpos_f = fits_get_sampling(flat_f,verbose = True) #get flat continuum position

    print(f"The continuum position of the flat field is at {cpos_f} index position")
    
    #--------
    # test if the science and flat have continuum at same position
    #--------

    flat = compare_cpos(flat,cpos_f,cpos_arr[0]) 

    flat_pmp_temp = str(header_flat['HPMPTSP1'])

    print(f"Flat PMP Temperature Set Point: {flat_pmp_temp}")

    #--------
    # correct for missing line in particular flat field
    #--------

    if flat_f[-15:] == '0162201100.fits':  # flat_f[-62:] == 'solo_L0_phi-hrt-flat_0667134081_V202103221851C_0162201100.fits'
        print("This flat has a missing line - filling in with neighbouring pixels")
        flat_copy = flat.copy()
        flat[:,:,1,1] = filling_data(flat_copy[:,:,1,1], 0, mode = {'exact rows':[1345,1346]}, axis=1)

        del flat_copy
        
    printc('--------------------------------------------------------------',bcolors.OKGREEN)
    printc(f"------------ Load flats time: {np.round(time.time() - start_time,3)} seconds",bcolors.OKGREEN)
    printc('--------------------------------------------------------------',bcolors.OKGREEN)

    return flat


def load_dark(dark_f) -> np.ndarray:
    """
    loads dark field from given path
    """
    print(" ")
    printc('-->>>>>>> Reading Darks',color=bcolors.OKGREEN)

    start_time = time.time()

    try:
        dark,_ = get_data(dark_f)

        dark_shape = dark.shape

        if dark_shape != (2048,2048):

            if dark.ndim > 2:
                printc("Dark Field Input File has more dimensions than the expected 2048,2048 format: {}",dark_f,color=bcolors.WARNING)
                raise ValueError
            
            printc("Dark Field Input File not in 2048,2048 format: {}",dark_f,color=bcolors.WARNING)
            printc("Attempting to correct ",color=bcolors.WARNING)

            
            try:
                if dark_shape[0] > 2048:
                    dark = dark[dark_shape[0]-2048:,:]
            
            except Exception:
                printc("ERROR, Unable to correct shape of dark field data: {}",dark_f,color=bcolors.FAIL)
                raise ValueError

        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        printc(f"------------ Load darks time: {np.round(time.time() - start_time,3)} seconds",bcolors.OKGREEN)
        printc('--------------------------------------------------------------',bcolors.OKGREEN)

        return dark

    except Exception:
        printc("ERROR, Unable to open and process darks file: {}",dark_f,color=bcolors.FAIL)


def apply_dark_correction(data, flat, dark, rows, cols) -> np.ndarray:
    """
    subtracts dark field from flat field and science data
    """
    print(" ")
    print("-->>>>>>> Subtracting dark field")
    
    start_time = time.time()

    data -= dark[rows,cols, np.newaxis, np.newaxis, np.newaxis] 
    #flat -= dark[..., np.newaxis, np.newaxis] - # all processed flat fields should already be dark corrected

    printc('--------------------------------------------------------------',bcolors.OKGREEN)
    printc(f"------------- Dark Field correction time: {np.round(time.time() - start_time,3)} seconds",bcolors.OKGREEN)
    printc('--------------------------------------------------------------',bcolors.OKGREEN)

    return data, flat


def normalise_flat(flat, flat_f, ceny, cenx) -> np.ndarray:
    """
    normalise flat fields at each wavelength position to remove the spectral line
    """
    print(" ")
    printc('-->>>>>>> Normalising Flats',color=bcolors.OKGREEN)

    start_time = time.time()

    try:
        norm_fac = np.mean(flat[ceny,cenx, :, :], axis = (0,1))[np.newaxis, np.newaxis, ...]  #mean of the central 1k x 1k
        flat /= norm_fac

        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        printc(f"------------- Normalising flat time: {np.round(time.time() - start_time,3)} seconds",bcolors.OKGREEN)
        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        
        return flat

    except Exception:
        printc("ERROR, Unable to normalise the flat fields: {}",flat_f,color=bcolors.FAIL)


def demod_hrt(data,pmp_temp, verbose = True) -> np.ndarray:
    '''
    Use constant demodulation matrices to demodulate input data
    '''
    if pmp_temp == '50':
        demod_data = np.array([[ 0.28037298,  0.18741922,  0.25307596,  0.28119895],
                     [ 0.40408596,  0.10412157, -0.7225681,   0.20825675],
                     [-0.19126636, -0.5348939,   0.08181918,  0.64422774],
                     [-0.56897295,  0.58620095, -0.2579202,   0.2414017 ]])
        
    elif pmp_temp == '40':
        demod_data = np.array([[ 0.26450154,  0.2839626,   0.12642948,  0.3216773 ],
                     [ 0.59873885,  0.11278069, -0.74991184,  0.03091451],
                     [ 0.10833212, -0.5317737,  -0.1677862,   0.5923593 ],
                     [-0.46916953,  0.47738808, -0.43824592,  0.42579797]])
    
    else:
        printc("Demodulation Matrix for PMP TEMP of {pmp_temp} deg is not available", color = bcolors.FAIL)
    if verbose:
        printc(f'Using a constant demodulation matrix for a PMP TEMP of {pmp_temp} deg',color = bcolors.OKGREEN)
    
    demod_data = demod_data.reshape((4,4))
    shape = data.shape
    demod = np.tile(demod_data, (shape[0],shape[1],1,1))

    if data.ndim == 5:
        #if data array has more than one scan
        data = np.moveaxis(data,-1,0) #moving number of scans to first dimension

        data = np.matmul(demod,data)
        data = np.moveaxis(data,0,-1) #move scans back to the end
    
    elif data.ndim == 4:
        #for if data has just one scan
        data = np.matmul(demod,data)
    
    return data, demod


def unsharp_masking(flat,sigma,flat_pmp_temp,cpos_arr,clean_mode,clean_f,pol_end=4,verbose=True):
    """
    unsharp masks the flat fields to blur our polarimetric structures due to solar rotation
    clean_f = ['blurring', 'fft']
    """
    flat_demod, demodM = demod_hrt(flat, flat_pmp_temp,verbose)

    norm_factor = np.mean(flat_demod[512:1536,512:1536,0,cpos_arr[0]])

    flat_demod /= norm_factor

    new_demod_flats = np.copy(flat_demod)
    
#     b_arr = np.zeros((2048,2048,3,5))

    if cpos_arr[0] == 0:
        wv_range = range(1,6)

    elif cpos_arr[0] == 5:
        wv_range = range(5)

    if clean_mode == "QUV":
        start_clean_pol = 1
        if verbose:
            print("Unsharp Masking Q,U,V")
        
    elif clean_mode == "UV":
        start_clean_pol = 2
        if verbose:
            print("Unsharp Masking U,V")
        
    elif clean_mode == "V":
        start_clean_pol = 3
        if verbose:
            print("Unsharp Masking V")
        
    if clean_f == 'blurring':
        blur = lambda a: gaussian_filter(a,sigma)
    elif clean_f == 'fft':
        x = np.fft.fftfreq(2048,1)
        fftgaus2d = np.exp(-2*np.pi**2*(x-0)**2*sigma**2)[:,np.newaxis] * np.exp(-2*np.pi**2*(x-0)**2*sigma**2)[np.newaxis]
        blur = lambda a : (np.fft.ifftn(fftgaus2d*np.fft.fftn(a.copy()))).real
    
    for pol in range(start_clean_pol,pol_end):

        for wv in wv_range: #not the continuum

            a = np.copy(np.clip(flat_demod[:,:,pol,wv], -0.02, 0.02))
            b = a - blur(a)
#             b_arr[:,:,pol-1,wv-1] = b
            c = a - b

            new_demod_flats[:,:,pol,wv] = c

    invM = np.linalg.inv(demodM)

    return np.matmul(invM, new_demod_flats*norm_factor)


def flat_correction(data,flat,flat_states,rows,cols) -> np.ndarray:
    """
    correct science data with flat fields
    """
    print(" ")
    printc('-->>>>>>> Correcting Flatfield',color=bcolors.OKGREEN)

    start_time = time.time()

    try: 
        if flat_states == 6:
            
            printc("Dividing by 6 flats, one for each wavelength",color=bcolors.OKGREEN)
                
            tmp = np.mean(flat,axis=-2) #avg over pol states for the wavelength

            return data / tmp[rows,cols, np.newaxis, :, np.newaxis]


        elif flat_states == 24:

            printc("Dividing by 24 flats, one for each image",color=bcolors.OKGREEN)

            return data / flat[rows,cols, :, :, np.newaxis] #only one new axis for the scans
                
        elif flat_states == 4:

            printc("Dividing by 4 flats, one for each pol state",color=bcolors.OKGREEN)

            tmp = np.mean(flat,axis=-1) #avg over wavelength

            return data / tmp[rows,cols, :, np.newaxis, np.newaxis]
        else:
            print(" ")
            printc('-->>>>>>> Unable to apply flat correction. Please insert valid flat_states',color=bcolors.WARNING)

            
        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        printc(f"------------- Flat Field correction time: {np.round(time.time() - start_time,3)} seconds ",bcolors.OKGREEN)
        printc('--------------------------------------------------------------',bcolors.OKGREEN)

        return data

    except: 
        printc("ERROR, Unable to apply flat fields",color=bcolors.FAIL)



def prefilter_correction(data,voltagesData_arr,prefilter,prefilter_voltages):
    """
    applies prefilter correction
    adapted from SPGPylibs
    """
    def _get_v1_index1(x):
        index1, v1 = min(enumerate([abs(i) for i in x]), key=itemgetter(1))
        return  v1, index1
    
    data_shape = data.shape
    
    for scan in range(data_shape[-1]):

        voltage_list = voltagesData_arr[scan]
        
        for wv in range(6):

            v = voltage_list[wv]

            vdif = [v - pf for pf in prefilter_voltages]
            
            v1, index1 = _get_v1_index1(vdif)
            
            if vdif[index1] >= 0:
                v2 = vdif[index1 + 1]
                index2 = index1 + 1
                
            else:
                v2 = vdif[index1-1]
                index2 = index1 - 1
                
            imprefilter = (prefilter[:,:, index1]*v1 + prefilter[:,:, index2]*v2)/(v1+v2) #interpolation between nearest voltages

            data[:,:,:,wv,scan] /= imprefilter[...,np.newaxis]
            
    return data

def apply_field_stop(data, rows, cols, header_imgdirx_exists, imgdirx_flipped) -> np.ndarray:
    """
    apply field stop mask to the science data
    """
    print(" ")
    printc("-->>>>>>> Applying field stop",color=bcolors.OKGREEN)

    start_time = time.time()
    
    field_stop,_ = load_fits('../field_stop/HRT_field_stop.fits')

    field_stop = np.where(field_stop > 0,1,0)

    if header_imgdirx_exists:
        if imgdirx_flipped == 'YES': #should be YES for any L1 data, but mistake in processing software
            field_stop = field_stop[:,::-1] #also need to flip the flat data after dark correction

    data *= field_stop[rows,cols,np.newaxis, np.newaxis, np.newaxis]

    printc('--------------------------------------------------------------',bcolors.OKGREEN)
    printc(f"------------- Field stop time: {np.round(time.time() - start_time,3)} seconds",bcolors.OKGREEN)
    printc('--------------------------------------------------------------',bcolors.OKGREEN)

    return data, field_stop

def crosstalk_auto_ItoQUV(data_demod,cpos,wl,roi=np.ones((2048,2048)),verbose=0,npoints=5000,limit=0.2):
    import random, statistics
    from scipy.optimize import curve_fit
    def linear(x,a,b):
        return a*x + b
    my = []
    sy = []
    
    x = data_demod[roi>0,0,cpos].flatten()
    ids = np.logical_and(x > limit, x < 1.5)
    x = x[ids].flatten()

    N = x.size
    idx = random.sample(range(N),npoints)
    mx = x[idx].mean() 
    sx = x[idx].std() 
    xp = np.linspace(x.min(), x.max(), 100)

    A = np.vstack([x, np.ones(len(x))]).T

    # I to Q
    yQ = data_demod[roi>0,1,wl].flatten()
    yQ = yQ[ids].flatten()
    my.append(yQ[idx].mean())
    sy.append(yQ[idx].std())
    cQ = curve_fit(linear,x,yQ,p0=[0,0])[0]
    pQ = np.poly1d(cQ)

    # I to U
    yU = data_demod[roi>0,2,wl].flatten()
    yU = yU[ids].flatten()
    my.append(yU[idx].mean())
    sy.append(yU[idx].std())
    cU = curve_fit(linear,x,yU,p0=[0,0])[0]
    pU = np.poly1d(cU)

    # I to V
    yV = data_demod[roi>0,3,wl].flatten()
    yV = yV[ids].flatten()
    my.append(yV[idx].mean())
    sy.append(yV[idx].std())
    cV = curve_fit(linear,x,yV,p0=[0,0])[0]
    pV = np.poly1d(cV)

    if verbose:
        
        PLT_RNG = 3
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(x[idx],yQ[idx],color='red',alpha=0.6,s=10)
        ax.plot(xp, pQ(xp), color='red', linestyle='dashed',linewidth=3.0)

        ax.scatter(x[idx],yU[idx],color='blue',alpha=0.6,s=10)
        ax.plot(xp, pU(xp), color='blue', linestyle='dashed',linewidth=3.0)

        ax.scatter(x[idx],yV[idx],color='green',alpha=0.6,s=10)
        ax.plot(xp, pV(xp), color='green', linestyle='dashed',linewidth=3.0)

        ax.set_xlim([mx - PLT_RNG * sx,mx + PLT_RNG * sx])
        ax.set_ylim([min(my) - 1.8*PLT_RNG * statistics.mean(sy),max(my) + PLT_RNG * statistics.mean(sy)])
        ax.set_xlabel('Stokes I')
        ax.set_ylabel('Stokes Q/U/V')
        ax.text(mx - 0.9*PLT_RNG * sx, min(my) - 1.4*PLT_RNG * statistics.mean(sy), 'Cross-talk from I to Q: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cQ[0],cQ[1],width=8,prec=4), style='italic',bbox={'facecolor': 'red', 'alpha': 0.1, 'pad': 1}, fontsize=15)
        ax.text(mx - 0.9*PLT_RNG * sx, min(my) - 1.55*PLT_RNG * statistics.mean(sy), 'Cross-talk from I to U: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cU[0],cU[1],width=8,prec=4), style='italic',bbox={'facecolor': 'blue', 'alpha': 0.1, 'pad': 1}, fontsize=15)
        ax.text(mx - 0.9*PLT_RNG * sx, min(my) - 1.7*PLT_RNG * statistics.mean(sy), 'Cross-talk from I to V: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cV[0],cV[1],width=8,prec=4), style='italic',bbox={'facecolor': 'green', 'alpha': 0.1, 'pad': 1}, fontsize=15)
#         fig.show()

        print('Cross-talk from I to Q: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cQ[0],cQ[1],width=8,prec=4))
        print('Cross-talk from I to U: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cU[0],cU[1],width=8,prec=4))
        print('Cross-talk from I to V: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cV[0],cV[1],width=8,prec=4))
    
#         return cQ,cU,cV, (idx,x,xp,yQ,yU,yV,pQ,pU,pV,mx,sx,my,sy)
    else:
        ct = np.asarray((cQ,cU,cV)).T
        return ct

def CT_ItoQUV(data, ctalk_params, norm_stokes, cpos_arr, Ic_mask):
    """
    performs cross talk correction for I -> Q,U,V
    """
    before_ctalk_data = np.copy(data)
    data_shape = data.shape
    
#     ceny = slice(data_shape[0]//2 - data_shape[0]//4, data_shape[0]//2 + data_shape[0]//4)
#     cenx = slice(data_shape[1]//2 - data_shape[1]//4, data_shape[1]//2 + data_shape[1]//4)

    cont_stokes = np.ones(data_shape[-1])
    printc(data_shape[-1],color=bcolors.WARNING)
    printc(Ic_mask.shape,color=bcolors.WARNING)
    for scan in range(data_shape[-1]):
        cont_stokes[scan] = np.mean(data[Ic_mask[...,scan],0,cpos_arr[0],scan])
    print('here1')   
    for i in range(6):
                
#         stokes_i_wv_avg = np.mean(data[ceny,cenx,0,i,:], axis = (0,1))
        stokes_i_wv_avg = np.ones(data_shape[-1])
        for scan in range(data_shape[-1]):
            stokes_i_wv_avg[scan] = np.mean(data[Ic_mask[...,scan],0,i,scan])
            
        print('here2')   
        if norm_stokes:
            #if normed, applies normalised offset to normed stokes

            tmp_param = ctalk_params*np.divide(stokes_i_wv_avg,cont_stokes)

            q_slope = tmp_param[0,0,:]
            u_slope = tmp_param[0,1,:]
            v_slope = tmp_param[0,2,:]

            q_int = tmp_param[1,0,:]
            u_int = tmp_param[1,1,:]
            v_int = tmp_param[1,2,:]

            data[:,:,1,i,:] = before_ctalk_data[:,:,1,i,:] - before_ctalk_data[:,:,0,i,:]*q_slope - q_int

            data[:,:,2,i,:] = before_ctalk_data[:,:,2,i,:] - before_ctalk_data[:,:,0,i,:]*u_slope - u_int

            data[:,:,3,i,:] = before_ctalk_data[:,:,3,i,:] - before_ctalk_data[:,:,0,i,:]*v_slope - v_int
            print('here3')   
        
        else:
            #if not normed, applies raw offset cross talk correction to raw stokes counts

            tmp_param = ctalk_params[0,:,:]*np.divide(stokes_i_wv_avg,cont_stokes)

            q_slope = tmp_param[0,:]
            u_slope = tmp_param[1,:]
            v_slope = tmp_param[2,:]

            q_int = ctalk_params[1,0,:]
            u_int = ctalk_params[1,1,:]
            v_int = ctalk_params[1,2,:]

            data[:,:,1,i,:] = before_ctalk_data[:,:,1,i,:] - before_ctalk_data[:,:,0,i,:]*q_slope - q_int*stokes_i_wv_avg 

            data[:,:,2,i,:] = before_ctalk_data[:,:,2,i,:] - before_ctalk_data[:,:,0,i,:]*u_slope - u_int*stokes_i_wv_avg 

            data[:,:,3,i,:] = before_ctalk_data[:,:,3,i,:] - before_ctalk_data[:,:,0,i,:]*v_slope - v_int*stokes_i_wv_avg
    
    return data


def cmilos(data_f, hdr_arr, wve_axis_arr, data_shape, cpos_arr, data, rte, field_stop, start_row, start_col, out_rte_filename, out_dir):
    """
    RTE inversion using CMILOS
    """
    print(" ")
    printc('-->>>>>>> RUNNING CMILOS ',color=bcolors.OKGREEN)
    
    try:
        CMILOS_LOC = os.path.realpath(__file__)

        CMILOS_LOC = CMILOS_LOC[:-15] + 'cmilos/' #-11 as hrt_pipe.py is 11 characters

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

        start_time = time.time()

        file_path = data_f[scan]
        wave_axis = wve_axis_arr[scan]
        hdr = hdr_arr[scan]

        #must invert each scan independently, as cmilos only takes in one dataset at a time

        #get wave_axis from the hdr information of the science scans
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

        """
        #vlos S/C vorrection
        v_x, v_y, v_z = hdr['HCIX_VOB']/1000, hdr['HCIY_VOB']/1000, hdr['HCIZ_VOB']/1000

        #need line of sight velocity, should be total HCI velocity in km/s, with sun at origin. 
        #need to take care for velocities moving towards the sun, (ie negative) #could use continuum position as if towards or away
    
        if cpos_arr[scan] == 5: #moving away, redshifted
            dir_factor = 1
        
        elif cpos_arr[scan] == 0: #moving towards, blueshifted
            dir_factor == -1
        
        v_tot = dir_factor*math.sqrt(v_x**2 + v_y**2+v_z**2) #in km/s

        rte_invs_noth[8,:,:] = rte_invs_noth[8,:,:] - v_tot
        """

        rte_data_products = np.zeros((6,rte_invs_noth.shape[1],rte_invs_noth.shape[2]))

        rte_data_products[0,:,:] = rte_invs_noth[9,:,:] + rte_invs_noth[10,:,:] #continuum
        rte_data_products[1,:,:] = rte_invs_noth[2,:,:] #b mag strength
        rte_data_products[2,:,:] = rte_invs_noth[3,:,:] #inclination
        rte_data_products[3,:,:] = rte_invs_noth[4,:,:] #azimuth
        rte_data_products[4,:,:] = rte_invs_noth[8,:,:] #vlos
        rte_data_products[5,:,:] = rte_invs_noth[2,:,:]*np.cos(rte_invs_noth[3,:,:]*np.pi/180.) #blos

        rte_data_products *= field_stop[np.newaxis,start_row:start_row + data.shape[0],start_col:start_col + data.shape[1],scan] #field stop, set outside to 0

        if out_rte_filename is None:
            filename_root = str(file_path.split('.fits')[0][-10:])
        else:
            if isinstance(out_rte_filename, list):
                filename_root = out_rte_filename[scan]

            elif isinstance(out_rte_filename, str):
                filename_root = out_rte_filename

            else:
                filename_root = str(file_path.split('.fits')[0][-10:])
                print(f"out_rte_filename neither string nor list, reverting to default: {filename_root}")

        with fits.open(file_path) as hdu_list:
            hdu_list[0].header = hdr
            hdu_list[0].data = rte_data_products
            hdu_list.writeto(out_dir+filename_root+'_rte_data_products.fits', overwrite=True)

        with fits.open(file_path) as hdu_list:
            hdu_list[0].hdrheader= hdr
            hdu_list[0].data = rte_data_products[5,:,:]
            hdu_list.writeto(out_dir+filename_root+'_blos_rte.fits', overwrite=True)
        # DC change 20211101 Gherdardo needs separate fits files from inversion
        with fits.open(file_path) as hdu_list:
            hdu_list[0].header = hdr
            hdu_list[0].data = rte_data_products[3,:,:]
            hdu_list.writeto(out_dir+filename_root+'_bazi_rte.fits', overwrite=True)

        with fits.open(file_path) as hdu_list:
            hdu_list[0].header = hdr
            hdu_list[0].data = rte_data_products[2,:,:]
            hdu_list.writeto(out_dir+filename_root+'_binc_rte.fits', overwrite=True)

        with fits.open(file_path) as hdu_list:
            hdu_list[0].header = hdr
            hdu_list[0].data = rte_data_products[1,:,:]
            hdu_list.writeto(out_dir+filename_root+'_bmag_rte.fits', overwrite=True)

        with fits.open(file_path) as hdu_list:
            hdu_list[0].header = hdr
            hdu_list[0].data = rte_data_products[4,:,:]
            hdu_list.writeto(out_dir+filename_root+'_vlos_rte.fits', overwrite=True)

        with fits.open(file_path) as hdu_list:
            hdu_list[0].header = hdr
            hdu_list[0].data = rte_data_products[0,:,:]
            hdu_list.writeto(out_dir+filename_root+'_Icont_rte.fits', overwrite=True)
            
        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        printc(f"------------- CMILOS RTE Run Time: {np.round(time.time() - start_time,3)} seconds ",bcolors.OKGREEN)
        printc('--------------------------------------------------------------',bcolors.OKGREEN)


def cmilos_fits(data_f, hdr_arr, wve_axis_arr, data_shape, cpos_arr, data, rte, field_stop, start_row, start_col, out_rte_filename, out_dir):
    """
    RTE inversion using CMILOS
    """
    print(" ")
    printc('-->>>>>>> RUNNING CMILOS ',color=bcolors.OKGREEN)
    
    try:
        CMILOS_LOC = os.path.realpath(__file__)

        CMILOS_LOC = CMILOS_LOC[:-15] + 'cmilos-fits/' #-11 as hrt_pipe.py is 11 characters

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

        start_time = time.time()

        file_path = data_f[scan]
        wave_axis = wve_axis_arr[scan]
        hdr = hdr_arr[scan]

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

        sdata = data[:,:,:,:,scan]
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
        
        # DC CHANGE (same number of digits of cmilos)
#         hdr['LAMBDA0'] = float('%e' % wave_axis[0])#needs it in Angstrom 6173.1 etc
#         hdr['LAMBDA1'] = float('%e' % wave_axis[1])
#         hdr['LAMBDA2'] = float('%e' % wave_axis[2])
#         hdr['LAMBDA3'] = float('%e' % wave_axis[3])
#         hdr['LAMBDA4'] = float('%e' % wave_axis[4])
#         hdr['LAMBDA5'] = float('%e' % wave_axis[5])
#         printc('-->>>>>>> CHANGING DATA PRECISION ',color=bcolors.OKGREEN)
#         for w in range(l):
#             for s in range(p):
#                 for i in range(y):
#                     for j in range(x):
#                         input_arr[w,s,i,j] = float('%e' % input_arr[w,s,i,j])
        # DC CHANGE END
        
        hdu1 = fits.PrimaryHDU(data=input_arr, header = hdr)

        #mask
        mask = np.ones((sdata.shape[0],sdata.shape[1])) #change this for fdt
        hdu2 = fits.ImageHDU(data=mask)

        #write out to temp fits
        hdul_tmp = fits.HDUList([hdu1, hdu2])
        hdul_tmp.writeto(out_dir+'temp_cmilos_io.fits', overwrite=True)
        
        del sdata

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
        """

        """
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

        """
        noise_in_V =  np.mean(data[:,:,3,cpos_arr[0],:])
        low_values_flags = np.max(np.abs(data[:,:,3,:,scan]),axis=-1) < noise_in_V  # Where values are low
        
        rte_out[2,low_values_flags] = 0 #not sure about 2,3,4 indexing here
        rte_out[3,low_values_flags] = 0
        rte_out[4,low_values_flags] = 0
        """
       
        rte_data_products = np.zeros((6,rte_out.shape[1],rte_out.shape[2]))

        rte_data_products[0,:,:] = rte_out[9,:,:] + rte_out[10,:,:] #continuum
        rte_data_products[1,:,:] = rte_out[1,:,:] #b mag strength
        rte_data_products[2,:,:] = rte_out[2,:,:] #inclination
        rte_data_products[3,:,:] = rte_out[3,:,:] #azimuth
        rte_data_products[4,:,:] = rte_out[7,:,:] #vlos
        rte_data_products[5,:,:] = rte_out[1,:,:]*np.cos(rte_out[2,:,:]*np.pi/180.) #blos

        rte_data_products *= field_stop[np.newaxis,start_row:start_row + data.shape[0],start_col:start_col + data.shape[1],scan] #field stop, set outside to 0

        if out_rte_filename is None:
            filename_root = str(file_path.split('.fits')[0][-10:])
        else:
            if isinstance(out_rte_filename, list):
                filename_root = out_rte_filename[scan]

            elif isinstance(out_rte_filename, str):
                filename_root = out_rte_filename

            else:
                filename_root = str(file_path.split('.fits')[0][-10:])
                print(f"out_rte_filename neither string nor list, reverting to default: {filename_root}")

        with fits.open(file_path) as hdu_list:
            hdu_list[0].header = hdr
            hdu_list[0].data = rte_data_products
            hdu_list.writeto(out_dir+filename_root+'_rte_data_products.fits', overwrite=True)

        with fits.open(file_path) as hdu_list:
            hdu_list[0].header = hdr
            hdu_list[0].data = rte_data_products[5,:,:]
            hdu_list.writeto(out_dir+filename_root+'_blos_rte.fits', overwrite=True)
        # DC change 20211101 Gherdardo needs separate fits files from inversion
        with fits.open(file_path) as hdu_list:
            hdu_list[0].header = hdr
            hdu_list[0].data = rte_data_products[3,:,:]
            hdu_list.writeto(out_dir+filename_root+'_bazi_rte.fits', overwrite=True)

        with fits.open(file_path) as hdu_list:
            hdu_list[0].header = hdr
            hdu_list[0].data = rte_data_products[2,:,:]
            hdu_list.writeto(out_dir+filename_root+'_binc_rte.fits', overwrite=True)

        with fits.open(file_path) as hdu_list:
            hdu_list[0].header = hdr
            hdu_list[0].data = rte_data_products[1,:,:]
            hdu_list.writeto(out_dir+filename_root+'_bmag_rte.fits', overwrite=True)

        with fits.open(file_path) as hdu_list:
            hdu_list[0].header = hdr
            hdu_list[0].data = rte_data_products[4,:,:]
            hdu_list.writeto(out_dir+filename_root+'_vlos_rte.fits', overwrite=True)

        with fits.open(file_path) as hdu_list:
            hdu_list[0].header = hdr
            hdu_list[0].data = rte_data_products[0,:,:]
            hdu_list.writeto(out_dir+filename_root+'_Icont_rte.fits', overwrite=True)

        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        printc(f"------------- CMILOS-FITS RTE Run Time: {np.round(time.time() - start_time,3)} seconds ",bcolors.OKGREEN)
        printc('--------------------------------------------------------------',bcolors.OKGREEN)



def pmilos(data_f, wve_axis_arr, data_shape, cpos_arr, data, rte, field_stop, start_row, start_col, out_rte_filename, out_dir):
    """
    RTE inversion using PMILOS
    """
    print(" ")
    printc('-->>>>>>> RUNNING PMILOS ',color=bcolors.OKGREEN)
    
    try:
        PMILOS_LOC = os.path.realpath(__file__)

        PMILOS_LOC = PMILOS_LOC[:-15] + 'p-milos/' #11 as hrt_pipe.py is 11 characters -8 if in utils.py

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

        start_time = time.time()

        file_path = data_f[scan]
        wave_axis = wve_axis_arr[scan]

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

        hdr = fits.Header()

        primary_hdu = fits.PrimaryHDU(wave_input, header = hdr)
        hdul = fits.HDUList([primary_hdu])
        hdul.writeto(f'./p-milos/run/wavelength_tmp.fits', overwrite=True)

        sdata = data[:,:,:,:,scan].T
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
        rte_data_products[2,:,:] = result[:,:,5] #inclination
        rte_data_products[3,:,:] = result[:,:,6] #azimuth
        rte_data_products[4,:,:] = result[:,:,2] #vlos
        rte_data_products[5,:,:] = result[:,:,1]*np.cos(result[:,:,5]*np.pi/180.) #blos

        rte_data_products *= field_stop[np.newaxis,start_row:start_row + data.shape[0],start_col:start_col + data.shape[1],scan] #field stop, set outside to 0

        #flipping taken care of for the field stop in the hrt_pipe 
        #printc(f'  ---- >>>>> and HERE now .... ',color=bcolors.WARNING)
        if out_rte_filename is None:
            filename_root = str(file_path.split('.fits')[0][-10:])
        else:
            filename_root = out_rte_filename

        with fits.open(file_path) as hdu_list:
            hdu_list[0].header = hdr
            hdu_list[0].data = rte_data_products
            hdu_list.writeto(out_dir+filename_root+'_rte_data_products.fits', overwrite=True)

        with fits.open(file_path) as hdu_list:
            hdu_list[0].header = hdr
            hdu_list[0].data = rte_data_products[5,:,:]
            hdu_list.writeto(out_dir+filename_root+'_blos_rte.fits', overwrite=True)
        # DC change 20211101 Gherdardo needs separate fits files from inversion
        with fits.open(file_path) as hdu_list:
            hdu_list[0].header = hdr
            hdu_list[0].data = rte_data_products[3,:,:]
            hdu_list.writeto(out_dir+filename_root+'_bazi_rte.fits', overwrite=True)

        with fits.open(file_path) as hdu_list:
            hdu_list[0].header = hdr
            hdu_list[0].data = rte_data_products[2,:,:]
            hdu_list.writeto(out_dir+filename_root+'_binc_rte.fits', overwrite=True)

        with fits.open(file_path) as hdu_list:
            hdu_list[0].header = hdr
            hdu_list[0].data = rte_data_products[1,:,:]
            hdu_list.writeto(out_dir+filename_root+'_bmag_rte.fits', overwrite=True)

        with fits.open(file_path) as hdu_list:
            hdu_list[0].header = hdr
            hdu_list[0].data = rte_data_products[4,:,:]
            hdu_list.writeto(out_dir+filename_root+'_vlos_rte.fits', overwrite=True)

        with fits.open(file_path) as hdu_list:
            hdu_list[0].header = hdr
            hdu_list[0].data = rte_data_products[0,:,:]
            hdu_list.writeto(out_dir+filename_root+'_Icont_rte.fits', overwrite=True)


    printc('--------------------------------------------------------------',bcolors.OKGREEN)
    printc(f"------------- PMILOS RTE Run Time: {np.round(time.time() - start_time,3)} seconds ",bcolors.OKGREEN)
    printc('--------------------------------------------------------------',bcolors.OKGREEN)
