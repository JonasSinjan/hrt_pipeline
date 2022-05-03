import numpy as np
from scipy.ndimage import gaussian_filter
from operator import itemgetter
from sophi_hrt_pipe.utils import *
import os
import time

def setup_header(hdr_arr):
    k = ['CAL_FLAT','CAL_USH','SIGM_USH',
    'CAL_PRE','CAL_GHST','CAL_REAL',
    'CAL_CRT0','CAL_CRT1','CAL_CRT2','CAL_CRT3','CAL_CRT4','CAL_CRT5',
    'CAL_CRT6','CAL_CRT7','CAL_CRT8','CAL_CRT9',
    'CAL_NORM','CAL_FRIN','CAL_PSF','CAL_IPOL',
    'CAL_SCIP','RTE_MOD','RTE_SW','RTE_ITER']

    v = [0,' ',' ',
    ' ','None ','NA',
    0,0,0,0,0,0,
    0,0,0,0,
    ' ','NA','NA',' ',
    'None',' ',' ',4294967295]

    c = ['Onboard calibrated for gain table','Unsharp masking correction','Sigma for unsharp masking [px]',
    'Prefilter correction (DID/file)','Ghost correction (name + version of module','Prealigment of images before demodulation',
    'cross-talk from I to Q (slope)','cross-talk from I to Q (offset)','cross-talk from I to U (slope)','cross-talk from I to U (offset)','cross-talk from I to V (slope)','cross-talk from I to V (offset)',
    'cross-talk from V to Q (slope)','cross-talk from V to Q (offset)','cross-talk from V to U (slope)','cross-talk from V to U (offset)',
    'Normalization (normalization constant PROC_Ic)','Fringe correction (name + version of module)','Onboard calibrated for instrumental PSF','Onboard calibrated for instrumental polarization',
    'Onboard scientific data analysis','Inversion mode','Inversion software','Number RTE inversion iterations']

    for h in hdr_arr:
        for i in range(len(k)):
            if k[i] in h:  # Check for existence
                pass # changed to avoid change of parameters after partial processing
                # h[k[i]] = v[i]
            else:
                if i==0:
                    h.set(k[i], v[i], c[i], after='CAL_DARK')
                else:
                    h.set(k[i], v[i], c[i], after=k[i-1])
    return hdr_arr

def data_hdr_kw(hdr, data):
    """
    add data descriptive header keywords
    """
    hdr['DATAMEDN'] = float(f"{np.median(data):.8g}")
    hdr['DATAMEAN'] = float(f"{np.mean(data):.8g}")
    #DATARMS
    #DATASKEW
    #DATAKURT
    return hdr

def load_flat(flat_f, accum_scaling, bit_conversion, scale_data, header_imgdirx_exists, imgdirx_flipped, cpos_arr) -> np.ndarray:
    """
    load, scale, flip and correct flat
    """
    print(" ")
    printc('-->>>>>>> Reading Flats',color=bcolors.OKGREEN)

    start_time = time.perf_counter()
    
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
    printc(f"------------ Load flats time: {np.round(time.perf_counter() - start_time,3)} seconds",bcolors.OKGREEN)
    printc('--------------------------------------------------------------',bcolors.OKGREEN)

    return flat


def load_dark(dark_f) -> np.ndarray:
    """
    loads dark field from given path
    """
    print(" ")
    printc('-->>>>>>> Reading Darks',color=bcolors.OKGREEN)

    start_time = time.perf_counter()

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
        printc(f"------------ Load darks time: {np.round(time.perf_counter() - start_time,3)} seconds",bcolors.OKGREEN)
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
    
    start_time = time.perf_counter()

    data -= dark[rows,cols, np.newaxis, np.newaxis, np.newaxis] 
    #flat -= dark[..., np.newaxis, np.newaxis] #- # all processed flat fields should already be dark corrected

    printc('--------------------------------------------------------------',bcolors.OKGREEN)
    printc(f"------------- Dark Field correction time: {np.round(time.perf_counter() - start_time,3)} seconds",bcolors.OKGREEN)
    printc('--------------------------------------------------------------',bcolors.OKGREEN)

    return data, flat


def normalise_flat(flat, flat_f, ceny, cenx) -> np.ndarray:
    """
    normalise flat fields at each wavelength position to remove the spectral line
    """
    print(" ")
    printc('-->>>>>>> Normalising Flats',color=bcolors.OKGREEN)

    start_time = time.perf_counter()

    try:
        norm_fac = np.mean(flat[ceny,cenx, :, :], axis = (0,1))[np.newaxis, np.newaxis, ...]  #mean of the central 1k x 1k
        flat /= norm_fac

        printc('--------------------------------------------------------------',bcolors.OKGREEN)
        printc(f"------------- Normalising flat time: {np.round(time.perf_counter() - start_time,3)} seconds",bcolors.OKGREEN)
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
#        demod_data = np.array([[ 0.26450154,  0.2839626,   0.12642948,  0.3216773 ],
#                     [ 0.59873885,  0.11278069, -0.74991184,  0.03091451],
#                     [ 0.10833212, -0.5317737,  -0.1677862,   0.5923593 ],
#                     [-0.46916953,  0.47738808, -0.43824592,  0.42579797]])
#Alberto 14/04/22
        printc(f'Using Alberto demodulation matrix ',color = bcolors.OKGREEN)
        mod_matrix = np.array([[ 0.99816  ,0.61485 , 0.010613 ,-0.77563 ],
                               [ 0.99192 , 0.08382 , 0.86254 , 0.46818],
                               [ 1.0042 , -0.84437 , 0.12872 ,-0.53972],
                               [ 1.0057 , -0.30576 ,-0.87969 , 0.40134]])
        demod_data = np.linalg.inv(mod_matrix)
    
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

    start_time = time.perf_counter()

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
        printc(f"------------- Flat Field correction time: {np.round(time.perf_counter() - start_time,3)} seconds ",bcolors.OKGREEN)
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
    # cop = np.copy(data)
    # new_data = np.zeros(data_shape)
    
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

    start_time = time.perf_counter()

    field_stop_loc = os.path.realpath(__file__)

    field_stop_loc = field_stop_loc.split('src/')[0] + 'field_stop/'

    field_stop,_ = load_fits(field_stop_loc + 'HRT_field_stop_new.fits')

    field_stop = np.where(field_stop > 0,1,0)

    if header_imgdirx_exists:
        if imgdirx_flipped == 'YES': #should be YES for any L1 data, but mistake in processing software
            field_stop = field_stop[:,::-1] #also need to flip the flat data after dark correction


    data *= field_stop[rows,cols,np.newaxis, np.newaxis, np.newaxis]
    printc('--------------------------------------------------------------',bcolors.OKGREEN)
    printc(f"------------- Field stop time: {np.round(time.perf_counter() - start_time,3)} seconds",bcolors.OKGREEN)
    printc('--------------------------------------------------------------',bcolors.OKGREEN)

    return data, field_stop


def load_ghost_field_stop(header_imgdirx_exists, imgdirx_flipped) -> np.ndarray:
    """
    apply field stop ghost mask to the science data
    """
    print(" ")
    printc("-->>>>>>> Loading ghost field stop",color=bcolors.OKGREEN)

    start_time = time.perf_counter()

    field_stop_loc = os.path.realpath(__file__)
    field_stop_loc = field_stop_loc.split('src/')[0] + 'field_stop/'

    field_stop_ghost,_ = load_fits(field_stop_loc + 'HRT_field_stop_ghost_new.fits')
    field_stop_ghost = np.where(field_stop_ghost > 0,1,0)

    if header_imgdirx_exists:
        if imgdirx_flipped == 'YES': #should be YES for any L1 data, but mistake in processing software
            field_stop_ghost = field_stop_ghost[:,::-1]

    printc('--------------------------------------------------------------',bcolors.OKGREEN)
    printc(f"------------- Load Ghost Field Stop time: {np.round(time.perf_counter() - start_time,3)} seconds",bcolors.OKGREEN)
    printc('--------------------------------------------------------------',bcolors.OKGREEN)
    return field_stop_ghost


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
        printc('Cross-talk from I to Q: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cQ[0],cQ[1],width=8,prec=4),color=bcolors.OKGREEN)
        printc('Cross-talk from I to U: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cU[0],cU[1],width=8,prec=4),color=bcolors.OKGREEN)
        printc('Cross-talk from I to V: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cV[0],cV[1],width=8,prec=4),color=bcolors.OKGREEN)
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
    
    for scan in range(data_shape[-1]):
        cont_stokes[scan] = np.mean(data[Ic_mask[...,scan],0,cpos_arr[0],scan])
    
    for i in range(6):
                
#         stokes_i_wv_avg = np.mean(data[ceny,cenx,0,i,:], axis = (0,1))
        stokes_i_wv_avg = np.ones(data_shape[-1])
        for scan in range(data_shape[-1]):
            stokes_i_wv_avg[scan] = np.mean(data[Ic_mask[...,scan],0,i,scan])
            
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


def hot_pixel_mask(data, rows, cols,mode='median'):
    """
    Apply hot pixel mask to the data, just after cross talk to remove pixels that diverge
    """
    file_loc = os.path.realpath(__file__)
    field_stop_fol = file_loc.split('src/')[0] + 'field_stop/'
    hot_pix_mask,_ = load_fits(field_stop_fol + 'bad_pixels.fits')
    hot_pix_cont,_ = load_fits(field_stop_fol + 'bad_pixels_contour.fits')
    
    s = data.shape # [y,x,p,l,s]
    
    if mode == 'median':
        func = lambda a: np.median(a,axis=0)
    elif mode == 'mean':
        func = lambda a: np.mean(a,axis=0)
    else:
        print('mode not found, input dataset not corrected')
        return new
    
    l = int(np.max(hot_pix_mask))
    
    for i in range(1,l+1):
        bad = (hot_pix_mask[rows,cols] == i)
        med = (hot_pix_cont[rows,cols] == i)
        data[bad] = func(data[med])
    
    return data

    
def crosstalk_auto_VtoQU(data_demod,cpos,wl,roi=np.ones((2048,2048)),verbose=0,npoints=5000,nlevel=0.3):
    import random, statistics
    from scipy.optimize import curve_fit
    def linear(x,a,b):
        return a*x + b
    my = []
    sy = []
    
    x = data_demod[roi>0,3,cpos].flatten()
    lx = data_demod[roi>0,0,cpos].flatten()
    lv = np.abs(data_demod[roi>0,3,cpos]).flatten()
    
    ids = (lv > nlevel/100.)
    x = x[ids].flatten()

    N = x.size
    idx = random.sample(range(N),npoints)
    mx = x[idx].mean() 
    sx = x[idx].std() 
    xp = np.linspace(x.min(), x.max(), 100)

    A = np.vstack([x, np.ones(len(x))]).T

    # V to Q
    yQ = data_demod[roi>0,1,wl].flatten()
    yQ = yQ[ids].flatten()
    my.append(yQ[idx].mean())
    sy.append(yQ[idx].std())
    cQ = curve_fit(linear,x,yQ,p0=[0,0])[0]
    pQ = np.poly1d(cQ)

    # V to U
    yU = data_demod[roi>0,2,wl].flatten()
    yU = yU[ids].flatten()
    my.append(yU[idx].mean())
    sy.append(yU[idx].std())
    cU = curve_fit(linear,x,yU,p0=[0,0])[0]
    pU = np.poly1d(cU)

    if verbose:
        
        PLT_RNG = 2
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(x[idx],yQ[idx],color='red',alpha=0.6,s=10)
        ax.plot(xp, pQ(xp), color='red', linestyle='dashed',linewidth=3.0)

        ax.scatter(x[idx],yU[idx],color='blue',alpha=0.6,s=10)
        ax.plot(xp, pU(xp), color='blue', linestyle='dashed',linewidth=3.0)

        ax.set_xlim([mx - PLT_RNG * sx,mx + PLT_RNG * sx])
        ax.set_ylim([min(my) - 1.8*PLT_RNG * statistics.mean(sy),max(my) + PLT_RNG * statistics.mean(sy)])
        ax.set_xlabel('Stokes V')
        ax.set_ylabel('Stokes Q/U')
        ax.text(mx - 0.9*PLT_RNG * sx, min(my) - 1.4*PLT_RNG * statistics.mean(sy), 'Cross-talk from V to Q: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cQ[0],cQ[1],width=8,prec=4), style='italic',bbox={'facecolor': 'red', 'alpha': 0.1, 'pad': 1}, fontsize=15)
        ax.text(mx - 0.9*PLT_RNG * sx, min(my) - 1.55*PLT_RNG * statistics.mean(sy), 'Cross-talk from V to U: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cU[0],cU[1],width=8,prec=4), style='italic',bbox={'facecolor': 'blue', 'alpha': 0.1, 'pad': 1}, fontsize=15)
#         fig.show()

        print('Cross-talk from V to Q: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cQ[0],cQ[1],width=8,prec=4))
        print('Cross-talk from V to U: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cU[0],cU[1],width=8,prec=4))
    
#         return cQ,cU,cV, (idx,x,xp,yQ,yU,yV,pQ,pU,pV,mx,sx,my,sy)
    else:
        printc('Cross-talk from V to Q: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cQ[0],cQ[1],width=8,prec=4),color=bcolors.OKGREEN)
        printc('Cross-talk from V to U: slope = {: {width}.{prec}f} ; off-set = {: {width}.{prec}f} '.format(cU[0],cU[1],width=8,prec=4),color=bcolors.OKGREEN)

    ct = np.asarray((cQ,cU)).T
    return ct


def CT_VtoQU(data, ctalk_params):
    """
    performs cross talk correction for I -> Q,U,V
    """
    before_ctalk_data = np.copy(data)
    data_shape = data.shape
    
#     ceny = slice(data_shape[0]//2 - data_shape[0]//4, data_shape[0]//2 + data_shape[0]//4)
#     cenx = slice(data_shape[1]//2 - data_shape[1]//4, data_shape[1]//2 + data_shape[1]//4)
    
    for i in range(6):
                

        tmp_param = ctalk_params#*stokes_i_wv_avg/cont_stokes

        q_slope = tmp_param[0,0]
        u_slope = tmp_param[0,1]
        
        q_int = tmp_param[1,0]
        u_int = tmp_param[1,1]
        
        data[:,:,1,i,:] = before_ctalk_data[:,:,1,i,:] - before_ctalk_data[:,:,3,i,:]*q_slope - q_int

        data[:,:,2,i,:] = before_ctalk_data[:,:,2,i,:] - before_ctalk_data[:,:,3,i,:]*u_slope - u_int

    
    return data

    
    
    
    
    
    
    
    
    
    
    
