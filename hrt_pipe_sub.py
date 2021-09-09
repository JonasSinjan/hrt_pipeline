import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from operator import itemgetter
from utils import *

def check_size(data_arr):
    first_shape = data_arr[0].shape
    result = all(element.shape == first_shape for element in data_arr)
    if (result):
        print("All the scan(s) have the same dimension")

    else:
        print("The scans have different dimensions! \n Ending process")

        exit()


def check_cpos(cpos_arr):
    first_cpos = cpos_arr[0]
    result = all(c_position == first_cpos for c_position in cpos_arr)
    if (result):
        print("All the scan(s) have the same continuum wavelength position")

    else:
        print("The scans have different continuum_wavelength postitions! Please fix \n Ending Process")

        exit()

def compare_cpos(data,cpos,cpos_ref):
    if cpos != cpos_ref:
        print("The flat field continuum position is not the same as the data, trying to correct.")

        if cpos == 5 and cpos_ref == 0:

            return np.roll(data, 1, axis = -1)

        elif cpos == 0 and cpos_ref == 5:

            return np.roll(data, -1, axis = -1)

        else:
            print("Cannot reconcile the different continuum positions. \n Ending Process.")

            exit()

def check_pmp_temp(hdr_arr):
    first_pmp_temp = hdr_arr[0]['HPMPTSP1']
    result = all(hdr['HPMPTSP1'] == first_pmp_temp for hdr in hdr_arr)
    if (result):
        print(f"All the scan(s) have the same PMP Temperature Set Point: {first_pmp_temp}")
        pmp_temp = str(first_pmp_temp)
        return pmp_temp
    else:
        print("The scans have different PMP Temperatures! Please fix \n Ending Process")

        exit()

def check_IMGDIRX(hdr_arr):
    if all('IMGDIRX' in hdr for hdr in hdr_arr):
        header_imgdirx_exists = True
        first_imgdirx = hdr_arr[0]['IMGDIRX']
        result = all(hdr['IMGDIRX'] == first_imgdirx for hdr in hdr_arr)
        if (result):
            print(f"All the scan(s) have the same IMGDIRX keyword: {first_imgdirx}")
            imgdirx_flipped = str(first_imgdirx)
            
            return header_imgdirx_exists, imgdirx_flipped
        else:
            print("The scans have different IMGDIRX keywords! Please fix \n Ending Process")
            exit()
    else:
        header_imgdirx_exists = False
        print("Not all the scan(s) contain the 'IMGDIRX' keyword! Assuming all not flipped - Proceed with caution")
        return header_imgdirx_exists, None

def compare_IMGDIRX(flat,header_imgdirx_exists,imgdirx_flipped,header_fltdirx_exists,fltdirx_flipped):
    if header_imgdirx_exists and imgdirx_flipped == 'YES':
        if header_fltdirx_exists:
            if fltdirx_flipped == 'YES':
                return flat
            else:
                return flat[:,:,::-1]
        else:
            return flat[:,:,::-1]
    elif (header_imgdirx_exists and imgdirx_flipped == 'NO') or not header_imgdirx_exists:
        if header_fltdirx_exists:
            if fltdirx_flipped == 'YES':
                return flat[:,:,::-1]
            else:
                return flat
        else:
            return flat

def stokes_reshape(data):
    data_shape = data.shape

    #converting to [y,x,pol,wv,scans]

    data = data.reshape(data_shape[0],data_shape[1],6,4,data_shape[-1]) #separate 24 images, into 6 wavelengths, with each 4 pol states
    data = np.moveaxis(data, 2,-2)
    
    return data

def unsharp_masking(flat,sigma,flat_pmp_temp,cpos_arr,clean_mode,pol_end=4):
    
    flat_demod, demodM = demod_hrt(flat, flat_pmp_temp)

    norm_factor = np.mean(flat_demod[512:1536,512:1536,0,0])

    flat_demod /= norm_factor

    new_demod_flats = np.copy(flat_demod)

    b_arr = np.zeros((2048,2048,3,5))

    if cpos_arr[0] == 0:
	    wv_range = range(1,6)

    elif cpos_arr[0] == 5:
	    wv_range = range(5)

    if clean_mode == "QUV":
	    start_clean_pol = 1
    elif clean_mode == "UV":
	    start_clean_pol = 2
    elif clean_mode == "V":
	    start_clean_pol = 3

    for pol in range(start_clean_pol,pol_end):

	    for wv in wv_range: #not the continuum

	        a = np.copy(np.clip(flat_demod[:,:,pol,wv], -0.02, 0.02))
	        b = a - gaussian_filter(a,sigma)
	        b_arr[:,:,pol-1,wv-1] = b
	        c = a - b

	        new_demod_flats[:,:,pol,wv] = c

    invM = np.linalg.inv(demodM)

    return np.matmul(invM, new_demod_flats*norm_factor)


def flat_correction(data,flat,flat_states,rows,cols):

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


def prefilter_correction(data,voltagesData_arr,prefilter,prefilter_voltages):
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
                
            imprefilter = (prefilter[:,:, index1]*v1 + prefilter[:,:, index2]*v2)/(v1+v2)

            data[:,:,:,wv,scan] /= imprefilter[...,np.newaxis]
            
    return data

def CT_ItoQUV(data, ctalk_params, norm_stokes, cpos_arr):

    before_ctalk_data = np.copy(data)
    data_shape = data.shape
    num_of_scans = data_shape[-1]
    ceny = slice(data_shape[0]//2 - data_shape[0]//4, data_shape[0]//2 + data_shape[0]//4)
    cenx = slice(data_shape[1]//2 - data_shape[1]//4, data_shape[1]//2 + data_shape[1]//4)
    cont_stokes = np.mean(data[ceny,cenx,0,cpos_arr[0],:], axis = (0,1))
    
    for i in range(6):
                
        stokes_i_wv_avg = np.mean(data[ceny,cenx,0,i,:], axis = (0,1))

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


