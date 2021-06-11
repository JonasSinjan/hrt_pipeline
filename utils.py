from astropy.io import fits
import numpy as np
import os

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
    print('-- Obtaining voltages......')
    fg_head = 3
    try:
        with fits.open(file) as hdu_list:
            header = hdu_list[fg_head].data
            j = 0
            dummy = 0
            voltagesData = np.zeros((6))
            tunning_constant = 0.0
            ref_wavelength = 0.0
            for v in header:
                if (j < 6):
                    if tunning_constant == 0:
                        tunning_constant = float(v[4])/1e9
                    if ref_wavelength == 0:
                        ref_wavelength = float(v[5])/1e3
                    if np.abs(np.abs(v[2]) - np.abs(dummy)) > 5:
                        voltagesData[j] = float(v[2])
                        dummy = voltagesData[j] 
                        j += 1
        
        d1 = voltagesData[0] - voltagesData[1]
        d2 = voltagesData[4] - voltagesData[5]
        if np.abs(d1) > np.abs(d2):
            cpos = 0
        else:
            cpos = 5
        if verbose:
            print('Continuum position at wave: ', cpos)
        wave_axis = voltagesData*tunning_constant + ref_wavelength  #6173.3356

        return wave_axis,voltagesData,tunning_constant,cpos

    except Exception:
        print("Unable to open fits file: {}",file)     

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