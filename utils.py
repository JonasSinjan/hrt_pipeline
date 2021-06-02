from astropy.io import fits
import numpy as np

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