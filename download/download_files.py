from dotenv import load_dotenv
import requests
import os

#########################################################
"""
To make this work need to create a .env file and install dotenv: 'pip install dotenv':
USER_NAME = YOUR LOGIN USERNAME (NORMALLY JUST YOUR LAST NAME)
PHIDATAPASSWORD = YOUR WINDOWS PASSWORD
"""
#########################################################


def download_file(url, dload_location = '../fits_files/'):
  """
  Download files from the given url and store locally
  
  Parameters
  ----------
  url:  url of the desired fits file
  """

  load_dotenv()
  username = os.environ.get('USER_NAME')
  password = os.environ.get('PHIDATAPASSWORD')
  local_filename = url.split('/')[-1]
  # NOTE the stream=True parameter below
  with requests.get(url, stream=True, auth=(username,password)) as r:
      r.raise_for_status()
      with open(dload_location + local_filename, 'wb') as f:
          for chunk in r.iter_content(chunk_size=8192): 
              # If you have chunk encoded response uncomment if
              # and set chunk_size parameter to None.
              #if chunk: 
              f.write(chunk)
  return dload_location + local_filename

if __name__ == "__main__":

  path = 'stp-144/'

  files = ['solo_L0_phi-hrt-ilam_20201117T170209_V202106300922C_0051170001.fits']
  """
  #['solo_L0_phi-hrt-ilam_20210421T120003_V202106080929C_0144210101.fits', 'solo_L0_phi-hrt-ilam_20210424T120003_V202106141014C_0144240101.fits', 
  'solo_L0_phi-hrt-ilam_20210425T120002_V202106141020C_0144250101.fits', 'solo_L0_phi-hrt-ilam_20210426T120002_V202106162118C_0144260101.fits',
  'solo_L0_phi-hrt-ilam_20210427T120002_V202106162052C_0144270101.fits', 'solo_L0_phi-hrt-ilam_20210427T120002_V202106171444C_0144270101.fits', 
  'solo_L0_phi-hrt-ilam_20210427T120002_V202106171517C_0144270101.fits']
  """

  url = 'https://www2.mps.mpg.de/services/proton/phi/fm/attic/'

  download_folder = '/data/slam/home/sinjan/fits_files/'

  for file in files:

    download_file(url + path + file, dload_location = download_folder)

