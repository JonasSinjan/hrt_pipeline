import subprocess
import os
from dotenv import load_dotenv

load_dotenv()
username = os.environ.get('USER_NAME')
password = os.environ.get('PHIDATAPASSWORD')

def get_file_list(text):
  text_str = str(text)
  text_split = text_str.split("\n")
  if text_split[0][0] == 'h':
    print("contains https links")
  print(text_split)
  return text_split

file = open('./file_names.txt')
text = file.read()

text_split = get_file_list(text)

#######################################################
#                 TARGET DIRECTORY
#######################################################

target_directory = '/data/slam/home/sinjan/hmi_hrt_cc/fdt_files/'

os.chdir(target_directory)

print(f"Downloading to target directory: {target_directory}")

for file in text_split:
  file = file.split("/")[-1]
  if len(file) < 20:
    print("Error occured with the file name, skipping to the next one")
    pass
  else:
    subprocess.call(["wget", "--user", f"{username}","--password", f"{password}", f"https://www2.mps.mpg.de/services/proton/phi/imgdb/{file}"])
    #subprocess.call(["gunzip", f"{file}"])
    subprocess.call(["chmod", "a+r", f"{file}"])

print("Download and unpacking complete")
