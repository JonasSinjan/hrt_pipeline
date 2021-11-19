# **SO/PHI-HRT PIPELINE**

Reduction software for SO/PHI-HRT instrument on the ESA Solar Orbiter
## **PHI-HRT data reduction**
1. read in science data (+scaling) open path option + open for several scans at once
2. read in flat field (+scaling)- accepts only one flat field fits file
3. read in dark field (+scaling)
4. apply dark field (to only science - assumes flat is dark fielded)
5. option to clean flat field with unsharp masking (Stokes QUV, UV or V)
6. normalise flat field
7. apply flat field
8. prefilter correction
9. apply field stop
10. demodulate with const demod matrix <br>
        a) option to output demod to fits file <br>
11. normalise to quiet sun
12. calibration <br>
        a) cross talk correction <br>
13. rte inversion with cmilos <br>
        a) output rte data products to fits files <br>




## **SETUP**

Operating System Required: Linux

If running on bob MPS server and you have access

Run bash script - setup.sh - to skip the first 3 steps

OTHERWISE:

1. Compile milos in both cmilos-fits folder and cmilos folder (cmilos-fits):

```bash
make clear
make
```

2. P-MILOS - multiple steps required - ASSUMING RUNNING ON **BOB** SERVER
- module load openmpi_gcc
- module load fftw
- load cfitsio (or export through .bashrc)

```bash
make clean
make
```    
3. Setup virtual environment from requirements.txt with PIP or CONDA

using pip - REQUIRES PYTHON >= 3.6
```bash
pip install -r requirements.txt
```
OR using conda (Anaconda3) - creates virtual environment called 'hrt_pipeline_env'
```bash
conda env create -f environment.yml
```

4. Activate 'hrt_pipeline_env'
```bash
source activate hrt_pipeline_env
```

5. Download files - see **DOWNLOAD INPUT FILES** Section

5. Generate json files with the science, dark and flat you desire to reduce, ensure that all keywords from in the example are used (for limb images you must know the limb in the FOV - can only reduce multiple limb files at the same runtime if all the same limb in FOV)

6. Make sure the correct input.json file is given to `hrt_pipe` in ```run.py```

7. Execute ```run.py```

```bash
python run.py
```

## **CONFIGURATION**

Any and all steps can be turned on or off as you wish using the keywords in the input json file passed to the `phihrt_pipe` function

See `/input_jsons/create_input_json.py` for example to create json file

## **DOWNLOAD INPUT FILES**

This needs the 'dataproc' environment - see **SETUP**

EITHER: download from the PHI Image Database (recommended): https://www2.mps.mpg.de/services/proton/phi/imgdb/

Suggested filters for HRT science data: 
- **KEYWORD DETECTOR = 'HRT'** <br >
- **Filename\* like \*L1_phi-hrt-ilam_date\***
        
To download via the command line (eg: if you want to save the files on a server and not locally)
```
wget --user yourusername --password yourpassword file_web_address
gunzip file.gz
```
Gunzip used to unpack the .gz to the file you want  <br>

Can also use `/download/download_from_db.py` to perform multi download from database:

Instructions:
  1. From the database find the files you wish to download
  2. Copy the 'Download File List' that the database will generate
  3. Paste into the `file_names.txt` file
  4. Create a `.env` file with your MPS Windows login: <br> 
      ```text=
      USER_NAME =
      PHIDATAPASSWORD =
      ```  
  5. Set the target download folder in the `download_from_db.py` file
  6. Run the file (will require dotenv python module to be installed - included in `dataproc`) 

OR : use `download_files.py` to download images from the attic repository: https://www2.mps.mpg.de/services/proton/phi/fm/attic/
## **OUTPUT**

#### **Stokes File**
Filename: `_reduced.fits `

Shape: [Y,X,POL,WAVE]

#### **RTE products**
- File: `_rte_data_products.fits`

  Shape: [6,Y,X] <br>
  First Index:
  - 0: Continuum Intensity
  - 1: Magnetic Field Strength |B| (Gauss)
  - 2: Inclination (degrees)
  - 3: Azimuth (degrees)
  - 4: Vlos (km/s)
  - 5: Blos (Gauss) </p>

- File: `_bmag_rte.fits`

  Shape: [1,Y,X] <br>
  First Index: <br>
  - 0: |B| (Gauss) </p>

- File: `_binc_rte.fits`

  Shape: [1,Y,X] <br>
  First Index: <br>
  - 0: Inclination (Degrees) </p>

- File: `_bazi_rte.fits`

  Shape: [1,Y,X] <br>
  First Index: <br>
  - 0: Aimuth (Degrees) </p>

- File: `_blos_rte.fits`

  Shape: [1,Y,X] <br>
  First Index: <br>
  - 0: Blos (Gauss) </p>

- File: `_vlos_rte.fits`

  Shape: [1,Y,X] <br>
  First Index: <br>
  - 0: Vlos (km/s) </p>

- File: `_Icont_rte.fits`

  Shape: [1,Y,X] <br>
  First Index:
  - 0: Continuum Intensity


***


### **Authors**: <br>

Jonas Sinjan - Max Planck Institute for Solar System Research, Goettingen, Germany

### **Contributors**: <br>

Daniele Calchetti - Max Planck Institute for Solar System Research, Goettingen, Germany

### **Credit**: <br>

- SPGPylibs for the foundation, from which it was expanded upon
- CMILOS: RTE INVERSION C code for SO-PHI (based on the IDL code MILOS by D. Orozco) Author: juanp (IAA-CSIC)
- CMILOS-FITS: CMILOS with fits interace, fits interfacing developed by Philipp Loeschl (MPS)
- P-MILOS:  Parallellised RTE INVERSION C code Authors: Manuel Cabrera, Juan P. Cobos, Luis Bellot Rubio (IAA-CSIC)