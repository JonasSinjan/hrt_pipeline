# **SO/PHI-HRT PIPELINE**

Reduction software for SO/PHI-HRT instrument on the ESA Solar Orbiter
## **PHI-HRT data reduction**
1. read in science data (+scaling) open path option + open for several scans at once
2. read in flat field (+scaling)- just accepts one flat field fits file
3. read in dark field (+scaling)
4. apply dark field
5. option to clean flat field with unsharp masking (Stokes V only)
6. normalise flat field
7. apply flat field
8. prefilter correction
9. read in field stop
10. apply field stop
11. demodulate with const demod matrix <br>
        a) option to output demod to fits file <br>
12. normalise to quiet sun
13. calibration <br>
        a) cross talk correction <br>
        (if required) b) ghost correction - **not implemented yet** <br>
14. rte inversion with cmilos <br>
        a) output rte data products to fits file <br>


#### **CONFIGURATION**

Any and all steps can be turned oon or off as you wish using the keywords in the `phihrt_pipe` function


## **DOWNLOAD INPUT FILES**


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

Can also use `download_from_db.py` to perform multi download from database

Instructions:
  1. From the database find the files you wish to download
  2. Copy the 'Download File List' that the database will generate
  3. Paste into the `file_names.txt` file
  4. Create a `.env` file with your MPS windows login: <br> 
      ```text=
      USER_NAME =
      PHIDATAPASSWORD =
      ```  
  5. Set the target download folder in the `download_from_db.py` file
  6. Run the file (will require dotenv python module to be installed) 

OR : use `download_files.py` to download images from the attic repository: https://www2.mps.mpg.de/services/proton/phi/fm/attic/

## **SETUP**

1. Compile milos:

```bash
make clear
make
```
        
2. Setup virtual environment from requirements.txt

using pip - REQUIRES PYTHON >= 3.6
```bash
pip install -r requirements.txt
```
using conda (Anaconda3) - creates virtual environment called 'dataproc'
```bash
conda env create -f environment.yml
```
2. Change fits files paths, desired processing steps and output directory in ```run.py```
 
3. Execute ```run.py```

```bash
python run.py
```
## **OUTPUT**

#### **Demod File**
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

### **Credit**: <br>

- SPGPylibs for the foundation, from which it was expanded upon
- CMILOS: RTE INVERSION C code for SOPHI (based on the ILD code MILOS by D. Orozco) Author: juanp (IAA-CSIC)