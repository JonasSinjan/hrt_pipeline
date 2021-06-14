# SO/PHI-HRT PIPELINE

Reduction software for SO/PHI-HRT instrument on the ESA Solar Orbiter
## PHI-HRT data reduction
1. read in science data (+scaling) open path option + open for several scans at once
2. read in flat field (+scaling)- just accepts one flat field fits file
3. read in dark field (+scaling)
4. apply dark field
5. option to clean flat field with unsharp masking (Stokes V only)
6. normalise flat field
7. apply flat field
8. prefilter correction - **not implemented yet**
9. read in field stop
10. apply field stop
11. demodulate with const demod matrix <br />
        a) option to output demod to fits file <br />
12. normalise to quiet sun
13. calibration <br />
        a) ghost correction - **not implemented yet** <br />
        b) cross talk correction - **not implemented yet** <br />
14. rte inversion with cmilos <br />
        a) output rte data products to fits file <br />
## SETUP

1. Compile milos:

```bash
make clean
make
```
        
2. Setup virtual environment from requirements.txt

using pip
```bash
pip install -r requirements.txt
```
using conda
```bash
conda create --name <env_name> --file requirements.txt
```
2. Change fits files paths in ```run.py```

3. Execute ```run.py```

```bash
python run.py
```

Authors: <br />

Jonas Sinjan - Max Planck Institute for Solar System Research, Goettingen Germany

Credit: <br />

SPGPylibs for the foundation, from which it was expanded upon