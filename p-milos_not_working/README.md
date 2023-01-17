# P-MILOS

Authors: Manuel Cabrera, Juan P. Cobos, Luis Bellot Rubio (IAA-CSIC). 

This work has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No 824135 (SOLARNET).

For questions, please contact Luis Bellot (lbellot@iaa.es).


## Introduction 


This repository contains P-MILOS, a state-of-the-art parallel Milne-Eddington inversion code written in C.  The code is capable of inverting full Stokes spectropolarimetric measurements of photospheric lines in real time using one-component Milne-Eddington atmospheres. P-MILOS is very fast, reaching speeds of up to 2400 pixels per second in sequential applications (one core) and 2000 pixels per second per core in parallel applications (multiple cores). These numbers refer to the inversion of the four Stokes profiles of a spectral line sampled at 30 wavelength positions, convolved wih the instrumental PSF, assuming 9 free parameters, with a maximum of 50 iterations, on an AMD EPYC 7742 2.25GHz 128-core server working at 100% of its capacity.

In this page we explain how to install and run the code. We also provide a brief overview of the input/output files. A complete user manual can be found [here](p-milos_manual.pdf). 


## Requeriments 

### Libraries

To run P-MILOS, the following libraries must be present in your system: 

- [OpenMPI](https://www.open-mpi.org/) (oldest version tested 1.4-4)
- [CFITSIO](https://heasarc.gsfc.nasa.gov/fitsio/) (oldest version tested 3.3.4.0)
- [FFTW](http://www.fftw.org/)  (oldest version tested 3.3.3)
- [GSL](https://www.gnu.org/software/gsl/) (oldest version tested 1.13-3)
  
There are many different ways to install the libraries depending on the operating system.  In Ubuntu 18.04, we used the following commands:  

OpenMPI 

```
sudo apt-get update -y 
sudo apt-get install openmpi-bin
```

CFITSIO

```
sudo apt-get update -y 
sudo apt-get install libcfitsio*
```

FFTW

```
sudo apt-get update -y 
sudo apt-get install libfftw3-*
```

GSL

```
sudo apt-get update -y 
sudo apt-get install libgsl*
```

## Compilation

The code needs to be compiled on the target machine. To do this, run the command 'make' in the directory where the distribution is located. We strongly recommend you to use the latest version of the Intel C compiler, to achieve maximum performance. This is particularly important when the objective is to invert data streams in real time. 

For AMD processors, please edit the makefile and change the compilation option -xHost to -march=core-avx2. 

Be warned that the use of the AVX-2 or AVX-512 instruction sets increases the code speed at the expense of less consistent floating-point results. For some pixels, this may lead to a different number of iterations performed. 

* Compile sequential code and create executable **milos.x**  
```
make milos.x
```
* Compile parallel code and create executable **pmilos.x**
```
make pmilos.x
```
* Compile and create executables **milos.x** and **pmilos.x**
```
make 
```
* Clean object files and executables
```
make clean
```

## Execution


### Sequential code: milos.x 

The inversion process is controlled with a  **.mtrol** file which specifies the inversion conditions (observed profiles, atomic parameter file, stray-light profile file, PSF file, wavelength file, initial model atmosphere, parameters to be inverted, maximum number of iterations, etc). The format is nearly the same as that of the **.trol** files used by SIR. Here, the extension **mtrol** stands for "Milne-Eddington control file". You can find an example in the run directory: [pmilos.mtrol](run/pmilos.mtrol). The user manual provides a detailed description of this file and its various parameters.

The sequential code must be executed by passing the control file as a parameter. From the run directory, type 

```
../milos.x pmilos.mtrol
```

### Parallel code: pmilos.x

To run the parallel code we need both an **.mtrol** file and an **.init** file,  as in the case of SIR-parallel. The init file is used to specify the time steps to invert, along with other parameters. An example can be found in the run directory: [pmilos.minit](run/pmilos.minit). The manual describes in more detail the format and parameters of this file.

The parallel code must be executed using the command **mpirun** or **mpiexec**.  In the local machine, one can specify the number of processors to be used with the *-np* option, as in the following example with 16 processors:

```
mpiexec -np 16 ../pmilos.x pmilos.minit
```

It is also possible to run the code on different machines simultaneously over local ethernet. In that case, the names of the machines or their IP addresses must be specified in a file *hostnames* using the option *-f*. The code should be run as follows:

```
mpiexec -f hostnames -np 600 ../pmilos.x pmilos.minit
```
Note that ssh keys must be properly installed on every machine, so that connections can be established between them without prompting the user for a password.  For more details, refer to the P-MILOS manual.


## Input/output files

Below we briefly describe the most important I/O files used by the code. A complete description of these and other files is given in the manual. 

#### Profile file (.fits) 

P-MILOS can be fed with cubes containing the Stokes profiles observed in the entire field of view. This is the usual way of inverting measurements from narrow-band filter imagers such as SST/CRISP. 

The data cubes must be written as 4-dimension arrays in FITS format, with one cube containing one spectral scan. The four dimensions correspond to the wavelength axis, the polarization axis, and two spatial coordinates (x, y). The number of elements in each dimension is n_lambdas, n_stokes, n_x, and n_y, respectively. The cube can be arranged in any order, but the FITS header must contain the keywords CTYPE1, CTYPE2, CTYPE3 and CTYPE4 to specify each dimension according to the SOLARNET standard:

  -  **HPLN-TAN** indicates a spatial coordinate axis 
  -  **HPLT-TAN** indicates a spatial coordinate axis
  - **WAVE-GRI** indicates the wavelength axis
  - **STOKES  '** indicates the Stokes parameter axis

The example below corresponds to a data cube with the wavelength grid in the first dimension, the polarization axis in the second dimension, the x-spatial coordinate in the third dimension, and the y-spatial coordinate in the fourth dimension. 

```
CTYPE1  = 'WAVE-GRI' 
CTYPE2  = 'STOKES  ' 
CTYPE3  = 'HPLN-TAN'
CTYPE4  = 'HPLT-TAN  ' 
```

When the FITS data cube does not have a header, the array is assumed to be ordered as (n_lambdas, n_stokes, n_x, n_y).

Individual spectral scans within a time sequence must be stored in separate FITS files numbered sequentially using a common name plus the string "_tnnn.fits", where nnn is the spectral scan number padded with zeros (three digits). This makes it easy to locate any time step within the sequence. No gaps are allowed. An example is given below: 

2014.09.28_09:18:00_xtalk_t000.fits, 2014.09.28_09:18:00_xtalk_t001.fits, 2014.09.28_09:18:00_xtalk_t002.fits, .... , 2014.09.28_09:18:00_xtalk_t031.fits, 2014.09.28_09:18:00_xtalk_t032.fits .... 


#### Profile files (.per)

The Stokes profiles of individual pixels can also be stored in ASCII files with extension **.per**. These files have the same format as SIR profile files. They are used as input when inverting one pixel and as output when synthesizing the profiles from a given model atmosphere. 

Profile files have one row per wavelength sample and 6 columns containing:

* The index of the spectral line in the atomic parameter file
* The wavelength offset with respect to the central wavelength (in mA) 
* The value of Stokes I
* The value of Stokes Q
* The value of Stokes U
* The value of Stokes V

This is an example of a file containing the Stokes parameters of spectral line number 1 in 30 wavelength positions, from -350 to + 665 mA:

```
1    -350.000000    9.836711e-01    6.600326e-04    4.649822e-04    -3.694108e-03
1    -315.000000    9.762496e-01    1.186279e-03    8.329745e-04    -6.497371e-03
1    -280.000000    9.651449e-01    2.305113e-03    1.581940e-03    -1.135160e-02
1    -245.000000    9.443904e-01    5.032997e-03    3.333831e-03    -2.191048e-02
1    -210.000000    9.018359e-01    1.146227e-02    7.265856e-03    -4.544966e-02
1    -175.000000    8.222064e-01    2.265146e-02    1.368713e-02    -8.623441e-02
1    -140.000000    7.066048e-01    3.263217e-02    1.847884e-02    -1.242511e-01
1    -105.000000    5.799722e-01    3.157282e-02    1.560218e-02    -1.238010e-01
1    -70.000000    4.711627e-01    2.068015e-02    6.887295e-03    -8.728459e-02
1    -35.000000    4.014441e-01    9.837587e-03    -1.054865e-03    -4.476189e-02
1    -0.000000    3.727264e-01    4.631597e-03    -4.830483e-03    -9.482273e-03
1    35.000000    3.799767e-01    5.985593e-03    -3.846331e-03    2.321622e-02
1    70.000000    4.249082e-01    1.378049e-02    1.806246e-03    6.157872e-02
1    105.000000    5.119950e-01    2.571598e-02    1.073034e-02    1.046772e-01
1    140.000000    6.316050e-01    3.361904e-02    1.782490e-02    1.297000e-01
1    175.000000    7.571660e-01    2.941514e-02    1.718121e-02    1.115998e-01
1    210.000000    8.600360e-01    1.762856e-02    1.087526e-02    6.786425e-02
1    245.000000    9.230015e-01    8.213106e-03    5.305210e-03    3.366419e-02
1    280.000000    9.545796e-01    3.605521e-03    2.427676e-03    1.654710e-02
1    315.000000    9.701367e-01    1.734934e-03    1.205792e-03    9.068003e-03
1    350.000000    9.786569e-01    9.418463e-04    6.697733e-04    5.552034e-03
1    385.000000    9.838719e-01    5.600237e-04    4.053978e-04    3.671347e-03
1    420.000000    9.873039e-01    3.608273e-04    2.646724e-04    2.540035e-03
1    455.000000    9.896545e-01    2.543154e-04    1.872806e-04    1.792397e-03
1    490.000000    9.912774e-01    1.968866e-04    1.437179e-04    1.263287e-03
1    525.000000    9.923597e-01    1.690016e-04    1.205914e-04    8.552026e-04
1    560.000000    9.929766e-01    1.638180e-04    1.128983e-04    4.949613e-04
1    595.000000    9.930763e-01    1.826333e-04    1.215294e-04    1.047099e-04
1    630.000000    9.923094e-01    2.385952e-04    1.569032e-04    -4.666936e-04
1    665.000000    9.895550e-01    3.734447e-04    2.531845e-04    -1.597078e-03
```


#### Wavelength grid file (.grid)

The wavelength grid file specifies the spectral line and the observed wavelength positions  (inversion mode) or the wavelength positions in which the profiles must be calculated (synthesis mode).  The line is identified by means of an index that must be present in the atomic parameter file. The wavelength range is given using three numbers: the initial wavelength, the wavelength step, and the final wavelength (all in mA). 

This file is written in ASCII and has the same format as the **.grid** SIR files. Here is an example:

```
Line indices            :   Initial lambda     Step     Final lambda
(in this order)                    (mA)          (mA)         (mA) 
-----------------------------------------------------------------------
1                       :        -350,            35,           665
```
This example corresponds to the file [malla.grid](run/malla.grid) in the *run* directory.


#### Wavelength grid file (.fits)

Different pixels in the field of view may have different wavelength grids (due, for example, to the telecentric or collimated setups of narrow-band filter imagers). A wavelength grid varying across the field of view can be specified as a 4-dimension array written in FITS format. The four dimensions are (line_index, x,y, lambda). The first dimension contains the index of the line in the atomic parameter file. Only one line can be inverted at a time with P-MILOS, so the number of elements in the first dimension is always one. For each pixel (x,y), the array of observed wavelengths must be given. 

If all pixels use the same wavelength grid, the FITS file should contain a 2-dimesion array with (1, n_lambdas) elements. Again, the first dimension contains the line index and the second the observed wavelengths.


#### Model atmosphere file (.mod)

Files with extension **.mod** are ASCII file containing the parameters of a Milne-Eddington model atmosphere. They are used in three situations:

1. To specify the initial model atmosphere in an inversion 
2. To store the best-fit model atmosphere resulting from the inversion of a profile provided as a **.per** file. 
3. To specify the model atmosphere in a spectral synthesis  

The following is an example of a model atmosphere file that can be used for the Fe I 6173 A line in the quiet Sun:

```
eta_0                :13.0
magnetic field [G]   :500.
LOS velocity [km/s]  :0.2
Doppler width [A]    :0.035
damping              :0.19
gamma [deg]          :30.
phi   [deg]          :30.
S_0                  :0.26
S_1                  :0.74
v_mac [km/s]         :0.
filling factor       :1
```

This file is different from the equivalent SIR file because Milne-Eddington atmospheres can be described with only 11 parameters. The units of the parameters are: Gauss (magnetic field strength), km/s (LOS velocity and macroturbulent velocity v_mac), Angstrom (Doppler width), and degrees (inclination gamma and azimuth phi). The rest of parameters do not have units. 


#### Model atmospheres (.fits) 

When full data cubes are inverted, the resulting model atmospheres are stored in FITS format as 3-dimension arrays of (n_x, n_y, 13) elements. The first two dimensions give the spatial coordinates (x,y). The third dimension contains the eleven parameters of the model, plus the number of interations used by the code to find the solution and the chisqr-value of the fit. Therefore, the 13 elements stored in the third dimension are: 

  1. eta0 = line-to-continuum absorption coefficient ratio         
  2. B = magnetic field strength       [Gauss]
  3. vlos = line-of-sight velocity       [km/s]         
  4. dopp = Doppler width               [Angstroms]
  5. aa = damping parameter
  6. gm = magnetic field inclination [deg]
  7. az = magnetic field azimuth      [deg]
  8. S0 = source function constant  
  9. S1 = source function gradient
  10. mac = macroturbulent velocity  [km/s]
  11. alpha = filling factor of the magnetic component [0->1]
  12. Number of iterations required 
  13. chisqr value of the fit
  
  The file name of the output models is constructed from the name of the Stokes profiles cube, adding the string '_mod.fits' and any prefix indicated in the .minit file (this prefix may also serve to set the directory where the output is written). For example, if the observed profiles are stored in the cube *2014.09.28_09:18:00_xtalk_t006.fits*, the atmospheres resulting from the inversion will be written in  *2014.09.28_09:18:00_xtalk_t006_mod.fits* and the best-fit profiles in *2014.09.28_09:18:00_xtalk_t006_stokes.fits.* To save on disk space, the user may decide not to write the best-fit profiles, setting the corresponding option in the .minit file.

