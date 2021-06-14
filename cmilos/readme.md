# C- MILOS#

# CMILOS v0.9 (2015) #
## RTE INVERSION C code for SOPHI (based on the ILD code MILOS by D. Orozco) ##
## juanp (IAA-CSIC) ##

#First recompile for your machine:

#If Unix based:
Make clean
Make

#How to use:#

```milos NLAMBDA MAX_ITER CLASSICAL_ESTIMATES RFS [FWHM DELTA NPOINTS] profiles_file.txt > output.txt```

* NLAMBDA : number of lambda of input profiles
* MAX_ITER : of inversion
* CLASSICAL_ESTIMATES :use classical estimates? 1 yes, 0 no, 2 only CE
* RFS : 0 RTE, 1 Spectral Synthesis, 2 Spectral Synthesis + Response Funcions
* [FWHM DELTA NPOINTS]: use convolution with a gaussian? if the tree parameteres are defined yes, else no. Units in A. NPOINTS has to be odd.
* profiles_file.txt : name of input profiles file
* output.txt : name of output file
