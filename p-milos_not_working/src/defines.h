#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include "/opt/local/cfitsio/cfitsio-3.350/include/fitsio.h" // 


#ifdef USE_DOUBLE_PRECISION

#define REAL double
#define SQRT(x)	sqrt(x)
#define SIN(x)	sin(x)
#define COS(x)	cos(x)
#define TAN(x)	tan(x)
#define ATAN(x)	atan(x)
#define ATAN2(x,y) atan2(x,y)
#define FABS(x)	fabs(x)
#define CREAL(x) creal(x)
#define CIMAG(x) cimag(x)
#define EXP(x) exp(x)

#else

#define REAL float
#define SQRT(x) sqrtf(x)
#define SIN(x) sinf(x)
#define COS(x) cosf(x)
#define TAN(x) tanf(x)
#define ATAN(x) atanf(x)
#define ATAN2(x,y) atan2f(x,y)
#define FABS(x) fabsf(x)
#define CREAL(x) crealf(x)
#define CIMAG(x) cimagf(x)
#define EXP(x) expf(x)

#endif /* USE_DOUBLE_PRECISION */

#define MOD(n, N) ((n<0)? N+n : n)


#ifndef DEFINES_H_
#define DEFINES_H_


//NumeroS cuanticos
#define CUANTIC_NWL 1
#define CUANTIC_SLOI 2
#define CUANTIC_LLOI 1
#define CUANTIC_JLOI 1
#define CUANTIC_SUPI 2
#define CUANTIC_LUPI 2
#define CUANTIC_JUPI 0


#define NOISE_SIGMA 0.001 

//#define NTERMS 11 

#define PRECISION double //double 

//#############################################


// DONT'T MODIFY ANYTHING BELOW OF THIS LINE

#define PI 	3.14159265358979323846264338327950288419716939937510
 		 	
#define ILAMBDA 0.1
#define TOPLIM 1e-12 
#define SLIGHT 0
#define NOISE 1e-10 //0.001

#define RR  0.5641895836

#define VLIGHT 2.99792458e+5 //;light speed (km/s); 

#define CTE4_6_13 4.6686411e-13
#define AH 1.0 //heliocentric angle

#define FFT_FORWARD -1 
#define FFT_BACKWARD +1

#define NPARMS 4 //(IQUV)

//INIT_MODEL=[eta0,magnet,vlos,landadopp,aa,gamma,azi,B1,B2,macro,alfa]
struct INIT_MODEL{
	PRECISION eta0; // 0
	PRECISION B;//magnetic field    
	PRECISION vlos;
	PRECISION dopp;
	PRECISION aa;
	PRECISION gm; //5
	PRECISION az;
	PRECISION S0;
	PRECISION S1;
	PRECISION mac; //9
	PRECISION alfa;		
};

struct CUANTIC{  

	REAL N_PI;
	REAL N_SIG;
	REAL * NUB;//size stored in  n_sig
	REAL * NUP;//size stored in n_pi
	REAL * NUR;//size stored in n_sig
	REAL * WEB;//size stored in n_sig
	REAL * WEP;//size stored in n_pi
	REAL * WER;//size stored in n_sig
	REAL GL;
	REAL GU;
	REAL GEFF;
	REAL FO;	

};

typedef struct INIT_MODEL Init_Model;
typedef struct CUANTIC Cuantic;

/******************************************************/

void AllocateMemoryDerivedSynthesis(int numl);
void FreeMemoryDerivedSynthesis();

/******************************************************/


Cuantic * create_cuantic(PRECISION * dat, int log);

int me_der(Cuantic *cuantic,Init_Model *initModel,PRECISION * wlines,PRECISION *lambda,int nlambda, REAL *d_spectra,REAL *spectra, REAL * spectra_slight,REAL ah,REAL * slight,int filter, int * fix);

int mil_sinrf(Cuantic *cuantic,Init_Model *initModel,PRECISION * wlines,PRECISION *lambda,int nlambda,REAL *spectra,
			REAL ah,REAL * slight,REAL * spectra_mc, REAL * spectra_slight, int filter);
			

PRECISION * fgauss(PRECISION MC, PRECISION * eje,int neje,PRECISION landa,int deriv);
PRECISION * fgauss_WL(PRECISION FWHM, PRECISION step_between_lw, PRECISION lambda0, PRECISION lambdaCentral, int nLambda, int * sizeG);

int fvoigt(PRECISION damp,REAL *vv,int nvv,REAL *h, REAL *f);



/******************* DEFINITIONS FOR READ FITS FILE *********************/

/* 
	Every fits_image will be store in memory like  4 Vector of PRECISION TYPE using this dimension: rows*cols*nLambdas. 
	If we want access a pixel in the vector: stockesI[ (numRow*numCol) + (rows*cols*nLamba)] (been lambda from 0 to nLambda-1)
*/



#define CTYPE1 "CTYPE1"
#define CTYPE2 "CTYPE2"
#define CTYPE3 "CTYPE3"
#define CTYPE4 "CTYPE4"
#define CUNIT1 "CUNIT1"
#define CUNIT2 "CUNIT2"
#define CUNIT3 "CUNIT3"

#define CUNIT_ANSTROM "Angstrom"
#define CUNIT_ARCSEC "arcsec"
#define CTYPE_WAVE "WAVE-GRI"
#define CTYPE_HPLN_TAN "HPLN-TAN"
#define CTYPE_HPLT_TAN "HPLT-TAN"
#define CTYPE_STOKES "STOKES"


/* This Sttructure will store information relative a pixel in the image to process by fuction lmils in C, MPI and CUDA. 
	In order to do more flexible the structure and pass this structure in the message to MPI and KERNEL of CUDA, we have decided 
	put an attribute for number of pixels to store in spectro. We will play with this parameter to pass more or less pixels through MPI and to the 
	KERNEL of CUDA. 
 */
struct VPIXEL {
	PRECISION * vLambda;
	float * spectro;
	int nLambda;
};

typedef struct VPIXEL vpixels;



struct FITS_IMAGE{  

	int rows, rows_original;  // number of rows in the image
	int cols, cols_original;  // number of cols in the image
	int nLambdas; // number of lambdas in the image 
	int numStokes; // number of stokes paramters, normally is 4

	/* POSITION OF EACH DATA IN THE DIMENSIONS OF FITS */
	int pos_lambda;
	int pos_row;
	int pos_col;
	int pos_stokes_parameters; 

	char ctype_1[FLEN_CARD];
	char ctype_2[FLEN_CARD];
	char ctype_3[FLEN_CARD];
	char ctype_4[FLEN_CARD];
	char cunit_1[FLEN_CARD];
	char cunit_2[FLEN_CARD];
	char cunit_3[FLEN_CARD];
	char cunit_4[FLEN_CARD];

	int numPixels;
	vpixels * pixels;
	PRECISION * vLambdaImagen;
	float * spectroImagen;
	long * naxes, * naxes_original;
	char ** vCard;
	char ** vKeyname;
	int nkeys;
	int naxis;
	int bitpix;
};

typedef struct FITS_IMAGE FitsImage;

#define NUMBER_PARAM_MODELS 11

#define PATH_MAX 4096

struct CONFIG_CONTROL{

	int NumberOfCycles;
	char ObservedProfiles[4096];
	char StrayLightFile[4096];
	char PSFFile[4096];
	char WavelengthFile[4096];
	char AtomicParametersFile[4096];
	char InitialGuessModel[4096];
	char InitialGuessModel_2[4096];
	REAL WeightForStokes[4];
	int InvertFillingFactor;
	int InvertStrayLightFactor;
	PRECISION mu;
	int EstimatedSNForI;
	int ContinuumContrast;
	REAL InitialDiagonalElement;
	int ConvolveWithPSF;
	int useFFT;
	PRECISION FWHM;
	PRECISION CentralWaveLenght;
	//INIT_MODEL=[eta0,magnet,vlos,landadopp,aa,gamma,azi,B1,B2,macro,alfa]
	int fix[11]; // eta0, B , vlos, dopp, aa, gm , az, S0, S1, mac, alpha
	int fix2[11]; // eta0, B, vlos, dopp, aa, gm , az, S0, S1, mac, alpha
	int saveChisqr;
	PRECISION toplim; // Optional minimum relative difference between two succesive merit-function values
	REAL sigma [4];
	REAL noise;
	int UseClassicalEstimates;
	int UseRTEInversion;
	int SaveSynthesisAdjusted;	
	int typeFileOutputModel; // 0 print to FITS , 1 print to TXT. 
	char OutputModelFile[4096];
	char OutputSynthesisFile[4096];
	char MallaGrid[4096];
	char AbundancesFile[4096];
	int useMallaGrid; // value 1 --> use malla grid, value 0 --> use fits fil
	int automaticSelectOfNodes;
	char controlFile [4096];
	char typeInputStokes [50];
	char typeInputStrayLight[50];
	int nx;
	int ny;
	int subx1;
	int subx2;
	int suby1;
	int suby2;	
	char outputPrefix[4096];	
	char MaskFile[4096];
	int t1;
	int t2;
	int logclambda;
	int loopInversion;
	int invertDirectory;	
};

typedef struct CONFIG_CONTROL ConfigControl;

// values for init guess model 

#define INITIAL_MODEL_ETHA0 "INITIAL_MODEL_ETHA0"
#define INITIAL_MODEL_B "INITIAL_MODEL_B"
#define INITIAL_MODEL_VLOS "INITIAL_MODEL_VLOS"
#define INITIAL_MODEL_LAMBDADOPP "INITIAL_MODEL_LAMBDADOPP"
#define INITIAL_MODEL_AA "INITIAL_MODEL_AA"
#define INITIAL_MODEL_GM "INITIAL_MODEL_GM"
#define INITIAL_MODEL_AZI "INITIAL_MODEL_AZI"
#define INITIAL_MODEL_S0 "INITIAL_MODEL_S0"
#define INITIAL_MODEL_S1 "INITIAL_MODEL_S1"
#define INITIAL_MODEL_MAC "INITIAL_MODEL_MAC"
#define INITIAL_MODEL_ALFA "INITIAL_MODEL_ALFA"


// TYPES OF CONVOLUTION 

#define CONVOLUTION_FFT "FFT"
#define CONVOLUTION_DIRECT "DIRECT"

// TIMEOUT FILE READ 

#define TIMEOUT_FILE 600 // SECONDS

// PERCENTAGE OF ITERATION
#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

struct NAME_FILE {
	char  name [PATH_MAX];
};

typedef struct NAME_FILE nameFile;



#define PER_FILE ".per"
#define GRID_FILE ".grid"
#define FITS_FILE ".fits"
#define TROL_FILE ".mtrol"
#define MOD_FILE ".mod"
#define PSF_FILE ".psf"
#define OUTPUT_MOD_FIT_EXT "_output.fits"
#define MOD_FITS "_mod.fits"
#define STOKES_FIT_EXT "_stokes.fits"
#define OUTPUT_MOD_TXT_EXT "_output_mod.txt"
#define STOKES_PER_EXT "_stokes.per"


// linked list to store name of files 
typedef struct snodo {
    char d_name [256];
    struct snodo * next;
} tnodo;

typedef tnodo *tpuntero;

#endif /*DEFINES_H_*/


