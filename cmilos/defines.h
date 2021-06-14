
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
//#include <fftw3.h> //siempre detras de complex.h!
//#include <math.h>
//#include <stdio.h>


#ifndef DEFINES_H_
#define DEFINES_H_

//---------------------------------------------------------
//---------------------------------------------------------
//---------------------------------------------------------
//---------------------------------------------------------
// USER CONFIGURATION

#define CENTRAL_WL 6173.335400


//##############################################
//SVD CONFIGURATION
#define USE_SVDCMP 1    //1 for using SVDCMP and 0 for using SVD_CORDIC  -->  Note: the SVDCMP doesn't work in float! only double

#define NORMALIZATION_SVD 0 //1 for using normalization matrixes ONLY  in the SVD_CORDIC

#define NUM_ITER_SVD_CORDIC 27 //9,18,27,36  --> 18 parece ok!

#define LIMITE_INFERIOR_PRECISION_SVD pow(2.0,-39)
#define LIMITE_INFERIOR_PRECISION_TRIG pow(2.0,-39)
#define LIMITE_INFERIOR_PRECISION_SINCOS pow(2.0,-39)
//#############################################

//INITIAL MODEL 
// #define INITIAL_MODEL_B 1200
// #define INITIAL_MODEL_GM 170
// #define INITIAL_MODEL_AZI 20
// #define INITIAL_MODEL_ETHA0 8
// #define INITIAL_MODEL_LAMBDADOPP 0.04  //en A
// #define INITIAL_MODEL_AA 0.18
// #define INITIAL_MODEL_VLOS 0.05 // Km/s
// #define INITIAL_MODEL_S0 0.35
// #define INITIAL_MODEL_S1 0.85

#define INITIAL_MODEL_B 400
#define INITIAL_MODEL_GM 30
#define INITIAL_MODEL_AZI 120
#define INITIAL_MODEL_ETHA0 3
#define INITIAL_MODEL_LAMBDADOPP 0.025  //en A
#define INITIAL_MODEL_AA 1.0
#define INITIAL_MODEL_VLOS 0.01 // Km/s
#define INITIAL_MODEL_S0 0.15
#define INITIAL_MODEL_S1 0.85


//NumeroS cuanticos
#define CUANTIC_NWL 1
#define CUANTIC_SLOI 2
#define CUANTIC_LLOI 1
#define CUANTIC_JLOI 1
#define CUANTIC_SUPI 2
#define CUANTIC_LUPI 2
#define CUANTIC_JUPI 0


#define NOISE_SIGMA 0.004 

#define CLASSICAL_ESTIMATES_SAMPLE_REF 4 //Muestra referencia para cambio de cuadrante de azimuth. Depende del numero de muestras y posicion Continuo


#define NTERMS 9  //ojo si es mayor q 10 casca el svdCordic (esta version)

#define	PRECISION double //double or float

// END USER CONFIGURATION
//---------------------------------------------------------
//---------------------------------------------------------
//---------------------------------------------------------
//---------------------------------------------------------

// DONT'T MODIFY ANYTHING BELOW OF THIS LINE

#define PI 	3.14159265358979323846264338327950288419716939937510
 		 	
#define ILAMBDA 10
#define TOPLIM 0.000000000001
#define SLIGHT 0
#define NOISE 0.001

#define RR  0.5641895836

#define VLIGHT 2.99792458e+5 //;light speed (km/s); 

#define CTE4_6_13 4.6686411e-13
#define AH 1.0 //angulo heliocentrico

#define FFT_FORWARD -1 
#define FFT_BACKWARD +1

#define NPARMS 4 //(IQUV)

#define LONG_PUNTERO_CALCULOS_COMPARTIDOS 100  //no se usa...

#define INSTRUMENTAL_CONVOLUTION_INTERPOLACION 0  //realizar interpolacion en la convolucion ?? //No funciona !

//INIT_MODEL=[eta0,magnet,vlos,landadopp,aa,gamma,azi,B1,B2,macro,alfa]
struct INIT_MODEL{
	double eta0; // 0
	double B;//magnetic field    
	double vlos;
	double dopp;
	double aa;
	double gm; //5
	double az;
	double S0;
	double S1;
	double mac; //9
	double alfa;		
};

struct CUANTIC{  
	
	double N_PI;
	double N_SIG;
	double *NUB;//tam n_sig
	double *NUP;//tam n_pi
	double *NUR;//tam n_sig
	double *WEB;//tam n_sig
	double *WEP;//tam n_pi
	double *WER;//tam n_sig
	double GL;
	double GU;
	double GEFF;
	double FO;	
	
};

typedef struct INIT_MODEL Init_Model;
typedef struct CUANTIC Cuantic;


/******************************************************/

void Inicializar_Puntero_Calculos_Compartidos();
void Resetear_Puntero_Calculos_Compartidos();
void Leer_Puntero_Calculos_Compartidos(int Numero,PRECISION ** a,...);
void Asignar_Puntero_Calculos_Compartidos(int Numero,PRECISION * a,...);
void Borrar_Calculos_Spectra();

void ReservarMemoriaSinteisisDerivadas(int numl);
void LiberarMemoriaSinteisisDerivadas();

/******************************************************/


Cuantic * create_cuantic(double * dat);

int me_der(Cuantic *cuantic,Init_Model *initModel,double * wlines,int nwlines,double *lambda,int nlambda,
			PRECISION *d_spectraOut,double ah,double slight,int triplete,int filter);

int mil_sinrf(Cuantic *cuantic,Init_Model *initModel,double * wlines,int nwlines,double *lambda,int nlambda,PRECISION *spectra,
			double ah,double slight,int triplete,int filter);

double * fgauss(double MC, double * eje,int neje,double landa,int deriv);

_Complex * fft_d(double * spectra, int nspectra,int direc);
_Complex * fft_c(_Complex * spectra, int nspectra,int direc);


int Guarda(char * nombre,PRECISION *v,int nv);
int GuardaC(char * nombre,_Complex *v,int nv,int a);

int fvoigt(PRECISION damp,PRECISION *vv,int nvv,PRECISION *h, PRECISION *f);

void direct_convolution(PRECISION * x, int nx,PRECISION * h, int nh,PRECISION delta);
PRECISION * vgauss(PRECISION fwhm,int nmuestras_G,PRECISION delta);


#endif /*DEFINES_H_*/
