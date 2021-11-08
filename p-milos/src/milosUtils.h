#include "defines.h"
#include <complex.h>
#include <fftw3.h> //siempre a continuacion de complex.h
#include <math.h>



/**
 * 
 */
void AplicaDelta(Init_Model *model, PRECISION *delta, int *fixed, Init_Model *modelout);
/**
 * 
 */
int check(Init_Model *Model);
/**
 * 
 */
void FijaACeroDerivadasNoNecesarias(REAL *d_spectra, int *fixed, int nlambda);

/**
 * 
 */
int mil_svd(PRECISION *h, REAL *beta, PRECISION *delta);



/*
*
*
* Cálculo de las estimaciones clásicas.
*
*
* lambda_0 :  centro de la línea
* lambda :    vector de muestras
* nlambda :   numero de muesras
* spectro :   vector [I,Q,U,V]
* initModel:  Modelo de atmosfera a ser modificado
*
*/
void estimacionesClasicas(PRECISION lambda_0, PRECISION *lambda, int nlambda, float *spectro, Init_Model *initModel, int forInitialUse);



/*
 *
 * nwlineas :   numero de lineas espectrales
 * wlines :		lineas spectrales
 * lambda :		wavelength axis in angstrom
			longitud nlambda
 * spectra : IQUV por filas, longitud ny=nlambda
 */

int lm_mils(Cuantic *cuantic, PRECISION *wlines, PRECISION *lambda, int nlambda, float *spectro, int nspectro,
				Init_Model *initModel, REAL *spectra, float *chisqrf,
				REAL * slight, PRECISION toplim, int miter, REAL *weight, int *fix,
				REAL *vSigma, REAL sigma, REAL ilambda, int * INSTRUMENTAL_CONVOLUTION, int * iter,REAL ah, int logclambda);

/**
 * Make the interpolation between deltaLambda and PSF where deltaLambda es x and PSF f(x)
 *  Return the array with the interpolation. 
 * */
int interpolationSplinePSF(PRECISION *deltaLambda, PRECISION * PSF, PRECISION * lambdasSamples, size_t N_PSF, PRECISION * fInterpolated, size_t NSamples);


/**
 * Make the interpolation between deltaLambda and PSF where deltaLambda es x and PSF f(x)
 *  Return the array with the interpolation. 
 * */
int interpolationLinearPSF(PRECISION *deltaLambda, PRECISION * PSF, PRECISION * lambdasSamples, size_t N_PSF, PRECISION * fInterpolated, size_t NSamples, double offset);
