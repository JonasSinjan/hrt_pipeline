#include <math.h>
#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include "convolution.h"
#include "defines.h"


extern PRECISION *dirConvPar;
extern REAL *resultConv;

/**
 * 
 * @param double * x signal to convolve
 * @param int nx size of signal to convolve 
 * @param double * h  kernel convolution 
 * @param int nh  size of kernel convolution 
 * 
 * Make convolution using direct method, the result is stored in @param * x
*/
void direct_convolution_double(PRECISION *x, int nx, PRECISION *h, int nh)
{
	int k, j;
	int mitad_nh = nh / 2;

	// fill auxiliar array
	for (k = 0; k < nx; k++)
	{
		dirConvPar[k + mitad_nh] = x[k];
	}

	// take only central convolution
	for (k = 0; k < nx; k++)
	{
		double aux = 0;
		for (j = 0; j < nh; j++)
		{
			aux += h[j] * dirConvPar[j + k];
		}
		x[k] = aux;
	}
}


/**
 * 
 * @param float * x signal to convolve
 * @param int nx size of signal to convolve 
 * @param double * h  kernel convolution 
 * @param int nh  size of kernel convolution 
 * 
 * Make convolution using direct method, the result is stored in @param * x
*/
void direct_convolution(REAL *x, int nx, PRECISION *h, int nh)
{

	int k, j;
	int mitad_nh = nh / 2;

	// fill auxiliar array 

	for (k = 0; k < nx; k++)
	{
		dirConvPar[k + mitad_nh] = x[k];
	}

	// take only central convolution
	double aux;
	for (k = 0; k < nx; k++)
	{
		aux = 0;
		for (j = 0; j < nh; j++)
		{
			aux += h[j] * dirConvPar[j + k];
		}
		x[k] = aux;
	}
}

/**
 * 
 * @param float * x signal to convolve
 * @param int nx size of signal to convolve 
 * @param double * h  kernel convolution 
 * @param int nh  size of kernel convolution 
 *	@param float ic value of intensity continuous
 * Make convolution using direct method, the result is stored in @param * x
 * 
*/
void direct_convolution_ic(REAL *x, int nx, PRECISION *h, int nh, REAL Ic)
{

	int k, j;

	int mitad_nh = nh / 2;

	for (k = 0; k < nx; k++)
	{
		dirConvPar[k + mitad_nh] = Ic - x[k];
	}

	// take only central convolution
	double aux;
	for (k = 0; k < nx; k++)
	{
		aux = 0;
		for (j = 0; j < nh; j++)
		{
			aux += h[j] * dirConvPar[j + k];
		}
		x[k] = Ic - aux;
	}
}


/**
 * @param float * x signal to convolve 
 * @param double * h kernel of convolution
 * @param int size size of signal and kernel 
 * @param float * result array to store result of convolution 
 * 
 * Method to do circular convolution over signal 'x'. We assume signal 'x' and 'h' has the same size. 
 * The result is stored in array 'result'
 * */

void convCircular(REAL *x, double *h, int size, REAL *result)
{
	int i,j,ishift,mod;
	double aux;

	int odd=(size%2);		
	int startShift = size/2;
	if(odd) startShift+=1;	
	ishift = startShift;

	for(i=0; i < size ; i++){
		aux = 0;
    	for(j=0; j < size; j++){
			mod = i-j;
			if(mod<0)
				mod = size+mod;
			aux += h[j] * x[mod];
    	}
		if(i < size/2)
			resultConv[ishift++] = aux;
		else
			resultConv[i-(size/2)] = aux;
	}
	for(i=0;i<size;i++){
		result[i] = resultConv[i];
	}

}

