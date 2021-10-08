#include "defines.h"

/**
 * 
 * @param double * x signal to convolve
 * @param int nx size of signal to convolve 
 * @param double * h  kernel convolution 
 * @param int nh  size of kernel convolution 
 * 
 * Make convolution using direct method, the result is stored in @param * x
*/
void direct_convolution_double(PRECISION *x, int nx, PRECISION *h, int nh);
/**
 * 
 * @param float * x signal to convolve
 * @param int nx size of signal to convolve 
 * @param double * h  kernel convolution 
 * @param int nh  size of kernel convolution 
 * 
 * Make convolution using direct method, the result is stored in @param * x
*/
void direct_convolution(REAL *x, int nx, PRECISION *h, int nh);
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
void direct_convolution_ic(REAL *x, int nx, PRECISION *h, int nh, REAL Ic);

/**
 * @param float * x signal to convolve 
 * @param double * h kernel of convolution
 * @param int size size of signal and kernel 
 * @param float * result array to store result of convolution 
 * 
 * Method to do circular convolution over signal 'x'. We assume signal 'x' and 'h' has the same size. 
 * The result is stored in array 'result'
 * */
void convCircular(REAL *x, double *h, int size, REAL *result);
