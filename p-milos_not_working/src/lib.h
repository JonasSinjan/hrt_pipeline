#include "defines.h"

/**
 * 
 */

int covarm(REAL *w,REAL *sig,float *spectro,int nspectro,REAL *spectra,REAL  *d_spectra,REAL *beta,REAL *alpha);
int covarm2(REAL *w,REAL *sig,float *spectro,int nspectro,REAL *spectra,REAL  *d_spectra,REAL *beta,REAL *alpha);

/**
 * 
 */
int multmatrix(PRECISION *a,int naf,int nac, PRECISION *b,int nbf,int nbc,PRECISION *result,int *fil,int *col);

/**
 * 
 */

REAL fchisqr(REAL * spectra,int nspectro,float *spectro,REAL *w,REAL *sig,REAL nfree);


/**
 * 
 */
int multmatrixIDLValue(REAL *a,int naf,int nac,REAL *b,int nbf,int nbc,REAL *result,int *fil,int *col,REAL value);




int multmatrixIDLValueSigma(REAL *a,int naf,int nac,REAL *b,int nbf,int nbc,REAL *result,int *fil,int *col, REAL * sigma);
/**
 * 
 */
void totalParcialMatrixf(REAL * A, int f,int c,int p,REAL *result);
/**
 * 
 */
void totalParcialf(REAL * A, int f,int c,REAL * result);
/**
 * 
 */
int multmatrix_transpose(REAL *a,int naf,int nac, REAL *b,int nbf,int nbc,REAL *result,int *fil,int *col,REAL value);

int multmatrix_transpose_sigma(REAL *a,int naf,int nac, REAL *b,int nbf,int nbc,REAL *result,int *fil,int *col,REAL weigth, REAL * sigma);
/**
 * 
 */
int CalculaNfree(int nspectro);


/*
* Check if path is a directory or not. 
*/
int isDirectory(const char *path);

void insert_in_linked_list (tpuntero *cabeza, char * fileName);
int checkNameInLista (tpuntero cabeza,char * fileName);
void deleteList (tpuntero *cabeza);
