#include "defines.h"
#include <string.h>
#include "lib.h"
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <sys/types.h>
#include <sys/stat.h>





extern REAL * opa;
extern int NTERMS;

/**
 * @param float * w 
 * @param float * sig
 * @param float * spectro 
 * @param int nspectro 
 * @param float * spectra
 * @param float * d_spectra 
 * @param float * beta 
 * @param float * alpha 
 * 
 * */
int covarm(REAL *w,REAL *sig,float *spectro,int nspectro,REAL *spectra,REAL  *d_spectra,REAL *beta,REAL *alpha){	
	
	int j,i,bt_nf,bt_nc,aux_nf,aux_nc;

	REAL AP[NTERMS*NTERMS*NPARMS],BT[NPARMS*NTERMS];
	
	REAL *BTaux,*APaux;

	//printf("\nVALORES DEL SIGMA SQUARE\n");

	for(j=0;j<NPARMS;j++){
		for(i=0;i<nspectro;i++){
			opa[i]= w[j]*(spectra[i+nspectro*j]-spectro[i+nspectro*j]);
		}

		BTaux=BT+(j*NTERMS);
		APaux=AP+(j*NTERMS*NTERMS);
		multmatrixIDLValueSigma(opa,nspectro,1,d_spectra+j*nspectro*NTERMS,NTERMS,nspectro,BTaux,&bt_nf,&bt_nc,sig+(nspectro*j)); //bt de tam NTERMS x 1
		multmatrix_transpose_sigma(d_spectra+j*nspectro*NTERMS,NTERMS,nspectro,d_spectra+j*nspectro*NTERMS,NTERMS,nspectro,APaux,&aux_nf,&aux_nc,w[j], sig+(nspectro*j));//ap de tam NTERMS x NTERMS
		
	}

	totalParcialf(BT,NPARMS,NTERMS,beta); //beta de tam 1 x NTERMS
	totalParcialMatrixf(AP,NTERMS,NTERMS,NPARMS,alpha); //alpha de tam NTERMS x NTERMS
	
	return 1;
}

/**
 * @param REAL * W 
 * @param REAL * sig
 * @param float * spectro 
 * @param int nspectro 
 * @param REAL * spectra 
 * @param REAL * d_spectra 
 * @param REAL * beta 
 * @param REAL * alpha 
 * 
 * */
int covarm2(REAL *w,REAL *sig,float *spectro,int nspectro,REAL *spectra,REAL  *d_spectra,REAL *beta,REAL *alpha){	
	
	int j,i,h,k,aux_nf,aux_nc;

	REAL AP[NTERMS*NTERMS*NPARMS],BT[NPARMS*NTERMS];
	
	REAL *BTaux,*APaux;
	REAL sum,sum2;

	for(j=0;j<NPARMS;j++){
		for(i=0;i<nspectro;i++){
			opa[i]= w[j]*(spectra[i+nspectro*j]-spectro[i+nspectro*j]);
		}

		BTaux=BT+(j*NTERMS);
		APaux=AP+(j*NTERMS*NTERMS);
		
		for ( i = 0; i < NTERMS; i++){
		    for ( h = 0; h < NTERMS; h++){
				sum=0;
				if(i==0){
					sum2=0;
				}
				for ( k = 0;  k < nspectro; k++){
					REAL dAux = (*(d_spectra+j*nspectro*NTERMS+h*nspectro+k));
					sum += (*(d_spectra+j*nspectro*NTERMS+i*nspectro+k) * dAux ) * (w[j]/sig[nspectro*j+k]);
					
					if(i==0){
						sum2+= ((opa[k] * dAux  ))/sig[nspectro*j+k];
					}
				}

				APaux[NTERMS*i+h] = sum;
				if(i==0){
					BTaux[h] = sum2;
				}				
     		} 
		
		}		
	}

	totalParcialf(BT,NPARMS,NTERMS,beta); //beta de tam 1 x NTERMS
	totalParcialMatrixf(AP,NTERMS,NTERMS,NPARMS,alpha); //alpha de tam NTERMS x NTERMS
	
	return 1;
}

/**
 * @param spectra: array with synthetic spectro 
 * @param nspectro: size of spectro
 * @param spectro: original spectro
 * @param w: array of weight for I,Q,U,V 
 * @param sig: array with sigma for I,Q,U,V 
 * @param nfree: (nspectro * NPARMS) - NTERMS, NPARAMs is 4 and NTERMS 11. 
 * */
REAL fchisqr(REAL * spectra,int nspectro,float *spectro,REAL *w,REAL *sig,REAL nfree){
	
	REAL TOT,dif1,dif2,dif3,dif4;	
	REAL opa1,opa2,opa3,opa4;
	int i,j;

	TOT=0;
	opa1=0;
	opa2=0;
	opa3=0;
	opa4=0;
	for(i=0;i<nspectro;i++){
		dif1=spectra[i]-spectro[i];
		dif2=spectra[i+nspectro]-spectro[i+nspectro];
		dif3=spectra[i+nspectro*2]-spectro[i+nspectro*2];
		dif4=spectra[i+nspectro*3]-spectro[i+nspectro*3];

		opa1+= (((dif1*dif1)*w[0])/(sig[i]));
		opa2+= (((dif2*dif2)*w[1])/(sig[i+nspectro]));
		opa3+= (((dif3*dif3)*w[2])/(sig[i+nspectro*2]));
		opa4+= (((dif4*dif4)*w[3])/(sig[i+nspectro*3]));
	}
	TOT+= opa1+opa2+opa3+opa4;

	return TOT/nfree;
	
}


/*

	Multiplica la matriz a (tamaño naf,nac)
	por la matriz b (de tamaño nbf,nbc)
	al estilo IDL, es decir, filas de a por columnas de b,
	el resultado se almacena en resultOut (de tamaño fil,col)

	El tamaño de salida (fil,col) corresponde con (nbf,nac).

	El tamaño de columnas de b, nbc, debe de ser igual al de filas de a, naf.

*/

/**
 * @param REAL * a 
 * @param int naf
 * @param int nac 
 * @param REAL * b
 * @param int nbf
 * @param int nbc 
 * @param REAL * result 
 * @param int * fil 
 * @param int * col 
 * @param REAL value 
 * */
int multmatrixIDLValue(REAL *a,int naf,int nac,REAL *b,int nbf,int nbc,REAL *result,int *fil,int *col,REAL value){
    
   int i,j,k;
   REAL sum;
	
	if(naf==nbc){
		(*fil)=nbf;
		(*col)=nac;

		for ( i = 0; i < nbf; i++){
		    for ( j = 0; j < nac; j++){
				sum=0;
				for ( k = 0;  k < naf; k++){
					sum += a[k*nac+j] * b[i*nbc+k];
				}
				result[((nac)*i)+j] = sum/value;
      		} 
		}
		return 1;
	}
	else
		printf("\n \n Error en multmatrixIDLValue no coinciden nac y nbf!!!! ..\n\n");
	return 0;
}

/**
 * 
 * */
int multmatrixIDLValueSigma(REAL *a,int naf,int nac,REAL *b,int nbf,int nbc,REAL *result,int *fil,int *col, REAL * sigma){
    
   int i,j,k;
   REAL sum;
	
	if(naf==nbc){
		(*fil)=nbf;
		(*col)=nac;

		for ( i = 0; i < nbf; i++){
				sum=0;
				for ( k = 0;  k < naf; k++){
						sum += (((a[k] * b[i*nbc+k])))/sigma[k];
				}
				result[i] = sum; 
		}
		return 1;
	}
	else
		printf("\n \n Error en multmatrixIDLValue no coinciden nac y nbf!!!! ..\n\n");
	return 0;
}

/**
 * @param REAL * A
 * @param int f 
 * @param int c
 * @param REAL * result
 * */
void totalParcialf(REAL * A, int f,int c,REAL * result){

	int i,j;
	REAL sum;
	for(i=0;i<c;i++){
		sum = 0;
		for(j=0;j<f;j++){
			sum+=A[j*c+i];
		}
		result[i] = sum;
	}
}

/**
 * @param REAL * A
 * @param int f
 * @param int c
 * @param int p
 * @param REAL * result 
 * */
void totalParcialMatrixf(REAL * A, int f,int c,int p,REAL *result){

	int i,j,k;
	REAL sum;
	for(i=0;i<f;i++)
		for(j=0;j<c;j++){
			sum=0;
			for(k=0;k<p;k++)
				sum+=A[i*c+j+f*c*k];
			result[i*c+j] = sum;
		}
}


/**
 * @param PRECISION * a 
 * @param int naf 
 * @param int nac 
 * @param PRECISION * b 
 * @param int nbf 
 * @param int nbc 
 * @param PRECISION * result 
 * @param int * fil 
 * @param int * col 
 * 
 * Multiply matrix "a"(naf,nac) with matrix "b" (nbf,nbc). Algebraic matrix multiplication style, that is, 
 * columns of "a" by rows of "b". The result is stored in "result"(fil,col). 
 * The size of columns of "a", nac, must be the same of rows of "b", nbf. 
 * */
int multmatrix(PRECISION *a,int naf,int nac, PRECISION *b,int nbf,int nbc,PRECISION *result,int *fil,int *col){
    
    int i,j,k;
    PRECISION sum;
    
	if(nac==nbf){
		(*fil)=naf;
		(*col)=nbc;

		for ( i = 0; i < naf; i++)
		    for ( j = 0; j < nbc; j++){
				sum=0;
				for ( k = 0;  k < nbf; k++){
					sum += a[i*nac+k] * b[k*nbc+j];
				}
				result[(*col)*i+j] = sum;
      		} 
		return 1;
	}
	return 0;

}



/**
 * @param REAL * a
 * @param int naf 
 * @param int nac 
 * @param REAL *b
 * @param int nbf
 * @param int nbc 
 * @param REAL * result 
 * @param int * fil 
 * @param int * col 
 * @param REAL value
 * */
int multmatrix_transpose(REAL *a,int naf,int nac, REAL *b,int nbf,int nbc,REAL *result,int *fil,int *col,REAL value){
    
    int i,j,k;
    REAL sum;
    
	if(nac==nbc){
		(*fil)=naf;
		(*col)=nbf;
		
		for ( i = 0; i < naf; i++){
		    for ( j = 0; j < nbf; j++){
				sum=0;
				for ( k = 0;  k < nbc; k++){
					sum += a[i*nac+k] * b[j*nbc+k];
				}

				result[(*col)*i+j] = (sum)*value;
     		} 
		
		}
		return 1;
	}else{
		printf("\n \n Error en multmatrix_transpose no coinciden nac y nbc!!!! ..\n\n");
	}

	return 0;
}

/**
 * @param REAL * a
 * @param int naf 
 * @param int nac 
 * @param REAL *b
 * @param int nbf
 * @param int nbc 
 * @param REAL * result 
 * @param int * fil 
 * @param int * col 
 * @param REAL weigth
 * @param REAL * sigma 
 * 
 * */
int multmatrix_transpose_sigma(REAL *a,int naf,int nac, REAL *b,int nbf,int nbc,REAL *result,int *fil,int *col,REAL weigth, REAL * sigma){
    
    int i,j,k;
    REAL sum;
    
	if(nac==nbc){
		(*fil)=naf;
		(*col)=nbf;
		
		for ( i = 0; i < naf; i++){
		    for ( j = 0; j < nbf; j++){
				sum=0;
				for ( k = 0;  k < nbc; k++){
						sum += (a[i*nac+k] * b[j*nbc+k]) * (weigth/sigma[k]);
				}

				result[(*col)*i+j] = sum;
     		} 
		
		}
		return 1;
	}else{
		printf("\n \n Error en multmatrix_transpose no coinciden nac y nbc!!!! ..\n\n");
	}

	return 0;
}
//Media de un vector de longitud numl



/**
 * @param int nspectro
 */
int CalculaNfree(int nspectro)
{
	int nfree;
	nfree = 0;

	nfree = (nspectro * NPARMS) - NTERMS;

	return nfree;
}

/**
 * @param const char * path
 * */
int isDirectory(const char *path) {
   struct stat statbuf;
   if (stat(path, &statbuf) != 0)
       return 0;
   return S_ISDIR(statbuf.st_mode);
}

/**
 * @param tpuntero * cabeza
 * @param char * fileName
 * */
void insert_in_linked_list (tpuntero *cabeza, char * fileName){
    tpuntero nuevo; 
    nuevo = malloc(sizeof(tnodo)); 
    strcpy(nuevo->d_name,fileName); 
    nuevo->next = *cabeza; 
    *cabeza = nuevo; 
}
 
/**
 * @param tpuntero  cabeza
 * @param char * fileName
 * */
int checkNameInLista(tpuntero cabeza,char * fileName){
	int found = 0;
    while(cabeza != NULL && !found){ //Mientras cabeza no sea NULL
		if(strcmp(cabeza->d_name,fileName)==0)
			found = 1;
		else
        	cabeza = cabeza->next; //Pasamos al siguiente nodo
    }
	return found;
}
 

/**
 * @param tpuntero * cabeza
 * */

void deleteList(tpuntero *cabeza){ 
    tpuntero actual; //Puntero auxiliar para eliminar correctamente la lista
  
    while(*cabeza != NULL){ //Mientras cabeza no sea NULL
        actual = *cabeza; //Actual toma el valor de cabeza
        *cabeza = (*cabeza)->next; //Cabeza avanza 1 posicion en la lista
        free(actual); //Se libera la memoria de la posicion de Actual (el primer nodo), y cabeza queda apuntando al que ahora es el primero
    }
}