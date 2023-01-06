
#include "defines.h"

double * leeVector(char *nombre,int tam);
int multmatrixIDL(double *a,int naf,int nac, double *b,int nbf,int nbc,double **resultOut,int *fil,int *col);
int multmatrix2(double *a,int naf,int nac, PRECISION *b,int nbf,int nbc,double **result,int *fil,int *col);
double * transpose(double *mat,int fil,int col);
double *totalParcial(double * A, int f,int c,int dire);
double *totalParcialMatrix(double * A, int f,int c,int p);
double total(double * A, int f,int c);
int multmatrix(PRECISION *a,int naf,int nac, PRECISION *b,int nbf,int nbc,PRECISION *result,int *fil,int *col);
double fchisqr(PRECISION * spectra,int nspectro,PRECISION *spectro,PRECISION *w,PRECISION *sig,double nfree);

PRECISION * transposef(PRECISION *mat,int fil,int col);

int multmatrixIDLf(PRECISION *a,int naf,int nac,PRECISION *b,int nbf,int nbc,PRECISION **resultOut,int *fil,int *col);
int multmatrixIDLValue(PRECISION *a,int naf,int nac,PRECISION *b,int nbf,int nbc,PRECISION *result,int *fil,int *col,PRECISION value);

void totalParcialMatrixf(PRECISION * A, int f,int c,int p,PRECISION *result);
void totalParcialf(PRECISION * A, int f,int c,int dire,PRECISION * result);
int multmatrix_transpose(PRECISION *a,int naf,int nac, PRECISION *b,int nbf,int nbc,PRECISION *result,int *fil,int *col,PRECISION value);
int multmatrix_transposeD(double *a,int naf,int nac, double *b,int nbf,int nbc,double *result,int *fil,int *col);
int multmatrix3(PRECISION *a,int naf,int nac,double *b,int nbf,int nbc,double **result,int *fil,int *col);
/*

 el tamaño de w es 	nlambda*NPARMS;

return 
	- beta de tam 1 x NTERMS
	- alpha de tam NTERMS x NTERMS

*/

int covarm(PRECISION *w,PRECISION *sig,int nsig,PRECISION *spectro,int nspectro,PRECISION *spectra,PRECISION  *d_spectra,
		PRECISION *beta,PRECISION *alpha){

	PRECISION *bt,*ap,*result,*trans;
	//static PRECISION opa[NLAMBDA];
	PRECISION *opa;
	int j,i,k,bt_nf,bt_nc,aux_nf,aux_nc;

	static PRECISION AP[NTERMS*NTERMS*NPARMS],BT[NPARMS*NTERMS];

	PRECISION *BTaux,*APaux;

	opa = calloc(nspectro,sizeof(PRECISION));
	
	for(j=0;j<NPARMS;j++){
		for(i=0;i<nspectro;i++){
			opa[i]=w[j]*(spectra[i+nspectro*j]-spectro[i+nspectro*j]);
		}

		BTaux=BT+(j*NTERMS);
		multmatrixIDLValue(opa,nspectro,1,d_spectra+j*nspectro*NTERMS,NTERMS,nspectro,BTaux,&bt_nf,&bt_nc,sig[j]*sig[j]); //bt de tam NTERMS x 1
	
		//
		APaux=AP+(j*NTERMS*NTERMS);
		multmatrix_transpose(d_spectra+j*nspectro*NTERMS,NTERMS,nspectro,d_spectra+j*nspectro*NTERMS,NTERMS,nspectro,
													APaux,&aux_nf,&aux_nc,w[j]/(sig[j]*sig[j]));//ap de tam NTERMS x NTERMS
	}

	free(opa);
	
	totalParcialf(BT,NPARMS,NTERMS,2,beta); //beta de tam 1 x NTERMS
	totalParcialMatrixf(AP,NTERMS,NTERMS,NPARMS,alpha); //alpha de tam NTERMS x NTERMS
	
	return 1;
}



double fchisqr(PRECISION * spectra,int nspectro,PRECISION *spectro,PRECISION *w,PRECISION *sig,double nfree){
		double TOT,dif;	
	double opa;
	int i,j;

	TOT=0;
	for(j=0;j<NPARMS;j++){
		opa=0;
		for(i=0;i<nspectro;i++){
			dif=spectra[i+nspectro*j]-spectro[i+nspectro*j];
			opa+= (dif*dif);
		}
		TOT+=((w[j]*opa)/(sig[j]*sig[j]));
		//TOT+= opa;///(sig[j]*sig[j]);
	}
		
	//return TOT/15;		
	return TOT/nfree;
	
}



double * leeVector(char *nombre,int tam)
{
	FILE *fichero;

	double *v,a;
	int n;

	//lectura
	fichero= fopen(nombre,"r");
	if(fichero==NULL){
		printf("leevector : Error de apertura, es posible que el fichero no exista.\n");
		return 0;
	}

	v=calloc(tam,sizeof(double));		

	n=0;
	while (fscanf(fichero,"%lf",&a)!= EOF ){
		v[n]=a;
		n++;			
	}
	fclose(fichero);

	return v;

}

/*

	Multiplica la matriz a (tamaño naf,nac)
	por la matriz b (de tamaño nbf,nbc)
	al estilo IDL, es decir, filas de a por columnas de b,
	el resultado se almacena en resultOut (de tamaño fil,col)

	El tamaño de salida (fil,col) corresponde con (nbf,nac).

	El tamaño de columnas de b, nbc, debe de ser igual al de filas de a, naf.

*/
int multmatrixIDL(double *a,int naf,int nac, double *b,int nbf,int nbc,double **resultOut,int *fil,int *col){
    
     int i,j,k;
    double sum;
	double *result;
	
	if(naf==nbc){
		(*fil)=nbf;
		(*col)=nac;
		
//		free(*result);
		result=calloc((nbf)*(nac),sizeof(double));
//		printf("a ver ..\n");

		for ( i = 0; i < nbf; i++){
		    for ( j = 0; j < nac; j++){
				sum=0;
				for ( k = 0;  k < naf; k++){
//					printf("i: %d,j:%d,k=%d .. a[%d][%d]:%f  .. b[%d][%d]:%f\n",i,j,k,k,j,a[k*nac+j],i,k,b[i*nbc+k]);
					sum += a[k*nac+j] * b[i*nbc+k];
				}
//				printf("Sum, result[%d][%d] : %f \n",i,j,sum);
				result[((nac)*i)+j] = sum;
      		} 
		}
		*resultOut=result;
		return 1;
	}
	else
		printf("\n \n Error en multmatrixIDL no coinciden nac y nbf!!!! ..\n\n");
	return 0;
}


int multmatrixIDLf(PRECISION *a,int naf,int nac,PRECISION *b,int nbf,int nbc,PRECISION **resultOut,int *fil,int *col){
    
     int i,j,k;
    PRECISION sum;
	PRECISION *result;
	
	if(naf==nbc){
		(*fil)=nbf;
		(*col)=nac;
		
//		free(*result);
		result=calloc((nbf)*(nac),sizeof(PRECISION));
//		printf("a ver ..\n");

		for ( i = 0; i < nbf; i++){
		    for ( j = 0; j < nac; j++){
				sum=0;
				for ( k = 0;  k < naf; k++){
//					printf("i: %d,j:%d,k=%d .. a[%d][%d]:%f  .. b[%d][%d]:%f\n",i,j,k,k,j,a[k*nac+j],i,k,b[i*nbc+k]);
					sum += a[k*nac+j] * b[i*nbc+k];
				}
//				printf("Sum, result[%d][%d] : %f \n",i,j,sum);
				result[((nac)*i)+j] = sum;
      		} 
		}
		*resultOut=result;
		return 1;
	}
	else
		printf("\n \n Error en multmatrixIDL no coinciden nac y nbf!!!! ..\n\n");
	return 0;
}

int multmatrixIDLValue(PRECISION *a,int naf,int nac,PRECISION *b,int nbf,int nbc,PRECISION *result,int *fil,int *col,PRECISION value){
    
     int i,j,k;
    PRECISION sum;
	
	if(naf==nbc){
		(*fil)=nbf;
		(*col)=nac;
		
//		free(*result);
//		result=calloc((nbf)*(nac),sizeof(PRECISION));
//		printf("a ver ..\n");

		for ( i = 0; i < nbf; i++){
		    for ( j = 0; j < nac; j++){
				sum=0;
				for ( k = 0;  k < naf; k++){
//					printf("i: %d,j:%d,k=%d .. a[%d][%d]:%f  .. b[%d][%d]:%f\n",i,j,k,k,j,a[k*nac+j],i,k,b[i*nbc+k]);
					sum += a[k*nac+j] * b[i*nbc+k];
				}
//				printf("Sum, result[%d][%d] : %f \n",i,j,sum);
				result[((nac)*i)+j] = sum/value;
      		} 
		}
		return 1;
	}
	else
		printf("\n \n Error en multmatrixIDL no coinciden nac y nbf!!!! ..\n\n");
	return 0;
}

double * transpose(double *mat,int fil,int col){
	
	int i,j;
	double *result;
	
	result=calloc(fil*col,sizeof(double));
	
	for(i=0;i<fil;i++){
		for(j=0;j<col;j++){
			result[j*fil+i]=mat[i*col+j];
		}
	}
	
	return result;
}

PRECISION * transposef(PRECISION *mat,int fil,int col){
	
	int i,j;
	PRECISION *result;
	
	result=calloc(fil*col,sizeof(PRECISION));
	
	for(i=0;i<fil;i++){
		for(j=0;j<col;j++){
			result[j*fil+i]=mat[i*col+j];
		}
	}
	
	return result;
}


/*
	dire:
		1: suma por filas, return double * de tam f
		2: suma por columnas, return double * de tam c
*/

double *totalParcial(double * A, int f,int c,int dire){

	int i,j;
//	double 	sum;
	double *result;	
	result=calloc(dire==1?f:c,sizeof(double));

	for(i=0;i<f;i++)
		for(j=0;j<c;j++){
			result[(dire==1)?i:j]+=A[i*c+j];
		}

	return result;
}

void totalParcialf(PRECISION * A, int f,int c,int dire,PRECISION * result){

	int i,j;
//	double 	sum;

//	result=calloc(dire==1?f:c,sizeof(double));
	int d;
	for(i=0;i<c;i++){
		result[i]=0;
		for(j=0;j<f;j++){
			result[i]+=A[j*c+i];
		}
	}
}


/*
return matriz de tam f*c
*/

double *totalParcialMatrix(double * A, int f,int c,int p){

	int i,j,k;
//	double 	sum;
	double *result;	
	result=calloc(f*c,sizeof(double));

	for(i=0;i<f;i++)
		for(j=0;j<c;j++){
			for(k=0;k<p;k++)
				result[i*c+j]+=A[i*c+j+f*c*k];
		}

	return result;
}

void totalParcialMatrixf(PRECISION * A, int f,int c,int p,PRECISION *result){

	int i,j,k;
//	double 	sum;
//	double *result;	
//	result=calloc(f*c,sizeof(double));

	for(i=0;i<f;i++)
		for(j=0;j<c;j++){
			result[i*c+j]=0;
			for(k=0;k<p;k++)
				result[i*c+j]+=A[i*c+j+f*c*k];
		}

//	return result;
}


double total(double * A, int f,int c){

	int i,j;
	double 	sum;
	sum=0;
	for(i=0;i<f;i++)
		for(j=0;j<c;j++)
			sum+=A[i*c+j];

	return sum;
}



/*
	Multiplica la matriz a (tamaño naf,nac)
	por la matriz b (de tamaño nbf,nbc)
	al estilo multiplicación algebraica de matrices, es decir, columnas de a por filas de b,
	el resultado se almacena en resultOut (de tamaño fil,col)

	El tamaño de salida (fil,col) corresponde con (nbf,nac).

	El tamaño de columnas de a, nac, debe de ser igual al de filas de b, nbf.
*/

int multmatrix(PRECISION *a,int naf,int nac, PRECISION *b,int nbf,int nbc,PRECISION *result,int *fil,int *col){
    
    int i,j,k;
    PRECISION sum;
    
	if(nac==nbf){
		(*fil)=naf;
		(*col)=nbc;
		
//		free(*result);
//		*result=calloc((*fil)*(*col),sizeof(double));
//		printf("a ver ..\n");

		for ( i = 0; i < naf; i++)
		    for ( j = 0; j < nbc; j++){
				sum=0;
				for ( k = 0;  k < nbf; k++){
//					printf("i: %d,j:%d,k=%d .. a[%d][%d]  .. b[%d][%d]\n",i,j,k,i,k,k,j);
					sum += a[i*nac+k] * b[k*nbc+j];
				}
//				printf("Sum\n");
				result[(*col)*i+j] = sum;

      		} 

		return 1;
	}
	return 0;

}

int multmatrix2(double *a,int naf,int nac, PRECISION *b,int nbf,int nbc,double **result,int *fil,int *col){
    
    int i,j,k;
    double sum;
    
	if(nac==nbf){
		(*fil)=naf;
		(*col)=nbc;
		
//		free(*result);
		*result=calloc((*fil)*(*col),sizeof(double));
//		printf("a ver ..\n");

		for ( i = 0; i < naf; i++)
		    for ( j = 0; j < nbc; j++){
				sum=0;
				for ( k = 0;  k < nbf; k++){
//					printf("i: %d,j:%d,k=%d .. a[%d][%d]  .. b[%d][%d]\n",i,j,k,i,k,k,j);
					sum += a[i*nac+k] * b[k*nbc+j];
				}
//				printf("Sum\n");
				(*result)[(*col)*i+j] = sum;

      		} 

		return 1;
	}
	return 0;
}

int multmatrix3(PRECISION *a,int naf,int nac,double *b,int nbf,int nbc,double **result,int *fil,int *col){
    
    int i,j,k;
    double sum;
    
	if(nac==nbf){
		(*fil)=naf;
		(*col)=nbc;
		
//		free(*result);
		*result=calloc((*fil)*(*col),sizeof(double));
//		printf("a ver ..\n");

		for ( i = 0; i < naf; i++)
		    for ( j = 0; j < nbc; j++){
				sum=0;
				for ( k = 0;  k < nbf; k++){
//					printf("i: %d,j:%d,k=%d .. a[%d][%d]  .. b[%d][%d]\n",i,j,k,i,k,k,j);
					sum += a[i*nac+k] * b[k*nbc+j];
				}
//				printf("Sum\n");
				(*result)[(*col)*i+j] = sum;

      		} 

		return 1;
	}
	return 0;
}



int multmatrix_transpose(PRECISION *a,int naf,int nac, PRECISION *b,int nbf,int nbc,PRECISION *result,int *fil,int *col,PRECISION value){
    
    int i,j,k;
    PRECISION sum;
    
	if(nac==nbc){
		(*fil)=naf;
		(*col)=nbf;
		
		for ( i = 0; i < naf; i++)
		    for ( j = 0; j < nbf; j++){
				sum=0;
				for ( k = 0;  k < nbc; k++){
					sum += a[i*nac+k] * b[j*nbc+k];
				}

				result[(*col)*i+j] = (sum)*value;
      		} 

		return 1;
	}else{
		printf("\n \n Error en multmatrix_transpose no coinciden nac y nbc!!!! ..\n\n");
	}

	return 0;
}

int multmatrix_transposeD(double *a,int naf,int nac, double *b,int nbf,int nbc,double *result,int *fil,int *col){
    
    int i,j,k;
    double sum;
    
	if(nac==nbc){
		(*fil)=naf;
		(*col)=nbf;
		
		for ( i = 0; i < naf; i++)
		    for ( j = 0; j < nbf; j++){
				sum=0;
				for ( k = 0;  k < nbc; k++){
					sum += a[i*nac+k] * b[j*nbc+k];
				}

				result[(*col)*i+j] = (sum);
      		} 

		return 1;
	}else{
		printf("\n \n Error en multmatrix_transposeD no coinciden nac y nbc!!!! ..\n\n");
	}

	return 0;
}



//Media de un vector de longitud numl
double mean(double *dat, int numl){
	
	double auxsum;
	int i;

	auxsum=0;
	for(i=0;i<numl;i++){
		auxsum=auxsum+dat[i];		
	}

	return auxsum/numl;
}



/*
 * Cambiamos la forma para tener en cada fila I Q U V
 * Tambien reajusta el tamaño para eliminar las posiciones vacias
 */

void reformarVector(PRECISION **spectro,int neje){
	
	PRECISION *aux;
	int i;
	aux=(PRECISION *)calloc(neje*4,sizeof(PRECISION));

	for(i=0;i<neje;i++){
		aux[i]=(*spectro)[i*4];		
		aux[neje+i]=(*spectro)[i*4+1];
		aux[2*neje+i]=(*spectro)[i*4+2];		
		aux[3*neje+i]=(*spectro)[i*4+3];		
	}
	
	free(*spectro);
	*spectro=aux;
}
  
