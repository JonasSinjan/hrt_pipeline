#include <complex.h>
#include <fftw3.h> //siempre a continuacion de complex.h
#include <math.h>
#include <stdio.h>

//#include "leePerfil.c"

#define FFT_FORWARD -1 
#define FFT_BACKWARD +1

_Complex * fft_d(double * spectra, int nspectra,int direc);
_Complex * fft_c(_Complex * spectra, int nspectra,int direc);

/*
int main(){

	int i;
   
//	double vv[]={1,1,1,1,1,1,1,1,1,0};
	_Complex * out,*out1;

	double *lambda;
	int nlambda;
	double *spectro;
	FILE *fichero;
	int n;
	double l;

	printf("FFTW ...\n");


	fichero= fopen("g1.txt","r");
	if(fichero==NULL){
		printf("Error de apertura, es posible que el fichero no exista.\n");
		return 0;
	}

	n=0;
	lambda=calloc(150,sizeof(double));
	while (fscanf(fichero,"%le",&l)!= EOF ){
		lambda[n]=l;
		n++;
	}
	fclose(fichero);
	
	out= fft_d(lambda,150,FFT_FORWARD);


//	printf("Orig ...\n");
//	for(i=0;i<25;i++){
//		printf("orig[%d]=(%le , %f) \n",i,lambda[i+64],0);
//	}


	printf("FORWARD ...%d\n",n);
	for(i=0;i<150;i++){
		printf("out[%d]=(%f , %le) \n",i,creal(out[i]),cimag(out[i]));
	}


//	printf("BACKWARD ...\n");
//	out1= fft_c(out, 10,FFT_BACKWARD);


//	for(i=0;i<10;i++){
//		printf("out1[%d]=(%f , %f) \n",i,creal(out1[i]),cimag(out1[i]));
//	}


	free(out);
//	free(out1);

	return 0;

}*/

/*
	
	direc : FFT_FORWARD (-1) or FFT_BACKWARD (+1)
*/


_Complex * fft_d(double * spectra, int nspectra,int direc){

	fftw_complex *in,*out;
	int i;

    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nspectra);

	for(i=0;i<nspectra;i++){
		in[i]=spectra[i] + 0 * _Complex_I;
	}

	out=fft_c(in,nspectra,direc);

    fftw_free(in);

	return (_Complex *)out;
}


_Complex * fft_c(_Complex * spectra, int nspectra,int direc){

	fftw_complex *out;
	fftw_plan p;
	int i;

    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nspectra);


    p = fftw_plan_dft_1d(nspectra, spectra, out, direc, FFTW_ESTIMATE);

    fftw_execute(p); /* repeat as needed */

	if(direc==FFT_FORWARD){
		for(i=0;i<nspectra;i++){
			out[i]=out[i]/nspectra;
		}
	}

    fftw_destroy_plan(p);

	return (_Complex *)out;
}









