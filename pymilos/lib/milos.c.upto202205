
//    _______             _______ _________ _        _______  _______
//   (  ____ \           (       )\__   __/( \      (  ___  )(  ____ \
//   | (    \/           | () () |   ) (   | (      | (   ) || (    \/
//   | |         _____   | || || |   | |   | |      | |   | || (_____
//   | |        (_____)  | |(_)| |   | |   | |      | |   | |(_____  )
//   | |                 | |   | |   | |   | |      | |   | |      ) |
//   | (____/\           | )   ( |___) (___| (____/\| (___) |/\____) |
//   (_______/           |/     \|\_______/(_______/(_______)\_______)
//
//
// CMILOS v0.9 (2015)
// RTE INVERSION C code for SOPHI (based on the ILD code MILOS by D. Orozco)
// juanp (IAA-CSIC)
//
// How to use:
//
//  >> milos NLAMBDA MAX_ITER CLASSICAL_ESTIMATES RFS [FWHM DELTA NPOINTS] profiles_file.txt > output.txt
//
//   NLAMBDA number of lambda of input profiles
//   MAX_ITER of inversion
//   CLASSICAL_ESTIMATES use classical estimates? 1 yes, 0 no, 2 only CE
//   RFS : 0 RTE, 1 Spectral Synthesis, 2 Spectral Synthesis + Response Funcions
//   [FWHM DELTA NPOINTS] use convolution with a gaussian? if the tree parameteres are defined yes, else no. Units in A. NPOINTS has to be odd.
//   profiles_file.txt name of input profiles file
//   output.txt name of output file
//
//

#include <time.h>
#include "defines.h"

#include "nrutil.h"
#include "svdcmp.c"
#include "svdcordic.c"
//#include "tridiagonal.c"
#include "convolution.c"

#include <string.h>

float pythag(float a, float b);

void weights_init(int nlambda,double *sigma,PRECISION *weight,int nweight,PRECISION **wOut,PRECISION **sigOut,double noise);

int check(Init_Model *Model);
int lm_mils(Cuantic *cuantic,double * wlines,int nwlines,double *lambda,int nlambda,PRECISION *spectro,int nspectro,
		Init_Model *initModel, PRECISION *spectra,int err,double *chisqrf, int *iterOut,
		double slight, double toplim, int miter, PRECISION * weight,int nweight, int * fix,
		PRECISION *sigma, double filter, double ilambda, double noise, double *pol,
		double getshi,int triplete);

int mil_svd(PRECISION *h,PRECISION *beta,PRECISION *delta);

int multmatrixIDL(double *a,int naf,int nac, double *b,int nbf,int nbc,double **resultOut,int *fil,int *col);
int multmatrix_transposeD(double *a,int naf,int nac, double *b,int nbf,int nbc,double *result,int *fil,int *col);
int multmatrix3(PRECISION *a,int naf,int nac,double *b,int nbf,int nbc,double **result,int *fil,int *col);
double * leeVector(char *nombre,int tam);
double * transpose(double *mat,int fil,int col);

double total(double * A, int f,int c);
int multmatrix(PRECISION *a,int naf,int nac, PRECISION *b,int nbf,int nbc,PRECISION *result,int *fil,int *col);
int multmatrix2(double *a,int naf,int nac, PRECISION *b,int nbf,int nbc,double **result,int *fil,int *col);

int covarm(PRECISION *w,PRECISION *sig,int nsig,PRECISION *spectro,int nspectro,PRECISION *spectra,PRECISION  *d_spectra,
		PRECISION *beta,PRECISION *alpha);

int CalculaNfree(PRECISION *spectro,int nspectro);

double fchisqr(PRECISION * spectra,int nspectro,PRECISION *spectro,PRECISION *w,PRECISION *sig,double nfree);

void AplicaDelta(Init_Model *model,PRECISION * delta,int * fixed,Init_Model *modelout);
void FijaACeroDerivadasNoNecesarias(PRECISION * d_spectra,int *fixed,int nlambda);
void reformarVector(PRECISION **spectro,int neje);
void spectral_synthesis_convolution();
void response_functions_convolution();

void estimacionesClasicas(PRECISION lambda_0,double *lambda,int nlambda,PRECISION *spectro,Init_Model *initModel);

#define tiempo(ciclos) asm volatile ("rdtsc \n\t":"=A"(ciclos))
long long int c1,c2,cd,semi,c1a,c2a,cda;			//variables de 64 bits para leer ciclos de reloj
long long int c1total,c2total,cdtotal,ctritotal;


Cuantic* cuantic;   // Variable global, está hecho así, de momento,para parecerse al original
char * concatena(char *a, int n,char*b);

PRECISION ** PUNTEROS_CALCULOS_COMPARTIDOS;
int POSW_PUNTERO_CALCULOS_COMPARTIDOS;
int POSR_PUNTERO_CALCULOS_COMPARTIDOS;

PRECISION * gp1,*gp2,*dt,*dti,*gp3,*gp4,*gp5,*gp6,*etai_2;

//PRECISION gp4_gp2_rhoq[NLAMBDA],gp5_gp2_rhou[NLAMBDA],gp6_gp2_rhov[NLAMBDA];
PRECISION *gp4_gp2_rhoq,*gp5_gp2_rhou,*gp6_gp2_rhov;


PRECISION *dgp1,*dgp2,*dgp3,*dgp4,*dgp5,*dgp6,*d_dt;
PRECISION * d_ei,*d_eq,*d_eu,*d_ev,*d_rq,*d_ru,*d_rv;
PRECISION *dfi,*dshi;
PRECISION CC,CC_2,sin_gm,azi_2,sinis,cosis,cosis_2,cosi,sina,cosa,sinda,cosda,sindi,cosdi,sinis_cosa,sinis_sina;
PRECISION *fi_p,*fi_b,*fi_r,*shi_p,*shi_b,*shi_r;
PRECISION *etain,*etaqn,*etaun,*etavn,*rhoqn,*rhoun,*rhovn;
PRECISION *etai,*etaq,*etau,*etav,*rhoq,*rhou,*rhov;
PRECISION *parcial1,*parcial2,*parcial3;
PRECISION *nubB,*nupB,*nurB;
PRECISION **uuGlobalInicial;
PRECISION **HGlobalInicial;
PRECISION **FGlobalInicial;
PRECISION *perfil_instrumental;
PRECISION * G;
int FGlobal,HGlobal,uuGlobal;

PRECISION *d_spectra,*spectra;

//Number of lambdas in the input profiles
int NLAMBDA = 0;

//Convolutions values
int NMUESTRAS_G	= 0;
PRECISION FWHM = 0;
PRECISION DELTA = 0;

int INSTRUMENTAL_CONVOLUTION = 0;
int INSTRUMENTAL_CONVOLUTION_WITH_PSF = 0;
int CLASSICAL_ESTIMATES = 0;
int RFS = 0;

// PSF obtenida desde los datos teoricos de CRISP CRISP_6173_28mA.psf
// Se usa el scrip interpolar_psf.m
const PRECISION crisp_psf[141] = {0.0004,0.0004,0.0005,0.0005,0.0005,0.0005,0.0006,0.0006,0.0006,0.0007,0.0007,0.0008,0.0008,0.0009,0.0009,0.0010,0.0010,0.0011,0.0012,0.0012,0.0013,0.0014,
							0.0015,0.0016,0.0017,0.0019,0.0020,0.0021,0.0023,0.0025,0.0027,0.0029,0.0031,0.0034,0.0037,0.0040,0.0044,0.0048,0.0052,0.0057,0.0062,0.0069,0.0076,0.0083,
							0.0092,0.0102,0.0114,0.0127,0.0142,0.0159,0.0178,0.0202,0.0229,0.0261,0.0299,0.0343,0.0398,0.0465,0.0546,0.0645,0.0765,0.0918,0.1113,0.1363,0.1678,0.2066,
							0.2501,0.2932,0.3306,0.3569,0.3669,0.3569,0.3306,0.2932,0.2501,0.2066,0.1678,0.1363,0.1113,0.0918,0.0765,0.0645,0.0546,0.0465,0.0398,0.0343,0.0299,0.0261,
							0.0229,0.0202,0.0178,0.0159,0.0142,0.0127,0.0114,0.0102,0.0092,0.0083,0.0076,0.0069,0.0062,0.0057,0.0052,0.0048,0.0044,0.0040,0.0037,0.0034,0.0031,0.0029,
							0.0027,0.0025,0.0023,0.0021,0.0020,0.0019,0.0017,0.0016,0.0015,0.0014,0.0013,0.0012,0.0012,0.0011,0.0010,0.0010,0.0009,0.0009,0.0008,0.0008,0.0007,0.0007,
							0.0006,0.0006,0.0006,0.0005,0.0005,0.0005,0.0005,0.0004,0.0004};

int main(int argc,char **argv){

	double * wlines;
	int nwlines;
	double *lambda;
	int nlambda;
	PRECISION *spectro;
	int ny,i,j;
	Init_Model initModel;
	int err;
	double chisqrf;
	int iter;
	double slight;
	double toplim;
	int miter;
	PRECISION weight[4]={1.,10.,10.,4.};
	int nweight;

	clock_t t_ini, t_fin;
	double secs, total_secs;

	double *chisqrf_array;
	chisqrf_array =(double*) calloc(883*894*4,sizeof(double));

	// CONFIGURACION DE PARAMETROS A INVERTIR
	//INIT_MODEL=[eta0,magnet,vlos,landadopp,aa,gamma,azi,B1,B2,macro,alfa]
	int fix[]={1.,1.,1.,1.,1.,1.,1.,1.,1.,0.,0.};  //Parametros invertidos
	//----------------------------------------------

	double sigma[NPARMS];
	double vsig;
	double filter;
	double ilambda;
	double noise;
	double *pol;
	double getshi;

	//for debugging
	double online_mean=0.0,online_variance=0.0;

	double dat[7]={CUANTIC_NWL,CUANTIC_SLOI,CUANTIC_LLOI,CUANTIC_JLOI,CUANTIC_SUPI,CUANTIC_LUPI,CUANTIC_JUPI};


	char *nombre,*input_iter;
	int Max_iter;

	//milos NLAMBDA MAX_ITER CLASSICAL_ESTIMATES RFS [FWHM DELTA NPOINTS] perfil.txt

	if(argc!=6 && argc != 7 && argc !=9){
		printf("milos: Error en el numero de parametros: %d .\n Pruebe: milos NLAMBDA MAX_ITER CLASSICAL_ESTIMATES RFS [FWHM(in A) DELTA(in A) NPOINTS] perfil.txt\n",argc);
		printf("O bien: milos NLAMBDA MAX_ITER CLASSICAL_ESTIMATES RFS [DELTA(in A)] perfil.txt\n  --> for using stored CRISP's PSF\n");
		printf("Note : CLASSICAL_ESTIMATES=> 0: Disabled, 1: Enabled, 2: Only Classical Estimates.\n");
		printf("RFS : 0: Disabled     1: Synthesis      2: Synthesis and Response Functions\n");
		printf("Note when RFS>0: perfil.txt is considered as models.txt. \n");

		return -1;
	}

	NLAMBDA = atoi(argv[1]);

	input_iter = argv[2];
	Max_iter = atoi(input_iter);
	CLASSICAL_ESTIMATES = atoi(argv[3]);
	RFS = atoi(argv[4]);

	if(CLASSICAL_ESTIMATES!=0 && CLASSICAL_ESTIMATES != 1 && CLASSICAL_ESTIMATES != 2){
		printf("milos: Error in CLASSICAL_ESTIMATES parameter. [0,1,2] are valid values. Not accepted: %d\n",CLASSICAL_ESTIMATES);
		return -1;
	}

	if(RFS != 0 && RFS != 1 && RFS != 2){
		printf("milos: Error in RFS parameter. [0,1,2] are valid values. Not accepted: %d\n",RFS);
		return -1;
	}

	if(argc ==6){ //if no filter declared
		nombre = argv[5]; //input file name
	}
	else{
		INSTRUMENTAL_CONVOLUTION = 1;
		if(argc ==7){
			INSTRUMENTAL_CONVOLUTION_WITH_PSF = 1;
			DELTA = atof(argv[5]);
			nombre = argv[6];
			FWHM = 0.035;
		}
		else{
			INSTRUMENTAL_CONVOLUTION_WITH_PSF = 0;
			FWHM = atof(argv[5]);
			DELTA = atof(argv[6]);
			NMUESTRAS_G = atoi(argv[7]);
			nombre = argv[8];
		}
	}


	nlambda=NLAMBDA;
	//Generamos la gaussiana -> perfil instrumental



	if(INSTRUMENTAL_CONVOLUTION){
		G=vgauss(FWHM,NMUESTRAS_G,DELTA);


		if(INSTRUMENTAL_CONVOLUTION_WITH_PSF){
			//if you wish to convolution with other instrumental profile you have to declare here and to asign it to "G"
			free(G);
			NMUESTRAS_G = 9;

			G=vgauss(FWHM,NMUESTRAS_G,DELTA); //solo para reservar memoria

			int kk;
			PRECISION sum =0;
			for(kk=0;kk<NMUESTRAS_G;kk++){
				int pos = 70 - (int)((DELTA/0.005)*(((int)(NMUESTRAS_G/2))-kk));

				if(pos<0 || pos > 140) //140 length de crisp_psf
					G[kk]=0;
				else
					G[kk] = crisp_psf[pos];  //70 es el centro de psf

				sum += G[kk];
			}

			for(kk=0;kk<NMUESTRAS_G;kk++)
				G[kk] /= sum;

		}

		/*
		int kk=0;
		printf("start G  %d \n",NMUESTRAS_G);
		for(kk=0;kk<NMUESTRAS_G;kk++){
			printf(" %f ,",G[kk]);
			if(kk==10)
				printf("\n");

		}

		printf("\nend G\n");


		exit(-1);
		*/


	}


	cuantic=create_cuantic(dat);
	Inicializar_Puntero_Calculos_Compartidos();

	toplim=1e-18;

	CC=PI/180.0;
	CC_2=CC*2;

	filter=0;
	getshi=0;
	nweight=4;

	nwlines=1;
	wlines=(double*) calloc(2,sizeof(double));
	wlines[0]=1;
	wlines[1]= CENTRAL_WL;

	vsig=NOISE_SIGMA; //original 0.001
	sigma[0]=vsig;
	sigma[1]=vsig;
	sigma[2]=vsig;
	sigma[3]=vsig;
	pol=NULL;

	noise=NOISE_SIGMA;
	ilambda=ILAMBDA;
	iter=0;
	miter=Max_iter;


	lambda=calloc(nlambda,sizeof(double));
	spectro=calloc(nlambda*4,sizeof(PRECISION));

	FILE *fichero;

	fichero= fopen(nombre,"r");
	if(fichero==NULL){
		printf("Error de apertura, es posible que el fichero no exista.\n");
		printf("Milos: Error de lectura del fichero. ++++++++++++++++++\n");
		return 1;
	}

	char * buf;
	buf=calloc(strlen(nombre)+15+19,sizeof(char));
	buf = strcat(buf,nombre);
	buf = strcat(buf,"_CL_ESTIMATES");

	int neje;
	double lin;
	double iin,qin,uin,vin;
	int rfscanf;
	int contador;

	int totalIter=0;

	contador=0;

	ReservarMemoriaSinteisisDerivadas(nlambda);

	//initializing weights
	PRECISION *w,*sig;
	weights_init(nlambda,sigma,weight,nweight,&w,&sig,noise);

	c2total=0;
	ctritotal=0;

	int nsub,indaux;
	indaux=0;

	//Probamos a usar los valores en single floating point que se generan en la FPGA
	// lambda[0] = 6.1732100e+003;
	// lambda[1] = 6.1732798e+003;
	// lambda[2] = 6.1733501e+003;
	// lambda[3] = 6.1734199e+003;
	// lambda[4] = 6.1734897e+003;
	// lambda[5] = 6.1737700e+003;


		// lambda[0] = 6.1732100e+003;
	// lambda[1] = 6.1732798e+003;
	// lambda[2] = 6.1733501e+003;
	// lambda[3] = 6.1734199e+003;
	// lambda[4] = 6.1734897e+003;
	// lambda[5] = 6.1737700e+003;




	if(!RFS){ // SI RFS ==0
		do{
			neje=0;
			nsub=0;
			 // // printf(" \n \n ");
			 while (neje<NLAMBDA && (rfscanf=fscanf(fichero,"%lf %lf %lf %lf %lf",&lin,&iin,&qin,&uin,&vin))!= EOF){ //para probar rte_validation_tests
			// while (neje<NLAMBDA && (rfscanf=fscanf(fichero,"%lf %le %le %le %le",&lin,&iin,&qin,&uin,&vin))!= EOF){

				lambda[nsub]=lin;	 //
				//lambda[NLAMBDA-1]=6173.341000 - 0.420;


				 // printf(" %f ",lambda[nsub]);
				spectro[nsub]=iin;
				spectro[nsub+NLAMBDA]=qin;
				spectro[nsub+NLAMBDA*2]=uin;
				spectro[nsub+NLAMBDA*3]=vin;
				nsub++;

				neje++;

				//printf("%.10e, %.10e, %.10e, %.10e \n",iin,qin,uin,vin);
			}
			 // exit(-1);
			if(rfscanf!=EOF ){  //   && contador==8


				//Initial Model

				initModel.eta0 = INITIAL_MODEL_ETHA0;
				initModel.B = INITIAL_MODEL_B; //200 700
				initModel.gm = INITIAL_MODEL_GM;
				initModel.az = INITIAL_MODEL_AZI;
				initModel.vlos = INITIAL_MODEL_VLOS; //km/s 0
				initModel.mac = 0.0;
				initModel.dopp = INITIAL_MODEL_LAMBDADOPP;
				initModel.aa = INITIAL_MODEL_AA;
				initModel.alfa = 1;							//0.38; //stray light factor
				initModel.S0 = INITIAL_MODEL_S0;
				initModel.S1 = INITIAL_MODEL_S1;


				if(CLASSICAL_ESTIMATES && !RFS){

					t_ini = clock();
					estimacionesClasicas(wlines[1],lambda,nlambda,spectro,&initModel);
					t_fin = clock();


					//Se comprueba si el resultado fue "nan" en las CE
					if(isnan(initModel.B))
						initModel.B = 1;
					if(isnan(initModel.vlos))
						initModel.vlos = 1e-3;
					if(isnan(initModel.gm))
						initModel.gm=1;
					if(isnan(initModel.az))
						initModel.az = 1;


					//Se inicializa S0 y S1 según el valor del continuo

					// initModel.S0 = lambda[nlambda-1] * 0.35;
					// initModel.S1 = lambda[nlambda-1] * 0.65;

					// Test C
					// if(spectro[nlambda-1]< 0.75){
						// initModel.S0 = 0.1;
						// initModel.S1 = 0.3;
					// }

					//Test B
					//Test Dic 2015
					// initModel.S0 = (lambda[nlambda-1]>0.25?(lambda[nlambda-1] * 0.35):0.1); //David 0I.3 y Ic*0.7
					// initModel.S1 = (lambda[nlambda-1]>0.25?(lambda[nlambda-1] * 0.65):0.25);

				}

				//inversion
				if(CLASSICAL_ESTIMATES!=2 ){

					//Se introduce en S0 el valor de Blos si solo se calculan estimaciones clásicas
					//Aqui se anula esa asignación porque se va a realizar la inversion RTE completa
					initModel.S0 = INITIAL_MODEL_S0;

					lm_mils(cuantic,wlines,nwlines,lambda, nlambda,spectro,nlambda,&initModel,spectra,err,&chisqrf,&iter,slight,toplim,miter,
						weight,nweight,fix,sig,filter,ilambda,noise,pol,getshi,0);
				}


				secs = (double)(t_fin - t_ini) / CLOCKS_PER_SEC;
				//printf("\n\n%.16g milisegundos\n", secs * 1000.0);

				total_secs += secs;

				totalIter+=iter;



				// [contador;iter;B;GM;AZI;etha0;lambdadopp;aa;vlos;S0;S1;final_chisqr];
				printf("%d\n",contador);
				printf("%d\n",iter);
				printf("%f\n",initModel.B);
				printf("%f\n",initModel.gm);
				printf("%f\n",initModel.az);
				printf("%f \n",initModel.eta0);
				printf("%f\n",initModel.dopp);
				printf("%f\n",initModel.aa);
				printf("%f\n",initModel.vlos); //km/s
				//printf("alfa \t:%f\n",initModel.alfa); //stay light factor
				printf("%f\n",initModel.S0);
				printf("%f\n",initModel.S1);
				printf("%.10e\n",chisqrf);

				contador++;

				//---------------------------------------------------
				//For debugging:online_mean,online_variance;
				// chisqrf_array[contador]=chisqrf;

				// double delta, delta2;
				// delta = chisqrf - online_mean;
				// online_mean += delta/contador;
				// delta2 = chisqrf - online_mean;
				// online_variance += delta * delta2;
				//---------------------------------------------------

				/*
				//For debugging: escribe tambien los perfiles
				int kk;
				for(kk=0;kk<NLAMBDA;kk++){
					printf("%lf %le %le %le %le \n",lambda[kk],spectra[kk],spectra[kk + NLAMBDA],spectra[kk + NLAMBDA*2],spectra[kk + NLAMBDA*3]);
				}
				*/

				//exit(-1);
			}

		}while(rfscanf!=EOF ); //&& contador<10000
	}
	else{   //when RFS is activated

		//Para leer el mustreo en el top del fichero de modelos
		// neje=0;
		// while (neje<NLAMBDA && (rfscanf=fscanf(fichero,"%lf",&lin))!= EOF){
			// lambda[neje]=lin;
			// // printf(" %f \n",lambda[neje]);
			// neje++;
		// }

		//Sino se lee el muestreo del top se pone uno a mano
		// lambda[0] =6.17320300e+003;
		// lambda[1] =6.17327200e+003;
		// lambda[2] =6.17334100e+003;
		// lambda[3] =6.17341000e+003;
		// lambda[4] =6.17347900e+003;
		// lambda[5] =6.17374100e+003;
		lambda[0] =6.17320100e+003;
		lambda[1] =6.1732710e+003;
		lambda[2] =6.17334130e+003;
		lambda[3] =6.17341110e+003;
		lambda[4] =6.17348100e+003;
		lambda[5] =6.17376100e+003;

		do{
			int contador,iter;
			double chisqr;
			int NMODEL=12; //Numero de parametros del modelo


			//num,iter,B,GM,AZ,ETA0,dopp,aa,vlos,S0,S1,chisqr,
			if((rfscanf=fscanf(fichero,"%d",&contador))!= EOF){
				//rfscanf=fscanf(fichero,"%d",&contador);
				rfscanf=fscanf(fichero,"%d",&iter);
				rfscanf=fscanf(fichero,"%lf",&initModel.B);
				rfscanf=fscanf(fichero,"%lf",&initModel.gm);
				rfscanf=fscanf(fichero,"%lf",&initModel.az);
				rfscanf=fscanf(fichero,"%lf",&initModel.eta0);
				rfscanf=fscanf(fichero,"%lf",&initModel.dopp);
				rfscanf=fscanf(fichero,"%lf",&initModel.aa);
				rfscanf=fscanf(fichero,"%lf",&initModel.vlos);
				rfscanf=fscanf(fichero,"%lf",&initModel.S0);
				rfscanf=fscanf(fichero,"%lf",&initModel.S1);
				rfscanf=fscanf(fichero,"%le",&chisqr);

				/* for debugging: escribe tambien los modelos
				//
				printf("%d\n",contador);
				printf("%d\n",iter);
				printf("%f\n",initModel.B);
				printf("%f\n",initModel.gm);
				printf("%f\n",initModel.az);
				printf("%f \n",initModel.eta0);
				printf("%f\n",initModel.dopp);
				printf("%f\n",initModel.aa);
				printf("%f\n",initModel.vlos); //km/s
				printf("%f\n",initModel.S0);
				printf("%f\n",initModel.S1);
				printf("%.10e\n",chisqr);
				// exit(-1);
				*/



				mil_sinrf(cuantic,&initModel,wlines,nwlines,lambda,nlambda,spectra,AH,slight,0,filter);

				// //Debug --------------------------------------
				// printf(" Acceso spectra \n");
				// int kka;
				// for(kka=0;kka<24;kka++)
					// printf(" withn MIL --------------- %d .... %e \n",kka,spectra[kka]);

				// printf(" Acceso G \n");
				// for(kka=0;kka<NMUESTRAS_G;kka++)
					// printf(" G - %d .... %e \n",kka,G[kka]);
				// //Debug --------------------------------------

				spectral_synthesis_convolution();

				// //Debug --------------------------------------
				// printf(" Acceso spectra \n");
				// for(kka=0;kka<24;kka++)
					// printf(" withn MIL --------------- %d .... %e \n",kka,spectra[kka]);

				// exit(-1);
				// //--------------------------------------

				me_der(cuantic,&initModel,wlines,nwlines,lambda,nlambda,d_spectra,AH,slight,0,filter);
				response_functions_convolution();

				int kk;
				for(kk=0;kk<NLAMBDA;kk++){
					printf("%lf %le %le %le %le \n",lambda[kk],spectra[kk],spectra[kk + NLAMBDA],spectra[kk + NLAMBDA*2],spectra[kk + NLAMBDA*3]);
				}

				if(RFS==2){
					int number_parametros = 0;
					for(number_parametros=0;number_parametros<NTERMS;number_parametros++){
						for(kk=0;kk<NLAMBDA;kk++){
							printf("%lf %le %le %le %le \n",lambda[kk],
												d_spectra[kk + NLAMBDA * number_parametros],
												d_spectra[kk + NLAMBDA * number_parametros + NLAMBDA * NTERMS],
												d_spectra[kk + NLAMBDA * number_parametros + NLAMBDA * NTERMS*2],
												d_spectra[kk + NLAMBDA * number_parametros + NLAMBDA * NTERMS*3]);
						}
					}
				}

			// if(contador==6){
				// exit(-1);
			// }

			}

		}while(rfscanf!=EOF );
	}
	// exit(-1);

	//------------------------------------------------------------------------------------------
	//For debugging ------------------------
	//contador -= 1; //Al terminar contador era N = length(X);

	// online_variance = online_variance / (contador -1);

	// printf("\n Number of points : %d",contador);
	// printf("\n Iterative Mean: %.10e   ||||| Iterative Variance %.10e \n",online_mean,online_variance);

	// double mean_traditional,sum_trad,variance_trad;
	// int kk;
	// for(kk=0;kk<contador-1;kk++)
		// sum_trad +=chisqrf_array[kk];

	// mean_traditional = sum_trad / (contador);

	// // int kk;
	// sum_trad = 0;
	// for(kk=0;kk<contador-1;kk++){
		// double par = chisqrf_array[kk] - mean_traditional;
		// sum_trad += par*par;
	// }
	// variance_trad = sum_trad / (contador-1);

	// printf("\n Traditional Mean: %.10e   ||||| Traditional Variance %.10e \n",mean_traditional,variance_trad);
	//------------------------------------------------------------------------------------------


	fclose(fichero);


	//printf("\n\n TOTAL sec : %.16g segundos\n", total_secs);

	free(spectro);
	free(lambda);
	free(cuantic);
	free(wlines);
	free(chisqrf_array);

	LiberarMemoriaSinteisisDerivadas();
	Liberar_Puntero_Calculos_Compartidos();

	free(G);

	return 0;
}




/*
 *
 * nwlineas :   numero de lineas espectrales
 * wlines :		lineas spectrales
 * lambda :		wavelength axis in angstrom
			longitud nlambda
 * spectra : IQUV por filas, longitud ny=nlambda
 */

int lm_mils(Cuantic *cuantic,double * wlines,int nwlines,double *lambda,int nlambda,PRECISION *spectro,int nspectro,
		Init_Model *initModel, PRECISION *spectra,int err,double *chisqrf, int *iterOut,
		double slight, double toplim, int miter, PRECISION * weight,int nweight, int * fix,
		PRECISION *sigma, double filter, double ilambda, double noise, double *pol,
		double getshi,int triplete)
{

	int * diag;
	int	iter;
	int i,j,In,*fixed,nfree;
	static PRECISION delta[NTERMS];
	double max[3],aux;
	int repite,pillado,nw,nsig;
	double *landa_store,flambda;
	static PRECISION beta[NTERMS],alpha[NTERMS*NTERMS];
	double chisqr,ochisqr;
	int nspectra,nd_spectra,clanda,ind;
	Init_Model model;

	//Genera aleatoriamente los componentes del vector
	tiempo(semi);					//semilla para  generar los valores de la lista de forma aleatoria con srand
	srand((char)semi);

	iter=0;


	//nterms= 11; //numero de elementomodel->gms de initmodel
	nfree=CalculaNfree(spectro,nspectro);
	//printf("\n nfree! %d:\n",nfree);
	//exit(-1);


	if(nfree==0){
		return -1; //'NOT ENOUGH POINTS'
	}

	flambda=ilambda;

	if(fix==NULL){
		fixed=calloc(NTERMS,sizeof(double));
		for(i=0;i<NTERMS;i++){
			fixed[i]=1;
		}
	}
	else{
		fixed=fix;
	}

	clanda=0;
	iter=0;
//	landa_store=calloc(miter+1,sizeof(double));
	repite=1;
	pillado=0;

	static PRECISION covar[NTERMS*NTERMS];
	static PRECISION betad[NTERMS];

	PRECISION chisqr_mem;
	int repite_chisqr=0;


	/**************************************************************************/
	mil_sinrf(cuantic,initModel,wlines,nwlines,lambda,nlambda,spectra,AH,slight,triplete,filter);


/*	printf(" Acceso spectra \n");
	int kk;
	for(kk=0;kk<24;kk++)
		printf(" withn MIL --------------- %d .... %e \n",kk,spectra[kk]);

	printf(" Acceso G \n");
	for(kk=0;kk<NMUESTRAS_G;kk++)
		printf(" G - %d .... %e \n",kk,G[kk]);
	*/

	//convolucionamos los perfiles IQUV (spectra)
	spectral_synthesis_convolution();

	/*printf(" Acceso spectra \n");
	for(kk=0;kk<24;kk++)
		printf(" withn MIL --------------- %d .... %e \n",kk,spectra[kk]);
	*/
	me_der(cuantic,initModel,wlines,nwlines,lambda,nlambda,d_spectra,AH,slight,triplete,filter);

	/*
	printf(" Acceso d_spectra \n");

	for(kk=0;kk<24;kk++)
		printf(" withn MIL --------------- %d .... %e \n",kk,d_spectra[kk]);
	printf(" return \n");
	*/
	//convolucionamos las funciones respuesta ( d_spectra )
	response_functions_convolution();

	//FijaACeroDerivadasNoNecesarias(d_spectra,fixed,nlambda);

	covarm(weight,sigma,nsig,spectro,nlambda,spectra,d_spectra,beta,alpha);

	for(i=0;i<NTERMS;i++)
		betad[i]=beta[i];

	for(i=0;i<NTERMS*NTERMS;i++)
		covar[i]=alpha[i];

	/**************************************************************************/

	ochisqr=fchisqr(spectra,nspectro,spectro,weight,sigma,nfree);


	model=*initModel;
	do{
		chisqr_mem=(PRECISION)ochisqr;

		/**************************************************************************/
		for(i=0;i<NTERMS;i++){
			ind=i*(NTERMS+1);
			covar[ind]=alpha[ind]*(1.0+flambda);
		}


		mil_svd(covar,betad,delta);

		AplicaDelta(initModel,delta,fixed,&model);

		check(&model);

		/**************************************************************************/

		mil_sinrf(cuantic,&model,wlines,nwlines,lambda,nlambda,spectra,AH,slight,triplete,filter);


		//convolucionamos los perfiles IQUV (spectra)
		spectral_synthesis_convolution();


		chisqr=fchisqr(spectra,nspectro,spectro,weight,sigma,nfree);

		/**************************************************************************/
		if(chisqr-ochisqr < 0){

			flambda=flambda/10.0;

			*initModel=model;


			// printf("iteration=%d , chisqr = %f CONVERGE	- lambda= %e \n",iter,chisqr,flambda);


			me_der(cuantic,initModel,wlines,nwlines,lambda,nlambda,d_spectra,AH,slight,triplete,filter);

			//convolucionamos las funciones respuesta ( d_spectra )
			response_functions_convolution();

			//FijaACeroDerivadasNoNecesarias(d_spectra,fixed,nlambda);

			covarm(weight,sigma,nsig,spectro,nlambda,spectra,d_spectra,beta,alpha);
			for(i=0;i<NTERMS;i++)
				betad[i]=beta[i];

			for(i=0;i<NTERMS*NTERMS;i++)
				covar[i]=alpha[i];

			ochisqr=chisqr;
		}
		else{
			flambda=flambda*10;//10;

			// printf("iteration=%d , chisqr = %f NOT CONVERGE	- lambda= %e \n",iter,ochisqr,flambda);

		}

		/**************************************************************************/
		iter++;

/*
		printf("\n-----------------------\n");
		printf("%d\n",iter);
		printf("%f\n",initModel->B);
		printf("%f\n",initModel->gm);
		printf("%f\n",initModel->az);
		printf("%f \n",initModel->eta0);
		printf("%f\n",initModel->dopp);
		printf("%f\n",initModel->aa);
		printf("%f\n",initModel->vlos); //km/s
		//printf("alfa \t:%f\n",initModel.alfa); //stay light factor
		printf("%f\n",initModel->S0);
		printf("%f\n",initModel->S1);
		printf("%.10e\n",ochisqr);
*/

	}while(iter<=miter); // && !clanda);

	*iterOut=iter;

	*chisqrf=ochisqr;

	//For debugging -> flambda en lugar de AA
	//initModel->aa = flambda;


	if(fix==NULL)
		free(fixed);


	return 1;
}

int CalculaNfree(PRECISION *spectro,int nspectro){
	int nfree,i,j;
	nfree=0;


	// for(j=0;j<4*nspectro;j++){
		// if(spectro[j]!=0.0){
			// nfree++;
		// }
	// }
	// nfree=nfree-NTERMS;//NTERMS;


	nfree = (nspectro*NPARMS) - NTERMS;


	return nfree;
}


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
*
*
* @Author: Juan Pedro Cobos Carrascosa (IAA-CSIC)
*		   jpedro@iaa.es
* @Date:  Nov. 2011
*
*/
void estimacionesClasicas(PRECISION lambda_0,double *lambda,int nlambda,PRECISION *spectro,Init_Model *initModel){

	// Modified by Daniele Calchetti (DC) calchetti@mps.mpg.de in March 2022 


	PRECISION x,y,aux,LM_lambda_plus,LM_lambda_minus,Blos,beta_B,Ic,Vlos;
	PRECISION *spectroI,*spectroQ,*spectroU,*spectroV;
	PRECISION L,m,gamma, gamma_rad,tan_gamma,maxV,minV,C,maxWh,minWh;
	int i,j;

	// added by DC for continuum position
	PRECISION d1, d2;
	int cont_pos, i0, ii;
	// end DC


	//Es necesario crear un lambda en FLOAT para probar como se hace en la FPGA
	PRECISION *lambda_aux;
	lambda_aux= (PRECISION*) calloc(nlambda,sizeof(PRECISION));

	// commented on March 2022 DOS (FPGA HERITAGE)
	// PRECISION lambda0,lambda1,lambda2,lambda3,lambda4;
	// lambda0 = 6.1732012e+3 + 0; // RTE_WL_0
	// lambda1 = lambda0 + 0.070000000; //RTE_WL_STEP
	// lambda2 = lambda1 + 0.070000000;
	// lambda3 = lambda2 + 0.070000000;
	// lambda4 = lambda3 + 0.070000000;

	// commented on March 2022 DOS (FPGA HERITAGE)
	// lambda_aux[0]=lambda0;
	// lambda_aux[1]=lambda1;
	// lambda_aux[2]=lambda2;
	// lambda_aux[3]=lambda3;
	// lambda_aux[4]=lambda4;

	// Ic= spectro[nlambda-1]; // Continuo ultimo valor de I
	// Ic= spectro[0]; // Continuo primer valor de I

	// added by DC for continuum position
	d1 = (PRECISION)lambda[0] - (PRECISION)lambda[1];
	d2 = (PRECISION)lambda[nlambda-2] - (PRECISION)lambda[nlambda-1];
	if (fabs(d1)>fabs(d2)){
			cont_pos = 0;
			i0 = 0;
			ii = 1;
	}
	else{
			cont_pos = nlambda -1;
			i0 = 1;
			ii = 0;
	}
	Ic = spectro[cont_pos]; // Continuo
	// end DC


	spectroI=spectro;
	spectroQ=spectro+nlambda;
	spectroU=spectro+nlambda*2;
	spectroV=spectro+nlambda*3;

	//Sino queremos usar el lambda de la FPGA
	for(i=0;i<nlambda-1;i++){
		lambda_aux[i] = (PRECISION)lambda[i+ii];// added by DC for continuum position
	}


	x=0;
	y=0;
	for(i=0;i<nlambda-1;i++){
		aux = ( Ic - (spectroI[i+ii]+ spectroV[i+ii])); // added by DC for continuum position
		x = x +  aux * (lambda_aux[i]-lambda_0);
		y = y + aux;
	}

	//Para evitar nan
	if(fabs(y)>1e-15)
		LM_lambda_plus	= x / y;
	else
		LM_lambda_plus = 0;
	// LM_lambda_plus	= x / y;

	x=0;
	y=0;
	for(i=0;i<nlambda-1;i++){
		aux = ( Ic - (spectroI[i+ii] - spectroV[i+ii]));// added by DC for continuum position
		x= x +  aux * (lambda_aux[i]-lambda_0);
		y = y + aux;
	}

	if(fabs(y)>1e-15)
		LM_lambda_minus	= x / y;
	else
		LM_lambda_minus = 0;
	// LM_lambda_minus	= x / y;

	C = (CTE4_6_13 * lambda_0 * lambda_0 * cuantic->GEFF);
	beta_B = 1 / C;

	// printf("beta_B %20.12e \n",beta_B/2);
	// printf("beta_v %20.12e \n",( VLIGHT / (lambda_0))/2);
	// printf("cuantic->GEFF %f \n",cuantic->GEFF);
	// exit(-1);

	Blos = beta_B * ((LM_lambda_plus - LM_lambda_minus)/2);
	Vlos = ( VLIGHT / (lambda_0)) * ((LM_lambda_plus + LM_lambda_minus)/2);


	//------------------------------------------------------------------------------------------------------------
	// //Para test del modo "non-polarimetric"	--> Calculo de Vlos solo con I

	// x=0;
	// y=0;
	// for(i=0;i<nlambda-1;i++){
	// 	aux = ( Ic - (spectroI[i]));
	// 	x= x +  aux * (lambda_aux[i]-lambda_0);
	// 	y = y + aux;
	// }
  //
	// //Para evitar nan
	// LM_lambda_plus	= x / y;
	// // if(fabs(y)>1e-12)
	// 	// LM_lambda_plus	= x / y;
	// // else
	// 	// LM_lambda_plus = 0;
  //
	// Vlos = ( VLIGHT / (lambda_0)) * ((LM_lambda_plus));

	//------------------------------------------------------------------------------------------------------------


	//------------------------------------------------------------------------------------------------------------
	// //Para test del modo "Longitudinal"	--> Calculo de Blos y Vlos solo con I+V y I-V
	// //Los datos vienen en la forma de siempre salvo que spectroI contiene I+V y spectroV contiene I-V
  //
	// //Para usar los mismos perfiles de siempre es necesario tunearlos y convertirlos a I+V y I-V
	// for(i=0;i<nlambda;i++){
	// 	aux = spectroI[i];
	// 	spectroI[i] = spectroI[i] + spectroV[i];
	// 	spectroV[i] = aux - spectroV[i];
	// }
  //
  //
	// //Calculo LONGITUDINAL
	// Ic= spectroI[nlambda-1]; // Continuo ultimo valor de I
  //
  //
	// x=0;
	// y=0;
	// for(i=0;i<nlambda-1;i++){
	// 	aux = ( Ic - (spectroI[i]));
	// 	x= x +  aux * ((float)lambda_aux[i]-(float)lambda_0);
	// 	y = y + aux;
	// }
  //
	// // // para probar si el orden de las operaciones afecta al redondeo
	// // PRECISION xb,dist0,dist1,dist2,dist3,dist4,aux0,aux1,aux2,aux3,aux4;
	// // PRECISION a3,a6,a9,a12,a15,a16,a17,a18,a19;
  // //
  // //
	// // lambda0 = 6.1732012e+3 + 0;
	// // lambda1 = lambda0 + 0.070000000;
	// // lambda2 = lambda1 + 0.070000000;
	// // lambda3 = lambda2 + 0.070000000;
	// // lambda4 = lambda3 + 0.070000000;
  // //
  // //
	// // dist0 = (lambda0-(float)lambda_0);
	// // dist1 = (lambda1-(float)lambda_0);
	// // dist2 = (lambda2-(float)lambda_0);
	// // dist3 = (lambda3-(float)lambda_0);
	// // dist4 = (lambda4-(float)lambda_0);
  // //
	// // aux0 = Ic - spectroI[0];
	// // aux1 = Ic - spectroI[1];
	// // aux2 = Ic - spectroI[2];
	// // aux3 = Ic - spectroI[3];
	// // aux4 = Ic - spectroI[4];
  // //
	// // a3 = aux0*dist0;
	// // a6 = aux1*dist1;
	// // a9 = aux2*dist2;
	// // a12 = aux3*dist3;
	// // a15 = aux4*dist4;
  // //
	// // a16= a3+a6;
	// // a17= a16+a9;
	// // a18= a17+a12;
	// // a19= a18+a15;
  // //
	// // printf("lambda_0 %20.7e \n",(float)lambda_0);
	// // printf("a19 B %20.7e \n",(float)a19);
	// // // fin prueba redondeo
  //
	// LM_lambda_plus	= x / y;
  //
	// // printf("a19 %20.12e \n",(float)x);
	// // printf("LM_lambda_plus %20.12e \n",(float)LM_lambda_plus);
  //
  //
	// // //Para evitar nan
	// // if(fabs(y)>1e-12)
	// 	// LM_lambda_plus	= x / y;
	// // else
	// 	// LM_lambda_plus = 0;
  //
  //
	// x=0;
	// y=0;
	// for(i=0;i<nlambda-1;i++){
	// 	aux = ( Ic - (spectroV[i]));
	// 	x= x +  aux * (lambda_aux[i]-lambda_0);
	// 	y = y + aux;
	// }
  //
  //
	// LM_lambda_minus	= x / y;
  //
	// // printf("b19 %20.12e \n",(float)x);
	// // printf("LM_lambda_minus %20.12e \n",(float)LM_lambda_minus);
  //
	// // //Para evitar nan
	// // if(fabs(y)>1e-12)
	// 	// LM_lambda_minus	= x / y;
	// // else
	// 	// LM_lambda_minus = 0;
  //
  //
	// C = ((float)CTE4_6_13 * (float)lambda_0 * (float)lambda_0 * (float)cuantic->GEFF);
	// beta_B = 1 / C;
  //
	// // printf("beta_B %20.12e \n",(float)beta_B/2);
	// // printf("beta_B (no float cast) %20.12e \n",((1/(CTE4_6_13 * lambda_0 * lambda_0 * cuantic->GEFF))/2));
	// // printf("beta_v %20.12e \n",(float)(( (float)VLIGHT / ((float)lambda_0))/2));
	// // printf("cuantic->GEFF %f \n",cuantic->GEFF);
	// // exit(-1);
  //
	// Blos = beta_B * ((LM_lambda_plus - LM_lambda_minus)/2);
	// Vlos = ( VLIGHT / (lambda_0)) * ((LM_lambda_plus + LM_lambda_minus)/2);
  //
	// // printf("spectroI_0 %20.7e \n",(float)spectroI[0]);
	// // printf("spectroI_1 %20.7e \n",(float)spectroI[1]);
	// // printf("spectroI_2 %20.7e \n",(float)spectroI[2]);
	// // printf("spectroI_3 %20.7e \n",(float)spectroI[3]);
	// // printf("spectroI_4 %20.7e \n",(float)spectroI[4]);
	// // printf("spectroI_5 %20.7e \n",(float)spectroI[5]);
	// // printf("Blos %20.12e \n",(float)Blos);
	// // printf("Vlos %20.12e \n",(float)Vlos);
  // //
	// // exit(-1);
	//------------------------------------------------------------------------------------------------------------

	//------------------------------------------------------------------------------------------------------------
	// //Para probar fórmulación propuesta por D. Orozco (Junio 2017)
	//La formula es la 2.7 que proviene del paper:
	// Diagnostics for spectropolarimetry and magnetography by Jose Carlos del Toro Iniesta and Valent´ýn Mart´ýnez Pillet
	//el 0.08 Es la anchura de la línea en lugar de la resuloción del etalón.

	//Vlos = ( 2*(VLIGHT)*0.08 / (PI*lambda_0)) * atan((spectroI[0]+spectroI[1]-spectroI[3]-spectroI[4])/(spectroI[0]-spectroI[1]-spectroI[3]+spectroI[4]));

	//------------------------------------------------------------------------------------------------------------


	Blos=Blos*1 ; //factor de correción x campo debil
	Vlos = Vlos * 1 ; //factor de correción ...

	//inclinacion
	x = 0;
	y = 0;
	for(i=0;i<nlambda-1;i++){
		L = fabs( sqrtf( spectroQ[i+ii]*spectroQ[i+ii] + spectroU[i+ii]*spectroU[i+ii] )); // added by DC for continuum position
		m = fabs( (4 * (lambda_aux[i]-lambda_0) * L ));// / (3*C*Blos) ); //2*3*C*Blos mod abril 2016 (en test!)

		x = x + fabs(spectroV[i+ii]) * m; // added by DC for continuum position
		y = y + fabs(spectroV[i+ii]) * fabs(spectroV[i+ii]); // added by DC for continuum position

//		printf("L %f \n",L);
//		printf("m : %f \n",m);
	}

	y = y * fabs((3*C*Blos));

	//for debuging
	// x = fabs(x) > 1e-7 ? x : 0;
	// y = fabs(y) > 1e-7 ? y : 0;

	tan_gamma = fabs(sqrtf(x/y));

	// tan_gamma = fabs(tan_gamma) > 1e-7 ? tan_gamma : 0;

	gamma_rad = atan(tan_gamma); //gamma en radianes

	// if(sqrt(y)<1e-12)
		// gamma_rad = PI/2;
	// else
		// gamma_rad = atan2(sqrt(x),sqrt(y)); //gamma en radianes

	//gamma_rad = atan(sqrtf(x),y)); //gamma en radianes

	// gamma_rad = fabs(gamma_rad) > 1e-7 ? gamma_rad : 0;

	gamma = gamma_rad * (180/ PI); //gamma en grados

	//correccion
	//utilizamos el signo de Blos para ver corregir el cuadrante
	PRECISION gamma_out = gamma;

    if (Blos<0)
        gamma = (180)-gamma;




	//azimuth

	PRECISION tan2phi,phi;
	int muestra;

	if(nlambda==6)
		muestra = CLASSICAL_ESTIMATES_SAMPLE_REF - i0; // added by DC for continuum position
	else
		muestra = nlambda*0.75;


	tan2phi=spectroU[muestra]/spectroQ[muestra];

	// printf("tan2phi : %f \n",tan2phi);
	// printf("%.10e \n",spectroU[muestra]);
	// printf("%.10e \n",spectroQ[muestra]);


	phi= (atan(tan2phi)*180/PI) / 2;  //atan con paso a grados

	//printf("atan : %f \n",phi*2);

	// printf("%.10e \n",atan(tan2phi));

	if(spectroU[muestra] > 0 && spectroQ[muestra] > 0 )
		phi=phi;
	else
	if (spectroU[muestra] < 0 && spectroQ[muestra] > 0 )
		phi=phi + 180;
	else
	if (spectroU[muestra] < 0 && spectroQ[muestra] < 0 )
		phi=phi + 90;
	else
	if (spectroU[muestra] > 0 && spectroQ[muestra]< 0 )
			phi=phi + 90;

	// printf("%.10e \n",phi);

	//printf("Blos : %f \n",Blos);
	//printf("vlos : %f \n",Vlos);
	//printf("gamma : %f \n",gamma);
	//printf("phi : %f \n",phi);


	PRECISION B_aux;

	B_aux = fabs(Blos/cos(gamma_rad)) * 2; // 2 factor de corrección

	//Vlos = Vlos * 1.5;
	if(Vlos < (-20))
		Vlos= -20;
	if(Vlos >(20))
		Vlos=(20);

	// if(phi< 0)
		// phi = 180 + (phi);
	// if(phi > 180){
		// phi = phi -180.0;
	// }

	// printf("%.10e \n",phi);

	initModel->B = (B_aux>4000?4000:B_aux);
	initModel->vlos=Vlos;//(Vlos*1.5);//1.5;
	initModel->gm=gamma;
	initModel->az=phi;
	initModel->S0= Blos;


	//Liberar memoria del vector de lambda auxiliar
	free(lambda_aux);

}

void FijaACeroDerivadasNoNecesarias(PRECISION * d_spectra,int *fixed,int nlambda){

	int In,j,i;
	for(In=0;In<NTERMS;In++)
		if(fixed[In]==0)
			for(j=0;j<4;j++)
				for(i=0;i<nlambda;i++)
					d_spectra[i+nlambda*In+j*nlambda*NTERMS]=0;
}

void AplicaDelta(Init_Model *model,PRECISION * delta,int * fixed,Init_Model *modelout){

	//INIT_MODEL=[eta0,magnet,vlos,landadopp,aa,gamma,azi,B1,B2,macro,alfa]

	if(fixed[0]){
		modelout->eta0=model->eta0-delta[0]; // 0
	}
	if(fixed[1]){
		if(delta[1]< -800) //300
			delta[1]=-800;
		else
			if(delta[1] >800)
				delta[1]=800;
		modelout->B=model->B-delta[1];//magnetic field
	}
	if(fixed[2]){

		 if(delta[2]>2)
			 delta[2] = 2;

		 if(delta[2]<-2)
			delta[2] = -2;

		modelout->vlos=model->vlos-delta[2];
	}

	if(fixed[3]){

		if(delta[3]>1e-2)
			delta[3] = 1e-2;
		else
			if(delta[3]<-1e-2)
				delta[3] = -1e-2;

		modelout->dopp=model->dopp-delta[3];
	}

	if(fixed[4])
		modelout->aa=model->aa-delta[4];

	if(fixed[5]){
		if(delta[5]< -15) //15
			delta[5]=-15;
		else
			if(delta[5] > 15)
				delta[5]=15;

		modelout->gm=model->gm-delta[5]; //5
	}
	if(fixed[6]){
		if(delta[6]< -15)
			delta[6]=-15;
		else
			if(delta[6] > 15)
				delta[6]= 15;

		modelout->az=model->az-delta[6];
	}
	if(fixed[7])
		modelout->S0=model->S0-delta[7];
	if(fixed[8])
		modelout->S1=model->S1-delta[8];
	if(fixed[9])
		modelout->mac=model->mac-delta[9]; //9
	if(fixed[10])
		modelout->alfa=model->alfa-delta[10];
}

/*
	Tamaño de H es 	 NTERMS x NTERMS
	Tamaño de beta es 1xNTERMS

	return en delta tam 1xNTERMS
*/

int mil_svd(PRECISION *h,PRECISION *beta,PRECISION *delta){

	double epsilon,top;
	static PRECISION v2[TAMANIO_SVD][TAMANIO_SVD],w2[TAMANIO_SVD],v[NTERMS*NTERMS],w[NTERMS];
	static PRECISION h1[NTERMS*NTERMS],h_svd[TAMANIO_SVD*TAMANIO_SVD];
	static PRECISION aux[NTERMS*NTERMS];
	int i,j;
//	static double aux2[NTERMS*NTERMS];
	static	PRECISION aux2[NTERMS];
	int aux_nf,aux_nc;
	PRECISION factor,maximo,minimo;
	int posi,posj;

	epsilon= 1e-12;
	top=1.0;

	factor=0;
	maximo=0;
	minimo=1000000000;


	/**/
	for(j=0;j<NTERMS*NTERMS;j++){
		h1[j]=h[j];
	}

	// //Para imprimir matrices
	// int i1,j1;
	// printf("#################################### h1 \n");
	// for(j1=0;j1<NTERMS;j1++){
		// for(i1=0;i1<NTERMS;i1++){
			// printf("%20.12e ,",h1[j1*NTERMS + i1]);
		// }
		// printf("\n");
	// }
	//exit(-1);


	if(USE_SVDCMP){

		svdcmp(h1,NTERMS,NTERMS,w,v);


	}
	else{
		//printf(" NORMALIZACION y CORDIC######################################\n");
		//	NORMALIZACION
		for(j=0;j<NTERMS*NTERMS;j++){
				if(fabs(h[j])>maximo){
					maximo=fabs(h[j]);
				}
			}

		factor=maximo;

		//printf("maximo : %.12e \n",maximo);
		//exit(-1);


		if(!NORMALIZATION_SVD)
			factor = 1;

		for(j=0;j<NTERMS*NTERMS;j++){
			h1[j]=h[j]/(factor );
		}


		for(i=0;i<TAMANIO_SVD-1;i++){
			for(j=0;j<TAMANIO_SVD;j++){
				if(j<NTERMS)
					h_svd[i*TAMANIO_SVD+j]=h1[i*NTERMS+j];
				else
					h_svd[i*TAMANIO_SVD+j]=0;
			}
		}

		for(j=0;j<TAMANIO_SVD;j++){
			h_svd[(TAMANIO_SVD-1)*TAMANIO_SVD+j]=0;
		}

		svdcordic(h_svd,TAMANIO_SVD,TAMANIO_SVD,w2,v2,NUM_ITER_SVD_CORDIC);

		for(i=0;i<TAMANIO_SVD-1;i++){
			for(j=0;j<TAMANIO_SVD-1;j++){
				v[i*NTERMS+j]=v2[i][j];
			}
		}

		for(j=0;j<TAMANIO_SVD-1;j++){
			w[j]=w2[j]*factor;
		}

	}

	static PRECISION vaux[NTERMS*NTERMS],waux[NTERMS];

	for(j=0;j<NTERMS*NTERMS;j++){
			vaux[j]=v[j];//*factor;
	}

	for(j=0;j<NTERMS;j++){
			waux[j]=w[j];//*factor;
	}

	// //Para imprimir matrices AUTOVAL y AUTOVEC
	// int i1,j1;
	// printf("#################################### w \n");
	// for(j1=0;j1<NTERMS;j1++){
		// printf("%20.12e ,",waux[j1]);
	// }
	// printf("\n");
	// printf("#################################### v \n");
	// for(j1=0;j1<NTERMS;j1++){
		// for(i1=0;i1<NTERMS;i1++){
			// printf("%20.12e ,",vaux[j1*NTERMS + i1]);
		// }
		// printf("\n");
	// }
	// exit(-1);

	multmatrix(beta,1,NTERMS,vaux,NTERMS,NTERMS,aux2,&aux_nf,&aux_nc);

	for(i=0;i<NTERMS;i++){
		aux2[i]= aux2[i]*((fabs(waux[i]) > epsilon) ? (1/waux[i]): 0.0);//((waux[i]>0)?(1/epsilon):(-1/epsilon))); //(1/waux[i]) : 0);//
//		aux2[i]= aux2[i]*((fabs(waux[i]) > epsilon) ? (1/waux[i]): (1/epsilon));//((waux[i]>0)?(1/epsilon):(-1/epsilon))); //(1/waux[i]) : 0);//
	}

	multmatrix(vaux,NTERMS,NTERMS,aux2,NTERMS,1,delta,&aux_nf,&aux_nc);

/*
	printf("\n");
	printf("#################################### delta \n");
	int j1;
	for(j1=0;j1<NTERMS;j1++){
		printf("%20.12e ,",delta[j1]);
	}
	// exit(-1);
*/
	return 1;

}



void weights_init(int nlambda,double *sigma,PRECISION *weight,int nweight,PRECISION **wOut,PRECISION **sigOut,double noise)
{
	int i,j;
	PRECISION *w,*sig;


	sig=calloc(4,sizeof(PRECISION));
	if(sigma==NULL){
		for(i=0;i<4;i++)
			sig[i]=	noise* noise;
	}
	else{

		for(i=0;i<4;i++)
			sig[i]=	(*sigma);// * (*sigma);
	}

	*wOut=w;
	*sigOut=sig;

}


int check(Init_Model *model){

	double offset=0;
	double inter;

	//Inclination
	/*	if(model->gm < 0)
		model->gm = -(model->gm);
	if(model->gm > 180)
		model->gm =180-(((int)floor(model->gm) % 180)+(model->gm-floor(model->gm)));//180-((int)model->gm % 180);*/

	//Magnetic field
	if(model->B < 0){
		//model->B = 190;
		model->B = -(model->B);
		model->gm = 180.0 -(model->gm);
	}
	if(model->B > 5000)
		model->B= 5000;

	//Inclination
	if(model->gm < 0)
		model->gm = -(model->gm);
	if(model->gm > 180){
		model->gm = 360.0 - model->gm;
		// model->gm = 179; //360.0 - model->gm;
	}

	//azimuth
	if(model->az < 0)
		model->az= 180 + (model->az); //model->az= 180 + (model->az);
	if(model->az > 180){
		model->az =model->az -180.0;
		// model->az = 179.0;
	}

	//RANGOS
	//Eta0
	if(model->eta0 < 0.1)
		model->eta0=0.1;

	// if(model->eta0 >8)
			// model->eta0=8;
	if(model->eta0 >2500)  //idl 2500
			model->eta0=2500;

	//velocity
	if(model->vlos < (-20)) //20
		model->vlos= (-20);
	if(model->vlos >20)
		model->vlos=20;

	//doppler width ;Do NOT CHANGE THIS
	if(model->dopp < 0.0001)
		model->dopp = 0.0001;

	if(model->dopp > 1.6)  // idl 0.6
		model->dopp = 1.6;


	if(model->aa < 0.0001)  // idl 1e-4
		model->aa = 0.0001;
	if(model->aa > 10)            //10
		model->aa = 10;

	//S0
	if(model->S0 < 0.0001)
		model->S0 = 0.0001;
	if(model->S0 > 1.500)
		model->S0 = 1.500;

	//S1
	if(model->S1 < 0.0001)
		model->S1 = 0.0001;
	if(model->S1 > 2.000)
		model->S1 = 2.000;

	//macroturbulence
	if(model->mac < 0)
		model->mac = 0;
	if(model->mac > 4)
		model->mac = 4;

	//filling factor
/*	if(model->S1 < 0)
		model->S1 = 0;
	if(model->S1 > 1)
		model->S1 = 1;*/

	return 1;
}

void spectral_synthesis_convolution(){

	int i;
	int nlambda=NLAMBDA;

	//convolucionamos los perfiles IQUV (spectra)
	if(INSTRUMENTAL_CONVOLUTION){

		PRECISION Ic;


		if(!INSTRUMENTAL_CONVOLUTION_INTERPOLACION){
			//convolucion de I
			Ic=spectra[nlambda-1];

			for(i=0;i<nlambda-1;i++)
				spectra[i]=Ic-spectra[i];



			direct_convolution(spectra,nlambda-1,G,NMUESTRAS_G,1);  //no convolucionamos el ultimo valor Ic


			for(i=0;i<nlambda-1;i++)
				spectra[i]=Ic-spectra[i];

			//convolucion QUV
			for(i=1;i<NPARMS;i++)
				direct_convolution(spectra+nlambda*i,nlambda-1,G,NMUESTRAS_G,1);  //no convolucionamos el ultimo valor
		}
		else{
			if(NLAMBDA == 6){

				//convolucion de I
				Ic=spectra[nlambda-1]; // CHANGE  HERE!!!!!!

				for(i=0;i<nlambda-1;i++) //CHANGE ITERABLE DEPENDENT ON IC
					spectra[i]=Ic-spectra[i];

				PRECISION *spectra_aux;
				spectra_aux =  (PRECISION*) calloc(nlambda*2-2,sizeof(PRECISION));

				int j=0;
				for(i=0,j=0;i<nlambda*2-2;i=i+2,j++) //AGAIN CHANGE HERE
					spectra_aux[i]=spectra[j];

				for(i=1,j=0;i<nlambda*2-2;i=i+2,j++) //AGAIN CHANGE HERE
					spectra_aux[i]=(spectra[j]+spectra[j+1])/2;


				// printf("spectraI_aux=[");
				// for(i=0;i<nlambda*2-2;i++){
					// printf("%f",Ic-spectra_aux[i]);
					// if(i<nlambda*2-2-1)
						// printf(", ");
				// }
				// printf("];\n");

				direct_convolution(spectra_aux,nlambda*2-2-1,G,NMUESTRAS_G,1);  //no convolucionamos el ultimo valor Ic //AGAIN CHANGE HERE

				// printf("spectraI_aux_conv=[");
				// for(i=0;i<nlambda*2-2;i++){
					// printf("%f",Ic-spectra_aux[i]);
					// if(i<nlambda*2-2-1)
						// printf(", ");
				// }
				// printf("];\n");


				for(i=0,j=0;i<nlambda*2-2;i=i+2,j++) //AGAIN CHANGE HERE
					spectra[j]=spectra_aux[i]; //AGAIN CHANGE HERE

				for(i=0;i<nlambda-1;i++) //AGAIN CHANGE HERE
					spectra[i]=Ic-spectra[i];

				free(spectra_aux);

				//convolucion QUV
				int k;
				for(k=1;k<NPARMS;k++){

					PRECISION *spectra_aux;
					spectra_aux =  (PRECISION*) calloc(nlambda*2-2,sizeof(PRECISION)); 

					int j=0;
					for(i=0,j=0;i<nlambda*2-2;i=i+2,j++) //AGAIN CHANGE HERE
						spectra_aux[i]=spectra[j+nlambda*k];

					for(i=1,j=0;i<nlambda*2-2;i=i+2,j++) //AGAIN CHANGE HERE
						spectra_aux[i]=(spectra[j+nlambda*k]+spectra[j+1+nlambda*k])/2;

					direct_convolution(spectra_aux,nlambda*2-2-1,G,NMUESTRAS_G,1);  //no convolucionamos el ultimo valor Ic //AGAIN CHANGE HERE

					for(i=0,j=0;i<nlambda*2-2;i=i+2,j++) //AGAIN CHANGE HERE
						spectra[j+nlambda*k]=spectra_aux[i];

					free(spectra_aux);
				}
			}
		}
	}
}



void response_functions_convolution(){

	int i,j;
	int nlambda=NLAMBDA;

	//convolucionamos las funciones respuesta ( d_spectra )
	if(INSTRUMENTAL_CONVOLUTION){
		if(!INSTRUMENTAL_CONVOLUTION_INTERPOLACION){


			for(j=0;j<NPARMS;j++){
				for(i=0;i<NTERMS;i++){
					if(i!=7) //no convolucionamos S0
						direct_convolution(d_spectra+nlambda*i+nlambda*NTERMS*j,nlambda-1,G,NMUESTRAS_G,1);  //no convolucionamos el ultimo valor
				}
			}


		}
		else{

			int k,m;
			for(k=0;k<NPARMS;k++){
				for(m=0;m<NTERMS;m++){

					PRECISION *spectra_aux;
					spectra_aux =  (PRECISION*) calloc(nlambda*2-2,sizeof(PRECISION));

					int j=0;
					for(i=0,j=0;i<nlambda*2-2;i=i+2,j++) //AGAIN CHANGE HERE
						spectra_aux[i]=d_spectra[j+nlambda*m+nlambda*NTERMS*k];

					for(i=1,j=0;i<nlambda*2-2;i=i+2,j++) //AGAIN CHANGE HERE
						spectra_aux[i]=(d_spectra[j+nlambda*m+nlambda*NTERMS*k]+d_spectra[j+nlambda*m+nlambda*NTERMS*k])/2;

					direct_convolution(spectra_aux,nlambda*2-2-1,G,NMUESTRAS_G,1);  //no convolucionamos el ultimo valor Ic //AGAIN CHANGE HERE

					for(i=0,j=0;i<nlambda*2-2;i=i+2,j++)
						d_spectra[j+nlambda*m+nlambda*NTERMS*k]=spectra_aux[i]; //AGAIN CHANGE HERE

					free(spectra_aux);
				}
			}
		}
	}

}
