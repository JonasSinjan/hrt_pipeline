
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
//  >> milos NLAMBDA MAX_ITER CLASSICAL_ESTIMATES [FWHM DELTA NPOINTS] profiles_file.txt > output.txt
//
//   NLAMBDA number of lambda of input profiles
//   MAX_ITER of inversion
//   CLASSICAL_ESTIMATES use classical estimates? 1 yes, 0 no
//   [FWHM DELTA NPOINTS] use convolution with a gaussian? if the tree parameteres are defined yes, else no. Units in A. NPOINTS has to be odd.
//   profiles_file.txt name of input profiles file
//   output.txt name of output file
//
//

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
int CLASSICAL_ESTIMATES = 0;
int RFS = 0;


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
	PRECISION weight[4]={1.,1.,1.,1.};
	int nweight;

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
	
	double dat[7]={CUANTIC_NWL,CUANTIC_SLOI,CUANTIC_LLOI,CUANTIC_JLOI,CUANTIC_SUPI,CUANTIC_LUPI,CUANTIC_JUPI};

	char *nombre,*input_iter;
	int Max_iter;

	//milos NLAMBDA MAX_ITER CLASSICAL_ESTIMATES RFS [FWHM DELTA NPOINTS] perfil.txt
	
	if(argc!=4 && argc!=5 && argc !=8 ){  // changed argc !=8 to !=9 (for RF output)
		printf("milos: Error en el numero de parametros: %d .\n Pruebe: milos NLAMBDA MAX_ITER CLASSICAL_ESTIMATES RFS [FWHM(in A) DELTA(in A) NPOINTS] perfil.txt\n",argc);
		return -1;
	}	
		
	NLAMBDA = atoi(argv[1]);
	
	input_iter = argv[2];
	Max_iter = atoi(input_iter);
	CLASSICAL_ESTIMATES = atoi(argv[3]);
	
	if(CLASSICAL_ESTIMATES!=0 && CLASSICAL_ESTIMATES != 1){
		printf("milos: Error in CLASSICAL_ESTIMATES parameter. 0 or 1 are valid values. Not accepted: \n",CLASSICAL_ESTIMATES);
		return -1;
	}
	
	if(argc ==5){		
		nombre = argv[4];
	}
    else if(argc == 6) {  //DOS
        RFS = 1;
    }
    else {
		INSTRUMENTAL_CONVOLUTION = 1;
		NMUESTRAS_G = atoi(argv[6]);
		FWHM = atof(argv[4]);
		DELTA = atof(argv[5]);
		nombre = argv[7];
	}

	
	//Generamos la gaussiana -> perfil instrumental			
	if(INSTRUMENTAL_CONVOLUTION){
		G=vgauss(FWHM,NMUESTRAS_G,DELTA);
		
		//if you wish to convolution with other instrumental profile you have to declare here and to asign it to "G"
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

	nlambda=NLAMBDA;
	lambda=calloc(nlambda,sizeof(double));
	spectro=calloc(nlambda*4,sizeof(PRECISION));

	FILE *fichero, *fichero_estimacioneClasicas;

	fichero= fopen(nombre,"r");
	if(fichero==NULL){   
		printf("Error de apertura, es posible que el fichero no exista.\n");
		printf("Milos: Error de lectura del fichero. ++++++++++++++++++\n");
		return 1;
	}

	char * buf;
	buf=calloc(strlen(nombre)+15,sizeof(char));
	buf = strcat(buf,nombre);
	buf = strcat(buf,"_CL_ESTIMATES");
	
	if(CLASSICAL_ESTIMATES){
		fichero_estimacioneClasicas= fopen(buf,"w");
		if(fichero_estimacioneClasicas==NULL){   
			printf("Error de apertura ESTIMACIONES CLASICAS, es posible que el fichero no exista.\n");
			printf("Milos: Error de lectura del fichero. ++++++++++++++++++\n");
			return 1;
		}		
	}
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
		
	
	
	do{
		neje=0;
		nsub=0;
		while (neje<NLAMBDA && (rfscanf=fscanf(fichero,"%lf %le %le %le %le",&lin,&iin,&qin,&uin,&vin))!= EOF){
			lambda[nsub]=lin;	
			//printf(" %f \n",lambda[nsub]);
			spectro[nsub]=iin;
			spectro[nsub+NLAMBDA]=qin;
			spectro[nsub+NLAMBDA*2]=uin;
			spectro[nsub+NLAMBDA*3]=vin;
			nsub++;

			neje++;
		}
	
		if(rfscanf!=EOF){  //   && contador==8


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

	
		if(CLASSICAL_ESTIMATES){
			estimacionesClasicas(wlines[1],lambda,nlambda,spectro,&initModel);
		
			//OJO CON LOS MENOS !!!

			if(isnan(initModel.B))
				initModel.B = 0;
			if(isnan(initModel.vlos))
				initModel.vlos = 0;
			if(isnan(initModel.gm))
				initModel.gm=0;
			if(isnan(initModel.az))
				initModel.az = 0;
			
			//Escribimos en fichero las estimaciones clásicas
			fprintf(fichero_estimacioneClasicas,"%e %e %e %e\n",initModel.B,initModel.gm,initModel.az,initModel.vlos);
		}	
			
		//inversion	
		lm_mils(cuantic,wlines,nwlines,lambda, nlambda,spectro,nlambda,&initModel,spectra,err,&chisqrf,&iter,slight,toplim,miter,
				weight,nweight,fix,sig,filter,ilambda,noise,pol,getshi,0);

		totalIter+=iter;

		//chisqrf_array[contador]=chisqrf;

		//[contador;iter;B;GM;AZI;etha0;lambdadopp;aa;vlos;S0;S1;final_chisqr];
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
		
		//exit(-1);
		
		}

		contador++;

	}while(rfscanf!=EOF);


	fclose(fichero);
	
	if(CLASSICAL_ESTIMATES)
		fclose(fichero_estimacioneClasicas);
	

	free(spectro);
	free(lambda);
	free(cuantic);
	free(wlines);

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
	
	//convolucionamos los perfiles IQUV (spectra)
	spectral_synthesis_convolution();
	
	
	me_der(cuantic,initModel,wlines,nwlines,lambda,nlambda,d_spectra,AH,slight,triplete,filter);

    if(RFS ==1){  // IF the caller adds RFS = 1 then, the program writes down the RFS of ITER 0
        printf("%f\n",model.B);
        printf("%f\n",model.gm);
        printf("%f\n",model.az);
        printf("%f \n",model.eta0);
        printf("%f\n",model.dopp);
        printf("%f\n",model.aa);
        printf("%f\n",model.vlos);
        printf("%f\n",model.S0);
        printf("%f\n",model.S1);
    int In,j,i;
    for(In=0;In<NTERMS;In++)  // Parameters
            for(j=0;j<4;j++)  // Stokes parameter
                for(i=0;i<nlambda;i++) //wavelength
                    printf("%f\n",d_spectra[i+nlambda*In+j*nlambda*NTERMS]);
                           }
                           
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

			
			//printf("iteration=%d , chisqr = %f CONVERGE	- lambda= %e \n",iter,chisqr,flambda);
			
			
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
			flambda=flambda*5;//5;

			//printf("iteration=%d , chisqr = %f NOT CONVERGE	- lambda= %e \n",iter,ochisqr,flambda);
		
		}

		/**************************************************************************/
		iter++;


	}while(iter<=miter); // && !clanda);

	*iterOut=iter;

	*chisqrf=ochisqr;

	

	if(fix==NULL)
		free(fixed);


	return 1;
}	

int CalculaNfree(PRECISION *spectro,int nspectro){
	int nfree,i,j;
	nfree=0;
	
	/*
	for(j=0;j<4*nspectro;j++){
		if(spectro[j]!=0.0){
			nfree++;				
		}
	}
	nfree=nfree-NTERMS;//NTERMS;
	*/
	
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

	PRECISION x,y,aux,LM_lambda_plus,LM_lambda_minus,Blos,beta_B,Ic,Vlos;
	PRECISION *spectroI,*spectroQ,*spectroU,*spectroV;
	PRECISION L,m,gamma, gamma_rad,tan_gamma,maxV,minV,C,maxWh,minWh;
	int i,j;

	spectroI=spectro;
	spectroQ=spectro+nlambda;
	spectroU=spectro+nlambda*2;
	spectroV=spectro+nlambda*3;

	Ic= spectro[nlambda-1]; // Continuo ultimo valor de I


	x=0;
	y=0;
	for(i=0;i<nlambda-1;i++){
		aux = ( Ic - (spectroI[i]+ spectroV[i]));
		x= x +  aux * (lambda[i]-lambda_0);
		y = y + aux;
	}
	
	//Para evitar nan
	if(y!=0)
		LM_lambda_plus	= x / y;
	else
		LM_lambda_plus = 0;


	x=0;
	y=0;
	for(i=0;i<nlambda-1;i++){
		aux = ( Ic - (spectroI[i] - spectroV[i]));
		x= x +  aux * (lambda[i]-lambda_0);
		y = y + aux;
	}

	if(y!=0)
		LM_lambda_minus	= x / y;
	else
		LM_lambda_minus = 0;

	C = (CTE4_6_13 * lambda_0 * lambda_0 * cuantic->GEFF);
	beta_B = 1 / C;

	Blos = beta_B * ((LM_lambda_plus - LM_lambda_minus)/2);
	Vlos = ( VLIGHT / lambda_0) * ((LM_lambda_plus + LM_lambda_minus)/2);
	
	

	Blos=Blos ; //factor de correción x campo debil       
	Vlos = Vlos ; //factor de correción ...

	//inclinacion	
	x = 0;
	y = 0;
	for(i=0;i<nlambda-1;i++){
		L = fabs( sqrt( spectroQ[i]*spectroQ[i] + spectroU[i]*spectroU[i] ));		
		m = fabs( (4 * (lambda[i]-lambda_0) * L ) / (2*3*C*Blos) );

		x = x + fabs(spectroV[i]) * m;
		y = y + fabs(spectroV[i]) * fabs(spectroV[i]);

//		printf("L %f \n",L);
//		printf("m : %f \n",m);
	}

	tan_gamma = fabs(sqrt(x/y));
	
	gamma_rad = atan(tan_gamma); //gamma en radianes
	
	gamma = gamma_rad * (180/ PI); //gamma en grados


	//correccion 
	//utilizamos el signo de Blos para ver corregir el cuadrante
    if (Blos<0)
        gamma = 180-gamma;
            
	

	//azimuth

	PRECISION tan2phi,phi;
	int muestra;
	
	if(nlambda==6)
		muestra = CLASSICAL_ESTIMATES_SAMPLE_REF;
	else
		muestra = nlambda*0.75;
	
	
	tan2phi=spectroU[muestra]/spectroQ[muestra];

//	printf("tan2phi : %f \n",tan2phi);

	phi= (atan(tan2phi)*180/PI) / 2;

//	printf("atan : %f \n",phi*2);
	
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



	//printf("Blos : %f \n",Blos);
	//printf("vlos : %f \n",Vlos);
	//printf("gamma : %f \n",gamma);
	//printf("phi : %f \n",phi);


	PRECISION B_aux;
	
	B_aux = fabs(Blos/cos(gamma_rad));//*2;
	

	if(Vlos < (-5))
		Vlos= -5;
	if(Vlos >(5))
		Vlos=(5);	
	
	if(phi< 0)
		phi = 180 + (phi);
	if(phi > 180){
		phi = phi -180.0;
	}
	
	initModel->B = (B_aux>4000?4000:B_aux);
	initModel->vlos=Vlos;//(Vlos*1.5);//1.5;
	initModel->gm=gamma;
	initModel->az=phi;

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
	
		 if(delta[2]>5)
			 delta[2] = 5;

		 if(delta[2]<-5)
			delta[2] = -5;

		modelout->vlos=model->vlos-delta[2];
	}
	if(fixed[3])
		modelout->dopp=model->dopp-delta[3];
	if(fixed[4])
		modelout->aa=model->aa-delta[4];
	if(fixed[5]){
		if(delta[5]< -45) //15
			delta[5]=-45;
		else
			if(delta[5] > 45)
				delta[5]=45;

		modelout->gm=model->gm-delta[5]; //5
	}
	if(fixed[6]){
		if(delta[6]< -15)
			delta[6]=-15;
		else
			if(delta[6] > 15)
				delta[6]=15;

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
	int i,j,j1;	
//	static double aux2[NTERMS*NTERMS];
	static	PRECISION aux2[NTERMS];
	int aux_nf,aux_nc;
	PRECISION factor,maximo,minimo;
	int posi,posj;

	epsilon= 1e-15;
	top=1.0;

	factor=0;
	maximo=0;
	minimo=1000000000;


	/**/
	for(j=0;j<NTERMS*NTERMS;j++){
		h1[j]=h[j];
	}
	
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
		
		if(!NORMALIZATION_SVD)
			factor = 1;

		for(j=0;j<NTERMS*NTERMS;j++){
			h1[j]=h[j]/factor;
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


	multmatrix(beta,1,NTERMS,vaux,NTERMS,NTERMS,aux2,&aux_nf,&aux_nc);

	for(i=0;i<NTERMS;i++){
		aux2[i]= aux2[i]*((fabs(waux[i]) > epsilon) ? (1/waux[i]) : 0.0);
	}

	multmatrix(vaux,NTERMS,NTERMS,aux2,NTERMS,1,delta,&aux_nf,&aux_nc);

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
		model->B = -(model->B);	
		model->gm = 180.0 -(model->gm);		
	} 
	if(model->B > 5000)
		model->B= 5000;			
	
	if(model->gm < 0)
		model->gm = -(model->gm);
	if(model->gm > 180)
		model->gm = 360.0 - model->gm;	

	//azimuth
	if(model->az < 0)
		model->az= 180 + (model->az);
	if(model->az > 180){
		model->az =model->az -180.0;
	}

	//RANGOS
	//Eta0
	if(model->eta0 < 0.1)
		model->eta0=0.1;
	if(model->eta0 >8)
			model->eta0=8;	
			
	//velocity
	if(model->vlos < (-5)) //20
		model->vlos= (-5);
	if(model->vlos >5)
		model->vlos=5;	

	//doppler width ;Do NOT CHANGE THIS
	if(model->dopp < 0.0001)
		model->dopp = 0.0001;
			
	if(model->dopp > 0.800)
		model->dopp = 0.800;	
	

	if(model->aa < 0.00001)
		model->aa = 0.00001;
	if(model->aa > 10)
		model->aa = 10;
	
	//S0
	if(model->S0 < 0.0001)
		model->S0 = 0.0001;
	if(model->S0 > 2.000)
		model->S0 = 2.000;

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
				Ic=spectra[nlambda-1];
				
				for(i=0;i<nlambda-1;i++)
					spectra[i]=Ic-spectra[i];				
				
				PRECISION *spectra_aux;
				spectra_aux =  (PRECISION*) calloc(nlambda*2-2,sizeof(PRECISION));				
				
				int j=0;
				for(i=0,j=0;i<nlambda*2-2;i=i+2,j++)
					spectra_aux[i]=spectra[j];

				for(i=1,j=0;i<nlambda*2-2;i=i+2,j++)
					spectra_aux[i]=(spectra[j]+spectra[j+1])/2;
							
									
				// printf("spectraI_aux=[");
				// for(i=0;i<nlambda*2-2;i++){
					// printf("%f",Ic-spectra_aux[i]);
					// if(i<nlambda*2-2-1)
						// printf(", ");
				// }
				// printf("];\n");				
							
				direct_convolution(spectra_aux,nlambda*2-2-1,G,NMUESTRAS_G,1);  //no convolucionamos el ultimo valor Ic
			
				// printf("spectraI_aux_conv=[");
				// for(i=0;i<nlambda*2-2;i++){
					// printf("%f",Ic-spectra_aux[i]);
					// if(i<nlambda*2-2-1)
						// printf(", ");
				// }
				// printf("];\n");		
			
			
				for(i=0,j=0;i<nlambda*2-2;i=i+2,j++)
					spectra[j]=spectra_aux[i];
				
				for(i=0;i<nlambda-1;i++)
					spectra[i]=Ic-spectra[i];		

				free(spectra_aux);
				
				//convolucion QUV
				int k;
				for(k=1;k<NPARMS;k++){
				
					PRECISION *spectra_aux;
					spectra_aux =  (PRECISION*) calloc(nlambda*2-2,sizeof(PRECISION));				
					
					int j=0;
					for(i=0,j=0;i<nlambda*2-2;i=i+2,j++)
						spectra_aux[i]=spectra[j+nlambda*k];

					for(i=1,j=0;i<nlambda*2-2;i=i+2,j++)
						spectra_aux[i]=(spectra[j+nlambda*k]+spectra[j+1+nlambda*k])/2;
								
					direct_convolution(spectra_aux,nlambda*2-2-1,G,NMUESTRAS_G,1);  //no convolucionamos el ultimo valor Ic
				
					for(i=0,j=0;i<nlambda*2-2;i=i+2,j++)
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
					for(i=0,j=0;i<nlambda*2-2;i=i+2,j++)
						spectra_aux[i]=d_spectra[j+nlambda*m+nlambda*NTERMS*k];

					for(i=1,j=0;i<nlambda*2-2;i=i+2,j++)
						spectra_aux[i]=(d_spectra[j+nlambda*m+nlambda*NTERMS*k]+d_spectra[j+nlambda*m+nlambda*NTERMS*k])/2;
								
					direct_convolution(spectra_aux,nlambda*2-2-1,G,NMUESTRAS_G,1);  //no convolucionamos el ultimo valor Ic
				
					for(i=0,j=0;i<nlambda*2-2;i=i+2,j++)
						d_spectra[j+nlambda*m+nlambda*NTERMS*k]=spectra_aux[i];
															
					free(spectra_aux);													
				}	
			}
		}		
	}

}
