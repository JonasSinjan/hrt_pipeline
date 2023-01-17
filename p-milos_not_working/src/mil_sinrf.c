#include <math.h>
#include "defines.h"
#include "lib.h"
#include <string.h>
#include "convolution.h"
#include "milosUtils.h"
#include <complex.h>
#include <fftw3.h> //siempre a continuacion de complex.h
#include "readConfig.h"



int funcionComponentFor_sinrf(REAL *u,int n_pi,int numl,REAL *wex,REAL *nuxB,REAL *fi_x,
												REAL *shi_x,PRECISION A,PRECISION MF);



extern REAL * gp1,*gp2,*dt,*dti,*gp3,*gp4,*gp5,*gp6,*etai_2;
extern REAL *gp4_gp2_rhoq,*gp5_gp2_rhou,*gp6_gp2_rhov;
extern REAL CC,CC_2,sin_gm,azi_2,sinis,cosis,cosis_2,cosi,sina,cosa,sinda,cosda,sindi,cosdi,sinis_cosa,sinis_sina;
extern REAL *fi_p,*fi_b,*fi_r,*shi_p,*shi_b,*shi_r;
extern REAL *etain,*etaqn,*etaun,*etavn,*rhoqn,*rhoun,*rhovn;
extern REAL *etai,*etaq,*etau,*etav,*rhoq,*rhou,*rhov;
extern REAL *parcial1,*parcial2,*parcial3;
extern REAL *nubB,*nupB,*nurB;
extern REAL **uuGlobalInicial;
extern REAL **HGlobalInicial;
extern REAL **FGlobalInicial;
extern int FGlobal,HGlobal,uuGlobal;
extern PRECISION *GMAC; // VECTOR WITH GAUSSIAN CREATED FOR CONVOLUTION 
extern PRECISION *G;
extern fftw_complex * inSpectraFwMAC, *inSpectraBwMAC, *outSpectraFwMAC, *outSpectraBwMAC;
extern fftw_complex * inFilterMAC, * inFilterMAC_DERIV, * outFilterMAC, * outFilterMAC_DERIV;
extern fftw_plan planForwardMAC, planBackwardMAC;
extern fftw_plan planFilterMAC, planFilterMAC_DERIV;
extern fftw_complex * fftw_G_PSF, * fftw_G_MAC_PSF, * fftw_G_MAC_DERIV_PSF;
extern fftw_complex * inPSF_MAC, * inMulMacPSF, * inPSF_MAC_DERIV, *inMulMacPSFDeriv, *outConvFilters, * outConvFiltersDeriv;
extern fftw_plan planForwardPSF_MAC, planForwardPSF_MAC_DERIV,planBackwardPSF_MAC, planBackwardPSF_MAC_DERIV;
extern fftw_complex * inSpectraFwPSF, *inSpectraBwPSF, *outSpectraFwPSF, *outSpectraBwPSF;
extern fftw_plan planForwardPSF, planBackwardPSF;
extern ConfigControl configCrontrolFile;


/**
 * */
int mil_sinrf(Cuantic *cuantic,Init_Model *initModel,PRECISION * wlines,PRECISION *lambda,int nlambda,REAL *spectra,
			REAL ah,REAL * slight, REAL * spectra_mc, REAL * spectra_slight, int filter)
{

	int offset,numl;
	int lineas;


	PRECISION E00,MF,VL,LD,A,GM,AZI,B0,B1,MC,ALF;
	int il,i,j;
	PRECISION E0;	
	REAL ulos;
	REAL u[nlambda];
   int odd,ishift;
	REAL  parcial;

	offset= 0;//10.0;

	E00=initModel->eta0; 
	MF=initModel->B;    
	VL=(initModel->vlos) - offset;
	LD=initModel->dopp;
	A=initModel->aa;
	GM=initModel->gm; 
	AZI=initModel->az;
	B0=initModel->S0;
	B1=-((initModel->S1)*ah);
	MC=initModel->mac;
	ALF=initModel->alfa;		
	
	
	numl=nlambda;   	

	lineas=(int)wlines[0];

	//Definicion de ctes.
	//a radianes	

	AZI=AZI*CC;
	GM=GM*CC;

	sin_gm=SIN(GM);
	cosi=COS(GM);
	sinis=sin_gm*sin_gm;
	cosis=cosi*cosi;
	cosis_2=(1+cosis)/2;
	azi_2=2*AZI;
	sina=SIN(azi_2);
	cosa=COS(azi_2);

	sinda=cosa*CC_2;
	cosda=-sina*CC_2;
	sindi=cosi*sin_gm*CC_2;
	cosdi=-sin_gm*CC;
	
	
	sinis_cosa=sinis*cosa;
	sinis_sina=sinis*sina;


    for(il=0;il<lineas;il++) {

		etain=etain+nlambda*il*sizeof(REAL);
		etaqn=etaqn+nlambda*il*sizeof(REAL);
		etaun=etaun+nlambda*il*sizeof(REAL);
		etavn=etavn+nlambda*il*sizeof(REAL);
		rhoqn=rhoqn+nlambda*il*sizeof(REAL);
		rhoun=rhoun+nlambda*il*sizeof(REAL);
		rhovn=rhovn+nlambda*il*sizeof(REAL);

		//Line strength
		E0=E00*cuantic[il].FO; 

		//frecuency shift for v line of sight
		ulos=(VL*wlines[il+1])/(VLIGHT*LD);

		//doppler velocity	    
		for(i=0;i<nlambda;i++){
			u[i]=((lambda[i]-wlines[il+1])/LD)-ulos;
		}

		fi_p=fi_p+nlambda*il*sizeof(REAL);
		fi_b=fi_b+nlambda*il*sizeof(REAL);
		fi_r=fi_r+nlambda*il*sizeof(REAL);
		shi_p=shi_p+nlambda*il*sizeof(REAL);
		shi_b=shi_b+nlambda*il*sizeof(REAL);
		shi_r=shi_r+nlambda*il*sizeof(REAL);

	    for(i=0;i<nlambda;i++){
			fi_p[i]=0;
			fi_b[i]=0;
			fi_r[i]=0;
		}
	    for(i=0;i<nlambda;i++){
			shi_p[i]=0;
			shi_b[i]=0;
			shi_r[i]=0;
		}


		// ******* GENERAL MULTIPLET CASE ********
		
		nubB=nubB+nlambda*il*sizeof(REAL);
		nurB=nurB+nlambda*il*sizeof(REAL);
		nupB=nupB+nlambda*il*sizeof(REAL);


		parcial=(((wlines[il+1]*wlines[il+1]))/LD)*(CTE4_6_13);
		
		//caso multiplete						
		for(i=0;i<cuantic[il].N_SIG;i++){
			nubB[i]=parcial*cuantic[il].NUB[i]; // Spliting	
		}

		for(i=0;i<cuantic[il].N_PI;i++){
			nupB[i]=parcial*cuantic[il].NUP[i]; // Spliting			    
		}						

		for(i=0;i<cuantic[il].N_SIG;i++){
			nurB[i]=-nubB[(int)cuantic[il].N_SIG-(i+1)]; // Spliting
		}						

		uuGlobal=0;
		FGlobal=0;
		HGlobal=0;

		//central component					    					
		funcionComponentFor_sinrf(u,cuantic[il].N_PI,numl,cuantic[il].WEP,nupB,fi_p,shi_p,A,MF);

		//blue component
		funcionComponentFor_sinrf(u,cuantic[il].N_SIG,numl,cuantic[il].WEB,nubB,fi_b,shi_b,A,MF);

		//red component
		funcionComponentFor_sinrf(u,cuantic[il].N_SIG,numl,cuantic[il].WER,nurB,fi_r,shi_r,A,MF);

		uuGlobal=0;
		FGlobal=0;
		HGlobal=0;

		//*****

		//dispersion profiles				
		REAL E0_2;
		E0_2=E0/2.0;

		for(i=0;i<numl;i++){
			parcial1[i]=fi_b[i]+fi_r[i];
			parcial2[i]=(E0_2)*(fi_p[i]-(parcial1[i])/2);
			parcial3[i]=(E0_2)*(shi_p[i]-(shi_b[i]+shi_r[i])/2);
		}	

		REAL cosi_E0_2;
		cosi_E0_2=E0_2*cosi;
		for(i=0;i<numl;i++){
			etain[i]=((E0_2)*(fi_p[i]*sinis+(parcial1[i])*cosis_2));
			etaqn[i]=(parcial2[i]*sinis_cosa);
			etaun[i]=(parcial2[i]*sinis_sina);
			etavn[i]=(fi_r[i]-fi_b[i])*cosi_E0_2;
		}
		for(i=0;i<numl;i++){
			rhoqn[i]=(parcial3[i]*sinis_cosa);
			rhoun[i]=(parcial3[i]*sinis_sina);
			rhovn[i]=(shi_r[i]-shi_b[i])*cosi_E0_2;
		}

		for(i=0;i<numl;i++){
			etai[i]=1.0 + etain[i];
			etaq[i]=etaqn[i];
			etau[i]=etaun[i];
			etav[i]=etavn[i];			
		}
		for(i=0;i<numl;i++){

			rhoq[i]=rhoqn[i];
			rhou[i]=rhoun[i];
			rhov[i]=rhovn[i];	
			
		}
		

	} //end for


	for(i=0;i<numl;i++){    	
		etai_2[i]=etai[i]*etai[i];
    } 

    for(i=0;i<numl;i++){
		REAL auxq,auxu,auxv;
		auxq=rhoq[i]*rhoq[i];
		auxu=rhou[i]*rhou[i];
		auxv=rhov[i]*rhov[i];
    	gp1[i]=etai_2[i]-etaq[i]*etaq[i]-etau[i]*etau[i]-etav[i]*etav[i]+auxq+auxu+auxv;
    	gp3[i]=etai_2[i]+auxq+auxu+auxv;
    }
    for(i=0;i<numl;i++){
        gp2[i]=etaq[i]*rhoq[i]+etau[i]*rhou[i]+etav[i]*rhov[i];
    }
    for(i=0;i<numl;i++){
        dt[i]=etai_2[i]*gp1[i]-gp2[i]*gp2[i];
    }
    for(i=0;i<numl;i++){
    	dti[i]=1.0/dt[i];
    }
    for(i=0;i<numl;i++){
    	gp4[i]=etai_2[i]*etaq[i]+etai[i]*(etav[i]*rhou[i]-etau[i]*rhov[i]);
    }    
    for(i=0;i<numl;i++){
    	gp5[i]=etai_2[i]*etau[i]+etai[i]*(etaq[i]*rhov[i]-etav[i]*rhoq[i]);
    }    
    for(i=0;i<numl;i++){
    	gp6[i]=(etai_2[i])*etav[i]+etai[i]*(etau[i]*rhoq[i]-etaq[i]*rhou[i]);
    }       
   
	REAL dtiaux[nlambda];

   for(i=0;i<numl;i++){
		gp4_gp2_rhoq[i] = gp4[i]+rhoq[i]*gp2[i];
		gp5_gp2_rhou[i] = gp5[i]+rhou[i]*gp2[i];
		gp6_gp2_rhov[i] = gp6[i]+rhov[i]*gp2[i];
	}

    for(i=0;i<numl;i++)
		dtiaux[i] = dti[i]*(B1);
    //espectro
    for(i=0;i<numl;i++){
    	spectra[i] = B0-dtiaux[i]*etai[i]*gp3[i];

        spectra[i+numl] = (dtiaux[i]*(gp4_gp2_rhoq[i]));

        spectra[i+numl*2] = (dtiaux[i]*(gp5_gp2_rhou[i]));

        spectra[i+numl*3] = (dtiaux[i]*(gp6_gp2_rhov[i]));
		if(spectra_mc!=NULL){
			spectra_mc[i] = spectra[i];
			spectra_mc[i+numl  ] = spectra[i+numl  ];
			spectra_mc[i+numl*2]=spectra[i+numl*2];
			spectra_mc[i+numl*3] = spectra[i+numl*3];
		}
    }

	int macApplied = 0;
    if(MC > 0.0001 && spectra_mc!=NULL){

		macApplied = 1;
		//MACROTURBULENCIA            
		odd=(numl%2);
		int startShift = numl/2;
		if(odd) startShift+=1;

    	fgauss(MC,lambda,numl,wlines[1],0);  // gauss kernel is stored in global array GMAC  		 

    	//convolution spectro

		int i;
		if(configCrontrolFile.useFFT){
			if(filter){// if there is PSF filter convolve both gaussian and use the result as the signal to convolve
				for(i=0;i<numl;i++){ // copy gmac to
					inPSF_MAC[i] = (GMAC[i]) + 0 * _Complex_I;
				}
				fftw_execute(planForwardPSF_MAC);
				for(i=0;i<numl;i++){ // multiply both fft gaussians
					inMulMacPSF[i] = fftw_G_PSF[i] * (fftw_G_MAC_PSF[i]/numl);
				}
				fftw_execute(planBackwardPSF_MAC);			
				for(i=0,ishift=startShift;i<numl/2;i++,ishift++){
					inFilterMAC[ishift]= outConvFilters[i]*numl;
				}
				for(i=(numl/2),ishift=0;i<numl;i++,ishift++){
					inFilterMAC[ishift]= outConvFilters[i]*numl;
				}
				
			}
			else{
				for(i=0;i<numl;i++){
					inFilterMAC[i] = GMAC[i] + 0 * _Complex_I;
				}
			}
			fftw_execute(planFilterMAC);

			//convolucion
			for(il=0;il<4;il++){
				for(i=0;i<numl;i++){
					inSpectraFwMAC[i] = spectra[numl*il+i] + 0 * _Complex_I;
				}				 
				fftw_execute(planForwardMAC);
				for(i=0;i<numl;i++){
					inSpectraBwMAC[i]=(outSpectraFwMAC[i]/numl)*(outFilterMAC[i]/numl);
				}
				fftw_execute(planBackwardMAC);
				//shift: -numl/2				
				for(i=0,ishift=startShift;i<numl/2;i++,ishift++){
					spectra[ishift+il*numl]=creal(outSpectraBwMAC[i])*numl;
				}
				for(i=(numl/2),ishift=0;i<numl;i++,ishift++){
					spectra[ishift+il*numl]=creal(outSpectraBwMAC[i])*numl;
				}
			}
		}
		else
		{ // direct convolution 
			//convolucion de I
			if(filter){
				direct_convolution_double(GMAC, nlambda, G, nlambda);
			}

			// FOR USE CIRCULAR CONVOLUTION 
			for (i = 0; i < NPARMS; i++)
				convCircular(spectra + nlambda * i, GMAC, nlambda,spectra + nlambda * i);				
		}

   }
    


	if(!macApplied && filter){
		if(configCrontrolFile.useFFT){
			odd=(numl%2);
			
			int startShift = numl/2;
			if(odd) startShift+=1;

			for (i = 0; i < NPARMS; i++){
				for(j=0;j<numl;j++){
					inSpectraFwPSF[j] = spectra[(numl*i)+j] + 0 * _Complex_I;
				}
				fftw_execute(planForwardPSF);
				// multiplication fft results 
				for(j=0;j<numl;j++){
					inSpectraBwPSF[j] = (outSpectraFwPSF[j]/(numl)) * fftw_G_PSF[j];						
				}
				fftw_execute(planBackwardPSF);
				//shift: -numln/2
				for(j=0,ishift=startShift;j<(numl)/2;j++,ishift++){
					spectra[ishift+i*(numl)]=creal(outSpectraBwPSF[j])*(numl);
				}
				for(j=(numl)/2,ishift=0;j<(numl);j++,ishift++){
					spectra[ishift+i*(numl)]=creal(outSpectraBwPSF[j])*(numl);
				}
			}
		}
		else{ // direct convolution
			//convolucion de I
			REAL Ic;
			if(spectra[0]>spectra[nlambda - 1])
				Ic = spectra[0];
			else				
				Ic = spectra[nlambda - 1];
			
			direct_convolution_ic(spectra, nlambda, G, nlambda,Ic);

			//convolucion QUV
			for (i = 1; i < NPARMS; i++){
				direct_convolution(spectra + nlambda * i, nlambda, G, nlambda);  
			}
			
		}
	}

	if(slight!=NULL){  //ADDING THE STRAY-LIGHT PROFILE

		for(i=0;i<numl*NPARMS;i++){
			spectra_slight[i] = spectra[i];
			spectra[i] = spectra[i]*ALF+slight[i]*(1.0-ALF);
		}

	}

	return 1;
}


int funcionComponentFor_sinrf(REAL *u,int n_pi,int numl,REAL *wex,REAL *nuxB,REAL *fi_x, REAL *shi_x,PRECISION A,PRECISION MF)
{
	REAL *uu,*F,*H;
	int i,j;
	//component
	for(i=0;i<n_pi;i++){

		uu=uuGlobalInicial[uuGlobal+i];
		F=FGlobalInicial[HGlobal+i];
		H=HGlobalInicial[FGlobal+i];

		for(j=0;j<numl;j++){
			uu[j]=u[j]-nuxB[i]*MF;
		}

		fvoigt(A,uu,numl,H,F);
			
		for(j=0;j<numl;j++){
			fi_x[j]=fi_x[j]+wex[i]*H[j];
		}

		for(j=0;j<numl;j++){
			shi_x[j]=(shi_x[j]+(wex[i]*F[j]*2));
		}

	}//end for 
	uuGlobal=uuGlobal+n_pi;
	HGlobal=HGlobal+n_pi;
	FGlobal=FGlobal+n_pi;

	return 1;	
}





