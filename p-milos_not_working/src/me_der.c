#include <time.h>
#include "defines.h"
#include "lib.h"
#include <string.h>
#include "milosUtils.h"
#include "convolution.h"
#include <complex.h>
#include <fftw3.h> //siempre a continuacion de complex.h

/**
 * Calculate central components
 * */
int funcionComponentFor(int n_pi,PRECISION iwlines,int numl,REAL *wex,REAL *nuxB,REAL *dfi,REAL *dshi,PRECISION LD,PRECISION A,int desp);


extern REAL *dtaux, *etai_gp3, *ext1, *ext2, *ext3, *ext4;
extern REAL *gp4_gp2_rhoq,*gp5_gp2_rhou,*gp6_gp2_rhov;
extern REAL * gp1,*gp2,*dt,*dti,*gp3,*gp4,*gp5,*gp6,*etai_2;
extern REAL *dgp1,*dgp2,*dgp3,*dgp4,*dgp5,*dgp6,*d_dt;
extern REAL * d_ei,*d_eq,*d_eu,*d_ev,*d_rq,*d_ru,*d_rv;
extern REAL *dfi,*dshi;
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
extern PRECISION *GMAC,*GMAC_DERIV; // VECTOR WITH GAUSSIAN CREATED FOR CONVOLUTION 
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
extern int NTERMS;



int me_der(Cuantic *cuantic,Init_Model *initModel,PRECISION * wlines,PRECISION *lambda,int nlambda,REAL *d_spectra,REAL *spectra, REAL * spectra_slight, REAL ah,REAL * slight,int filter, int * fix)
{

	int nterms,numl;
	int lineas;
	PRECISION E00,LD,A,B1,MC,ALF;
	int il,i,j,k;
	PRECISION E0;	
   int odd,ishift,par;
    
	E00=initModel->eta0; 
	LD=initModel->dopp;
	A=initModel->aa;
	B1=-((initModel->S1)*ah);
	MC=initModel->mac;
	ALF=initModel->alfa;

	nterms=NTERMS; 
	numl=nlambda;
	lineas=wlines[0];

	


    
    for(il=0;il<lineas;il++) {
		//Line strength
	    E0=E00*cuantic[il].FO; 

		fi_p=fi_p+nlambda*il*sizeof(REAL);
		fi_b=fi_b+nlambda*il*sizeof(REAL);
		fi_r=fi_r+nlambda*il*sizeof(REAL);
		shi_p=shi_p+nlambda*il*sizeof(REAL);
		shi_b=shi_b+nlambda*il*sizeof(REAL);
		shi_r=shi_r+nlambda*il*sizeof(REAL);

		//central component					    					
		funcionComponentFor(cuantic[il].N_PI,wlines[il+1],numl,cuantic[il].WEP,nupB,dfi,dshi,LD,A,0);

		//blue component
		funcionComponentFor(cuantic[il].N_SIG,wlines[il+1],numl,cuantic[il].WEB,nubB,dfi,dshi,LD,A,1);

		//red component
		funcionComponentFor(cuantic[il].N_SIG,wlines[il+1],numl,cuantic[il].WER,nurB,dfi,dshi,LD,A,2);

		for(i=0;i<numl;i++){
			d_ei[i]=etain[i]/E0;
		}
		for(i=0;i<numl;i++){
			d_eq[i]=etaqn[i]/E0;
		}
		for(i=0;i<numl;i++){
			d_eu[i]=etaun[i]/E0;
		}
		for(i=0;i<numl;i++){
			d_ev[i]=etavn[i]/E0;
		}
		for(i=0;i<numl;i++){
			d_rq[i]=rhoqn[i]/E0;
		}
		for(i=0;i<numl;i++){
			d_ru[i]=rhoun[i]/E0;
		}
		for(i=0;i<numl;i++){
			d_rv[i]=rhovn[i]/E0;
		}

		//dispersion profiles
		REAL E0_2;
		REAL cosi_2_E0;
		sinis_cosa=E0*sinis_cosa/2;
		sinis_sina=E0*sinis_sina/2;
		E0_2=E0/2.0;
		REAL sindi_cosa,sindi_sina,cosdi_E0_2,cosis_2_E0_2,sinis_E0_2;
		sindi_cosa=sindi*cosa;
		sindi_sina=sindi*sina;
		cosdi_E0_2=(E0_2)*cosdi;
		cosi_2_E0=E0*cosi/2.0;
		cosis_2_E0_2=cosis_2*E0_2;
		sinis_E0_2=sinis*E0_2;
	
		
		for(j=1;j<5;j++){
			//derived from the dispersion profiles with respect to de B,VL,LDOPP,A 
			for(i=0;i<numl;i++){
				REAL dfisum,aux;
				dfisum=	dfi[i + (j-1)*numl+ (numl*4)]+dfi[i + (j-1)*numl + (numl*4*2)];
				d_ei[j*numl+i] = (dfi[i+ (j-1)*numl] * sinis_E0_2 + dfisum * cosis_2_E0_2);
				
				aux=dfi[(j-1)*numl+i]-dfisum/2;
				d_eq[j*numl+i]=(aux)*sinis_cosa;
				d_eu[j*numl+i]=(aux)*sinis_sina;
			}
			for(i=0;i<numl;i++){
				d_ev[j*numl+i]= (dfi[(j-1)*numl+i+(numl*4*2)]-dfi[(j-1)*numl+i+(numl*4)])*cosi_2_E0;
			}
		}
		for(j=1;j<5;j++){
			for(i=0;i<numl;i++){
				REAL aux=dshi[(j-1)*numl+i]-(dshi[(j-1)*numl+i+(numl*4)]+dshi[(j-1)*numl+i+(numl*4*2)])/2.0;
				d_rq[j*numl+i]=(aux)*sinis_cosa;
				d_ru[j*numl+i]=(aux)*sinis_sina;
			}
			for(i=0;i<numl;i++){
				d_rv[j*numl+i]=((dshi[(j-1)*numl+i+(numl*4*2)]-dshi[(j-1)*numl+i+(numl*4)]))*cosi_2_E0;
			}	
		}
		
		//derived from the dispersion profiles with respect to de GAMMA
		REAL cosi_cosdi,sindi_E0_2;
		cosi_cosdi=cosi*cosdi*E0_2;
		sindi_E0_2=sindi*E0_2;
		for(i=0;i<numl;i++){
			d_ei[5*numl+i]=fi_p[i]*sindi_E0_2+(parcial1[i])*cosi_cosdi;
		}
		for(i=0;i<numl;i++){
			d_eq[5*numl+i]=parcial2[i]*sindi_cosa;
		}
		for(i=0;i<numl;i++){
			d_eu[5*numl+i]=parcial2[i]*sindi_sina;
		}
		for(i=0;i<numl;i++){
			d_ev[5*numl+i]=(fi_r[i]-fi_b[i])*cosdi_E0_2;
		}
		for(i=0;i<numl;i++){
			d_rq[5*numl+i]=parcial3[i]*sindi_cosa;
		}
		for(i=0;i<numl;i++){
			d_ru[5*numl+i]=parcial3[i]*sindi_sina;
		}
		for(i=0;i<numl;i++){
			d_rv[5*numl+i]=(shi_r[i]-shi_b[i])*cosdi_E0_2;
		}
		

		//derivadas de los perfiles de dispersion respecto de AZI
		REAL sinis_cosda,sinis_sinda;
		sinis_cosda=sinis*cosda;
		sinis_sinda=sinis*sinda;		
		for(i=0;i<numl;i++){				
			d_eq[6*numl+i]=parcial2[i]*sinis_cosda;
		}

		for(i=0;i<numl;i++){
			d_eu[6*numl+i]=parcial2[i]*sinis_sinda;
		}
		for(i=0;i<numl;i++){
			d_rq[6*numl+i]=parcial3[i]*sinis_cosda;
		}
		for(i=0;i<numl;i++){
			d_ru[6*numl+i]=parcial3[i]*sinis_sinda;
		}

	} //end for



    //derivative of spectra with respect to  E0,MF,VL,LD,A,gamma,azi

   for(i=0;i<numl;i++)
		dtaux[i]=(B1)/(dt[i]*dt[i]);

   for(i=0;i<numl;i++){
		etai_gp3[i]=etai[i]*gp3[i];
	}

   for(i=0;i<numl;i++){
		REAL aux=2*etai[i];
		ext1[i]=aux*etaq[i]+etav[i]*rhou[i]-etau[i]*rhov[i];
		ext2[i]=aux*etau[i]+etaq[i]*rhov[i]-etav[i]*rhoq[i];
		ext3[i]=aux*etav[i]+etau[i]*rhoq[i]-etaq[i]*rhou[i];
		ext4[i]=aux*gp1[i];
	}
	
	
    for(il=0;il<7;il++){
    	for(i=0;i<numl;i++){
    		dgp1[i]=2.0*(etai[i]*d_ei[i+numl*il]-etaq[i]*d_eq[i+numl*il]-etau[i]*d_eu[i+numl*il]-etav[i]*d_ev[i+numl*il]  
				 +rhoq[i]*d_rq[i+numl*il]+rhou[i]*d_ru[i+numl*il]+rhov[i]*d_rv[i+numl*il]);
    	}
    
    	for(i=0;i<numl;i++){
    		dgp2[i]=rhoq[i]*d_eq[i+numl*il]+etaq[i]*d_rq[i+numl*il]+rhou[i]*d_eu[i+numl*il]+etau[i]*d_ru[i+numl*il]+
    		                    rhov[i]*d_ev[i+numl*il]+etav[i]*d_rv[i+numl*il];
    	}
	
    	for(i=0;i<numl;i++){
    		d_dt[i]=ext4[i]*d_ei[i+numl*il]+etai_2[i]*dgp1[i]-2*gp2[i]*dgp2[i];
    	}
		
    	for(i=0;i<numl;i++){
    		dgp3[i]=2.0*(etai[i]*d_ei[i+numl*il]+rhoq[i]*d_rq[i+numl*il]+rhou[i]*d_ru[i+numl*il]+rhov[i]*d_rv[i+numl*il]);
    	}

    	for(i=0;i<numl;i++){    		
    		d_spectra[i+numl*il]=-(((d_ei[i+numl*il]*gp3[i]+etai[i]*dgp3[i])*dt[i]-d_dt[i]*etai_gp3[i])*(dtaux[i]));
    	}
	
    	for(i=0;i<numl;i++){
    		dgp4[i]=d_ei[i+numl*il]*(ext1[i])+(etai_2[i])*d_eq[i+numl*il]+
    		etai[i]*(rhou[i]*d_ev[i+numl*il]+etav[i]*d_ru[i+numl*il]-rhov[i]*d_eu[i+numl*il]-etau[i]*d_rv[i+numl*il]);
    	}
		
    	for(i=0;i<numl;i++){
    		d_spectra[i+numl*il+numl*nterms]=((dgp4[i]+d_rq[i+numl*il]*gp2[i]+rhoq[i]*dgp2[i])*dt[i]-
    				d_dt[i]*(gp4_gp2_rhoq[i]))*(dtaux[i]);
    	}    
	
    	for(i=0;i<numl;i++){
    		dgp5[i]=d_ei[i+numl*il]*(ext2[i])+(etai_2[i])*d_eu[i+numl*il]+
    		etai[i]*(rhov[i]*d_eq[i+numl*il]+etaq[i]*d_rv[i+numl*il]-rhoq[i]*d_ev[i+numl*il]-etav[i]*d_rq[i+numl*il]);
    	}

    	for(i=0;i<numl;i++){
    		d_spectra[i+numl*il+(numl*nterms*2)]=((dgp5[i]+d_ru[i+numl*il]*gp2[i]+rhou[i]*dgp2[i])*dt[i]-
    				d_dt[i]*(gp5_gp2_rhou[i]))*(dtaux[i]);
    	}    

    	for(i=0;i<numl;i++){
    		dgp6[i]=d_ei[i+numl*il]*(ext3[i])+(etai_2[i])*d_ev[i+numl*il]+
    		etai[i]*(rhoq[i]*d_eu[i+numl*il]+etau[i]*d_rq[i+numl*il]-rhou[i]*d_eq[i+numl*il]-etaq[i]*d_ru[i+numl*il]);
    	}

		for(i=0;i<numl;i++){
    		d_spectra[i+numl*il+(numl*nterms*3)]=((dgp6[i]+d_rv[i+numl*il]*gp2[i]+rhov[i]*dgp2[i])*dt[i]-
    				d_dt[i]*(gp6_gp2_rhov[i]))*(dtaux[i]);
    	} 

		
    }
	
    //LA 7-8 RESPECTO B0 Y B1

    for(i=0;i<numl;i++)
		dti[i]=-(dti[i]*ah);

    for(i=0;i<numl;i++){
    	d_spectra[i+numl*8]=-dti[i]*etai_gp3[i];
    }

    for(i=0;i<numl;i++){
    	d_spectra[i+numl*8+(numl*nterms)]= dti[i]*(gp4_gp2_rhoq[i]);   		
    }

    for(i=0;i<numl;i++){
    	d_spectra[i+numl*8+(numl*nterms*2)]= dti[i]*(gp5_gp2_rhou[i]);
    }

    for(i=0;i<numl;i++){
    	d_spectra[i+numl*8+(numl*nterms*3)]= dti[i]*(gp6_gp2_rhov[i]);
    }

	//S0
   for(i=0;i<numl;i++){
    	d_spectra[i+numl*7]=1;
    	d_spectra[i+numl*7+(numl*nterms)]=0;
    	d_spectra[i+numl*7+(numl*nterms*2)]=0;
    	d_spectra[i+numl*7+(numl*nterms*3)]=0;
	}

	//azimuth stokes I &V
    for(i=0;i<numl;i++){
    	d_spectra[i+numl*6]=0;
    	d_spectra[i+numl*6+(numl*nterms*3)]=0;				  
	}
	
    //MACROTURBULENCIA
                
	int macApplied = 0;
    if(MC > 0.0001){
		 
		macApplied = 1;
		odd=(numl%2);		
		int startShift = numl/2;
		if(odd) startShift+=1;		
		
    	//convolution original spectro
    	
		fgauss(MC,lambda,numl,wlines[1],0);  // Gauss Function stored in global variable GMAC 
		// VARIABLES USED TO CALCULATE DERIVATE OF G1
		PRECISION ild = (wlines[1] * MC) / 2.99792458e5; //Sigma
		PRECISION centro = lambda[(int)numl / 2];		  //center of the axis

		if(configCrontrolFile.useFFT){
			if(filter){// if there is PSF filter convolve both gaussian and use the result as the signal to convolve
				for(i=0;i<numl;i++){ // copy gmac to
					inPSF_MAC[i] = (GMAC[i]) + 0 * _Complex_I;
					inPSF_MAC_DERIV[i] = (GMAC[i] / MC * ((((lambda[i] - centro) / ild) * ((lambda[i] - centro) / ild)) - 1.0)) + 0 * _Complex_I;
				}
				fftw_execute(planForwardPSF_MAC);
				fftw_execute(planForwardPSF_MAC_DERIV);
				for(i=0;i<numl;i++){ // multiply both fft gaussians
					inMulMacPSF[i] = fftw_G_PSF[i] * (fftw_G_MAC_PSF[i]/numl);
					inMulMacPSFDeriv[i] = fftw_G_PSF[i] * (fftw_G_MAC_DERIV_PSF[i]/numl);
				}
				fftw_execute(planBackwardPSF_MAC);
				fftw_execute(planBackwardPSF_MAC_DERIV);
				for(i=0,ishift=startShift;i<numl/2;i++,ishift++){
					inFilterMAC[ishift]= outConvFilters[i]*numl;
					inFilterMAC_DERIV[ishift]= outConvFiltersDeriv[i]*numl;
				}
				for(i=(numl/2),ishift=0;i<numl;i++,ishift++){
					inFilterMAC[ishift]= outConvFilters[i]*numl;
					inFilterMAC_DERIV[ishift]= outConvFiltersDeriv[i]*numl;
				}
			}
			else{
				for(i=0;i<numl;i++){
					inFilterMAC[i] = GMAC[i] + 0 * _Complex_I;
					inFilterMAC_DERIV[i] = (GMAC[i] / MC * ((((lambda[i] - centro) / ild) * ((lambda[i] - centro) / ild)) - 1.0)) + 0 * _Complex_I;
				}
			}
			fftw_execute(planFilterMAC);
			fftw_execute(planFilterMAC_DERIV);

			
			for(il=0;il<4;il++){
				for(i=0;i<numl;i++){
					inSpectraFwMAC[i] = spectra[numl*il+i] + 0 * _Complex_I;
				} 
				fftw_execute(planForwardMAC);
				for(i=0;i<numl;i++){
					inSpectraBwMAC[i]=(outSpectraFwMAC[i]/numl)*(outFilterMAC_DERIV[i]/numl);
				}
				fftw_execute(planBackwardMAC);
				//shift: -numl/2
				for(i=0,ishift=startShift;i<numl/2;i++,ishift++){
					d_spectra[ishift+9*numl+numl*nterms*il]=creal(outSpectraBwMAC[i])*numl;
				}
				for(i=(numl/2),ishift=0;i<numl;i++,ishift++){
					d_spectra[ishift+9*numl+numl*nterms*il]=creal(outSpectraBwMAC[i])*numl;
				}		
			}
			
			for(par=0;par<4;par++){
				//Go until the eigth because macros is not convolve 

				for(il=0;il<9;il++){
					if(il!=7){
						for(i=0;i<numl;i++){
							inSpectraFwMAC[i] = d_spectra[(numl*il+numl*nterms*par)+i] + 0 * _Complex_I;
						} 
						fftw_execute(planForwardMAC);
						for(i=0;i<numl;i++){
							inSpectraBwMAC[i]=(outSpectraFwMAC[i]/numl)*(outFilterMAC[i]/numl);
						}
						fftw_execute(planBackwardMAC);  			

						//shift 
						for(i=0,ishift=startShift;i<numl/2;i++,ishift++){
							d_spectra[ishift+il*numl+numl*nterms*par]=creal(outSpectraBwMAC[i])*numl;
						}
						for(i=(numl/2),ishift=0;i<numl;i++,ishift++){
							d_spectra[ishift+il*numl+numl*nterms*par]=creal(outSpectraBwMAC[i])*numl;
						}  
					}
				}
			}		 

		} // END USE FFT
		else{ // USE DIRECT CONVOLUTION 
			for(i=0;i<numl;i++){
				GMAC_DERIV[i] = (GMAC[i] / MC * ((((lambda[i] - centro) / ild) * ((lambda[i] - centro) / ild)) - 1.0));
			}
			if(filter){
				// convolve both gaussians and use this to convolve this with spectra 
				direct_convolution_double(GMAC_DERIV, nlambda, G, nlambda); 
				direct_convolution_double(GMAC, nlambda, G, nlambda); 
			}

			for(il=0;il<4;il++){
				convCircular(spectra+nlambda*il, GMAC_DERIV, nlambda,d_spectra+(9*numl)+(numl*nterms*il)); 
			}

			
			for(il=0;il<4;il++){
				convCircular(spectra+nlambda*il, GMAC_DERIV, nlambda,d_spectra+(9*numl)+(numl*nterms*il)); 
			}

			for (j = 0; j < NPARMS; j++)
			{
				for (i = 0; i < 9; i++)
				{
					if (i != 7)																															 //no convolucionamos S0
						convCircular(d_spectra + (nlambda * i) + (nlambda * NTERMS * j), GMAC, nlambda,d_spectra + (nlambda * i) + (nlambda * NTERMS * j)); 
				}
			}

		}  // END DIRECT CONVOLUTION 
   
    }//end if(MC > 0.0001)
	if(!macApplied && filter){
		int h;
		if(configCrontrolFile.useFFT){
			int odd=(numl%2);
			int startShift = (numl)/2;
			if(odd) startShift+=1;

			for (j = 0; j < NPARMS; j++)
			{
				for (i = 0; i < NTERMS; i++)
				{
					if (i != 7)	{																														 //no convolucionamos S0
						// copy to inSpectra
						for(k=0;k<(numl);k++){
							inSpectraFwPSF[k] = d_spectra[(numl * i + numl * NTERMS * j) + k] + 0 * _Complex_I;
						}
						fftw_execute(planForwardPSF);
						for(h=0;h<numl;h++){
							inSpectraBwPSF[h] = (outSpectraFwPSF[h]/numl) * fftw_G_PSF[h];
						}
						fftw_execute(planBackwardPSF);   			
						//shift 
						for(h=0,ishift=startShift;h<numl/2;h++,ishift++){
							d_spectra[ishift+ numl * i + numl * NTERMS * j]=creal(outSpectraBwPSF[h])*numl;
						}
						for(h=(numl/2),ishift=0;h<numl;h++,ishift++){
							d_spectra[ishift+numl * i + numl * NTERMS * j]=creal(outSpectraBwPSF[h])*numl;
						}
					}
				}
			}
		}
		else{ // DIRECT CONVOLUTION 
			REAL Ic;
			
			for (i = 0; i < NTERMS; i++)
			{
				// invert continuous
				if (i != 7){	
					if(d_spectra[(nlambda * i)]>d_spectra[(nlambda * i) + (nlambda - 1)])
						Ic = d_spectra[(nlambda * i)];	
					else
						Ic = d_spectra[(nlambda * i) + (nlambda - 1)];

					direct_convolution_ic(d_spectra + (nlambda * i), nlambda, G, nlambda,Ic);
				}	
			}

			for (j = 1; j < NPARMS; j++)
			{
				for (i = 0; i < NTERMS; i++)
				{
					if (i != 7)																															 //no convolucionamos S0
						direct_convolution(d_spectra + (nlambda * i) + (nlambda * NTERMS * j), nlambda, G, nlambda);
				}
			}

		}
	}
	// stray light factor 
	if(slight!=NULL){
		// Response Functions 
	   for(par=0;par<NPARMS;par++){
	    	for(il=0;il<NTERMS;il++){
				for(i=0;i<numl;i++){
					d_spectra[(numl*il+numl*nterms*par)+i]=d_spectra[(numl*il+numl*nterms*par)+i]*ALF;
					if(NTERMS==11){
						if(il==10){ //Magnetic filling factor Response function
							d_spectra[(numl*il+numl*nterms*par)+i]=spectra_slight[numl*par+i]-slight[numl*par+i];
						}
					}
					else{
						if(fix[9]){ // if there is mac 
							if(il==10){ //Magnetic filling factor Response function
								d_spectra[numl*il+numl*nterms*par+i]=spectra_slight[numl*par+i]-slight[numl*par+i];
							}
						}
						else{
							if(il==9){ //Magnetic filling factor Response function
								d_spectra[numl*il+numl*nterms*par+i]=spectra_slight[numl*par+i]-slight[numl*par+i];
							}
						}
					}
				}
	    	}
    	}
	}

	
	return 1;
	
}


/*
 * 
 */
int funcionComponentFor(int n_pi,PRECISION iwlines,int numl,REAL *wex,REAL *nuxB,REAL *dfi,REAL *dshi,PRECISION LD,PRECISION A,int desp)
{
	REAL *uu;
	int i,j;
	REAL dH_u[numl],dF_u[numl],auxCte[numl];
	
	REAL *H,*F;
	
	for(j=0;j<numl;j++){
		auxCte[j]=(-iwlines)/(VLIGHT*LD);
	}


	//component
	for(i=0;i<n_pi;i++){

		uu=uuGlobalInicial[uuGlobal+i];
		F=FGlobalInicial[HGlobal+i];
		H=HGlobalInicial[FGlobal+i];

		for(j=0;j<numl;j++){
			dH_u[j]=((4*A*F[j])-(2*uu[j]*H[j]))*wex[i];
		}

		for(j=0;j<numl;j++){
			dF_u[j]=(RR-A*H[j]-2*uu[j]*F[j])*wex[i];//
		}
		
		for(j=0;j<numl;j++){
			uu[j]=-uu[j]/LD;
		}


		for(j=0;j<numl;j++){
			dfi[j+(numl*4*desp)]=dH_u[j]*(-nuxB[i]);
		}
		
		for(j=0;j<numl;j++){
			dfi[numl+j+(numl*4*desp)]=dH_u[j]*auxCte[j];
		}

		for(j=0;j<numl;j++){
			dfi[2*numl+j+(numl*4*desp)]=(dH_u[j]*uu[j]); 											
		}								

		for(j=0;j<numl;j++){
			dfi[3*numl+j+(numl*4*desp)]=(-2*dF_u[j]);						
		}
		
		//dshi
		for(j=0;j<numl;j++){
			dshi[j+(numl*4*desp)]=(dF_u[j])*(-nuxB[i]);
		}
						
		for(j=0;j<numl;j++){
			dshi[numl+j+(numl*4*desp)]=dF_u[j]*auxCte[j];
		}

		for(j=0;j<numl;j++){
			dshi[2*numl+j+(numl*4*desp)]=(dF_u[j]*uu[j]); 											
		}								
		
		for(j=0;j<numl;j++){
			dshi[3*numl+j+(numl*4*desp)]=(dH_u[j]/2);						
		}									

	}

	uuGlobal=uuGlobal+n_pi;
	HGlobal=HGlobal+n_pi;
	FGlobal=FGlobal+n_pi;

	return 1;	
}
