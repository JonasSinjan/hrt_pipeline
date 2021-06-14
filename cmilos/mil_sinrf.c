
#include "defines.h"
#include <string.h>

//pro me_der,param,wl,lmb,spectra,d_spectra,triplete=triplete,ah=ah,slight=slight,$
  //         filter=filter

int funcionComponent_sinrf(PRECISION *u,PRECISION ulos,PRECISION shift,int numl,PRECISION *fi_x,
		PRECISION *shi_x,PRECISION A);

int funcionComponentFor_sinrf(PRECISION *u,PRECISION ulos,int n_pi,int numl,double *wex,PRECISION *nuxB,PRECISION *fi_x,
												PRECISION *shi_x,PRECISION A,PRECISION MF);

char * concatenasin(char *a, int n,char*b);

PRECISION mean(PRECISION *dat, int numl);
/*
	E00	int eta0; // 0
	MF	int B;    
	VL	PRECISION vlos;
	LD	PRECISION dopp;
	A	PRECISION aa;
	GM	int gm; //5
	AZI	int az;
	B0	PRECISION S0;
	B1	PRECISION S1;
	MC	PRECISION mac; //9
		PRECISION alfa;		
*/


extern PRECISION * gp1,*gp2,*dt,*dti,*gp3,*gp4,*gp5,*gp6,*etai_2;
//extern PRECISION gp4_gp2_rhoq[NLAMBDA],gp5_gp2_rhou[NLAMBDA],gp6_gp2_rhov[NLAMBDA];

extern PRECISION *gp4_gp2_rhoq,*gp5_gp2_rhou,*gp6_gp2_rhov;

extern PRECISION CC,CC_2,sin_gm,azi_2,sinis,cosis,cosis_2,cosi,sina,cosa,sinda,cosda,sindi,cosdi,sinis_cosa,sinis_sina;
extern PRECISION *fi_p,*fi_b,*fi_r,*shi_p,*shi_b,*shi_r;
extern PRECISION *etain,*etaqn,*etaun,*etavn,*rhoqn,*rhoun,*rhovn;
extern PRECISION *etai,*etaq,*etau,*etav,*rhoq,*rhou,*rhov;
extern PRECISION *parcial1,*parcial2,*parcial3;
extern PRECISION *nubB,*nupB,*nurB;
PRECISION **uuGlobalInicial;
PRECISION **HGlobalInicial;
PRECISION **FGlobalInicial;
extern int FGlobal,HGlobal,uuGlobal;


int mil_sinrf(Cuantic *cuantic,Init_Model *initModel,double * wlines,int nwlines,double *lambda,int nlambda,PRECISION *spectra,
			double ah,double slight,int triplete,int filter)
{

	int offset,nterms,numl;
	int lineas;

	PRECISION E00,MF,VL,LD,A,GM,AZI,B0,B1,MC,ALF;
	int il,i,j;
	PRECISION E0;	
	PRECISION ulos,*g;
	//static PRECISION u[NLAMBDA];
	PRECISION *u;

	u = calloc(nlambda,sizeof(PRECISION));
	
    PRECISION *dgp1,*dgp2,*dgp3,*dgp4,*dgp5,*dgp6,*d_dt;
    int edge,numln,ishift,par;
    PRECISION *axis,*g1,*g2;
	_Complex *fftg,*ffts,*fftaux,*fftaux2,*fftd;    
	PRECISION  parcial;

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
	
	nterms=NTERMS; // realmente el tamaño necesario es 11 ??
	numl=nlambda;   	

	lineas=(int)wlines[0];



/*	etai=calloc(numl,sizeof(PRECISION));
	etaq=calloc(numl,sizeof(PRECISION));
	etau=calloc(numl,sizeof(PRECISION));
	etav=calloc(numl,sizeof(PRECISION));
	rhoq=calloc(numl,sizeof(PRECISION));
	rhou=calloc(numl,sizeof(PRECISION));
	rhov=calloc(numl,sizeof(PRECISION));	
*/

	for(j=0;j<numl;j++){
		etai[j]=1.0;
		etaq[j]=0;
		etau[j]=0;
		etav[j]=0;
		rhoq[j]=0;
		rhou[j]=0;
		rhov[j]=0;
	}

	//Definicion de ctes.
	//a radianes	

	AZI=AZI*CC;
	GM=GM*CC;

	sin_gm=sin(GM);
	cosi=cos(GM);
	sinis=sin_gm*sin_gm;
	cosis=cosi*cosi;
	cosis_2=(1+cosis)/2;
	azi_2=2*AZI;
	sina=sin(azi_2);
	cosa=cos(azi_2);
	sinda=cosa*CC_2;
	cosda=-sina*CC_2;
	sindi=cosi*sin_gm*CC_2;
	cosdi=-sin_gm*CC;
	sinis_cosa=sinis*cosa;
	sinis_sina=sinis*sina;

//    u=calloc(nlambda,sizeof(PRECISION));	
    for(il=0;il<lineas;il++) {
		//reserva de memoria para vectores auxiliares

/*		etain=calloc(numl,sizeof(PRECISION));
		etaqn=calloc(numl,sizeof(PRECISION));
		etaun=calloc(numl,sizeof(PRECISION));
		etavn=calloc(numl,sizeof(PRECISION));
		rhoqn=calloc(numl,sizeof(PRECISION));
		rhoun=calloc(numl,sizeof(PRECISION));
		rhovn=calloc(numl,sizeof(PRECISION));
*/

		etain=etain+nlambda*il*sizeof(PRECISION);
		etaqn=etaqn+nlambda*il*sizeof(PRECISION);
		etaun=etaun+nlambda*il*sizeof(PRECISION);
		etavn=etavn+nlambda*il*sizeof(PRECISION);
		rhoqn=rhoqn+nlambda*il*sizeof(PRECISION);
		rhoun=rhoun+nlambda*il*sizeof(PRECISION);
		rhovn=rhovn+nlambda*il*sizeof(PRECISION);

		//Line strength
	    E0=E00*cuantic[il].FO; //y sino se definio Fo que debe de pasar 0 o 1 ...??


	    //frecuency shift for v line of sight
	    ulos=(VL*wlines[il+1])/(VLIGHT*LD);


	    //doppler velocity	    
	    for(i=0;i<nlambda;i++){
	    	u[i]=((lambda[i]-wlines[il+1])/LD)-ulos;
	    }

/*		fi_p=calloc(numl,sizeof(PRECISION));//VNULO
		fi_b=calloc(numl,sizeof(PRECISION));//VNULO
		fi_r=calloc(numl,sizeof(PRECISION));//VNULO
		shi_p=calloc(numl,sizeof(PRECISION));//VNULO
		shi_b=calloc(numl,sizeof(PRECISION));//VNULO
		shi_r=calloc(numl,sizeof(PRECISION));//VNULO
*/

		fi_p=fi_p+nlambda*il*sizeof(PRECISION);
		fi_b=fi_b+nlambda*il*sizeof(PRECISION);
		fi_r=fi_r+nlambda*il*sizeof(PRECISION);
		shi_p=shi_p+nlambda*il*sizeof(PRECISION);
		shi_b=shi_b+nlambda*il*sizeof(PRECISION);
		shi_r=shi_r+nlambda*il*sizeof(PRECISION);

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

		if(!triplete){
			nubB=nubB+nlambda*il*sizeof(PRECISION);
			nurB=nurB+nlambda*il*sizeof(PRECISION);
			nupB=nupB+nlambda*il*sizeof(PRECISION);

//			parcial=((MF*(wlines[il+1]*wlines[il+1]))/LD)*(CTE4_6_13);
			parcial=(((wlines[il+1]*wlines[il+1]))/LD)*(CTE4_6_13);

			// printf("parcial %.8e\n",parcial);
			
			//caso multiplete						
			for(i=0;i<cuantic[il].N_SIG;i++){
				nubB[i]=parcial*cuantic[il].NUB[i]; // Spliting	
				// printf("nub %.8e\n",nubB[i]);
			}

			for(i=0;i<cuantic[il].N_PI;i++){
				nupB[i]=parcial*cuantic[il].NUP[i]; // Spliting			    
			}						

			for(i=0;i<cuantic[il].N_SIG;i++){
//				nurB[i]=parcial*cuantic[il].NUR[i]; // Spliting							
				nurB[i]=-nubB[(int)cuantic[il].N_SIG-(i+1)]; // Spliting
				// printf("nur %.8e\n",nurB[i]);				
			}						
//			Asignar_Puntero_Calculos_Compartidos(3,nubB,nupB,nurB);	

			uuGlobal=0;
			FGlobal=0;
			HGlobal=0;

			//central component					    					
			funcionComponentFor_sinrf(u,ulos,cuantic[il].N_PI,numl,cuantic[il].WEP,nupB,fi_p,shi_p,A,MF);

			//blue component
			funcionComponentFor_sinrf(u,ulos,cuantic[il].N_SIG,numl,cuantic[il].WEB,nubB,fi_b,shi_b,A,MF);
			//red component
			funcionComponentFor_sinrf(u,ulos,cuantic[il].N_SIG,numl,cuantic[il].WER,nurB,fi_r,shi_r,A,MF);



			uuGlobal=0;
			FGlobal=0;
			HGlobal=0;


		}
		else{
			printf("error :parte no terminada\n\n");
			exit(-1);
			//caso triplete
			PRECISION shift;

			shift=((MF*wlines[il+1]*wlines[il+1])/LD)*(CTE4_6_13*cuantic[il].GEFF);
			//central component			
			funcionComponent_sinrf(u,ulos,0,numl,fi_p,shi_p,A);
			//blue component
			funcionComponent_sinrf(u,ulos,shift,numl,fi_b,shi_b,A);
			//red component
			funcionComponent_sinrf(u,ulos,-shift,numl,fi_r,shi_r,A);
		}

		//dispersion profiles				
		PRECISION E0_2;
		E0_2=E0/2.0;

		for(i=0;i<numl;i++){
			parcial1[i]=fi_b[i]+fi_r[i];
			parcial2[i]=(E0_2)*(fi_p[i]-(parcial1[i])/2);
			parcial3[i]=(E0_2)*(shi_p[i]-(shi_b[i]+shi_r[i])/2);
		}

//		Asignar_Puntero_Calculos_Compartidos(3,parcial1,parcial2,parcial3);	
		

		PRECISION cosi_E0_2;
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
			etai[i]=etai[i]+etain[i];
			etaq[i]=etaq[i]+etaqn[i];
			etau[i]=etau[i]+etaun[i];
			etav[i]=etav[i]+etavn[i];
		}
		for(i=0;i<numl;i++){
			rhoq[i]=rhoq[i]+rhoqn[i];
			rhou[i]=rhou[i]+rhoun[i];
			rhov[i]=rhov[i]+rhovn[i];
		}
		
		// for(i=0;i<numl;i++){
			// printf("etain %.8e\n",etain[i]);
			

		// for(i=0;i<numl;i++){
			// printf("etaqn %.8e\n",etaqn[i]);
			
			
	//}
	
	

//		Asignar_Puntero_Calculos_Compartidos(7,etain,etaqn,etaun,etavn,rhoqn,rhoun,rhovn);

	} //end for

//	Asignar_Puntero_Calculos_Compartidos(7,etai,etaq,etau,etav,rhoq,rhou,rhov);

    //liberamos memoria
	free(u);
            
    //Los parametros de Stokes estan normalizados a la intensidad en el continuo (no) ??
    
	
	// for(i=0;i<numl;i++){
			// printf("etai %.8e\n",etai[i]);
	// }

	for(i=0;i<numl;i++){    	
		etai_2[i]=etai[i]*etai[i];
    } 

    for(i=0;i<numl;i++){
		PRECISION auxq,auxu,auxv;
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
   
	//static PRECISION dtiaux[nlambda];
	
	PRECISION *dtiaux;
	dtiaux = calloc(nlambda,sizeof(PRECISION));

   	for(i=0;i<numl;i++){
		gp4_gp2_rhoq[i]=gp4[i]+rhoq[i]*gp2[i];
		gp5_gp2_rhou[i]=gp5[i]+rhou[i]*gp2[i];
		gp6_gp2_rhov[i]=gp6[i]+rhov[i]*gp2[i];
	}

    for(i=0;i<numl;i++)
		dtiaux[i]=dti[i]*(B1);
    //espectro
    for(i=0;i<numl;i++){
    	spectra[i]=B0-dtiaux[i]*etai[i]*gp3[i];
//    }
//    for(i=0;i<numl;i++){
        spectra[i+numl  ]=(dtiaux[i]*(gp4_gp2_rhoq[i]));
//    }
//    for(i=0;i<numl;i++){
        spectra[i+numl*2]=(dtiaux[i]*(gp5_gp2_rhou[i]));
//    }
//    for(i=0;i<numl;i++){
        spectra[i+numl*3]=(dtiaux[i]*(gp6_gp2_rhov[i]));
    }

	free(dtiaux);
	
//	Asignar_Puntero_Calculos_Compartidos(9,etai_2,gp1,gp2,gp3,gp4,gp5,gp6,dt,dti);

 /*   free(etai);
	free(etai_2);
    free(etaq);
    free(etau);
    free(etav);

	free(rhoq);
    free(rhou);
    free(rhov);
	free(gp1);
    free(gp2);
    free(gp3);
    free(gp4);
    free(gp5);
    free(gp6);
    free(dt);
    free(dti);        
*/

   //MACROTURBULENCIA            
/*    edge=(numl%2)+1;
    numln=numl-edge+1;

   
    axis=calloc(numln,sizeof(double));
    

    for(i=0;i<numln;i++){
    	axis[i]=lambda[i];
    }

//    g=calloc(numln,sizeof(double));    
    
	char *buf;

    if(MC > 0.0001){
    	//la 9 respecto MC
    	//convolucion del espectro original
    	g=fgauss(MC,axis,numl,wlines[1],0);

    	fftg=fft_d(g,numln,FFT_FORWARD);

		fftaux=calloc(numln,sizeof(_Complex));

    	//convolucion
    	for(il=0;il<4;il++){
    		if(fabs(mean(spectra+numl*il,numl)) > 1.e-25	){
    			ffts=fft_d(spectra+numl*il,numln,FFT_FORWARD);
    			

    			for(i=0;i<numln;i++){
    				fftaux[i]=ffts[i]*fftg[i];
    			}

		
    			fftaux2=fft_c(fftaux,numln,FFT_BACKWARD);

    			//shift: -numln/2
    			for(i=0,ishift=numln/2;i<numln/2;i++,ishift++){
    				spectra[ishift+il*numl]=creal(fftaux2[i])*numln;
    			}
    			for(i=numln/2,ishift=0;i<numln;i++,ishift++){
    				spectra[ishift+il*numl]=creal(fftaux2[i])*numln;
    			}

				free(ffts);
				free(fftaux2);
    		}
    	}
		free(fftaux);


    }//end if(MC > 0.0001)
    	 
    if(filter=1){
    	//falta
    }
    
    //if(slight)
        
	//liberar memoria	 ??

	free(axis);
*/

	return 1;
}


/*
 * 
 */

int funcionComponentFor_sinrf(PRECISION *u,PRECISION ulos,int n_pi,int numl,double *wex,PRECISION *nuxB,PRECISION *fi_x,
												PRECISION *shi_x,PRECISION A,PRECISION MF)
{
	PRECISION *uu,*F,*H;
	int i,j;

//	H=NULL;
//	F=NULL;

//	printf("MILSINRF\n");

	//component
	for(i=0;i<n_pi;i++){

/*		uu=uuGlobal+nlambda*sizeof(PRECISION)*i;
		H=HGlobal+nlambda*sizeof(PRECISION)*i;
		F=FGlobal+nlambda*sizeof(PRECISION)*i;
*/
		uu=uuGlobalInicial[uuGlobal+i];
		F=FGlobalInicial[HGlobal+i];
		H=HGlobalInicial[FGlobal+i];
		//uu=calloc(numl,sizeof(PRECISION));

/*		printf("uu %d\n",uu);
		printf("H %d\n",H);
		printf("F %d\n",F);
*/
		for(j=0;j<numl;j++){
			uu[j]=u[j]-nuxB[i]*MF;
		}

		//free(H);
		//free(F);

		fvoigt(A,uu,numl,H,F);

		/*
		printf("--->>H \n");
		for(j=0;j<numl;j++)
			printf("%.16e\n",H[j]);

		printf("--->>F \n");
		for(j=0;j<numl;j++)
			printf("%.16e\n",F[j]);
		*/
			
		for(j=0;j<numl;j++){
			fi_x[j]=fi_x[j]+wex[i]*H[j];
		}

		for(j=0;j<numl;j++){
			shi_x[j]=(shi_x[j]+(wex[i]*F[j]*2));
		}

//		Asignar_Puntero_Calculos_Compartidos(2,H,F);

	}//end for 
	uuGlobal=uuGlobal+n_pi;
	HGlobal=HGlobal+n_pi;
	FGlobal=FGlobal+n_pi;

//	for(j=0;j<numl;j++){
	//	shi_x[j]=(shi_x[j])*2;//SHI_P=2d0*SHI_P
	//}

//	Asignar_Puntero_Calculos_Compartidos(2,fi_x,shi_x);

	/*free(uu);
	free(H);
	free(F);*/

	return 1;	
}

/*
 * 
 */
int funcionComponent_sinrf(PRECISION *u,PRECISION ulos,PRECISION shift,int numl,PRECISION *fi_x,
		PRECISION *shi_x,PRECISION A){
	
	PRECISION *H,*F,*uu;
	int j;
					
	uu=calloc(numl,sizeof(PRECISION));
	H=calloc(numl,sizeof(PRECISION));
	F=calloc(numl,sizeof(PRECISION));
	

	for(j=0;j<numl;j++){
		uu[j]=u[j]-ulos+shift;
	}	
	
	fvoigt(A,uu,numl,H,F);

	for(j=0;j<numl;j++){
		fi_x[j]=H[j];
	}

	for(j=0;j<numl;j++){
		shi_x[j]=F[j]*2.0;
	}


	free(uu);
	free(H);
	free(F);


	return 1;
	
}



char * concatenasin(char *a, int n,char*b){

	char *buf;
	char *c;

	buf=calloc(strlen(a)+strlen(b)+2,sizeof(char));
	c=calloc(1,sizeof(char));

	c[0]=(char)48+n; 	// "0" ascii 48

	buf=strcpy(buf,a);
	buf[strlen(a)]=95;   // "_" ascii 95

	buf=strcat(buf,c);
	buf=strcat(buf,b);

	free(c);
	return buf;

}
