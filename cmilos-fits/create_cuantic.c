
#include "defines.h"

int Cuanten(Cuantic *cuantic,double sl,double ll,double jl,double su,double lu,double ju,double fos);

//data -> [lines, sl,ll,jl,su,lu,ju, fos]  (fos only if lines>1)

Cuantic * create_cuantic(double * dat)
{	
	Cuantic * cuantic;
	
	int i;
	double lines;
	
	lines=dat[0];
	
	cuantic=calloc(lines,sizeof(Cuantic));

	
	for(i=0;i<lines;i++){
		Cuanten(&(cuantic[i]),dat[i*7+1],dat[i*7+2],dat[i*7+3],
				dat[i*7+4],
				dat[i*7+5],
				dat[i*7+6],
				(lines>1?dat[i*7+7]:1));
	}



	return cuantic;
}

int Cuanten(Cuantic *cuantic,double sl,double ll,double jl,double su,double lu,double ju,double fos){
								//SL(I),     SU(I),     LL(I),    LU(I),    JL(I),   JU(I)
										//,NUB,NUP,NUR,WEB,WEP,WER,GLO,GUP,GEF
//	;1-> LOW
//	;2-> UP
		
	int * m1,* m2,im,i,j;
	int jj,n_pi,n_sig;
	double g1,g2,geff;	
	double *pi,*sig1,*sig2,*mpi,*msig1,*msig2;	
	int ipi,isig1,isig2;
	double sumpi,sumsig1,sumsig2;
	
	
	//lande factors (with 'if' because j could be cero)	
	if (jl != 0)  
		g1=(3.0*jl*(jl+1)+sl*(sl+1)-ll*(ll+1))/(2*jl*(jl+1)); 
	else 
		g1=0;
		
	if (ju != 0) 
		g2=(3.0*ju*(ju+1.0)+su*(su+1.0)-lu*(lu+1.0))/(2.0*ju*(ju+1.0)); 
	else 
		g2=0;
	
	geff=(g1+g2)/2+(g1-g2)*(jl*(jl+1.)-ju*(ju+1.))/4;
	
//	printf("g1 : %f\n",g1);
//	printf("g2 : %f\n",g2);


	//magnetic quanten number Mlo Mup	
	m1=calloc(2*jl+1,sizeof(int));
	for(i=0;i<2*jl+1;i++){
		m1[i]=i-(int)jl;
//		printf("m1 (%d) : %d \n",i,m1[i]);
	}
	m2=calloc(2*ju+1,sizeof(int));
	for(i=0;i<2*ju+1;i++){
		m2[i]=i-(int)ju;
//		printf("m2 (%d) :%d\n",i,m2[i]);
	}

	
	n_pi=(2*(jl<ju?jl:ju))+1; //Number of pi components
	n_sig=jl+ju; //Number of sigma components	
		
//	printf("n_pi :%d\n",n_pi);
//	printf("n_sig :%d\n",n_sig);

	//BLUE COMPONENT => Mlo-Mup = +1
	//RED COMPONENT => Mlo-Mup = -1
	//CENTRAL COMPONENT => Mlo-Mup = 0

	pi=calloc(n_pi,sizeof(double));
	sig1=calloc(n_sig,sizeof(double));
	sig2=calloc(n_sig,sizeof(double));	
	mpi=calloc(n_pi,sizeof(double));	
	msig1=calloc(n_sig,sizeof(double));
	msig2=calloc(n_sig,sizeof(double));	
	
	
	//counters for the components
	ipi=0;
	isig1=0;
	isig2=0;
	
	jj=ju-jl;
	
	for(j=0;j<=2*jl;j++){
		for(i=0;i<=2*ju;i++){
			im=m2[i]-m1[j];
			switch(im){
			case 0:  	//M -> M  ;CENTRAL COMPONENT
				switch(jj){
				case -1:					
					//j -> j-1
	                pi[ipi]=jl*jl-m1[j]*m1[j];
	                break;
				case 0:
					//  j -> j
					pi[ipi]=m1[j]*m1[j];
					break;
				case 1:
					//  j -> j+1
                    pi[ipi]=(jl+1)*(jl+1)-m1[j]*m1[j];
                    break;
				}
                mpi[ipi]=g1*m1[j]-g2*m2[i];
                ipi=ipi+1;				
				break;
			case 1:		//M -> M+1  ;BLUE COMPONENT
				switch(jj){
				case -1:					
					//j -> j-1
                	sig1[isig1]=(jl-m1[j])*(jl-m1[j]-1)/4;
	                break;
				case 0:
					//  j -> j
					sig1[isig1]=(jl-m1[j])*(jl+m1[j]+1)/4;
					break;
				case 1:
					//  j -> j+1
					sig1[isig1]=(jl+m1[j]+1)*(jl+m1[j]+2)/4;
                    break;
				}
				msig1[isig1]=g1*m1[j]-g2*m2[i];
				isig1=isig1+1;				                				
				break;			
			case -1:		//M -> M-1   ;RED COMPONENT
				switch(jj){
				case -1:					
					//j -> j-1
                	sig2[isig2]=(jl+m1[j])*(jl+m1[j]-1)/4;
	                break;
				case 0:
					//  j -> j
					sig2[isig2]=(jl+m1[j])*(jl-m1[j]+1)/4;
					break;
				case 1:
					//  j -> j+1
					sig2[isig2]=(jl-m1[j]+1)*(jl-m1[j]+2)/4;
                    break;
				}
				msig2[isig2]=g1*m1[j]-g2*m2[i];
				isig2=isig2+1;				                				
				break;				
			}//end switch
		}
	}	
	
	//normalization OF EACH COMPONENT
	


	sumpi=0;
	for(i=0;i<n_pi;i++){	
		sumpi=sumpi+pi[i];
	}

//C_N(IL).wep(i)
	for(i=0;i<n_pi;i++){	
		pi[i]=pi[i]/sumpi;
	}
	
	sumsig1=0;
	for(i=0;i<n_sig;i++){		
		sumsig1=sumsig1+sig1[i];
	}
	for(i=0;i<n_sig;i++){	
		sig1[i]=sig1[i]/sumsig1;
	}

	sumsig2=0;
	for(i=0;i<n_sig;i++){		
		sumsig2=sumsig2+sig2[i];
	}
	for(i=0;i<n_sig;i++){	
		sig2[i]=sig2[i]/sumsig2;
	}


//Cuanten,S1,S2,L1,L2,J1,J2,msig1,mpi,msig2,sig1,pi, sig2,g1, g2 , geff
//CUANTEN,SL,SU,LL,LU,JL,JU,NUB,  NUP,NUR,  WEB, WEP,WER, GLO,GUP,GEF	
	
	cuantic->N_PI=n_pi;
	cuantic->N_SIG=n_sig;
	cuantic->NUB=msig1;
	cuantic->NUP=mpi;
	cuantic->NUR=msig2;	
	cuantic->WEB=sig1;
	cuantic->WEP=pi;	
	cuantic->WER=sig2;
	cuantic->GL=g1;
	cuantic->GU=g2;
	cuantic->GEFF=geff;
	cuantic->FO=fos;

	/*printf("geff %f\n",geff);
	printf("n_pi %d\n",n_pi);
	printf("n_sig %d\n",n_sig);
	printf("nub %f\n",msig1[0]);
	printf("nur %f\n",msig2[0]);
	printf("nup %f\n",mpi[0]);
	
	printf("web %f\n",sig1[0]);
	printf("wer %f\n",sig2[0]);
	printf("wep %f\n",pi[0]);

	printf("GL %f\n",g1);
	printf("GU %f\n",g2);*/

	free(m1);
	free(m2);	
	
	return 1; 
}

