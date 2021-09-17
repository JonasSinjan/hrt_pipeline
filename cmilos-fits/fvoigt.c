

#include "defines.h"

//pro fvoigt,damp,vv,h,f
int fvoigt(PRECISION damp,PRECISION *vv,int nvv,PRECISION *h, PRECISION *f){
	
	int i,j;
//	PRECISION *f,*h;
	static PRECISION a[]={122.607931777104326, 214.382388694706425, 181.928533092181549,
		   93.155580458138441, 30.180142196210589, 5.912626209773153,
		   0.564189583562615};

	static PRECISION b[]={122.60793177387535, 352.730625110963558, 457.334478783897737, 
	   	   348.703917719495792, 170.354001821091472, 53.992906912940207, 
		   10.479857114260399,1.};
	
	//static _Complex PRECISION z[NLAMBDA],zden[NLAMBDA],zdiv[NLAMBDA];
	_Complex PRECISION *z,*zden,*zdiv;
	
	
	z=calloc(nvv,sizeof(_Complex PRECISION));
	zden=calloc(nvv,sizeof(_Complex PRECISION));
	zdiv=calloc(nvv,sizeof(_Complex PRECISION));
	
	for(i=0;i<nvv;i++){
		z[i]= damp -fabs(vv[i]) * _Complex_I;
//		zden[i]=0 + 0 * _Complex_I;//z[i];
//		zdiv[i]=1 + 0 * _Complex_I;//z[i];
//	printf("z(i) %d , real %f imag %f \n",i,creal(z[i]),cimag(z[i]));
	}
	
	//
	for(i=0;i<nvv;i++){
		zden[i]=a[6];
	}

	for(j=5;j>=0;j--){
		for(i=0;i<nvv;i++){
			zden[i]=zden[i]*z[i]+a[j];
		}
	}

	//
	for(i=0;i<nvv;i++){
		zdiv[i]=z[i]+b[6];
	}

	for(j=5;j>=0;j--){
		for(i=0;i<nvv;i++){
			zdiv[i]=zdiv[i]*z[i]+b[j];
		}
	}

	
	for(i=0;i<nvv;i++){
		z[i]=zden[i]/zdiv[i];
	}

	free(zden);
	free(zdiv);
	
//	h=calloc(nvv,sizeof(PRECISION));
//	f=calloc(nvv,sizeof(PRECISION));

	for(i=0;i<nvv;i++){
		h[i]=creal(z[i]);
	}

	for(i=0;i<nvv;i++){
		f[i]= vv[i]>=0 ? (PRECISION) cimag(z[i])*0.5 : (PRECISION) cimag(z[i])*-0.5;
	}


	free(z);
	
	return 1;
	
}

