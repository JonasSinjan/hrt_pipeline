
#include "defines.h"


/*
 * 
 * deriv : 1 true, 0 false
 */

//;this function builds a gauss function
//;landa(amstrong) ;Central wavelength 
//;eje(amstrong) ;Wavelength axis
//;macro ;Macroturbulence in km/s


double * fgauss(double MC, double * eje,int neje,double landa,int deriv){
//int fgauss(double MC, double * eje,int neje,double landa,int deriv,double * mtb,int nmtb){	

	double centro,*mtb;
	double ild;
	double * term,*loai;
	int i;
	int nloai,nmtb;
	double cte;
	
	centro=eje[(int)neje/2]; //center of the axis
	ild=(landa*MC)/2.99792458e5; //Sigma
	
//	printf("ild-> %f  ...\n",ild);

/*
	for(i=0;i<neje;i++){
		printf("eje (%d) %f  ...\n",i,eje[i]);
	}
*/
	term=(double *)calloc(neje,sizeof(double));

	for(i=0;i<neje;i++){
		term[i]=(((eje[i]-centro)/ild)*((eje[i]-centro)/ild))/2; //exponent
//		printf("term (%d) %f  ...\n",i,term[i]);
	}

	nloai=0;
	loai=calloc(neje,sizeof(double));
	for(i=0;i<neje;i++){
		if(term[i]<1e30){
			nloai++;
			loai[i]=1;
		}
	}

	

	if(nloai>0){
		nmtb=nloai;
		mtb=calloc(nmtb,sizeof(double));
		for(i=0;i<neje;i++){
			if(loai[i]){
				mtb[i]=exp(-term[i]);
			}
		}
	}
	else{

		nmtb=neje;
		mtb=calloc(nmtb,sizeof(double));
		for(i=0;i<neje;i++){
				mtb[i]=exp(-term[i]);
		}
	}
	
	cte=0;
	//normalization
	for(i=0;i<nmtb;i++){
		cte+=mtb[i];
	}
	for(i=0;i<neje;i++){
		mtb[i]/=cte;
	}

	free(loai);
	free(term);

	//In case we need the deriv of f gauss /deriv
	if(deriv==1){
		for(i=0;i<nmtb;i++){
			//mtb2=mtb/macro*(((eje-centro)/ILd)^2d0-1d0)
			mtb[i]=mtb[i]/MC*((((eje[i]-centro)/ild)*((eje[i]-centro)/ild))-1.0);
		}
	}



	return mtb;

}

