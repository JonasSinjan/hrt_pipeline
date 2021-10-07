
#include "defines.h"


extern PRECISION *GMAC;

/**
 * 
 * @param double MC
 * @param double * eje
 * @param int neje
 * @param double landa
 * @param int deriv 
 * 
 * */
PRECISION * fgauss(PRECISION MC, PRECISION *eje, int neje, PRECISION landa, int deriv)
{
	//int fgauss(PRECISION MC, PRECISION * eje,int neje,PRECISION landa,int deriv,PRECISION * mtb,int nmtb){

	PRECISION centro;
	PRECISION ild;
	PRECISION term[neje];
	int i;
	PRECISION cte;

	centro = eje[(int)neje / 2];		  //center of the axis
	ild = (landa * MC) / 2.99792458e5; //Sigma


	for (i = 0; i < neje; i++)
	{
		PRECISION aux = ((eje[i] - centro) / ild);
		term[i] = ( aux * aux) / 2; //exponent
	}


	for (i = 0; i < neje; i++)
	{
		if(term[i]< 1e30)
			GMAC[i] = exp(-term[i]);
		else 
			GMAC[i] = 0;
	}

	cte = 0;
	//normalization
	for (i = 0; i < neje; i++)
	{
		cte += GMAC[i];
	}
	for (i = 0; i < neje; i++)
	{
		GMAC[i] /= cte;
	}

	//In case we need the deriv of f gauss /deriv
	if (deriv == 1)
	{
		for (i = 0; i < neje; i++)
		{
			GMAC[i] = GMAC[i] / MC * ((((eje[i] - centro) / ild) * ((eje[i] - centro) / ild)) - 1.0);			
		}
	}

	return NULL;
}


/**
 * @param double FWHM
 * @param double step_between_lw
 * @param double lambda0
 * @param double lambdaCentral
 * @param int nLambda
 * @param int * sizeG
 * 
 * */

PRECISION * fgauss_WL(PRECISION FWHM, PRECISION step_between_lw, PRECISION lambda0, PRECISION lambdaCentral, int nLambda, int * sizeG)
{

	REAL *mtb_final;
	PRECISION *mtb ;
	PRECISION *term, *loai;
	int i;
	int nloai;
	int nmtb;
	PRECISION cte;
	*sizeG = nLambda;

	///Conversion from FWHM to Gaussian sigma (1./(2*sqrt(2*alog2)))
	PRECISION sigma=FWHM*0.42466090/1000.0; // in Angstroms
	//PRECISION sigma = FWHM * (2 * sqrt(2 * log(2)))/1000;

	term = (PRECISION *)calloc(*sizeG, sizeof(PRECISION));

	for (i = 0; i < *sizeG; i++)
	{
		PRECISION lambdaX = lambda0 +i*step_between_lw;
		PRECISION aux = ((lambdaX - lambdaCentral) / sigma);
		term[i] = ( aux * aux) / 2; //exponent
	}

	nloai = 0;
	loai = calloc(*sizeG, sizeof(PRECISION));
	for (i = 0; i < *sizeG; i++)
	{
		if (term[i] < 1e30)
		{
			nloai++;
			loai[i] = 1;
		}
	}

	if (nloai > 0)
	{
		nmtb = nloai;
		mtb = calloc(nmtb, sizeof(PRECISION));
		for (i = 0; i < *sizeG; i++)
		{
			if (loai[i])
			{
				mtb[i] = exp(-term[i]);
			}
		}
	}
	else
	{

		nmtb = *sizeG;
		mtb = malloc ( (*sizeG)* sizeof(PRECISION));
		for (i = 0; i < *sizeG; i++)
		{
			mtb[i] = exp(-term[i]);
		}
	}

	cte = 0;
	//normalization
	for (i = 0; i < nmtb; i++)
	{
		cte += mtb[i];
	}

	for (i = 0; i < *sizeG; i++)
	{
		mtb[i] /= cte;
	}

	free(loai);
	free(term);

	return mtb;
}
