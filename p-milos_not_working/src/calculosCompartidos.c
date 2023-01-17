#include "defines.h"
#include <stdarg.h>
extern REAL *dtaux, *etai_gp3, *ext1, *ext2, *ext3, *ext4;
extern REAL *gp4_gp2_rhoq, *gp5_gp2_rhou, *gp6_gp2_rhov;
extern REAL *gp1, *gp2, *dt, *dti, *gp3, *gp4, *gp5, *gp6, *etai_2;
extern REAL *dgp1, *dgp2, *dgp3, *dgp4, *dgp5, *dgp6, *d_dt;
extern REAL *d_ei, *d_eq, *d_eu, *d_ev, *d_rq, *d_ru, *d_rv;
extern REAL *dfi, *dshi;
extern REAL *fi_p, *fi_b, *fi_r, *shi_p, *shi_b, *shi_r;
extern REAL *spectra, *d_spectra,*spectra_mac, *spectra_slight;

extern REAL *etain, *etaqn, *etaun, *etavn, *rhoqn, *rhoun, *rhovn;
extern REAL *etai, *etaq, *etau, *etav, *rhoq, *rhou, *rhov;
extern REAL *parcial1, *parcial2, *parcial3;
extern REAL *nubB, *nupB, *nurB;
extern REAL **uuGlobalInicial;
extern REAL **HGlobalInicial;
extern REAL **FGlobalInicial;
extern int FGlobal, HGlobal, uuGlobal;
extern PRECISION *GMAC,*GMAC_DERIV;
extern Cuantic *cuantic;

extern REAL * opa;
extern PRECISION *dirConvPar;
extern REAL *resultConv;
extern _Complex double *z,* zden, * zdiv;
extern int NTERMS;

/**
 * @param int numl indicates the number of wavelenghts and the size of array 
 * 
 * This method allocate the memory neccesary for inversion of one pixel
 * */
void AllocateMemoryDerivedSynthesis(int numl)
{

	/************* SVD VARIABLES *************************************/
	
	/************* ME DER *************************************/
	dtaux = calloc(numl,sizeof(REAL));
	etai_gp3 = calloc(numl,sizeof(REAL));
	ext1 = calloc(numl,sizeof(REAL));
	ext2 = calloc(numl,sizeof(REAL));
	ext3 = calloc(numl,sizeof(REAL));
	ext4 = calloc(numl,sizeof(REAL));
	/**********************************************************/

	//***** VARIABLES FOR FVOIGT ****************************//
	z = malloc (numl * sizeof(_Complex double));
	zden = malloc (numl * sizeof(_Complex double));
	zdiv = malloc (numl * sizeof(_Complex double));	
	/********************************************************/


	GMAC = calloc(numl, sizeof(PRECISION));
	GMAC_DERIV = calloc(numl,sizeof(PRECISION));
	dirConvPar = calloc((numl+numl)+1,sizeof(PRECISION));
	
	resultConv = calloc(numl,sizeof(REAL));

	spectra = calloc(numl * NPARMS, sizeof(REAL));
	spectra_mac = calloc(numl * NPARMS, sizeof(REAL));
	spectra_slight = calloc(numl * NPARMS, sizeof(REAL));
	d_spectra = calloc(numl * NTERMS * NPARMS, sizeof(REAL));
	
	
	opa = calloc(numl,sizeof(REAL));

	gp4_gp2_rhoq = calloc(numl, sizeof(REAL));
	gp5_gp2_rhou = calloc(numl, sizeof(REAL));
	gp6_gp2_rhov = calloc(numl, sizeof(REAL));

	gp1 = calloc(numl, sizeof(REAL));
	gp2 = calloc(numl, sizeof(REAL));
	gp3 = calloc(numl, sizeof(REAL));
	gp4 = calloc(numl, sizeof(REAL));
	gp5 = calloc(numl, sizeof(REAL));
	gp6 = calloc(numl, sizeof(REAL));
	dt = calloc(numl, sizeof(REAL));
	dti = calloc(numl, sizeof(REAL));

	etai_2 = calloc(numl, sizeof(REAL));

	dgp1 = calloc(numl, sizeof(REAL));
	dgp2 = calloc(numl, sizeof(REAL));
	dgp3 = calloc(numl, sizeof(REAL));
	dgp4 = calloc(numl, sizeof(REAL));
	dgp5 = calloc(numl, sizeof(REAL));
	dgp6 = calloc(numl, sizeof(REAL));
	d_dt = calloc(numl, sizeof(REAL));

	d_ei = calloc(numl * 7, sizeof(REAL));
	d_eq = calloc(numl * 7, sizeof(REAL));
	d_eu = calloc(numl * 7, sizeof(REAL));
	d_ev = calloc(numl * 7, sizeof(REAL));
	d_rq = calloc(numl * 7, sizeof(REAL));
	d_ru = calloc(numl * 7, sizeof(REAL));
	d_rv = calloc(numl * 7, sizeof(REAL));
	dfi = calloc(numl * 4 * 3, sizeof(REAL));  //DNULO
	dshi = calloc(numl * 4 * 3, sizeof(REAL)); //DNULO

	fi_p = calloc(numl * 2, sizeof(REAL));
	fi_b = calloc(numl * 2, sizeof(REAL));
	fi_r = calloc(numl * 2, sizeof(REAL));
	shi_p = calloc(numl * 2, sizeof(REAL));
	shi_b = calloc(numl * 2, sizeof(REAL));
	shi_r = calloc(numl * 2, sizeof(REAL));

	etain = calloc(numl * 2, sizeof(REAL));
	etaqn = calloc(numl * 2, sizeof(REAL));
	etaun = calloc(numl * 2, sizeof(REAL));
	etavn = calloc(numl * 2, sizeof(REAL));
	rhoqn = calloc(numl * 2, sizeof(REAL));
	rhoun = calloc(numl * 2, sizeof(REAL));
	rhovn = calloc(numl * 2, sizeof(REAL));

	etai = calloc(numl, sizeof(REAL));
	etaq = calloc(numl, sizeof(REAL));
	etau = calloc(numl, sizeof(REAL));
	etav = calloc(numl, sizeof(REAL));
	rhoq = calloc(numl, sizeof(REAL));
	rhou = calloc(numl, sizeof(REAL));
	rhov = calloc(numl, sizeof(REAL));

	parcial1 = calloc(numl, sizeof(REAL));
	parcial2 = calloc(numl, sizeof(REAL));
	parcial3 = calloc(numl, sizeof(REAL));

	nubB = calloc(cuantic[0].N_SIG, sizeof(REAL));
	nurB = calloc(cuantic[0].N_SIG, sizeof(REAL));
	nupB = calloc(cuantic[0].N_PI, sizeof(REAL));

	uuGlobalInicial = calloc((int)(cuantic[0].N_PI + cuantic[0].N_SIG * 2), sizeof(REAL *));
	uuGlobal = 0;
	int i = 0;
	for (i = 0; i < (int)(cuantic[0].N_PI + cuantic[0].N_SIG * 2); i++)
	{
		uuGlobalInicial[i] = calloc(numl, sizeof(REAL));
	}

	HGlobalInicial = calloc((int)(cuantic[0].N_PI + cuantic[0].N_SIG * 2), sizeof(REAL *));
	HGlobal = 0;
	for (i = 0; i < (int)(cuantic[0].N_PI + cuantic[0].N_SIG * 2); i++)
	{
		HGlobalInicial[i] = calloc(numl, sizeof(REAL));
	}

	FGlobalInicial = calloc((int)(cuantic[0].N_PI + cuantic[0].N_SIG * 2), sizeof(REAL *));
	for (i = 0; i < (int)(cuantic[0].N_PI + cuantic[0].N_SIG * 2); i++)
	{
		FGlobalInicial[i] = calloc(numl, sizeof(REAL));
	}
	FGlobal = 0;
}


/**
 * This method free memory used to calculate the inversion of one pixel. 
 * */
void FreeMemoryDerivedSynthesis()
{
	
	/**************************************************/
	free(dtaux);
	free(etai_gp3);
	free(ext1);
	free(ext2);
	free(ext3);
	free(ext4);	

	free(zden);
	free(zdiv);
	free(z);

	free(gp1);
	free(gp2);
	free(gp3);
	free(gp4);
	free(gp5);
	free(gp6);
	free(dt);
	free(dti);

	free(etai_2);

	free(dgp1);
	free(dgp2);
	free(dgp3);
	free(dgp4);
	free(dgp5);
	free(dgp6);
	free(d_dt);

	free(d_ei);
	free(d_eq);
	free(d_ev);
	free(d_eu);
	free(d_rq);
	free(d_ru);
	free(d_rv);

	free(dfi);
	free(dshi);

	free(GMAC);
	free(GMAC_DERIV);
	free(dirConvPar);
	free(resultConv);

	free(opa);
	free(spectra);
	free(spectra_mac);
	free(spectra_slight);
	free(d_spectra);
	
	

	free(fi_p);
	free(fi_b);
	free(fi_r);
	free(shi_p);
	free(shi_b);
	free(shi_r);

	free(etain);
	free(etaqn);
	free(etaun);
	free(etavn);
	free(rhoqn);
	free(rhoun);
	free(rhovn);

	free(etai);
	free(etaq);
	free(etau);
	free(etav);

	free(rhoq);
	free(rhou);
	free(rhov);

	free(parcial1);
	free(parcial2);
	free(parcial3);

	free(nubB);
	free(nurB);
	free(nupB);

	free(gp4_gp2_rhoq);
	free(gp5_gp2_rhou);
	free(gp6_gp2_rhov);

	int i;
	for (i = 0; i < (int)(cuantic[0].N_PI + cuantic[0].N_SIG * 2); i++)
	{
		free(uuGlobalInicial[i]);
	}

	for (i = 0; i < (int)(cuantic[0].N_PI + cuantic[0].N_SIG * 2); i++)
	{
		free(HGlobalInicial[i]);
	}

	for (i = 0; i < (int)(cuantic[0].N_PI + cuantic[0].N_SIG * 2); i++)
	{
		free(FGlobalInicial[i]);
	}

	free(uuGlobalInicial);
	free(HGlobalInicial);
	free(FGlobalInicial);
}
