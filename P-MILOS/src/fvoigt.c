

#include "defines.h"
extern  _Complex double *z,* zden, * zdiv;

/**
 * @param double damp
 * @param float * vv
 * @param int nvv
 * @param float * h 
 * @param float * f
 * 
 * 
 * */
int fvoigt(PRECISION damp, REAL *vv, int nvv, REAL *h, REAL *f)
{

	int i, j;

	static PRECISION a[] = {122.607931777104326, 214.382388694706425, 181.928533092181549,
									93.155580458138441, 30.180142196210589, 5.912626209773153,
									0.564189583562615};

	static PRECISION b[] = {122.60793177387535, 352.730625110963558, 457.334478783897737,
									348.703917719495792, 170.354001821091472, 53.992906912940207,
									10.479857114260399, 1.};



	for (i = 0; i < nvv; i++)
	{
		z[i] = damp - vv[i] * _Complex_I;
	}

	//
	for (i = 0; i < nvv; i++)
	{
		zden[i] = a[6];
	}

	for (j = 5; j >= 0; j--)
	{
		for (i = 0; i < nvv; i++)
		{
			zden[i] = zden[i] * z[i] + a[j];
		}
	}

	//
	for (i = 0; i < nvv; i++)
	{
		zdiv[i] = z[i] + b[6];
	}

	for (j = 5; j >= 0; j--)
	{
		for (i = 0; i < nvv; i++)
		{
			zdiv[i] = zdiv[i] * z[i] + b[j];
		}
	}

	for (i = 0; i < nvv; i++)
	{
		z[i] = zden[i] / zdiv[i];
	}


	for (i = 0; i < nvv; i++)
	{
		h[i] = creal(z[i]);
	}

	for (i = 0; i < nvv; i++)
	{
		f[i] = (PRECISION)cimag(z[i]) * 0.5;
	}

	return 1;
}
