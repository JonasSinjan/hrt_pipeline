/*

  (C) 2000 Minghong Gilbert Wu and Michael W. Deem

  This program is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 2
  of the License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307,
  USA.

  http://www.fsf.org/copyleft/gpl.html


  (C) 2000 Minghong Gilbert Wu and Michael W. Deem

  UCLA

 */
#include "svdcmp.h"
//#include "errh.h"

/*  This  subroutine is  a  modified version  of  svdcmp  in the  book
   "Numerical   Recipes   in  C",   2nd   edition.   Given  a   matrix
   a[0..m-1][0..n-1],  this   routine  computes  its   singluar  value
   decomposition,A = U x W x V^T.  The matrix V  (not the  transpose) is
   output  as v[0..n-1][0..n-1].  The  singular values  are output  as
   w[0..n-1] */

int svdcmp(PRECISION *a, int m, int n, PRECISION w[], PRECISION *v)
{
  int flag,i,its,j,jj,k,l=0,nm=0;
  PRECISION anorm,c,f,g,h,s,scale,x,y,z;//,*rv1;
  static double rv1[NTERMS];
  
  
  //rv1=  malloc(sizeof(PRECISION) *n);
  g=scale=anorm=0.0;
  for (i=0;i<n;i++) {
    l=i+1;
    rv1[i]=scale*g;
    g=s=scale=0.0;
    if (i < m) {
      for (k=i;k<m;k++) scale += fabs(a[k*n+i]); //a[k][i]

		if (scale) {
			for (k=i;k<m;k++) {
				a[k*n+i] /= scale;
				s += a[k*n+i]*a[k*n+i]; //a[k][i]*a[k][i]
			}
			f=a[i*n+i];
			g = -SIGN(sqrt(s),f);
			h=f*g-s;
			a[i*n+i]=f-g;
			for (j=l;j<n;j++) {
				for (s=0.0,k=i;k<m;k++) s += a[k*n+i]*a[k*n+j];
					f=s/h;
					for (k=i;k<m;k++) a[k*n+j] += f*a[k*n+i];
			}
			for (k=i;k<m;k++) a[k*n+i] *= scale;
		}
    }
    w[i]=scale *g;
    g=s=scale=0.0;
    if (i < m && i != n-1) {
		for (k=l;k<n;k++) scale += fabs(a[i*n+k]);
		if (scale) {
		for (k=l;k<n;k++) {
			a[i*n+k] /= scale;
			s += a[i*n+k]*a[i*n+k];
		}
	f=a[i*n+l];
	g = -SIGN(sqrt(s),f);
	h=f*g-s;
	a[i*n+l]=f-g;
	for (k=l;k<n;k++) rv1[k]=a[i*n+k]/h;
	for (j=l;j<m;j++) {
	  for (s=0.0,k=l;k<n;k++) s += a[j*n+k]*a[i*n+k];
	  for (k=l;k<n;k++) a[j*n+k] += s*rv1[k];
	}
	for (k=l;k<n;k++) a[i*n+k] *= scale;
      }
    }
    anorm = MAX(anorm, fabs(w[i])+fabs(rv1[i]));
  }
  for (i=n-1;i>=0;i--) {
    if (i < n-1) {
      if (g) {
	for (j=l;j<n;j++) v[j*n+i]=(a[i*n+j]/a[i*n+l])/g;
	for (j=l;j<n;j++) {
	  for (s=0.0,k=l;k<n;k++) s += a[i*n+k]*v[k*n+j];
	  for (k=l;k<n;k++) v[k*n+j] += s*v[k*n+i];
	}
      }
      for (j=l;j<n;j++) v[i*n+j]=v[j*n+i]=0.0;
    }
    v[i*n+i]=1.0;
    g=rv1[i];
    l=i;
  }
  for (i=MIN(m, n)-1;i>=0;i--) {
    l=i+1;
    g=w[i];
    for (j=l;j<n;j++) a[i*n+j]=0.0;
    if (g) {
      g=1.0/g;
      for (j=l;j<n;j++) {
	for (s=0.0,k=l;k<m;k++) s += a[k*n+i]*a[k*n+j];
	f=(s/a[i*n+i])*g;
	for (k=i;k<m;k++) a[k*n+j] += f*a[k*n+i];
      }
      for (j=i;j<m;j++) a[j*n+i] *= g;
    } else for (j=i;j<m;j++) a[j*n+i]=0.0;
    ++a[i*n+i];
  }
  for (k=n-1;k>=0;k--) {
    for (its=1;its<=30;its++) {
      flag=1;
      for (l=k;l>=0;l--) {
	nm=l-1;
	if ((PRECISION)(fabs(rv1[l])+anorm) == anorm) {
	  flag=0;
	  break;
	}
	if ((PRECISION)(fabs(w[nm])+anorm) == anorm) break;
      }
      if (flag) {
	c=0.0;
	s=1.0;
	for (i=l;i<=k;i++) {
	  f=s*rv1[i];
	  rv1[i]=c*rv1[i];
	  if ((PRECISION)(fabs(f)+anorm) == anorm) break;
	  g=w[i];
	  h=dpythag(f,g);
	  w[i]=h;
	  h=1.0/h;
	  c=g*h;
	  s = -f*h;
	  for (j=0;j<m;j++) {
	    y=a[j*n+nm];
	    z=a[j*n+i];
	    a[j*n+nm]=y*c+z*s;
	    a[j*n+i]=z*c-y*s;
	  }
	}
      }
      z=w[k];
      if (l == k) {
	if (z < 0.0) {
	  w[k] = -z;
	  for (j=0;j<n;j++) v[j*n+k] = -v[j*n+k];
	}
	break;
      }
      if (its == 30){  //nrerror("no convergence in 30 dsvdcmp iterations");
		//free(rv1); 
		return -1;
	  };
      x=w[l];
      nm=k-1;
      y=w[nm];
      g=rv1[nm];
      h=rv1[k];
      f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
      g=dpythag(f,1.0);
      f=((x-z)*(x+z)+h*((y/(f+SIGN(g,f)))-h))/x;
      c=s=1.0;
      for (j=l;j<=nm;j++) {
	i=j+1;
	g=rv1[i];
	y=w[i];
	h=s*g;
	g=c*g;
	z=dpythag(f,h);
	rv1[j]=z;
	c=f/z;
	s=h/z;
	f=x*c+g*s;
	g = g*c-x*s;
	h=y*s;
	y *= c;
	for (jj=0;jj<n;jj++) {
	  x=v[jj*n+j];
	  z=v[jj*n+i];
	  v[jj*n+j]=x*c+z*s;
	  v[jj*n+i]=z*c-x*s;
	}
	z=dpythag(f,h);
	w[j]=z;
	if (z) {
	  z=1.0/z;
	  c=f*z;
	  s=h*z;
	}
	f=c*g+s*y;
	x=c*y-s*g;
	for (jj=0;jj<m;jj++) {
	  y=a[jj*n+j];
	  z=a[jj*n+i];
	  a[jj*n+j]=y*c+z*s;
	  a[jj*n+i]=z*c-y*s;
	}
      }
      rv1[l]=0.0;
      rv1[k]=f;
      w[k]=x;
    }
  }
  //free(rv1);
  
  return 0;
}

PRECISION dpythag(PRECISION a, PRECISION b)
{
  PRECISION absa,absb;
  absa=fabs(a);
  absb=fabs(b);
  if (absa > absb) return absa*sqrt(1.0+ SQR(absb/absa));
  else return (absb == 0.0 ? 0.0 : absb*sqrt(1.0+ SQR(absa/absb)));
}

#undef NRANSI


