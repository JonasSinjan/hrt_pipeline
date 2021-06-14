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


#ifndef SVDCMP
#define SVDCMP
#include <math.h>
#include <stdlib.h>
//#include <defines.h>
#include "errh.h"


#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
#define MAX(a,b) (((a) > (b)) ?  (a) : (b))
#define MIN(a,b) (((a) < (b)) ?  (a) : (b))
//#define SQR(a) ((a) *(a))

int svdcmp(PRECISION *a, int m, int n, PRECISION w[], PRECISION *v);
PRECISION dpythag(PRECISION a, PRECISION b);

#endif

