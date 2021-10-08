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


#include <stdio.h>
#include <stdlib.h>
#ifndef ERRH
#define ERRH
#define out_of_memory { printf("out of memory!\n"); abort(); }
#define nrerror(error_text) { printf("%s\n", error_text); abort(); }
#define error_print(file, fname, error_text) { printf("%s: %s: %s\n", file, fname, error_text); abort(); }
#endif
