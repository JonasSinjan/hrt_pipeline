
#include <time.h>

clock_t t_ini, t_fin;
double secs, total_secs;

t_ini = clock();

call FUNCTION to TEST!

t_fin = clock();								

secs = (double)(t_fin - t_ini) / CLOCKS_PER_SEC;
printf("\n\n%.16g milisegundos\n", secs * 1000.0);