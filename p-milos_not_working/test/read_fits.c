#include "/opt/local/cfitsio/cfitsio-3.350/include/fitsio.h"

int main() {
  int numberOfFileSpectra;
  nameFile * vInputFileSpectra = NULL;
  FitsImage * fitsImage = NULL;
  FILE *fp;
  //open file in read more
  fp=fopen("./run/data/input_tmp.fits","r");

  numberOfFileSpectra = 1;
  vInputFileSpectra = (nameFile *)malloc(numberOfFileSpectra*sizeof(nameFile));
  strcpy(vInputFileSpectra[0].name,"./run/data/input_tmp.fits"); //configCrontrolFile.ObservedProfiles
}