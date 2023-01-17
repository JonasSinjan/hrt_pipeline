
#include "defines.h"


PRECISION readFileCuanticLines(const char * inputLineFile, PRECISION * cuanticDat, int line2Read, int printLog);

int readInitialModel(Init_Model * INIT_MODEL, char * fileInitModel);

int readMallaGrid(const char * fileMallaGrid, PRECISION * initialLambda, PRECISION * step, PRECISION * finalLambda, int printLog);

int readPSFFile(PRECISION * deltaLambda, PRECISION * PSF, const char * nameInputPSF, PRECISION centralWaveLenght);

void loadInitialValues(ConfigControl * configControlFile);



int readTrolFile(char * fileParameters,  ConfigControl * trolConfig, int printLog);

int readInitFile(char * fileParameters,  ConfigControl * trolConfig, int printLog);

char* file_ext(const char *string);

char * get_basefilename (const char * fname);
int checkInitialModel (Init_Model * INIT_MODEL);

char * mySubString (char* input, int offset, int len, char* dest);

