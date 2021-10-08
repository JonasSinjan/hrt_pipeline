
#include "defines.h"



/**
 * 
 * Clean the memory of the array "image"
 * @param image --> Array to clean memory 
 * @param numPixels --> Number of elements of array of image to clean. 
 */
void freeVpixels(vpixels * image, int numPixels);

/**
 * This function read the spectro image from the file "fitsFileSpectra" and store it into a struct of FitsImage
 * @param fitsFileSpectra --> name of the fits file to read 
 * Return the image read or NULL if something was wrong during the lecture. 
 */
FitsImage *  readFitsSpectroImage (const char * fitsFileSpectra, int forParallel, int nLambdaGrid);


/**
 * This function read the spectro image from the file "fitsFileSpectra" and store it into a struct of FitsImage
 * @param fitsFileSpectra --> name of the fits file to read 
 * @param configControlFile
 * @param forParallel
 * Return the image read or NULL if something was wrong during the lecture. 
 */
FitsImage * readFitsSpectroImageRectangular (const char * fitsFileSpectra, ConfigControl * configCrontrolFile, int forParallel, int nLambdaGrid);


/**
 * 
 * */
PRECISION * readFitsLambdaToArray (const char * fitsFileLambda,  int * indexLine, int * nLambda);
/**
 * This function read the Stray Light values from the file "perFileStrayLight" and store it into an array of length nlambda * NPARAMS
 *  
 * be read it before call this method. 
 * @param perFileStrayLight --> name of the PER file to read with lambda values 
 * @param nlambda --> struct of image 
 * Return 1 If the image has been read corectly if not return 0 
 */
float * readPerStrayLightFile (const char * perFileStrayLight, int nlambda, PRECISION *  vOffsetsLambda);

/**
 * This function read the Stray Light values from the file "fitsFileStrayLight" and store it into a vector , in dimStrayLight will be stored the tam of fits file:
 *  (n lambdas or n lambdas X numPixels)
 * be read it before call this method. 
 * @param fitsFileLambda --> name of the fits file to read with lambda values 
 * @param fitsImage --> struct of image 
 * Return 1 If the image has been read corectly if not return 0 
 */
float * readFitsStrayLightFile (ConfigControl * configCrontrolFile,int * nl_straylight,int * ns_straylight,int * nx_straylight,int * ny_straylight);

float * readFitsStrayLightFileSubSet (ConfigControl * configCrontrolFile,int * nl_straylight,int * ns_straylight,int * nx_straylight,int * ny_straylight);
/**
 * This function read from a fits file a mask matrix with the same size of spectral image. 
 * If size is different to the spectral imgen return a null value showing this issue, in normal case return 
 * an array of numRowsXnumCols with integer values of mask. 
 * 
 * @param fitsMask --> name of the fits Mask 
 * @param numRows --> num of rows of spectral image
 * @param numCols --> num of cols of spectral image
 * 
 * @return an array with mask integer values or null in error case
 */
int * readFitsMaskFile (const char * fitsMask, int * numRows, int * numCols);


int * readFitsMaskFileSubSet (const char * fitsMask, int * numRows, int * numCols,  ConfigControl * configCrontrolFile);

/**
 * Clean the memory reserved for the image
 * @param image --> Image to clean memory . 
 */
void freeFitsImage(FitsImage * image);

/**
 * Write the models resutls in a fils file with 3 dimensiones: number of models x number of cols x number of rows. 
 * 
 * @param fitsFile --> Name of the file to store the image. 
 * @param numRows --> Number of rows of original image.
 * @param numCols --> Number of cols of original image.
 * @param vInitModel --> Array with the models obtained from the inversion. Each element of the array is a stucture with the models for one
 * pixel in the image. 
 * @param vChisqrf --> Array with the chisqrf calculated for each pixel in the image. 
 * @param vNumIterPixel
 * @param addChisqr -->  to know if add chisqr at output model file 
 * 
 */
int writeFitsImageModels(const char * fitsFile, int numRows, int numCols, Init_Model * vInitModel, float * vChisqrf, int * vNumIterPixel, int addChiqr);
/**
 * Write the models resutls in a fils file with 3 dimensiones: number of models x number of cols x number of rows. 
 * 
 * @param fitsFile --> Name of the file to store the image. 
 * @param numRowsOriginal --> Number of rows of original image.
 * @param numColsOriginal --> Number of cols of original image.
 * @param rowRowsWrite --> Number of rows to write.
 * @param numColsWrite --> Number of cols to write.
 * @param configCrontrolFile --> Config control structure with sub images parameters.
 * @param vInitModel --> Array with the models obtained from the inversion. Each element of the array is a stucture with the models for one
 * pixel in the image. 
 * @param vChisqrf --> Array with the chisqrf calculated for each pixel in the image.
 * @param vNumIterPixel --> Array with number of iterations to write  
 * @param addChisqr -->  to know if add chisqr at output model file 
 * 
 * */
int writeFitsImageModelsSubSet(const char * fitsFile, int numRowsOriginal, int numColsOriginal, ConfigControl configCrontrolFile, Init_Model * vInitModel, float * vChisqrf, int * vNumIterPixel, int addChiqr);

int writeFitsImageModelsWithArray(char * fitsFile, int numRows, int numCols, PRECISION * eta0, PRECISION * B, PRECISION * vlos, PRECISION * dopp, PRECISION * aa, PRECISION * gm, PRECISION * az, PRECISION * S0, PRECISION * S1, PRECISION * mac, PRECISION * alfa, PRECISION * vChisqrf);
/**
 * Print the error status of status. 
 * @param status -> int code with the status error. 
 */
void printerror( int status);

/**
 * Write the image of profiles in a fits file with the same header of the fits spectro file. 
 * 
 * @param fitsProfileFile --> Name of the file to store the fits profile Image. 
 * @param fitsFileOrigin --> Name of the file with the origin spectro image. This image is used to copy the same header to the profile file. 
 * @param image --> Struct with the image procesed to store in the profile file. 
 * 
 */
int writeFitsImageProfiles(const char * fitsProfileFile, const char * fitsFileOrigin, FitsImage * image);


/**
 * Write the image of profiles in a fits file with the same header of the fits spectro file. 
 * 
 * @param fitsProfileFile --> Name of the file to store the fits profile Image. 
 * @param fitsFileOrigin --> Name of the file with the origin spectro image. This image is used to copy the same header to the profile file. 
 * @param image --> Struct with the image procesed to store in the profile file. 
 * 
 */
int writeFitsImageProfilesSubSet(const char * fitsProfileFile, const char * fitsFileOrigin, FitsImage * image, ConfigControl configCrontrolFile);



/**
 * 
 * @param: fitsFile
 * @param: numRows
 * @param: numCols
 *  
 * */
int readSizeImageSpectro(const char * fitsFile, int * numRows, int * numCols);
