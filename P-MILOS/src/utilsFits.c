#include "utilsFits.h"
#include "/opt/local/cfitsio/cfitsio-3.350/include/fitsio.h" ///opt/local/cfitsio/cfitsio-3.350/include/
#include <time.h>
#include <string.h>
#include <stdio.h>
#include <stddef.h>
#include <locale.h>
#include <unistd.h>

/**
 * Clean memory from fits image
 */
void freeVpixels(vpixels * image, int numPixels){
	int i;
	for(i=9;i<numPixels;i++){
		free(image[i].spectro);
		free(image[i].vLambda);
	}
	free(image);
}




FitsImage *  readFitsSpectroImage (const char * fitsFileSpectra, int forParallel, int nLambdaGrid){
	fitsfile *fptr;   /* FITS file pointer, defined in fitsio.h */
	FitsImage * image =  malloc(sizeof(FitsImage));
	int status = 0, header = 1;   /* CFITSIO status value MUST be initialized to zero! */
	PRECISION nulval = 0.; // define null value to 0 because the performance to read from fits file is better doing this. 
	int bitpix, naxis, anynul, numPixelsFitsFile, nkeys;
	long naxes [4] = {1,1,1,1}; /* The maximun number of dimension that we will read is 4*/
	char comment[FLEN_CARD];   /* Standard string lengths defined in fitsio.h */
	char value [FLEN_CARD];
	
	int i, j, k, h;
   // OPEN THE FITS FILE TO READ THE DEPTH OF EACH DIMENSION
	if (!fits_open_file(&fptr, fitsFileSpectra, READONLY, &status)){
		// READ THE HDU PARAMETER FROM THE FITS FILE
		int hdutype;
		fits_get_hdu_type(fptr, &hdutype, &status);

		// We want only fits image 
		if(hdutype==IMAGE_HDU){
			// We assume that we have only on HDU as primary 
			if(fits_read_key(fptr, TSTRING, CTYPE1, image->ctype_1, comment, &status)){
				header = 0 ;
				status = 0;
			} 
			if(fits_read_key(fptr, TSTRING, CTYPE2, image->ctype_2, comment, &status)){ 
				header = 0 ;
				status = 0;
			}
			if(fits_read_key(fptr, TSTRING, CTYPE3, image->ctype_3, comment, &status)){
				header = 0 ;
				status = 0;
			}
			if(fits_read_key(fptr, TSTRING, CTYPE4, image->ctype_4, comment, &status)){
				header = 0 ;
				status = 0;
			}
			
			// ORDER MUST BE CTYPE1->'HPLN-TAN',CTYPE2->'HPLT-TAN',CTYPE3->'WAVE-GRI',CTYPE4->'STOKES'
			// int pos_row = 0, pos_col = 1, pos_lambda = 2, pos_stokes_parameters = 3;
			int correctOrder =0;
			// GET THE CURRENT POSITION OF EVERY PARAMETER
			int pos_lambda; 
			int pos_row;
			int pos_col;
			int pos_stokes_parameters;
			if(header){
				// LAMBDA POSITION
				if(strcmp(image->ctype_1,CTYPE_WAVE)==0) pos_lambda = 0;
				if(strcmp(image->ctype_2,CTYPE_WAVE)==0) pos_lambda = 1;
				if(strcmp(image->ctype_3,CTYPE_WAVE)==0) pos_lambda = 2;
				if(strcmp(image->ctype_4,CTYPE_WAVE)==0) pos_lambda = 3;

				// HPLN TAN 
				if(strcmp(image->ctype_1,CTYPE_HPLN_TAN)==0) pos_row = 0;
				if(strcmp(image->ctype_2,CTYPE_HPLN_TAN)==0) pos_row = 1;
				if(strcmp(image->ctype_3,CTYPE_HPLN_TAN)==0) pos_row = 2;
				if(strcmp(image->ctype_4,CTYPE_HPLN_TAN)==0) pos_row = 3;

				// HPLT TAN 
				if(strcmp(image->ctype_1,CTYPE_HPLT_TAN)==0) pos_col = 0;
				if(strcmp(image->ctype_2,CTYPE_HPLT_TAN)==0) pos_col = 1;
				if(strcmp(image->ctype_3,CTYPE_HPLT_TAN)==0) pos_col = 2;
				if(strcmp(image->ctype_4,CTYPE_HPLT_TAN)==0) pos_col = 3;			

				// Stokes paramter position , 
				if(strcmp(image->ctype_1,CTYPE_STOKES)==0) pos_stokes_parameters = 0;
				if(strcmp(image->ctype_2,CTYPE_STOKES)==0) pos_stokes_parameters = 1;
				if(strcmp(image->ctype_3,CTYPE_STOKES)==0) pos_stokes_parameters = 2;
				if(strcmp(image->ctype_4,CTYPE_STOKES)==0) pos_stokes_parameters = 3;
			}
			// GET THE CURRENT POSITION OF EVERY PARAMETER
			/*int pos_row = 2;
			int pos_col = 3;
			int pos_lambda = 0; 
			int pos_stokes_parameters = 1;*/

			// READ IMAGE AND STORAGE IN STRUCTURE IMAGE 
			if (!fits_get_img_param(fptr, 4, &bitpix, &naxis, naxes, &status) ){
				// printf("%d\n",bitpix);
				// printf("%d\n",FLOAT_IMG);
				if(bitpix != FLOAT_IMG){
					printf("\n ERROR: the datatype of FITS spectro image must be FLOAT\n");
					printf("\n EXITING THE PROGRAM\n");
					fits_close_file(fptr, &status);
					exit(EXIT_FAILURE);
				}
				if(!header){
					
					int num_stokes = 10000;
					for(i=0;i<naxis;i++){
						if(naxes[i]<num_stokes){
							pos_stokes_parameters =i;
							num_stokes = naxes[i];
						}
					}
					
					int num_lambda = 10000;
					for(i=0;i<naxis;i++){
						if(i!=pos_stokes_parameters){
							if(naxes[i]<num_lambda){
								pos_lambda =i;
								num_lambda = naxes[i];
							}
						}
					}
					

					if( (pos_stokes_parameters == 0 && pos_lambda == 1) || (pos_stokes_parameters == 1 && pos_lambda == 0) ){ 
						pos_row = 2;
						pos_col = 3;
					}
					if( (pos_stokes_parameters == 0 && pos_lambda == 2) || (pos_stokes_parameters == 2 && pos_lambda == 0) ){ 
						pos_row = 1;
						pos_col = 3;
					} 
					if( (pos_stokes_parameters == 0 && pos_lambda == 3) || (pos_stokes_parameters == 3 && pos_lambda == 0) ){ 
						pos_row = 1;
						pos_col = 2;					
					}
					if( (pos_stokes_parameters == 1 && pos_lambda == 2) || (pos_stokes_parameters == 2 && pos_lambda == 1) ){ 
						pos_row = 0;
						pos_col = 3;					
					}
					if( (pos_stokes_parameters == 1 && pos_lambda == 3) || (pos_stokes_parameters == 3 && pos_lambda == 1) ){ 
						pos_row = 0;
						pos_col = 2;					
					}
					if( (pos_stokes_parameters == 2 && pos_lambda == 3) || (pos_stokes_parameters == 3 && pos_lambda == 2) ){ 
						pos_row = 0;
						pos_col = 1;					
					}

				}

				if(pos_row==0 && pos_col ==1 && pos_lambda == 2 && pos_stokes_parameters == 3)
				//if(pos_row==2 && pos_col ==3 && pos_lambda == 0 && pos_stokes_parameters == 1)
					correctOrder = 1;
				// for(i=0;i<naxis;i++){
				// printf("%d\n", naxes[i]);
				// }
				// printf("%d\n", pos_lambda);
				// printf("%d\n", correctOrder);
				if(naxes[pos_lambda] != nLambdaGrid){
					printf("\n ERROR: The dimension of wavelength is different in FITS file  %ld to give dimension in grid file %d \n",naxes[pos_lambda],nLambdaGrid);
					printf("\n EXITING THE PROGRAM\n");
					fits_close_file(fptr, &status);
					exit(EXIT_FAILURE);
				}
				fits_get_hdrspace(fptr, &nkeys, NULL, &status); 
				image->nkeys = nkeys;
				image->vCard = (char**) malloc( nkeys * sizeof(char*));
				image->vKeyname = (char**) malloc( nkeys * sizeof(char*));
				for (i = 1; i <= nkeys; i++) {
					char * card = malloc(FLEN_CARD * sizeof(char));
					char * keyname = malloc(FLEN_CARD * sizeof(char));
					fits_read_record(fptr, i, card, &status);
					fits_read_keyn(fptr,i,keyname, value, NULL, &status);
					image->vCard[i-1] = card;
					image->vKeyname[i-1] = keyname;
				}
				image->bitpix = bitpix;
				image->naxis  = naxis;

				image->rows=naxes[pos_row];
				image->cols=naxes[pos_col];
				image->nLambdas=naxes[pos_lambda];
				image->numStokes=naxes[pos_stokes_parameters];
				image->naxes = calloc(4,sizeof(long));
				image->naxes[pos_row] = naxes[pos_row];
				image->naxes[pos_col] = naxes[pos_col];
				image->naxes[pos_lambda] = naxes[pos_lambda];
				image->naxes[pos_stokes_parameters] = naxes[pos_stokes_parameters];
				if(image->numStokes!=4){
					printf("\n************** PLEASE REVIEW THE ORDER OF HEADER NAXIS. DIMENSION OF STOKES MUST HAVE AS VALUE: 4 \n");
					exit(EXIT_FAILURE);
				}
				image->numPixels = naxes[pos_col] * naxes[pos_row]; // we will read the image by columns 
				image->pos_lambda = pos_lambda;
				image->pos_col = pos_col;
				image->pos_row = pos_row;
				image->pos_stokes_parameters = pos_stokes_parameters;
				numPixelsFitsFile = naxes[pos_row]*naxes[pos_col]*naxes[pos_lambda]*naxes[pos_stokes_parameters];
				// allocate memory to read all pixels in the same array 
				float * imageTemp = calloc(numPixelsFitsFile, sizeof(float));
				if (!imageTemp)  {
					printf("ERROR ALLOCATION MEMORY FOR TEMP IMAGE");
					return NULL;
          		}
				
				
				long fpixel [4] = {1,1,1,1}; 
				fits_read_pix(fptr, TFLOAT, fpixel, numPixelsFitsFile, &nulval, imageTemp, &anynul, &status);
				if(status){
					fits_report_error(stderr, status);
               return NULL;	
				}

				// allocate memory for reorder the image
				
				if(forParallel){
					//image->vLambdaImagen = calloc(image->numPixels*image->nLambdas, sizeof(PRECISION));
					image->vLambdaImagen = NULL;
					image->pixels = NULL;
					image->spectroImagen = calloc(image->numPixels*image->nLambdas*image->numStokes, sizeof(float));
				}
				else{
					image->pixels = calloc(image->numPixels, sizeof(vpixels));
					for( i=0;i<image->numPixels;i++){
						image->pixels[i].spectro = calloc ((image->numStokes*image->nLambdas),sizeof(float));
						image->pixels[i].vLambda = NULL;
						image->pixels[i].nLambda = image->nLambdas;
					}					
					image->vLambdaImagen = NULL;
					image->spectroImagen = NULL;
				}


				
				//PRECISION pixel;
				if(naxis==4){ // image with 4 dimension 
					if(correctOrder){
						// i = cols, j = rows, k = stokes, h = lambda
						for( i=0; i<naxes[3];i++){
							for( j=0; j<naxes[2];j++){
								for( k=0;k<naxes[1];k++){
									for( h=0;h<naxes[0];h++){
										if(forParallel){
											image->spectroImagen[ (((i*naxes[2]) + j)*(image->nLambdas*image->numStokes)) + (image->nLambdas * k) + h] = imageTemp [(i*naxes[2]*naxes[1]*naxes[0]) + (j*naxes[1]*naxes[0]) + (k*naxes[0]) + h];  // I =0, Q = 1, U = 2, V = 3
										}
										else{
											image->pixels[(i*naxes[2]) + j].spectro[h + (image->nLambdas * k)] = imageTemp [(i*naxes[2]*naxes[1]*naxes[0]) + (j*naxes[1]*naxes[0]) + (k*naxes[0]) + h];  // I =0, Q = 1, U = 2, V = 3
										}
									}
								}
							}
						}					
					}
					else{
						int currentLambda = 0, currentRow = 0, currentStokeParameter=0, currentCol = 0, currentPixel;
						for( i=0; i<naxes[3];i++){
							for( j=0; j<naxes[2];j++){
								for( k=0;k<naxes[1];k++){
									for( h=0;h<naxes[0];h++){
										PRECISION pixel = 0.0;
										// I NEED TO KNOW THE CURRENT POSITION OF EACH ITERATOR 
										switch (pos_lambda)
										{
											case 0:
												currentLambda = h;
												break;
											case 1:
												currentLambda = k;
												break;
											case 2:
												currentLambda = j;
												break;
											case 3:
												currentLambda = i;
												break;																						
										}
										switch (pos_stokes_parameters)
										{
											case 0:
												currentStokeParameter = h;
												break;
											case 1:
												currentStokeParameter = k;
												break;
											case 2:
												currentStokeParameter = j;
												break;
											case 3:
												currentStokeParameter = i;
												break;																						
										}
										switch (pos_row)
										{
											case 0:
												currentRow = h;
												break;
											case 1:
												currentRow = k;
												break;
											case 2:
												currentRow = j;
												break;
											case 3:
												currentRow = i;
												break;																						
										}
										switch (pos_col)
										{
											case 0:
												currentCol = h;
												break;
											case 1:
												currentCol = k;
												break;
											case 2:
												currentCol = j;
												break;
											case 3:
												currentCol = i;
												break;																						
										}			
										pixel = imageTemp [(i*naxes[2]*naxes[1]*naxes[0]) + (j*naxes[1]*naxes[0]) + (k*naxes[0]) + h];
										currentPixel = (currentCol*naxes[pos_row]) + currentRow;
										if(forParallel){
											image->spectroImagen[(currentPixel *(image->nLambdas*image->numStokes)) + (image->nLambdas * currentStokeParameter) + currentLambda] = pixel;
										}
										else
										{
											image->pixels[currentPixel].spectro[currentLambda + (image->nLambdas * currentStokeParameter)] = pixel;  // I =0, Q = 1, U = 2, V = 3
										}
										
										
									}
								}
							}
						}
					}
				}
				free(imageTemp);
				fits_close_file(fptr, &status);
				if (status){
					fits_report_error(stderr, status);
					return NULL;
				}				
			}
			else{
				printf("\n IMPOSSIBLE READ IMAGE \n ");
			}
		}
		else{
			return NULL;  // we are interested only in FITS image
		}
	}
   else{ // IN CASE AN ERROR OPENING THE FILE RETURN THE ERROR CODE
      if (status) fits_report_error(stderr, status); /* print any error message */
      return NULL;
   }
	
	return image; 
}





FitsImage * readFitsSpectroImageRectangular (const char * fitsFileSpectra, ConfigControl * configCrontrolFile, int forParallel, int nLambdaGrid){



	fitsfile *fptr;   
	FitsImage * image =  malloc(sizeof(FitsImage));
   	int status = 0, headers = 1;   
   	PRECISION nulval = 0.; // define null value to 0 because the performance to read from fits file is better doing this. 
   	int bitpix, naxis, anynul, numPixelsFitsFile,nkeys;
   	long naxes [4] = {1,1,1,1}; 
		char comment[FLEN_CARD];   
		char value [FLEN_CARD];

		int i, j, k, h;
   	// OPEN THE FITS FILE TO READ THE DEPTH OF EACH DIMENSION
   	if (!fits_open_file(&fptr, fitsFileSpectra, READONLY, &status)){
    	// READ THE HDU PARAMETER FROM THE FITS FILE
    	int hdutype;
    	fits_get_hdu_type(fptr, &hdutype, &status);

		// We want only fits image 
		if(hdutype==IMAGE_HDU){
			// We assume that we have only on HDU as primary 

			if(fits_read_key(fptr, TSTRING, CTYPE1, image->ctype_1, comment, &status)){
				headers=0;
				status = 0;
			}
			if(fits_read_key(fptr, TSTRING, CTYPE2, image->ctype_2, comment, &status)){ 
				headers=0;
				status = 0;
			}
			if(fits_read_key(fptr, TSTRING, CTYPE3, image->ctype_3, comment, &status)){
				headers=0;
				status = 0;
			} 
			if(fits_read_key(fptr, TSTRING, CTYPE4, image->ctype_4, comment, &status)){
				headers=0;
				status = 0;
			}

			// ORDER MUST BE CTYPE1->'HPLN-TAN',CTYPE2->'HPLT-TAN',CTYPE3->'WAVE-GRI',CTYPE4->'STOKES'
			// int pos_row = 0, pos_col = 1, pos_lambda = 2, pos_stokes_parameters = 3;
			int correctOrder =0;
			// GET THE CURRENT POSITION OF EVERY PARAMETER
			int pos_lambda; 
			int pos_row;
			int pos_col;
			int pos_stokes_parameters;
			if(headers){
				// LAMBDA POSITION
				if(strcmp(image->ctype_1,CTYPE_WAVE)==0) pos_lambda = 0;
				if(strcmp(image->ctype_2,CTYPE_WAVE)==0) pos_lambda = 1;
				if(strcmp(image->ctype_3,CTYPE_WAVE)==0) pos_lambda = 2;
				if(strcmp(image->ctype_4,CTYPE_WAVE)==0) pos_lambda = 3;

				// HPLN TAN 
				if(strcmp(image->ctype_1,CTYPE_HPLN_TAN)==0) pos_row = 0;
				if(strcmp(image->ctype_2,CTYPE_HPLN_TAN)==0) pos_row = 1;
				if(strcmp(image->ctype_3,CTYPE_HPLN_TAN)==0) pos_row = 2;
				if(strcmp(image->ctype_4,CTYPE_HPLN_TAN)==0) pos_row = 3;

				// HPLT TAN 
				if(strcmp(image->ctype_1,CTYPE_HPLT_TAN)==0) pos_col = 0;
				if(strcmp(image->ctype_2,CTYPE_HPLT_TAN)==0) pos_col = 1;
				if(strcmp(image->ctype_3,CTYPE_HPLT_TAN)==0) pos_col = 2;
				if(strcmp(image->ctype_4,CTYPE_HPLT_TAN)==0) pos_col = 3;			

				// Stokes paramter position , 
				if(strcmp(image->ctype_1,CTYPE_STOKES)==0) pos_stokes_parameters = 0;
				if(strcmp(image->ctype_2,CTYPE_STOKES)==0) pos_stokes_parameters = 1;
				if(strcmp(image->ctype_3,CTYPE_STOKES)==0) pos_stokes_parameters = 2;
				if(strcmp(image->ctype_4,CTYPE_STOKES)==0) pos_stokes_parameters = 3;
			}

			// GET THE CURRENT POSITION OF EVERY PARAMETER
			/*int pos_row = 0;
			int pos_col = 1;
			int pos_lambda = 2; 
			int pos_stokes_parameters = 3;*/


			// READ IMAGE AND STORAGE IN STRUCTURE IMAGE 
			if (!fits_get_img_param(fptr, 4, &bitpix, &naxis, naxes, &status) ){

				if(bitpix != FLOAT_IMG){
					printf("\n ERROR: the datatype of FITS spectro image must be FLOAT\n");
					printf("\n EXITING THE PROGRAM\n");
					fits_close_file(fptr, &status);
					exit(EXIT_FAILURE);
				}

				if(!headers){
					
					int num_stokes = 10000;
					for(i=0;i<naxis;i++){
						if(naxes[i]<num_stokes){
							pos_stokes_parameters =i;
							num_stokes = naxes[i];
						}
					}
					int num_lambda = 10000;
					for(i=0;i<naxis;i++){
						if(i!=pos_stokes_parameters){
							if(naxes[i]<num_lambda){
								pos_lambda =i;
								num_lambda = naxes[i];
							}
						}
					}

					if( (pos_stokes_parameters == 0 && pos_lambda == 1) || (pos_stokes_parameters == 1 && pos_lambda == 0) ){ 
						pos_row = 2;
						pos_col = 3;
					}
					if( (pos_stokes_parameters == 0 && pos_lambda == 2) || (pos_stokes_parameters == 2 && pos_lambda == 0) ){ 
						pos_row = 1;
						pos_col = 3;
					} 
					if( (pos_stokes_parameters == 0 && pos_lambda == 3) || (pos_stokes_parameters == 3 && pos_lambda == 0) ){ 
						pos_row = 1;
						pos_col = 2;					
					}
					if( (pos_stokes_parameters == 1 && pos_lambda == 2) || (pos_stokes_parameters == 2 && pos_lambda == 1) ){ 
						pos_row = 0;
						pos_col = 3;					
					}
					if( (pos_stokes_parameters == 1 && pos_lambda == 3) || (pos_stokes_parameters == 3 && pos_lambda == 1) ){ 
						pos_row = 0;
						pos_col = 2;					
					}
					if( (pos_stokes_parameters == 2 && pos_lambda == 3) || (pos_stokes_parameters == 3 && pos_lambda == 2) ){ 
						pos_row = 0;
						pos_col = 1;					
					}
				}

				if(naxes[pos_lambda] != nLambdaGrid){
					printf("\n ERROR: The dimension of wavelenght is different in FITS file  %ld to give dimension in grid file %d \n",naxes[pos_lambda],nLambdaGrid);
					printf("\n EXITING THE PROGRAM\n");
					fits_close_file(fptr, &status);
					exit(EXIT_FAILURE);
				}

				if( configCrontrolFile->subx2 > naxes[pos_row] || configCrontrolFile->subx1>configCrontrolFile->subx2 || configCrontrolFile->suby2 > naxes[pos_col] || configCrontrolFile->suby1 > configCrontrolFile->suby2){
					printf("\n ERROR IN THE DIMENSIONS, PLEASE CHECK GIVEN VALUES \n ");
					exit(EXIT_FAILURE);
				}
				if(pos_row==0 && pos_col ==1 && pos_lambda == 2 && pos_stokes_parameters == 3)
				//if(pos_row==2 && pos_col ==3 && pos_lambda == 0 && pos_stokes_parameters == 1)
					correctOrder = 1;

				fits_get_hdrspace(fptr, &nkeys, NULL, &status); 
				image->nkeys = nkeys;
				image->vCard = (char**) malloc( nkeys * sizeof(char*));
				image->vKeyname = (char**) malloc( nkeys * sizeof(char*));
				for (i = 1; i <= nkeys; i++) {
					char * card = malloc(FLEN_CARD * sizeof(char));
					char * keyname = malloc(FLEN_CARD * sizeof(char));
					fits_read_record(fptr, i, card, &status);
					fits_read_keyn(fptr,i,keyname, value, NULL, &status);
					image->vCard[i-1] = card;
					image->vKeyname[i-1] = keyname;
				}
				image->bitpix = bitpix;
				image->naxis  = naxis;
				image->rows=(configCrontrolFile->subx2-configCrontrolFile->subx1)+1;
				image->rows_original=naxes[pos_row];
				image->cols= (configCrontrolFile->suby2-configCrontrolFile->suby1)+1;
				image->cols_original=naxes[pos_col];

				image->nLambdas=naxes[pos_lambda];
				image->numStokes=naxes[pos_stokes_parameters];
				if(image->numStokes!=4){
					printf("\n************** PLEASE REVIEW THE ORDER OF HEADER NAXIS. DIMENSION OF STOKES MUST HAVE AS VALUE: 4 \n");
					exit(EXIT_FAILURE);
				}
				image->numPixels = image->cols * image->rows ; // we will read the image by columns 

				image->pos_lambda = pos_lambda;
				image->pos_col = pos_col;
				image->pos_row = pos_row;
				image->pos_stokes_parameters = pos_stokes_parameters;
				numPixelsFitsFile = image->rows*image->cols*image->nLambdas*image->numStokes;
				
				
				// allocate memory to read all pixels in the same array 
				float * imageTemp = calloc(numPixelsFitsFile, sizeof(float));
				if (!imageTemp)  {
					printf("ERROR ALLOCATION MEMORY FOR TEMP IMAGE");
					return NULL;
          		}
				
				long fpixelBegin [4] = {1,1,1,1}; 
				long fpixelEnd [4] = {1,1,1,1}; 
				long inc [4] = {1,1,1,1};
				fpixelBegin[pos_row] = configCrontrolFile->subx1;
				fpixelEnd[pos_row] = configCrontrolFile->subx2;
				fpixelBegin[pos_col] = configCrontrolFile->suby1;
				fpixelEnd[pos_col] = configCrontrolFile->suby2;

				fpixelEnd[pos_lambda] = naxes[pos_lambda];
				fpixelEnd[pos_stokes_parameters] = naxes[pos_stokes_parameters];

				fits_read_subset(fptr, TFLOAT, fpixelBegin, fpixelEnd, inc, &nulval, imageTemp, &anynul, &status);
				if(status){
					fits_report_error(stderr, status);
               		return NULL;	
				}

				// allocate memory for reorder the image
				image->pixels = calloc(image->numPixels, sizeof(vpixels));
				if(forParallel){
					image->vLambdaImagen = NULL;
					image->pixels = NULL;
					image->spectroImagen = calloc(image->numPixels*image->nLambdas*image->numStokes, sizeof(float));
				}
				else{
					image->pixels = calloc(image->numPixels, sizeof(vpixels));
					for( i=0;i<image->numPixels;i++){
						image->pixels[i].spectro = calloc ((image->numStokes*image->nLambdas),sizeof(float));
						image->pixels[i].vLambda = NULL;
						image->pixels[i].nLambda = image->nLambdas;
					}					
					image->vLambdaImagen = NULL;
					image->spectroImagen = NULL;
				}
				//printf("\n Número de pixeles: %d", image->numPixels);
				//printf("\n ***********************************************");

				
				if(naxis==4){ // image with 4 dimension 
					
					int sizeDim0 = (fpixelEnd[0]-(fpixelBegin[0]-1));
					int sizeDim1 = (fpixelEnd[1]-(fpixelBegin[1]-1));
					int sizeDim2 = (fpixelEnd[2]-(fpixelBegin[2]-1));
					int sizeDim3 = (fpixelEnd[3]-(fpixelBegin[3]-1));
					image->naxes = calloc(4,sizeof(long));
					image->naxes_original = calloc(4,sizeof(long));
					image->naxes[pos_row] = (fpixelEnd[pos_row]-(fpixelBegin[pos_row]-1));
					image->naxes_original[pos_row] = naxes[pos_row];
					image->naxes[pos_col] = (fpixelEnd[pos_col]-(fpixelBegin[pos_col]-1));
					image->naxes_original[pos_col] = naxes[pos_col];
					image->naxes[pos_lambda] = (fpixelEnd[pos_lambda]-(fpixelBegin[pos_lambda]-1));
					image->naxes_original[pos_lambda] = naxes[pos_lambda];
					image->naxes[pos_stokes_parameters] = (fpixelEnd[pos_stokes_parameters]-(fpixelBegin[pos_stokes_parameters]-1));
					image->naxes_original[pos_stokes_parameters] = naxes[pos_stokes_parameters];


					if(correctOrder){
						for( i=0; i<sizeDim3;i++){
							for( j=0; j<sizeDim2;j++){
								for( k=0;k<sizeDim1;k++){
									for( h=0;h<sizeDim0;h++){
										if(forParallel){
											image->spectroImagen[ (((i*sizeDim2) + j)*(image->nLambdas*image->numStokes)) + (image->nLambdas * k) + h] = imageTemp [(i*sizeDim2*sizeDim1*sizeDim1) + (j*sizeDim1*sizeDim0) + (k*sizeDim0) + h];  // I =0, Q = 1, U = 2, V = 3
										}
										else{
											image->pixels[(i*sizeDim2) + j].spectro[h + (image->nLambdas * k)] = imageTemp [(i*sizeDim2*sizeDim1*sizeDim0) + (j*sizeDim1*sizeDim0) + (k*sizeDim0) + h];  // I =0, Q = 1, U = 2, V = 3
										}
									}
								}
							}
						}						
					}
					else{						
						int currentLambda = 0, currentRow = 0, currentStokeParameter=0, currentCol = 0, currentPixel;
						for( i=0; i<sizeDim3;i++){
							for( j=0; j<sizeDim2;j++){
								for( k=0;k<sizeDim1;k++){
									for( h=0;h<sizeDim0;h++){
										PRECISION pixel = 0.0;
										//fits_read_pix(fptr, datatype, fpixel, 1, &nulval, &pixel, &anynul, &status);
										// I NEED TO KNOW THE CURRENT POSITION OF EACH ITERATOR 
										switch (pos_lambda)
										{
											case 0:
												currentLambda = h;
												break;
											case 1:
												currentLambda = k;
												break;
											case 2:
												currentLambda = j;
												break;
											case 3:
												currentLambda = i;
												break;																						
										}
										switch (pos_stokes_parameters)
										{
											case 0:
												currentStokeParameter = h;
												break;
											case 1:
												currentStokeParameter = k;
												break;
											case 2:
												currentStokeParameter = j;
												break;
											case 3:
												currentStokeParameter = i;
												break;																						
										}
										switch (pos_row)
										{
											case 0:
												currentRow = h;
												break;
											case 1:
												currentRow = k;
												break;
											case 2:
												currentRow = j;
												break;
											case 3:
												currentRow = i;
												break;																						
										}
										switch (pos_col)
										{
											case 0:
												currentCol = h;
												break;
											case 1:
												currentCol = k;
												break;
											case 2:
												currentCol = j;
												break;
											case 3:
												currentCol = i;
												break;																						
										}			
										pixel = imageTemp [(i*sizeDim2*sizeDim1*sizeDim0) + (j*sizeDim1*sizeDim0) + (k*sizeDim0) + h];
										currentPixel = (currentCol*(fpixelEnd[pos_row]-(fpixelBegin[pos_row]-1))) + currentRow;
										if(forParallel){
											image->spectroImagen[(currentPixel *(image->nLambdas*image->numStokes)) + (image->nLambdas * currentStokeParameter) + currentLambda] = pixel;
										}
										else
										{
											image->pixels[currentPixel].spectro[currentLambda + (image->nLambdas * currentStokeParameter)] = pixel;  // I =0, Q = 1, U = 2, V = 3
										}
									}
								}
							}
						}
					}
				}

				free(imageTemp);
				fits_close_file(fptr, &status);
				if (status){
					fits_report_error(stderr, status);
					return NULL;
				}				
			}
		}
		else{
			return NULL;  // we are interested only in FITS image
		}
	}
   else{ // IN CASE AN ERROR OPENING THE FILE RETURN THE ERROR CODE
      if (status) fits_report_error(stderr, status); 
      return NULL;
   }
	
	return image; 
}


/**
 * This function read the lambda values for the image from the file "fitsFileLambda" and store it into a struct of FitsImage. The file of spectro must
 * be read it before call this method. 
 * fitsFileLambda --> name of the fits file to read with lambda values 
 * fitsImage --> struct of image 
 * @return Vector with 
 */

PRECISION * readFitsLambdaToArray (const char * fitsFileLambda,  int * indexLine, int * nLambda){
	int i;
	fitsfile *fptr;   /* FITS file pointer, defined in fitsio.h */
	int status = 0;   /* CFITSIO status value MUST be initialized to zero! */
	PRECISION  nulval = 0.; // define null value to 0 because the performance to read from fits file is better doing this. 
	PRECISION * vLambda = NULL;
	int bitpix, naxis, anynul;
	long naxes [4] = {1,1,1,1}; /* The maximun number of dimension that we will read is 4*/
	
	/*printf("\n READING IMAGE WITH LAMBDA ");
	printf("\n**********");*/
	if (!fits_open_file(&fptr, fitsFileLambda, READONLY, &status)){
		if (!fits_get_img_param(fptr, 4, &bitpix, &naxis, naxes, &status) ){
			if(naxis!=2  || naxis!=4){
				if(naxis == 2){ // array of lambads 

					*nLambda = naxes[0];
					long fpixel [2] = {1,1};
					i=0;
					vLambda = calloc(*nLambda,sizeof(PRECISION));
					for(fpixel[1]=1;fpixel[1]<=naxes[1];fpixel[1]++){
						for(fpixel[0]=1;fpixel[0]<=naxes[0];fpixel[0]++){
							PRECISION lambdaAux;
							fits_read_pix(fptr, TDOUBLE, fpixel, 1, &nulval, &lambdaAux, &anynul, &status) ;		
							if(fpixel[1]==1)
								*indexLine = (int) lambdaAux;
							else
								vLambda[i++] = lambdaAux;
						}
					}
					printf("\nLAMBDAS LEIDOS:\n");
					for(i=0;i<*nLambda;i++){
						printf(" %lf ",vLambda[i]);
					}
					printf("\n");
					//fits_read_pix(fptr, TDOUBLE, fpixel, naxes[0]*naxes[1], &nulval, data, &anynul, &status) ;
					if(status){
						fits_report_error(stderr, status);
						free(vLambda);
						return 0;	
					}
				}
				else if(naxis == 4){  // matrix of lambdas  
					// READ ALL FILE IN ONLY ONE ARRAY 
					// WE ASSUME THAT DATA COMES IN THE FORMAT ROW x COL x LAMBDA					
					int numLambdas2Read = naxes[0]*naxes[1]*naxes[2];
					//fits_read_img(fptr, datatype, first, numLambdas2Read, &nulval, vAuxLambdas, &anynul, &status);
					long fpixel [3] = {1,1,1};
					fits_read_pix(fptr, TDOUBLE, fpixel, numLambdas2Read, &nulval, vLambda, &anynul, &status);
					if(status){
						fits_report_error(stderr, status);
						free(vLambda);
						return 0;	
					}
				}
			}
			else{
				printf("\n NAXIS FROM LAMBA FILE IS NOT VALID %d ** \n", naxis);
				return 0;
			}
			// CLOSE FILE FITS LAMBDAS
			fits_close_file(fptr, &status);
			if (status){
				fits_report_error(stderr, status);
				return 0;
			}
		}
		else {
			printf("\n WE CAN NOT OPEN FILE OF LAMBAS ** \n");
			if (status) fits_report_error(stderr, status); /* print any error message */
			return 0;
		}
	}
	else {
		printf("\n WE CAN NOT READ PARAMETERS FROM THE FILE  %s \n",fitsFileLambda);
		if (status) fits_report_error(stderr, status); /* print any error message */
		return 0;
	}
	/*printf("\n LAMBDA IMAGE READ");
	printf("\n**********");*/

	return vLambda;

}


float * readPerStrayLightFile (const char * perFileStrayLight, int nlambda, PRECISION *  vOffsetsLambda){

	REAL * slightPER = NULL;
	FILE * fReadStrayLight;
	char * line = NULL;
	size_t len = 0;
	ssize_t read;
	fReadStrayLight = fopen(perFileStrayLight, "r");
	
	int contLine=0;
	if (fReadStrayLight == NULL)
	{
		printf("Error opening file stray light it's possible that the file doesn't exist. Please verify it. \n");
		printf("\n ******* THIS IS THE NAME OF THE FILE RECEVIED : %s \n", perFileStrayLight);
		fclose(fReadStrayLight);
		return NULL;
	}
	slightPER = calloc(nlambda*NPARMS,sizeof(float));
	float aux1, aux2,aux3,aux4,aux5,aux6;
	int correcto = 1;
	while ((read = getline(&line, &len, fReadStrayLight)) != -1 && contLine<nlambda && correcto) {
		sscanf(line,"%e %e %e %e %e %e",&aux1,&aux2,&aux3,&aux4,&aux5,&aux6);
		if(aux2 != trunc(vOffsetsLambda[contLine]) ){ 
			correcto = 0;
			printf("\n WAVE LENGTH: %f   -   %f", aux2, trunc(vOffsetsLambda[contLine]));
		}
		slightPER[contLine] = aux3;
		slightPER[contLine + nlambda] = aux4;
		slightPER[contLine + nlambda * 2] = aux5;
		slightPER[contLine + nlambda * 3] = aux6;
		contLine++;
	}
	fclose(fReadStrayLight);
	if(!correcto){
		printf("\nThe observed profiles and the stray light profiles do not have the same wavelengths and/or line index\n");
		exit(EXIT_FAILURE);
	}
	/*printf("\n");
	for(int i=0;i<nlambda*NPARMS;i++){					
		printf("%f\t",slightPER[i]);
	}				
	printf("\n");*/
	return slightPER;
}


float * readFitsStrayLightFile (ConfigControl * configCrontrolFile,int * nl_straylight,int * ns_straylight,int * nx_straylight,int * ny_straylight){
	
	fitsfile *fptr;   /* FITS file pointer, defined in fitsio.h */
	
	int status = 0;   /* CFITSIO status value MUST be initialized to zero! */
	float nulval = 0.; // define null value to 0 because the performance to read from fits file is better doing this. 
	int bitpix, naxis, anynul;
	long naxes [4] = {1,1,1,1}; /* The maximun number of dimension that we will read is 4*/
	int i, j, k, h;
	float * slight = NULL;
	/*printf("\n READING IMAGE WITH LAMBDA ");
	printf("\n**********");*/
	if (!fits_open_file(&fptr, configCrontrolFile->StrayLightFile, READONLY, &status)){
		if (!fits_get_img_param(fptr, 4, &bitpix, &naxis, naxes, &status) ){
			if(bitpix != FLOAT_IMG){
				printf("\n ERROR: the datatype of FITS spectro image must be FLOAT\n");
				printf("\n EXITING THE PROGRAM\n");
				fits_close_file(fptr, &status);
				exit(EXIT_FAILURE);
			}

			if(naxis==2){
				if(naxes[0]>naxes[1]){
					*nl_straylight = naxes[0];
					*ns_straylight = naxes[1];
				}
				else
				{
					*nl_straylight = naxes[1];
					*ns_straylight = naxes[0];
				}
				
				// READ ALL FILE IN ONLY ONE ARRAY 
				// WE ASSUME THAT DATA COMES IN THE FORMAT ROW x COL x LAMBDA
				slight = calloc(naxes[0]*naxes[1], sizeof(float));
				long fpixel [2] = {1,1};
				fits_read_pix(fptr, TFLOAT, fpixel, naxes[0]*naxes[1], &nulval, slight, &anynul, &status);
				if(status){
					fits_report_error(stderr, status);
					slight = NULL;	
				}
				
				int index=0;
				if(*nl_straylight!=naxes[0]){
					float * vStrayLight_aux = calloc(naxes[0]*naxes[1], sizeof(float));
					for(i=0;i<naxes[0];i++){
						for(j=0;j<naxes[1];j++){
							vStrayLight_aux[index++] = slight[i + j*(naxes[0])];
						}
					}
					free(slight);
					slight = vStrayLight_aux;
				}
				
			}
			else if (naxis==4){
				FitsImage * image =  malloc(sizeof(FitsImage));
				char comment[FLEN_CARD];   /* Standard string lengths defined in fitsio.h */
				// We assume that we have only on HDU as primary 
				int header = 1;
				if(fits_read_key(fptr, TSTRING, CTYPE1, image->ctype_1, comment, &status)){
					header = 0 ;
					status = 0;
				} 
				if(fits_read_key(fptr, TSTRING, CTYPE2, image->ctype_2, comment, &status)){ 
					header = 0 ;
					status = 0;
				}
				if(fits_read_key(fptr, TSTRING, CTYPE3, image->ctype_3, comment, &status)){
					header = 0 ;
					status = 0;
				}
				if(fits_read_key(fptr, TSTRING, CTYPE4, image->ctype_4, comment, &status)){
					header = 0 ;
					status = 0;
				}
				// ORDER MUST BE CTYPE1->'HPLN-TAN',CTYPE2->'HPLT-TAN',CTYPE3->'WAVE-GRI',CTYPE4->'STOKES'
				// int pos_row = 0, pos_col = 1, pos_lambda = 2, pos_stokes_parameters = 3;
				int correctOrder =0;
				// GET THE CURRENT POSITION OF EVERY PARAMETER
				int pos_lambda; 
				int pos_row;
				int pos_col;
				int pos_stokes_parameters;
				if(header){
					// LAMBDA POSITION
					if(strcmp(image->ctype_1,CTYPE_WAVE)==0) pos_lambda = 0;
					if(strcmp(image->ctype_2,CTYPE_WAVE)==0) pos_lambda = 1;
					if(strcmp(image->ctype_3,CTYPE_WAVE)==0) pos_lambda = 2;
					if(strcmp(image->ctype_4,CTYPE_WAVE)==0) pos_lambda = 3;

					// HPLN TAN 
					if(strcmp(image->ctype_1,CTYPE_HPLN_TAN)==0) pos_row = 0;
					if(strcmp(image->ctype_2,CTYPE_HPLN_TAN)==0) pos_row = 1;
					if(strcmp(image->ctype_3,CTYPE_HPLN_TAN)==0) pos_row = 2;
					if(strcmp(image->ctype_4,CTYPE_HPLN_TAN)==0) pos_row = 3;

					// HPLT TAN 
					if(strcmp(image->ctype_1,CTYPE_HPLT_TAN)==0) pos_col = 0;
					if(strcmp(image->ctype_2,CTYPE_HPLT_TAN)==0) pos_col = 1;
					if(strcmp(image->ctype_3,CTYPE_HPLT_TAN)==0) pos_col = 2;
					if(strcmp(image->ctype_4,CTYPE_HPLT_TAN)==0) pos_col = 3;			

					// Stokes paramter position , 
					if(strcmp(image->ctype_1,CTYPE_STOKES)==0) pos_stokes_parameters = 0;
					if(strcmp(image->ctype_2,CTYPE_STOKES)==0) pos_stokes_parameters = 1;
					if(strcmp(image->ctype_3,CTYPE_STOKES)==0) pos_stokes_parameters = 2;
					if(strcmp(image->ctype_4,CTYPE_STOKES)==0) pos_stokes_parameters = 3;
				}
				else{
					int num_stokes = 10000;
					for(i=0;i<naxis;i++){
						if(naxes[i]<num_stokes){
							pos_stokes_parameters =i;
							num_stokes = naxes[i];
						}
					}
					int num_lambda = 10000;
					for(i=0;i<naxis;i++){
						if(i!=pos_stokes_parameters){
							if(naxes[i]<num_lambda){
								pos_lambda =i;
								num_lambda = naxes[i];
							}
						}
					}
					if( (pos_stokes_parameters == 0 && pos_lambda == 1) || (pos_stokes_parameters == 1 && pos_lambda == 0) ){ 
						pos_row = 2;
						pos_col = 3;
					}
					if( (pos_stokes_parameters == 0 && pos_lambda == 2) || (pos_stokes_parameters == 2 && pos_lambda == 0) ){ 
						pos_row = 1;
						pos_col = 3;
					} 
					if( (pos_stokes_parameters == 0 && pos_lambda == 3) || (pos_stokes_parameters == 3 && pos_lambda == 0) ){ 
						pos_row = 1;
						pos_col = 2;					
					}
					if( (pos_stokes_parameters == 1 && pos_lambda == 2) || (pos_stokes_parameters == 2 && pos_lambda == 1) ){ 
						pos_row = 0;
						pos_col = 3;					
					}
					if( (pos_stokes_parameters == 1 && pos_lambda == 3) || (pos_stokes_parameters == 3 && pos_lambda == 1) ){ 
						pos_row = 0;
						pos_col = 2;					
					}
					if( (pos_stokes_parameters == 2 && pos_lambda == 3) || (pos_stokes_parameters == 3 && pos_lambda == 2) ){ 
						pos_row = 0;
						pos_col = 1;					
					}					
				}
				if(pos_row==0 && pos_col ==1 && pos_lambda == 2 && pos_stokes_parameters == 3)
				//if(pos_row==2 && pos_col ==3 && pos_lambda == 0 && pos_stokes_parameters == 1)
					correctOrder = 1;

				float * imageTemp;
				
				image->rows=naxes[pos_row];
				image->cols=naxes[pos_col];
				image->nLambdas=naxes[pos_lambda];
				image->numStokes=naxes[pos_stokes_parameters];
				if(image->numStokes!=4){
					printf("\n************** PLEASE REVIEW THE ORDER OF HEADER NAXIS. DIMENSION OF STOKES MUST HAVE AS VALUE: 4 . STRAY LIGHT FILE \n");
					exit(EXIT_FAILURE);
				}
				image->numPixels = image->cols * image->rows; // we will read the image by columns 
				image->pos_lambda = pos_lambda;
				image->pos_col = pos_col;
				image->pos_row = pos_row;
				image->pos_stokes_parameters = pos_stokes_parameters;
				int numPixelsFitsFile = image->rows*image->cols*image->nLambdas*image->numStokes;
				imageTemp = calloc(numPixelsFitsFile, sizeof(float));
				if (!imageTemp)  {
					printf("ERROR ALLOCATION MEMORY FOR TEMP IMAGE");
					image= NULL;
					slight = NULL;
          		}
				long fpixel [4] = {1,1,1,1}; 
				fits_read_pix(fptr, TFLOAT, fpixel, numPixelsFitsFile, &nulval, imageTemp, &anynul, &status);
				if(status){
					fits_report_error(stderr, status);
               		image= NULL;
					slight = NULL;	
				}
				//image->spectroImagen = calloc(image->numPixels*image->nLambdas*image->numStokes, sizeof(float));
				slight = calloc(image->numPixels*image->nLambdas*image->numStokes, sizeof(float));
				
				*nx_straylight = image->rows;
				*ny_straylight = image->cols;
				*ns_straylight = image->numStokes;
				*nl_straylight = image->nLambdas;

				if(correctOrder){
					// i = cols, j = rows, k = stokes, h = lambda
					for( i=0; i<naxes[3];i++){
						for( j=0; j<naxes[2];j++){
							for( k=0;k<naxes[1];k++){
								for( h=0;h<naxes[0];h++){
									//image->spectroImagen[ (((i*naxes[2]) + j)*(image->nLambdas*image->numStokes)) + (image->nLambdas * k) + h] = imageTemp [(i*naxes[2]*naxes[1]*naxes[0]) + (j*naxes[1]*naxes[0]) + (k*naxes[0]) + h];  // I =0, Q = 1, U = 2, V = 3
									slight[ (((i*naxes[2]) + j)*(image->nLambdas*image->numStokes)) + (image->nLambdas * k) + h] = imageTemp [(i*naxes[2]*naxes[1]*naxes[0]) + (j*naxes[1]*naxes[0]) + (k*naxes[0]) + h];  // I =0, Q = 1, U = 2, V = 3
								}
							}
						}
					}					
				}
				else{
					int currentLambda = 0, currentRow = 0, currentStokeParameter=0, currentCol = 0, currentPixel;
					for( i=0; i<naxes[3];i++){
						for( j=0; j<naxes[2];j++){
							for( k=0;k<naxes[1];k++){
								for( h=0;h<naxes[0];h++){
									float pixel = 0.0;
									//fits_read_pix(fptr, datatype, fpixel, 1, &nulval, &pixel, &anynul, &status);
									// I NEED TO KNOW THE CURRENT POSITION OF EACH ITERATOR 
									switch (pos_lambda)
									{
										case 0:
											currentLambda = h;
											break;
										case 1:
											currentLambda = k;
											break;
										case 2:
											currentLambda = j;
											break;
										case 3:
											currentLambda = i;
											break;																						
									}
									switch (pos_stokes_parameters)
									{
										case 0:
											currentStokeParameter = h;
											break;
										case 1:
											currentStokeParameter = k;
											break;
										case 2:
											currentStokeParameter = j;
											break;
										case 3:
											currentStokeParameter = i;
											break;																						
									}
									switch (pos_row)
									{
										case 0:
											currentRow = h;
											break;
										case 1:
											currentRow = k;
											break;
										case 2:
											currentRow = j;
											break;
										case 3:
											currentRow = i;
											break;																						
									}
									switch (pos_col)
									{
										case 0:
											currentCol = h;
											break;
										case 1:
											currentCol = k;
											break;
										case 2:
											currentCol = j;
											break;
										case 3:
											currentCol = i;
											break;																						
									}			
									pixel = imageTemp [(i*naxes[2]*naxes[1]*naxes[0]) + (j*naxes[1]*naxes[0]) + (k*naxes[0]) + h];
									currentPixel = (currentCol*naxes[pos_row]) + currentRow;
									slight[(currentPixel *(image->nLambdas*image->numStokes)) + (image->nLambdas * currentStokeParameter) + currentLambda] = pixel;
								}
							}
						}
					}
				}

				free(imageTemp);
				if (status){
					fits_close_file(fptr, &status);
					fits_report_error(stderr, status);
					image= NULL;
					slight = NULL;
				}	
			}
			else {
				printf("\n NAXIS FROM STRAY LIGHT FILE IS NOT VALID %d ** \n", naxis);
				slight = NULL;
			}
			// CLOSE FILE FITS LAMBDAS
			fits_close_file(fptr, &status);
			if (status){
				fits_report_error(stderr, status);
				slight = NULL;
			}
		}
		else {
			printf("\n WE CAN NOT OPEN FILE OF STRAY LIGHT ** \n");
			if (status) fits_report_error(stderr, status); /* print any error message */
			slight = NULL;
		}
	}
	else {
		printf("\n WE CAN NOT READ PARAMETERS FROM THE FILE  %s \n", configCrontrolFile->StrayLightFile);
		if (status) fits_report_error(stderr, status); /* print any error message */
		slight = NULL;
	}
	

	return slight;
	
}


float * readFitsStrayLightFileSubSet (ConfigControl * configCrontrolFile,int * nl_straylight,int * ns_straylight,int * nx_straylight,int * ny_straylight){
	
	fitsfile *fptr;   /* FITS file pointer, defined in fitsio.h */
	
	int status = 0;   /* CFITSIO status value MUST be initialized to zero! */
	float nulval = 0.; // define null value to 0 because the performance to read from fits file is better doing this. 
	int bitpix, naxis, anynul;
	long naxes [4] = {1,1,1,1}; /* The maximun number of dimension that we will read is 4*/
	int i, j, k, h;
	float * slight = NULL;
	/*printf("\n READING IMAGE WITH LAMBDA ");
	printf("\n**********");*/
	if (!fits_open_file(&fptr, configCrontrolFile->StrayLightFile, READONLY, &status)){
		if (!fits_get_img_param(fptr, 4, &bitpix, &naxis, naxes, &status) ){
			if(bitpix != FLOAT_IMG){
				printf("\n ERROR: the datatype of FITS spectro image must be FLOAT\n");
				printf("\n EXITING THE PROGRAM\n");
				fits_close_file(fptr, &status);
				exit(EXIT_FAILURE);
			}

			if(naxis==2){
				if(naxes[0]>naxes[1]){
					*nl_straylight = naxes[0];
					*ns_straylight = naxes[1];
				}
				else
				{
					*nl_straylight = naxes[1];
					*ns_straylight = naxes[0];
				}
				
				// READ ALL FILE IN ONLY ONE ARRAY 
				// WE ASSUME THAT DATA COMES IN THE FORMAT ROW x COL x LAMBDA
				slight = calloc(naxes[0]*naxes[1], sizeof(float));
				long fpixel [2] = {1,1};
				fits_read_pix(fptr, TFLOAT, fpixel, naxes[0]*naxes[1], &nulval, slight, &anynul, &status);
				if(status){
					fits_report_error(stderr, status);
					slight = NULL;	
				}
				
				int index=0;
				if(*nl_straylight!=naxes[0]){
					float * vStrayLight_aux = calloc(naxes[0]*naxes[1], sizeof(float));
					for(i=0;i<naxes[0];i++){
						for(j=0;j<naxes[1];j++){
							vStrayLight_aux[index++] = slight[i + j*(naxes[0])];
						}
					}
					free(slight);
					slight = vStrayLight_aux;
				}
				
			}
			else if (naxis==4){
				FitsImage * image =  malloc(sizeof(FitsImage));
				char comment[FLEN_CARD];   /* Standard string lengths defined in fitsio.h */
				// We assume that we have only on HDU as primary 
				int header = 1;
				if(fits_read_key(fptr, TSTRING, CTYPE1, image->ctype_1, comment, &status)){
					header = 0 ;
					status = 0;
				} 
				if(fits_read_key(fptr, TSTRING, CTYPE2, image->ctype_2, comment, &status)){ 
					header = 0 ;
					status = 0;
				}
				if(fits_read_key(fptr, TSTRING, CTYPE3, image->ctype_3, comment, &status)){
					header = 0 ;
					status = 0;
				}
				if(fits_read_key(fptr, TSTRING, CTYPE4, image->ctype_4, comment, &status)){
					header = 0 ;
					status = 0;
				}
				// ORDER MUST BE CTYPE1->'HPLN-TAN',CTYPE2->'HPLT-TAN',CTYPE3->'WAVE-GRI',CTYPE4->'STOKES'
				// int pos_row = 0, pos_col = 1, pos_lambda = 2, pos_stokes_parameters = 3;
				int correctOrder =0;
				// GET THE CURRENT POSITION OF EVERY PARAMETER
				int pos_lambda; 
				int pos_row;
				int pos_col;
				int pos_stokes_parameters;
				if(header){
					// LAMBDA POSITION
					if(strcmp(image->ctype_1,CTYPE_WAVE)==0) pos_lambda = 0;
					if(strcmp(image->ctype_2,CTYPE_WAVE)==0) pos_lambda = 1;
					if(strcmp(image->ctype_3,CTYPE_WAVE)==0) pos_lambda = 2;
					if(strcmp(image->ctype_4,CTYPE_WAVE)==0) pos_lambda = 3;

					// HPLN TAN 
					if(strcmp(image->ctype_1,CTYPE_HPLN_TAN)==0) pos_row = 0;
					if(strcmp(image->ctype_2,CTYPE_HPLN_TAN)==0) pos_row = 1;
					if(strcmp(image->ctype_3,CTYPE_HPLN_TAN)==0) pos_row = 2;
					if(strcmp(image->ctype_4,CTYPE_HPLN_TAN)==0) pos_row = 3;

					// HPLT TAN 
					if(strcmp(image->ctype_1,CTYPE_HPLT_TAN)==0) pos_col = 0;
					if(strcmp(image->ctype_2,CTYPE_HPLT_TAN)==0) pos_col = 1;
					if(strcmp(image->ctype_3,CTYPE_HPLT_TAN)==0) pos_col = 2;
					if(strcmp(image->ctype_4,CTYPE_HPLT_TAN)==0) pos_col = 3;			

					// Stokes paramter position , 
					if(strcmp(image->ctype_1,CTYPE_STOKES)==0) pos_stokes_parameters = 0;
					if(strcmp(image->ctype_2,CTYPE_STOKES)==0) pos_stokes_parameters = 1;
					if(strcmp(image->ctype_3,CTYPE_STOKES)==0) pos_stokes_parameters = 2;
					if(strcmp(image->ctype_4,CTYPE_STOKES)==0) pos_stokes_parameters = 3;
				}
				else{
					int num_stokes = 10000;
					for(i=0;i<naxis;i++){
						if(naxes[i]<num_stokes){
							pos_stokes_parameters =i;
							num_stokes = naxes[i];
						}
					}
					int num_lambda = 10000;
					for(i=0;i<naxis;i++){
						if(i!=pos_stokes_parameters){
							if(naxes[i]<num_lambda){
								pos_lambda =i;
								num_lambda = naxes[i];
							}
						}
					}
					if( (pos_stokes_parameters == 0 && pos_lambda == 1) || (pos_stokes_parameters == 1 && pos_lambda == 0) ){ 
						pos_row = 2;
						pos_col = 3;
					}
					if( (pos_stokes_parameters == 0 && pos_lambda == 2) || (pos_stokes_parameters == 2 && pos_lambda == 0) ){ 
						pos_row = 1;
						pos_col = 3;
					} 
					if( (pos_stokes_parameters == 0 && pos_lambda == 3) || (pos_stokes_parameters == 3 && pos_lambda == 0) ){ 
						pos_row = 1;
						pos_col = 2;					
					}
					if( (pos_stokes_parameters == 1 && pos_lambda == 2) || (pos_stokes_parameters == 2 && pos_lambda == 1) ){ 
						pos_row = 0;
						pos_col = 3;					
					}
					if( (pos_stokes_parameters == 1 && pos_lambda == 3) || (pos_stokes_parameters == 3 && pos_lambda == 1) ){ 
						pos_row = 0;
						pos_col = 2;					
					}
					if( (pos_stokes_parameters == 2 && pos_lambda == 3) || (pos_stokes_parameters == 3 && pos_lambda == 2) ){ 
						pos_row = 0;
						pos_col = 1;					
					}					
				}
				if(pos_row==0 && pos_col ==1 && pos_lambda == 2 && pos_stokes_parameters == 3)
				//if(pos_row==2 && pos_col ==3 && pos_lambda == 0 && pos_stokes_parameters == 1)
					correctOrder = 1;

				float * imageTemp;
				
				if( configCrontrolFile->subx2 > naxes[pos_row] || configCrontrolFile->subx1>configCrontrolFile->subx2 || configCrontrolFile->suby2 > naxes[pos_col] || configCrontrolFile->suby1 > configCrontrolFile->suby2){
					printf("\n ERROR IN THE DIMENSIONS, PLEASE CHECK GIVEN VALUES \n ");
					exit(EXIT_FAILURE);
				}
				
	
				
				image->rows=(configCrontrolFile->subx2-configCrontrolFile->subx1)+1;
				image->cols= (configCrontrolFile->suby2-configCrontrolFile->suby1)+1;
				image->nLambdas=naxes[pos_lambda];
				image->numStokes=naxes[pos_stokes_parameters];
				if(image->numStokes!=4){
					printf("\n************** PLEASE REVIEW THE ORDER OF HEADER NAXIS. DIMENSION OF STOKES MUST HAVE AS VALUE: 4 . STRAY LIGHT FILE \n");
					exit(EXIT_FAILURE);
				}
				image->numPixels = image->cols * image->rows; // we will read the image by columns 
				image->pos_lambda = pos_lambda;
				image->pos_col = pos_col;
				image->pos_row = pos_row;
				image->pos_stokes_parameters = pos_stokes_parameters;
				int numPixelsFitsFile = image->rows*image->cols*image->nLambdas*image->numStokes;
				imageTemp = calloc(numPixelsFitsFile, sizeof(float));
				if (!imageTemp)  {
					printf("ERROR ALLOCATION MEMORY FOR TEMP IMAGE");
					image= NULL;
					slight = NULL;
          		}
				long fpixelBegin [4] = {1,1,1,1}; 
				long fpixelEnd [4] = {1,1,1,1}; 
				long inc [4] = {1,1,1,1};
				fpixelBegin[pos_row] = configCrontrolFile->subx1;
				fpixelEnd[pos_row] = configCrontrolFile->subx2;
				fpixelBegin[pos_col] = configCrontrolFile->suby1;
				fpixelEnd[pos_col] = configCrontrolFile->suby2;

				fpixelEnd[pos_lambda] = naxes[pos_lambda];
				fpixelEnd[pos_stokes_parameters] = naxes[pos_stokes_parameters];

				fits_read_subset(fptr, TFLOAT, fpixelBegin, fpixelEnd, inc, &nulval, imageTemp, &anynul, &status);
				if(status){
					fits_report_error(stderr, status);
               		image= NULL;
					slight = NULL;	
				}
				//image->spectroImagen = calloc(image->numPixels*image->nLambdas*image->numStokes, sizeof(float));
				slight = calloc(image->numPixels*image->nLambdas*image->numStokes, sizeof(float));
				
				*nx_straylight = image->rows;
				*ny_straylight = image->cols;
				*ns_straylight = image->numStokes;
				*nl_straylight = image->nLambdas;
				int sizeDim0 = (fpixelEnd[0]-(fpixelBegin[0]-1));
				int sizeDim1 = (fpixelEnd[1]-(fpixelBegin[1]-1));
				int sizeDim2 = (fpixelEnd[2]-(fpixelBegin[2]-1));
				int sizeDim3 = (fpixelEnd[3]-(fpixelBegin[3]-1));

				if(correctOrder){
					// i = cols, j = rows, k = stokes, h = lambda
					for( i=0; i<sizeDim3;i++){
						for( j=0; j<sizeDim2;j++){
							for( k=0;k<sizeDim1;k++){
								for( h=0;h<sizeDim0;h++){
									slight[ (((i*sizeDim2) + j)*(image->nLambdas*image->numStokes)) + (image->nLambdas * k) + h] = imageTemp [(i*sizeDim2*sizeDim1*sizeDim1) + (j*sizeDim1*sizeDim0) + (k*sizeDim0) + h];  // I =0, Q = 1, U = 2, V = 3
								}
							}
						}
					}
				}
				else{
					int currentLambda = 0, currentRow = 0, currentStokeParameter=0, currentCol = 0, currentPixel;
					for( i=0; i<sizeDim3;i++){
						for( j=0; j<sizeDim2;j++){
							for( k=0;k<sizeDim1;k++){
								for( h=0;h<sizeDim0;h++){
									PRECISION pixel = 0.0;
									//fits_read_pix(fptr, datatype, fpixel, 1, &nulval, &pixel, &anynul, &status);
									// I NEED TO KNOW THE CURRENT POSITION OF EACH ITERATOR 
									switch (pos_lambda)
									{
										case 0:
											currentLambda = h;
											break;
										case 1:
											currentLambda = k;
											break;
										case 2:
											currentLambda = j;
											break;
										case 3:
											currentLambda = i;
											break;																						
									}
									switch (pos_stokes_parameters)
									{
										case 0:
											currentStokeParameter = h;
											break;
										case 1:
											currentStokeParameter = k;
											break;
										case 2:
											currentStokeParameter = j;
											break;
										case 3:
											currentStokeParameter = i;
											break;																						
									}
									switch (pos_row)
									{
										case 0:
											currentRow = h;
											break;
										case 1:
											currentRow = k;
											break;
										case 2:
											currentRow = j;
											break;
										case 3:
											currentRow = i;
											break;																						
									}
									switch (pos_col)
									{
										case 0:
											currentCol = h;
											break;
										case 1:
											currentCol = k;
											break;
										case 2:
											currentCol = j;
											break;
										case 3:
											currentCol = i;
											break;																						
									}			
									pixel = imageTemp [(i*sizeDim2*sizeDim1*sizeDim0) + (j*sizeDim1*sizeDim0) + (k*sizeDim0) + h];
									currentPixel = (currentCol*(fpixelEnd[pos_row]-(fpixelBegin[pos_row]-1))) + currentRow;
									slight[(currentPixel *(image->nLambdas*image->numStokes)) + (image->nLambdas * currentStokeParameter) + currentLambda] = pixel;
								}
							}
						}
					}
				}
				
				free(imageTemp);
				if (status){
					fits_close_file(fptr, &status);
					fits_report_error(stderr, status);
					image= NULL;
					slight = NULL;
				}	
			}
			else {
				printf("\n NAXIS FROM STRAY LIGHT FILE IS NOT VALID %d ** \n", naxis);
				slight = NULL;
			}
			// CLOSE FILE FITS LAMBDAS
			fits_close_file(fptr, &status);
			if (status){
				fits_report_error(stderr, status);
				slight = NULL;
			}
		}
		else {
			printf("\n WE CAN NOT OPEN FILE OF STRAY LIGHT ** \n");
			if (status) fits_report_error(stderr, status); /* print any error message */
			slight = NULL;
		}
	}
	else {
		printf("\n WE CAN NOT READ PARAMETERS FROM THE FILE  %s \n", configCrontrolFile->StrayLightFile);
		if (status) fits_report_error(stderr, status); /* print any error message */
		slight = NULL;
	}


	return slight;
	
}


int * readFitsMaskFile (const char * fitsMask, int * numRows, int * numCols){
	
	fitsfile *fptr;   /* FITS file pointer, defined in fitsio.h */
	int status = 0;   /* CFITSIO status value MUST be initialized to zero! */
	int nulval = 0; // define null value to 0 because the performance to read from fits file is better doing this. 
	int bitpix, naxis, anynul;
	long naxes [2] = {1,1}; /* The maximun number of dimension that we will read is 4*/
	
	int * vMask = NULL;
	/*printf("\n READING IMAGE WITH LAMBDA ");
	printf("\n**********");*/
	if (!fits_open_file(&fptr, fitsMask, READONLY, &status)){
		if (!fits_get_img_param(fptr, 2, &bitpix, &naxis, naxes, &status) ){

			if(naxis==2){

				/*if( naxes[0]!= numRows || naxes[1]!= numCols){ // stray light has different size of spectra image 
					printf("\n MASK FILE HAS DIFFERENT SIZE OF SPECTRA IMAGE. SIZE SPECTRA %d X %d X . MASK SIZE %ld X %ld ", numRows, numCols , naxes[0], naxes[1]);
					return NULL;
				}*/
				// READ ALL FILE IN ONLY ONE ARRAY 
				// WE ASSUME THAT DATA COMES IN THE FORMAT ROW x COL x LAMBDA
				*numRows = naxes[0];
				*numCols = naxes[1];
				int dimMask = (*numRows)*(*numCols);
				vMask = calloc(dimMask, sizeof(int));
				long fpixel [2] = {1,1};
				fits_read_pix(fptr, TINT, fpixel, dimMask, &nulval, vMask, &anynul, &status);
				if(status){
					fits_report_error(stderr, status);
					return NULL;	
				}
				
			}
			else{
				printf("\n Naxis for MASK file must be 2, current naxis %d ** \n", naxis);
				return 0;
			}
			// CLOSE FILE FITS LAMBDAS
			fits_close_file(fptr, &status);
			if (status){
				fits_report_error(stderr, status);
				return  NULL;
			}
		}
		else {
			printf("\n WE CAN NOT OPEN FILE OF STRAY LIGHT ** \n");
			if (status) fits_report_error(stderr, status); /* print any error message */
			return NULL;
		}
	}
	else {
		printf("\n WE CAN NOT READ PARAMETERS FROM THE FILE  %s \n",fitsMask);
		if (status) fits_report_error(stderr, status); /* print any error message */
		return NULL;
	}


	return vMask;
}

/**
 * 
 * 
 * 
 * 
 * 
 * */
int * readFitsMaskFileSubSet (const char * fitsMask, int * numRows, int * numCols,  ConfigControl * configCrontrolFile){
	
	fitsfile *fptr;   /* FITS file pointer, defined in fitsio.h */
	int status = 0;   /* CFITSIO status value MUST be initialized to zero! */
	int nulval = 0; // define null value to 0 because the performance to read from fits file is better doing this. 
	int bitpix, naxis, anynul;
	long naxes [2] = {1,1}; /* The maximun number of dimension that we will read is 4*/
	
	int * vMask = NULL;
	/*printf("\n READING IMAGE WITH LAMBDA ");
	printf("\n**********");*/
	if (!fits_open_file(&fptr, fitsMask, READONLY, &status)){
		if (!fits_get_img_param(fptr, 2, &bitpix, &naxis, naxes, &status) ){

			if(naxis==2){

				if( configCrontrolFile->subx2 > naxes[0] || configCrontrolFile->subx1>configCrontrolFile->subx2 || configCrontrolFile->suby2 > naxes[1] || configCrontrolFile->suby1 > configCrontrolFile->suby2){
					printf("\n ERROR IN THE DIMENSIONS FOR SUBSET WHEN READ MASK FILE \n ");
					exit(EXIT_FAILURE);
				}
				// READ ALL FILE IN ONLY ONE ARRAY 
				// WE ASSUME THAT DATA COMES IN THE FORMAT ROW x COL x LAMBDA
				*numRows = (configCrontrolFile->subx2-configCrontrolFile->subx1)+1;
				*numCols = (configCrontrolFile->suby2-configCrontrolFile->suby1)+1;

				int dimMask = (*numRows)*(*numCols);
				vMask = calloc(dimMask, sizeof(int));
				long fpixelBegin [4] = {1,1}; 
				long fpixelEnd [4] = {1,1}; 
				long inc [4] = {1,1};
				fpixelBegin[0] = configCrontrolFile->subx1;
				fpixelEnd[0] = configCrontrolFile->subx2;
				fpixelBegin[1] = configCrontrolFile->suby1;
				fpixelEnd[1] = configCrontrolFile->suby2;
				fits_read_subset(fptr, TINT, fpixelBegin, fpixelEnd, inc, &nulval, vMask, &anynul, &status);
				
				if(status){
					fits_report_error(stderr, status);
					return NULL;	
				}
				
			}
			else{
				printf("\n Naxis for MASK file must be 2, current naxis %d ** \n", naxis);
				return 0;
			}
			// CLOSE FILE FITS LAMBDAS
			fits_close_file(fptr, &status);
			if (status){
				fits_report_error(stderr, status);
				return  NULL;
			}
		}
		else {
			printf("\n WE CAN NOT OPEN FILE OF STRAY LIGHT ** \n");
			if (status) fits_report_error(stderr, status); /* print any error message */
			return NULL;
		}
	}
	else {
		printf("\n WE CAN NOT READ PARAMETERS FROM THE FILE  %s \n",fitsMask);
		if (status) fits_report_error(stderr, status); /* print any error message */
		return NULL;
	}


	return vMask;
}

void freeFitsImage(FitsImage * image){
	int i;
	if(image!=NULL){
		if(image->pixels!=NULL){
			for( i=0;i<image->numPixels;i++){
				if(image->pixels[i].spectro!=NULL)
					free(image->pixels[i].spectro);
				if(image->pixels[i].vLambda!=NULL)
					free(image->pixels[i].vLambda);
			}

			free(image->pixels);
		}
		if(image->spectroImagen!=NULL)
			free(image->spectroImagen);
		if(image->vLambdaImagen!=NULL)
			free(image->vLambdaImagen);
		free(image);
	}
	
}


/**
 * 
 * fixed = array with positions to write in the file, Positions are in the following order: 
 * [Eta0,Strength,Vlos,Lambdadopp,Damp,Gamma,Azimuth,S0,S1,Macro,Alpha]
 * */
int writeFitsImageModels(const char * fitsFile, int numRows, int numCols, Init_Model * vInitModel, float * vChisqrf, int * vNumIterPixel, int addChiqr){

	fitsfile *fptr;       /* pointer to the FITS file, defined in fitsio.h */
	int status;
	int i, j, h; // indexes for loops
	long  fpixel;
	int indexModel = 0; 


	int bitpix =  FLOAT_IMG; 
	long naxis =   3;  /* 2-dimensional image */    
	long naxes[3] = { numRows, numCols, NUMBER_PARAM_MODELS+1};   /* Image of numRows X numCols x 10 parameters of model and chisqrf */


	if(addChiqr){
		naxes[2]++;
	}
   

   remove(fitsFile);               /* Delete old file if it already exists */
   status = 0;         /* initialize status before calling fitsio routines */
   if (fits_create_file(&fptr, fitsFile, &status)) /* create new FITS file */
   	printerror( status );           /* call printerror if error occurs */
	
	 /* write the required keywords for the primary array image.     */
    /* Since bitpix = FLOAT_IMG, this will cause cfitsio to create */
    /* a FITS image with BITPIX = -32 (float) .Note that the BSCALE  */
    /* and BZERO keywords will be automatically written by cfitsio  */
    /* in this case.                                                */
	if ( fits_create_img(fptr,  bitpix, naxis, naxes, &status) ){
		printerror( status );
		return 0;
	}

	float * vModel = calloc(naxes[0] * naxes[1] * naxes[2], sizeof(float));

	for( i=0;i<naxes[2];i++){
		for( j=0;j<naxes[0];j++){
			for( h=0; h<naxes[1];h++){
				//[Eta0,Strength,Vlos,Lambdadopp,Damp,Gamma,Azimuth,S0,S1,Macro,Alpha]
				switch (i)
				{
				case 0:
					vModel[indexModel++] = vInitModel[( j*naxes[1]) + h].eta0;
					break;
				case 1:
					vModel[indexModel++] = vInitModel[( j*naxes[1]) + h].B;
					break;
				case 2:
					vModel[indexModel++] = vInitModel[( j*naxes[1]) + h].vlos;
					break;
				case 3:
					vModel[indexModel++] = vInitModel[( j*naxes[1]) + h].dopp;
					break;
				case 4:
					vModel[indexModel++] = vInitModel[( j*naxes[1]) + h].aa;
					break;
				case 5:
					vModel[indexModel++] = vInitModel[( j*naxes[1]) + h].gm;
					break;					
				case 6:
					vModel[indexModel++] = vInitModel[( j*naxes[1]) + h].az;
					break;					
				case 7:
					vModel[indexModel++] = vInitModel[( j*naxes[1]) + h].S0;
					break;					
				case 8:
					vModel[indexModel++] = vInitModel[( j*naxes[1]) + h].S1;
					break;					
				case 9:
					vModel[indexModel++] = vInitModel[( j*naxes[1]) + h].mac;
					break;					
				case 10:
					vModel[indexModel++] = vInitModel[( j*naxes[1]) + h].alfa;
					break;
				case 11: // NUMBER OF ITERATIONS
					vModel[indexModel++] = vNumIterPixel[( j*naxes[1]) + h];
					break;
				case 12: // CHISQR 
					vModel[indexModel++] = vChisqrf[( j*naxes[1]) + h];
					break;										
				default:
					break;
				}
			}
		}
	}

   fpixel = 1;                               /* first pixel to write      */
   //nelements = naxes[0] * naxes[1] * naxes[2];          /* number of pixels to write */


   if ( fits_write_img(fptr, TFLOAT, fpixel, indexModel, vModel, &status) ){
		printerror( status );
		free(vModel);
		return 0;
	}

	// CLEAN MEMORY 
	free(vModel);

	    /* write another optional keyword to the header */
    /* Note that the ADDRESS of the value is passed in the routine */
   /*exposure = 1500;
	if ( fits_update_key(fptr, TLONG, "EXPOSURE", &exposure,
		"Total Exposure Time", &status) ){
		printerror( status );           
		return 0;
	}*/
	
	if ( fits_close_file(fptr, &status) ){        
		printerror( status );
		return 0;
	}
	
	return 1;

}

/**
 * 
 * fixed = array with positions to write in the file, Positions are in the following order: 
 * [Eta0,Strength,Vlos,Lambdadopp,Damp,Gamma,Azimuth,S0,S1,Macro,Alpha]
 * */
int writeFitsImageModelsSubSet(const char * fitsFile, int numRowsOriginal, int numColsOriginal,  ConfigControl configCrontrolFile, Init_Model * vInitModel, float * vChisqrf, int * vNumIterPixel, int addChiqr){
	
	fitsfile *fptr;       /* pointer to the FITS file, defined in fitsio.h */
	int status;
	int i, j, h; // indexes for loops
	long  fpixel;
	int indexModel = 0; 
	int bitpix =  FLOAT_IMG; 
	long naxis =   3;  /* 2-dimensional image */    
	long naxes[3] = { numRowsOriginal, numColsOriginal, NUMBER_PARAM_MODELS+1};   /* Image of numRows X numCols x 10 parameters of model and chisqrf */
	if(addChiqr){
		naxes[2]++;
	}	
	remove(fitsFile);               /* Delete old file if it already exists */
	status = 0;         /* initialize status before calling fitsio routines */
	if (fits_create_file(&fptr, fitsFile, &status)) /* create new FITS file */
   		printerror( status );           /* call printerror if error occurs */
	 /* write the required keywords for the primary array image.     */
    /* Since bitpix = FLOAT_IMG, this will cause cfitsio to create */
    /* a FITS image with BITPIX = -32 (float) .Note that the BSCALE  */
    /* and BZERO keywords will be automatically written by cfitsio  */
    /* in this case.                                                */
	if ( fits_create_img(fptr,  bitpix, naxis, naxes, &status) ){
		printerror( status );
		return 0;
	}
	
	// initialize image to 0 
	float * vModel = calloc(naxes[0] * naxes[1] * naxes[2], sizeof(float));
	indexModel=naxes[0] * naxes[1] * naxes[2];
	fpixel = 1;                               /* first pixel to write      */
	//nelements = naxes[0] * naxes[1] * naxes[2];          /* number of pixels to write */
	if ( fits_write_img(fptr, TFLOAT, fpixel, indexModel, vModel, &status) ){
		printerror( status );
		free(vModel);
		return 0;
	}
	free(vModel);


	int numRowsSub = (configCrontrolFile.subx2-configCrontrolFile.subx1)+1;
	int numColsSub = (configCrontrolFile.suby2-configCrontrolFile.suby1)+1;
	float * vModelSub = calloc(numRowsSub * numColsSub * naxes[2], sizeof(float));
	long fpixelBegin [3] = {1,1,1}; 
	long fpixelEnd [3] = {1,1,1}; 
	fpixelBegin[0] = configCrontrolFile.subx1;
	fpixelEnd[0] = configCrontrolFile.subx2;
	fpixelBegin[1] = configCrontrolFile.suby1;
	fpixelEnd[1] = configCrontrolFile.suby2;
	fpixelEnd[2] = naxes[2];
	
	indexModel=0;
	for( i=0;i<naxes[2];i++){
		for( j=0;j<numRowsSub;j++){
			for( h=0; h<numColsSub;h++){
				//[Eta0,Strength,Vlos,Lambdadopp,Damp,Gamma,Azimuth,S0,S1,Macro,Alpha]
				switch (i)
				{
				case 0:
					vModelSub[indexModel++] = vInitModel[( j*numColsSub) + h].eta0;
					break;
				case 1:
					vModelSub[indexModel++] = vInitModel[( j*numColsSub) + h].B;
					break;
				case 2:
					vModelSub[indexModel++] = vInitModel[( j*numColsSub) + h].vlos;
					break;
				case 3:
					vModelSub[indexModel++] = vInitModel[( j*numColsSub) + h].dopp;
					break;
				case 4:
					vModelSub[indexModel++] = vInitModel[( j*numColsSub) + h].aa;
					break;
				case 5:
					vModelSub[indexModel++] = vInitModel[( j*numColsSub) + h].gm;
					break;					
				case 6:
					vModelSub[indexModel++] = vInitModel[( j*numColsSub) + h].az;
					break;					
				case 7:
					vModelSub[indexModel++] = vInitModel[( j*numColsSub) + h].S0;
					break;					
				case 8:
					vModelSub[indexModel++] = vInitModel[( j*numColsSub) + h].S1;
					break;					
				case 9:
					vModelSub[indexModel++] = vInitModel[( j*numColsSub) + h].mac;
					break;					
				case 10:
					vModelSub[indexModel++] = vInitModel[( j*numColsSub) + h].alfa;
					break;
				case 11: // NUMBER OF ITERATIONS
					vModelSub[indexModel++] = vNumIterPixel[( j*numColsSub) + h];
					break;
				case 12: // CHISQR 
					vModelSub[indexModel++] = vChisqrf[( j*numColsSub) + h];
					break;										
				default:
					break;
				}
			}
		}
	}
	
	if ( fits_write_subset(fptr, TFLOAT, fpixelBegin, fpixelEnd, vModelSub, &status) ){
		printerror( status );
		free(vModelSub);
		return 0;
	}
	free(vModelSub);

	if ( fits_close_file(fptr, &status) ){        
		printerror( status );
		return 0;
	}
	
	return 1;

}



int writeFitsImageProfiles(const char * fitsProfileFile, const char * fitsFileOrigin, FitsImage * image){

	fitsfile *infptr, *outfptr;   /* FITS file pointers defined in fitsio.h */
	int status = 0, ii = 1;
	int i, j, k, h; // indexes for loops
	int bitpix,naxis, nkeys;
	long naxes_read [4] = {1,1,1,1}; /* The maximun number of dimension that we will read is 4*/
	long naxes [4] = {image->naxes[0],image->naxes[1],image->naxes[2],image->naxes[3]};
	char card[FLEN_CARD];
	char keyname [FLEN_CARD];
	char value [FLEN_CARD];
	remove(fitsProfileFile);               /* Delete old file if it already exists */
	/* Open the input file and create output file */

   fits_create_file(&outfptr, fitsProfileFile, &status);
	if (status != 0) {    
		fits_report_error(stderr, status);
		return(status);
	}

	// read a maximun of 4 dimensions 
	// CREATE A NEW IMAGE 
    fits_create_img(outfptr, image->bitpix, image->naxis, naxes, &status);
	if (status) {
		fits_report_error(stderr, status);
		return(status);
	}
	/* copy all the user keywords (not the structural keywords) */
	for (ii = 0; ii < image->nkeys; ii++) {
		if(strcmp(image->vKeyname[ii],"NAXIS1")!=0 && strcmp(image->vKeyname[ii],"NAXIS2")!=0 && strcmp(image->vKeyname[ii],"NAXIS3")!=0 && strcmp(image->vKeyname[ii],"NAXIS4")!=0)
			fits_update_card(outfptr, image->vKeyname[ii],image->vCard[ii], &status);
	}

	// CLOSE THE ORIGIN FILE, HE HAVE ALREADY THE INFORMATION OF KEYWORDS. 
	if (status){
		fits_report_error(stderr, status);
		return 0;
	}

	// ALLOCATE MEMORY TO WRITE THE IMAGE
	int numElemWrite = naxes[3]*naxes[2]*naxes[1]*naxes[0];

	float * outputImage = calloc(numElemWrite, sizeof(float));
	int currentLambda = 0, currentRow = 0, currentStokeParameter=0, currentCol = 0;
	 
	int pos_lambda = image->pos_lambda;
	int pos_col = image->pos_col;
	int pos_row = image->pos_row;
	int pos_stokes_parameters = image->pos_stokes_parameters;
	
		
	for( i=0; i <naxes[3]; i++){
		for( j=0;j <naxes[2]; j++){
			for( k=0; k<naxes[1]; k++){
				for( h=0; h<naxes[0]; h++){
					// I NEED TO KNOW THE CURRENT POSITION OF EACH ITERATOR 
					switch (pos_lambda)
					{
						case 0:
							currentLambda = h;
							break;
						case 1:
							currentLambda = k;
							break;
						case 2:
							currentLambda = j;
							break;
						case 3:
							currentLambda = i;
							break;																						
					}
					switch (pos_stokes_parameters)
					{
						case 0:
							currentStokeParameter = h;
							break;
						case 1:
							currentStokeParameter = k;
							break;
						case 2:
							currentStokeParameter = j;
							break;
						case 3:
							currentStokeParameter = i;
							break;																						
					}
					switch (pos_row)
					{
						case 0:
							currentRow = h;
							break;
						case 1:
							currentRow = k;
							break;
						case 2:
							currentRow = j;
							break;
						case 3:
							currentRow = i;
							break;																						
					}
					switch (pos_col)
					{
						case 0:
							currentCol = h;
							break;
						case 1:
							currentCol = k;
							break;
						case 2:
							currentCol = j;
							break;
						case 3:
							currentCol = i;
							break;																						
					}
					outputImage[(i*naxes[2]*naxes[1]*naxes[0]) + (j*naxes[1]*naxes[0]) + (k*naxes[0]) + h] = image->pixels[(currentCol*image->rows) + currentRow].spectro[currentLambda+(image->nLambdas * currentStokeParameter)];
				}
			}
		}
	}

    /* write the array of unsigned integers to the FITS file */
	if ( fits_write_img(outfptr, TFLOAT, 1, numElemWrite, outputImage, &status) ){
		printerror( status );
		free(outputImage);
		return 0;
	}

	free(outputImage);
	fits_close_file(outfptr,  &status);
	if(status){
		printerror( status );
		return 0;
	}
	return 1;
}



int writeFitsImageProfilesSubSet(const char * fitsProfileFile, const char * fitsFileOrigin, FitsImage * image, ConfigControl configCrontrolFile){
		fitsfile *infptr, *outfptr;   /* FITS file pointers defined in fitsio.h */
	int status = 0, ii = 1;
	int i, j, k, h; // indexes for loops
	int bitpix,naxis, nkeys;
	long naxes_read [4] = {1,1,1,1}; /* The maximun number of dimension that we will read is 4*/
	long naxes [4] = {image->naxes_original[0],image->naxes_original[1],image->naxes_original[2],image->naxes_original[3]};
	long naxes_sub [4] = {image->naxes[0],image->naxes[1],image->naxes[2],image->naxes[3]};
	char card[FLEN_CARD];
	char keyname [FLEN_CARD];
	char value [FLEN_CARD];
	remove(fitsProfileFile);               /* Delete old file if it already exists */
	/* Open the input file and create output file */
   //fits_open_file(&infptr, fitsFileOrigin, READONLY, &status);
   fits_create_file(&outfptr, fitsProfileFile, &status);
	if (status != 0) {    
		fits_report_error(stderr, status);
		return(status);
	}

	// read a maximun of 4 dimensions 
	//fits_get_img_param(infptr, 4, &bitpix, &naxis, naxes_read, &status);
	// CREATE A NEW IMAGE 
    fits_create_img(outfptr, image->bitpix, image->naxis, naxes, &status);
	if (status) {
		fits_report_error(stderr, status);
		return(status);
	}
	/* copy all the user keywords (not the structural keywords) */


	for (ii = 0; ii < image->nkeys; ii++) {
		//printf("\n  Card %s  keyname %s",image->vCard[ii],image->vKeyname[ii]);
		if(strcmp(image->vKeyname[ii],"NAXIS1")!=0 && strcmp(image->vKeyname[ii],"NAXIS2")!=0 && strcmp(image->vKeyname[ii],"NAXIS3")!=0 && strcmp(image->vKeyname[ii],"NAXIS4")!=0)
			fits_update_card(outfptr, image->vKeyname[ii],image->vCard[ii], &status);
	}

	// CLOSE THE ORIGIN FILE, HE HAVE ALREADY THE INFORMATION OF KEYWORDS. 

	//fits_close_file(infptr, &status);
	if (status){
		fits_report_error(stderr, status);
		return 0;
	}

	// ALLOCATE MEMORY TO WRITE THE IMAGE
	int numElemWrite = naxes[3]*naxes[2]*naxes[1]*naxes[0];

	float * outputImage = calloc(numElemWrite, sizeof(float));

	if ( fits_write_img(outfptr, TFLOAT, 1, numElemWrite, outputImage, &status) ){
		printerror( status );
		free(outputImage);
		return 0;
	}

	free(outputImage);


	
	numElemWrite = naxes_sub[3]*naxes_sub[2]*naxes_sub[1]*naxes_sub[0];
	outputImage = calloc(numElemWrite, sizeof(float));

	int currentLambda = 0, currentRow = 0, currentStokeParameter=0, currentCol = 0;
	int pos_lambda = image->pos_lambda;
	int pos_col = image->pos_col;
	int pos_row = image->pos_row;
	int pos_stokes_parameters = image->pos_stokes_parameters;
	for( i=0; i <naxes_sub[3]; i++){
		for( j=0;j <naxes_sub[2]; j++){
			for( k=0; k<naxes_sub[1]; k++){
				for( h=0; h<naxes_sub[0]; h++){
					// I NEED TO KNOW THE CURRENT POSITION OF EACH ITERATOR 
					switch (pos_lambda)
					{
						case 0:
							currentLambda = h;
							break;
						case 1:
							currentLambda = k;
							break;
						case 2:
							currentLambda = j;
							break;
						case 3:
							currentLambda = i;
							break;																						
					}
					switch (pos_stokes_parameters)
					{
						case 0:
							currentStokeParameter = h;
							break;
						case 1:
							currentStokeParameter = k;
							break;
						case 2:
							currentStokeParameter = j;
							break;
						case 3:
							currentStokeParameter = i;
							break;																						
					}
					switch (pos_row)
					{
						case 0:
							currentRow = h;
							break;
						case 1:
							currentRow = k;
							break;
						case 2:
							currentRow = j;
							break;
						case 3:
							currentRow = i;
							break;																						
					}
					switch (pos_col)
					{
						case 0:
							currentCol = h;
							break;
						case 1:
							currentCol = k;
							break;
						case 2:
							currentCol = j;
							break;
						case 3:
							currentCol = i;
							break;																						
					}
					
					outputImage[(i*naxes_sub[2]*naxes_sub[1]*naxes_sub[0]) + (j*naxes_sub[1]*naxes_sub[0]) + (k*naxes_sub[0]) + h] = image->pixels[(currentCol*image->rows) + currentRow].spectro[currentLambda+(image->nLambdas * currentStokeParameter)];					

				}
			}
		}
	}

	long fpixelBegin [4] = {1,1,1,1}; 
	long fpixelEnd [4] = {1,1,1,1}; 
	fpixelBegin[pos_row] = configCrontrolFile.subx1;
	fpixelEnd[pos_row] = configCrontrolFile.subx2;
	fpixelBegin[pos_col] = configCrontrolFile.suby1;
	fpixelEnd[pos_col] = configCrontrolFile.suby2;

	fpixelEnd[pos_lambda] = naxes[image->pos_lambda];
	fpixelEnd[pos_stokes_parameters] = naxes[pos_stokes_parameters];

	if ( fits_write_subset(outfptr, TFLOAT, fpixelBegin, fpixelEnd, outputImage, &status) ){
		printerror( status );
		free(outputImage);
		return 0;
	}

	free(outputImage);
    

	fits_close_file(outfptr,  &status);
	if(status){
		printerror( status );
		return 0;
	}
	return 1;
}






int writeFitsImageModelsWithArray(char * fitsFile, int numRows, int numCols, PRECISION * eta0, PRECISION * B, PRECISION * vlos, PRECISION * dopp, PRECISION * aa, PRECISION * gm, PRECISION * az, PRECISION * S0, PRECISION * S1, PRECISION * mac, PRECISION * alfa, PRECISION * vChisqrf){

	fitsfile *fptr;       /* pointer to the FITS file, defined in fitsio.h */
   int status;
	int i, j, h; // indexes for loops
   long  fpixel, nelements, exposure;

	int bitpix =  FLOAT_IMG; /* 16-bit unsigned short pixel values       */
   long naxis =   3;  /* 2-dimensional image                            */    
   long naxes[3] = { numRows, numCols, NUMBER_PARAM_MODELS };   /* Image of numRows X numCols x 10 parameters of model and chisqrf */

   remove(fitsFile);               /* Delete old file if it already exists */
   status = 0;         /* initialize status before calling fitsio routines */
   if (fits_create_file(&fptr, fitsFile, &status)) /* create new FITS file */
   	printerror( status );           /* call printerror if error occurs */
	
	 /* write the required keywords for the primary array image.     */
    /* Since bitpix = FLOAT_IMG, this will cause cfitsio to create */
    /* a FITS image with BITPIX = -32 (float) .Note that the BSCALE  */
    /* and BZERO keywords will be automatically written by cfitsio  */
    /* in this case.                                                */
	if ( fits_create_img(fptr,  bitpix, naxis, naxes, &status) ){
		printerror( status );
		return 0;
	}

	float * vModel = malloc(numRows * numCols * NUMBER_PARAM_MODELS);

	int indexModel = 0;
	for( i=0;i<NUMBER_PARAM_MODELS;i++){
		for( j=0;j<numCols;j++){
			for( h=0; h<numRows;h++){
				switch (i)
				{
				case 0:
					vModel[indexModel++] = B[( j*numRows) + h];
					break;
				case 1:
					vModel[indexModel++] = gm[( j*numRows) + h];
					break;
				case 2:
					vModel[indexModel++] = az[( j*numRows) + h];
					break;
				case 3:
					vModel[indexModel++] = eta0[( j*numRows) + h];
					break;
				case 4:
					vModel[indexModel++] = dopp[( j*numRows) + h];
					break;
				case 5:
					vModel[indexModel++] = aa[( j*numRows) + h];
					break;					
				case 6:
					vModel[indexModel++] = vlos[( j*numRows) + h];
					break;					
				case 7:
					vModel[indexModel++] = alfa[( j*numRows) + h];
					break;					
				case 8:
					vModel[indexModel++] = S0[( j*numRows) + h];
					break;					
				case 9:
					vModel[indexModel++] = S1[( j*numRows) + h];
					break;					
				case 10:
					vModel[indexModel++] = mac[( j*numRows) + h];
					break;										
				case 11: // READ FROM CHISQR
					vModel[indexModel++] = vChisqrf[( j*numRows) + h];
					break;					
				default:
					break;
				}
			}
		}
	}

   fpixel = 1;                               /* first pixel to write      */
   nelements = naxes[0] * naxes[1] * naxes[2];          /* number of pixels to write */

    /* write the array of unsigned integers to the FITS file */
   if ( fits_write_img(fptr, TFLOAT, fpixel, nelements, vModel, &status) ){
		printerror( status );
		free(vModel);
		return 0;
	}

	// CLEAN MEMORY 
	free(vModel);

	    /* write another optional keyword to the header */
    /* Note that the ADDRESS of the value is passed in the routine */
    exposure = 1500;
	if ( fits_update_key(fptr, TLONG, "EXPOSURE", &exposure,
		"Total Exposure Time", &status) ){
		printerror( status );           
		return 0;
	}

	if ( fits_close_file(fptr, &status) ){                /* close the file */
		printerror( status );
		return 0;
	}
	
	return 1;

}


int readSizeImageSpectro(const char * fitsFile, int * numRows, int * numCols){
	fitsfile *fptr;   
   int status = 0;   
	FitsImage * image =  malloc(sizeof(FitsImage));
	PRECISION nulval = 0.; // define null value to 0 because the performance to read from fits file is better doing this. 
   int bitpix, naxis, anynul, numPixelsFitsFile;
   long naxes [4] = {1,1,1,1}; 
	char comment[FLEN_CARD];

	if (!fits_open_file(&fptr, fitsFile, READONLY, &status)){
      // READ THE HDU PARAMETER FROM THE FITS FILE
      int hdutype;
      fits_get_hdu_type(fptr, &hdutype, &status);
		// We want only fits image 
		if(hdutype==IMAGE_HDU){
			// We assume that we have only on HDU as primary 
			if(fits_read_key(fptr, TSTRING, CTYPE1, image->ctype_1, comment, &status)) return 0;
			if(fits_read_key(fptr, TSTRING, CTYPE2, image->ctype_2, comment, &status)) return 0;
			if(fits_read_key(fptr, TSTRING, CTYPE3, image->ctype_3, comment, &status)) return 0;
			if(fits_read_key(fptr, TSTRING, CTYPE4, image->ctype_4, comment, &status)) return 0;
			// GET THE CURRENT POSITION OF EVERY PARAMETER
			int pos_lambda; 
			int pos_row;
			int pos_col;
			int pos_stokes_parameters;
			// LAMBDA POSITION
			if(strcmp(image->ctype_1,CTYPE_WAVE)==0) pos_lambda = 0;
			if(strcmp(image->ctype_2,CTYPE_WAVE)==0) pos_lambda = 1;
			if(strcmp(image->ctype_3,CTYPE_WAVE)==0) pos_lambda = 2;
			if(strcmp(image->ctype_4,CTYPE_WAVE)==0) pos_lambda = 3;

			// HPLN TAN 
			if(strcmp(image->ctype_1,CTYPE_HPLN_TAN)==0) pos_row = 0;
			if(strcmp(image->ctype_2,CTYPE_HPLN_TAN)==0) pos_row = 1;
			if(strcmp(image->ctype_3,CTYPE_HPLN_TAN)==0) pos_row = 2;
			if(strcmp(image->ctype_4,CTYPE_HPLN_TAN)==0) pos_row = 3;

			// HPLT TAN 
			if(strcmp(image->ctype_1,CTYPE_HPLT_TAN)==0) pos_col = 0;
			if(strcmp(image->ctype_2,CTYPE_HPLT_TAN)==0) pos_col = 1;
			if(strcmp(image->ctype_3,CTYPE_HPLT_TAN)==0) pos_col = 2;
			if(strcmp(image->ctype_4,CTYPE_HPLT_TAN)==0) pos_col = 3;			

			// Stokes paramter position , 
			if(strcmp(image->ctype_1,CTYPE_STOKES)==0) pos_stokes_parameters = 0;
			if(strcmp(image->ctype_2,CTYPE_STOKES)==0) pos_stokes_parameters = 1;
			if(strcmp(image->ctype_3,CTYPE_STOKES)==0) pos_stokes_parameters = 2;
			if(strcmp(image->ctype_4,CTYPE_STOKES)==0) pos_stokes_parameters = 3;

			if (!fits_get_img_param(fptr, 4, &bitpix, &naxis, naxes, &status) ){
				*numRows=naxes[pos_row];
				*numCols=naxes[pos_col];
				free(image);
				fits_close_file(fptr, &status);
				if (status){
					fits_report_error(stderr, status);
					return 0;
				}
			}
			else{
				printf("\n************ ERROR getting the image from the fits file:  %s",fitsFile);
				fits_close_file(fptr, &status);
				free(image);
				if (status){
					fits_report_error(stderr, status);
					return 0;
				}			
				return 0;				
			}

		}
		else
		{
			printf("\n************ ERROR: Fits file: %s could not be a fits image.",fitsFile);
			fits_close_file(fptr, &status);
			free(image);
			if (status){
				fits_report_error(stderr, status);
				return 0;
			}			
			return 0;
		}
		
	}
	else{
		printf("\n************ ERROR openning fits file %s",fitsFile);
		fits_close_file(fptr, &status);
		free(image);
		if (status){
			fits_report_error(stderr, status);
			return 0;
		}			
	}
	free(image);
	return 1;

}

/*--------------------------------------------------------------------------*/
void printerror( int status)
{
    /*****************************************************/
    /* Print out cfitsio error messages and exit program */
    /*****************************************************/
    if (status)
    {
       fits_report_error(stderr, status); /* print error report */
       //exit( status );    /* terminate the program, returning error status */
    }
    return;
}



