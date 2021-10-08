
#include "stdio.h"
#include <fitsio.h>
#include <math.h>

#include "defines.h"

// TODO MOVE TO SEPARATE FILE
int load_fits(struct img_t*, char*);
int write_fits(struct img_t*, struct inv_t*, char*);


int init_img_struct(struct img_t *, char *);
int free_img_struct(struct img_t *);
int init_mask_struct(struct mask_t *, char *);
int free_mask_struct(struct mask_t *);
int init_inv_struct(struct inv_t *, struct img_t*);
int free_inv_struct(struct inv_t *);


int init_inv_struct(struct inv_t *inv, struct img_t* img){

    long npix;

    npix = img->xdim * img->ydim;

    // 5376 mb RAM!
    inv->iter = malloc(npix*sizeof(int));
    inv->B    = malloc(npix*sizeof(double));
    inv->gm   = malloc(npix*sizeof(double));
    inv->az   = malloc(npix*sizeof(double));
    inv->eta0 = malloc(npix*sizeof(double));
    inv->dopp = malloc(npix*sizeof(double));
    inv->aa   = malloc(npix*sizeof(double));
    inv->vlos = malloc(npix*sizeof(double)); //km/s
    inv->alfa = malloc(npix*sizeof(double)); //stay light factor
    inv->S0   = malloc(npix*sizeof(double));
    inv->S1   = malloc(npix*sizeof(double));
    inv->nchisqrf = malloc(npix*sizeof(double));

  return 0;
}

int free_inv_struct(struct inv_t *inv){

		free(inv->iter);
		free(inv->B);
		free(inv->gm);
		free(inv->az);
		free(inv->eta0);
		free(inv->dopp);
		free(inv->aa);
		free(inv->vlos); //km/s
        free(inv->alfa); //stay light factor
		free(inv->S0);
		free(inv->S1);
		free(inv->nchisqrf);

  return 0;
}

int free_img_struct(struct img_t *img){

    int w;

    for (w=0;w<img->wl;w++){
        free(img->i[w]);
        free(img->q[w]);
        free(img->u[w]);
        free(img->v[w]);
    }

    free(img->i);
    free(img->q);
    free(img->u);
    free(img->v);

    free(img->lambda);

    free(img->mask);

  return 0;
}


int init_img_struct(struct img_t *img, char *path){
  
  int nkeys;
  int status=0, nulval=0, anynul=0, naxis=0, result=0;
  int xdim=0, ydim=0, wl=0, pol=0, w;

  fitsfile *fptr;
  char comment[256], ext[256], lambda[6][16];
  
  sprintf(ext, "%s%s", path, "[0]");

  // FITS file IO
  if(!fits_open_image(&fptr, ext, READONLY, &status)){
      
    // read hdu header
    if(fits_get_hdrspace(fptr, &nkeys, NULL, &status)){
        printf("Error: Could not read %s header!\n", path);      
                
        fits_close_file(fptr, &status);
        return 1;
    }

    // read naxis from fits header
    fits_read_key(fptr, TINT, "NAXIS", &naxis, comment, &status);

    fits_read_key(fptr, TINT, "NAXIS1", &ydim, comment, &status);
    fits_read_key(fptr, TINT, "NAXIS2", &xdim, comment, &status);
    fits_read_key(fptr, TINT, "NAXIS3", &pol,  comment, &status);
    fits_read_key(fptr, TINT, "NAXIS4", &wl,   comment, &status);
    
    fits_read_key(fptr, TSTRING, "LAMBDA0", &lambda[0], comment, &status);
    fits_read_key(fptr, TSTRING, "LAMBDA1", &lambda[1], comment, &status);
    fits_read_key(fptr, TSTRING, "LAMBDA2", &lambda[2], comment, &status);
    fits_read_key(fptr, TSTRING, "LAMBDA3", &lambda[3], comment, &status);
    fits_read_key(fptr, TSTRING, "LAMBDA4", &lambda[4], comment, &status);
    fits_read_key(fptr, TSTRING, "LAMBDA5", &lambda[5], comment, &status);

    img->i = malloc(wl*sizeof(double*));
    img->q = malloc(wl*sizeof(double*));
    img->u = malloc(wl*sizeof(double*));
    img->v = malloc(wl*sizeof(double*));

    img->lambda = malloc(wl*sizeof(double*));

    for (w=0;w<wl;w++){
      img->i[w] = malloc(ydim*xdim*sizeof(double));
      img->q[w] = malloc(ydim*xdim*sizeof(double));
      img->u[w] = malloc(ydim*xdim*sizeof(double));
      img->v[w] = malloc(ydim*xdim*sizeof(double));

      img->lambda[w] = atof(lambda[w]);
    }

    img->xdim = xdim;
    img->ydim = ydim;
    img->pol  = pol;
    img->wl   = wl;
    img->npix = ydim*xdim*pol*wl;
    img->naxis = naxis;

    fits_close_file(fptr, &status);
      
  }
  else
  { 
      fits_report_error(stderr, status);
      //free(buffer);
      return 1; 
  } 
  

  // Load mask
  sprintf(ext, "%s%s", path,"[1]");

  if(!fits_open_image(&fptr, ext, READONLY, &status)){
      
    // read hdu header
    if(fits_get_hdrspace(fptr, &nkeys, NULL, &status)){
        printf("Error: Could not read %s header!\n", path);      
        
        fits_close_file(fptr, &status);
        return 1;
    }

    // read naxis from fits header
    fits_read_key(fptr, TINT, "NAXIS", &naxis, comment, &status);
    fits_read_key(fptr, TINT, "NAXIS1", &ydim, comment, &status);
    fits_read_key(fptr, TINT, "NAXIS2", &xdim, comment, &status);

    img->mask = malloc(ydim*xdim*sizeof(int));
 
    fits_close_file(fptr, &status);

    }
    else
    {
      printf("File format not compatible. Abort.\n");
      fits_report_error(stderr, status);
      //free(buffer);
      return 1; 
    }

  return 0;
}


int load_fits(struct img_t *img, char *path){

  int nkeys;
  int status=0, nulval=0, anynul=0, naxis=0;
  int xdim=0, ydim=0, wl=0, pol=0, x, y, w, p, i;
  long *fpix, npix;

  float *buffer;
  fitsfile *fptr;
  char comment[256], ext[256];
  
  sprintf(ext, "%s%s", path,"[0]");

  // FITS file IO
  if(!fits_open_image(&fptr, ext, READONLY, &status)){
      
    fpix   = malloc(img->naxis*sizeof(long));
    buffer = malloc(img->npix*sizeof(double));

    for(i=0; i<img->naxis; i++){
      fpix[i] = 1;
    }

    if (fits_read_pix(fptr, TFLOAT, fpix, img->npix, &nulval, buffer, &anynul, &status)){
        
        fits_report_error(stderr, status);  
        fits_close_file(fptr, &status);
       
        free(buffer);
        free(fpix);

        return 1; 
    }

    // (y*ydim+x) + (xdim*ydim*w) + (xdim*ydim*wl*p) old version for wrong order

    // (y*ydim+x) + (xdim*ydim*p) + (xdim*ydim*pol*w)
    // works with order [0,0,:,:], [0,1,:,:], [0,2,:,:]....[1,0,:,:], [1,1,:,:] etc

    xdim = img->xdim;
    ydim = img->ydim;
    pol  = img->pol;
    wl   = img->wl;
    
    // primary HDU with inversion data
    for(x=0; x<xdim; x++){ 
        for(y=0; y<ydim; y++){ 
            for(p=0; p<pol; p++){ 
                for (w=0; w<wl; w++){ 
                    switch(p){
                        case 0: img->i[w][y*ydim+x] = buffer[(y*ydim+x) + (xdim*ydim*p) + (xdim*ydim*pol*w)];
                                break;
                        case 1: img->q[w][y*ydim+x] = buffer[(y*ydim+x) + (xdim*ydim*p) + (xdim*ydim*pol*w)];
                                break;
                        case 2: img->u[w][y*ydim+x] = buffer[(y*ydim+x) + (xdim*ydim*p) + (xdim*ydim*pol*w)];
                                break;
                        case 3: img->v[w][y*ydim+x] = buffer[(y*ydim+x) + (xdim*ydim*p) + (xdim*ydim*pol*w)];      
                                break;               
                        }   
                    }
                }
            }
        }
    
    free(buffer);
    free(fpix);

    fits_close_file(fptr, &status);
  }
  else
  { 
      fits_report_error(stderr, status);
      //free(buffer);
      return 1; 
  } 

  sprintf(ext, "%s%s", path,"[1]");

  // FITS file IO
  if(!fits_open_image(&fptr, ext, READONLY, &status)){
      
    fpix   = malloc(2*sizeof(long));
    buffer = malloc(xdim*ydim*sizeof(double));

    for(i=0; i<2; i++){
      fpix[i] = 1;
    }

    if (fits_read_pix(fptr, TFLOAT, fpix, xdim*ydim, &nulval, buffer, &anynul, &status)){
        
        fits_report_error(stderr, status);  
        fits_close_file(fptr, &status);
       
        free(buffer);
        free(fpix);

        return 1; 
    }
      
    // image extension with pixel mask
    for(x=0; x<xdim; x++){ 
        for(y=0; y<ydim; y++){ 
            img->mask[y*ydim+x] = buffer[y*ydim+x];
        }
    }
    free(fpix);
    free(buffer);
  }
 
  return 0;
}


int write_fits(struct img_t *img, struct inv_t *inv, char* opath){
  
    int i, xdim, ydim, pol, wl, x, y, p, w, n, nimg;
    int nkeys, bash;
    int status=0, nulval=0, anynul=0, naxis=0, result=0;
    long *fpix, npix;
 
    char comment[256];

    float *buffer;
    fitsfile *ofptr;

    bash = remove(opath);

    nimg  = 12;
    naxis = 3;
    npix  = img->xdim * img->ydim;
    
    xdim = img->xdim;
    ydim = img->ydim;

    fpix   = malloc(naxis*sizeof(long));
    buffer = malloc(nimg*npix*sizeof(double));
    
    for(i=0; i<naxis; i++){
      fpix[i] = 1;
    }

    long naxes[3] = {img->xdim, img->ydim, nimg};

    // primary HDU with inversion data
    for(x=0; x<xdim; x++){ 
      for(y=0; y<ydim; y++){
            
            buffer[(0*xdim*ydim)  + (y*ydim+x)] = inv->iter[y*ydim+x];
            buffer[(1*xdim*ydim)  + (y*ydim+x)] = inv->B[y*ydim+x];
            buffer[(2*xdim*ydim)  + (y*ydim+x)] = inv->gm[y*ydim+x];
            buffer[(3*xdim*ydim)  + (y*ydim+x)] = inv->az[y*ydim+x];
            buffer[(4*xdim*ydim)  + (y*ydim+x)] = inv->eta0[y*ydim+x];
            buffer[(5*xdim*ydim)  + (y*ydim+x)] = inv->dopp[y*ydim+x];
            buffer[(6*xdim*ydim)  + (y*ydim+x)] = inv->aa[y*ydim+x];
            buffer[(7*xdim*ydim)  + (y*ydim+x)] = inv->vlos[y*ydim+x];
            buffer[(8*xdim*ydim)  + (y*ydim+x)] = inv->alfa[y*ydim+x];
            buffer[(9*xdim*ydim)  + (y*ydim+x)] = inv->S0[y*ydim+x];
            buffer[(10*xdim*ydim) + (y*ydim+x)] = inv->S1[y*ydim+x];
            buffer[(11*xdim*ydim) + (y*ydim+x)] = inv->nchisqrf[y*ydim+x];
        
      }
    }


  if (fits_create_file(&ofptr, opath, &status)) /* create new FITS file */
      fits_report_error(stderr, status);           /* call printerror if error occurs */

  if ( fits_create_img(ofptr, FLOAT_IMG, naxis, naxes, &status) )
      fits_report_error(stderr, status);         

  if ( fits_write_pix(ofptr, TFLOAT, fpix, nimg*npix, buffer, &status) )
      fits_report_error(stderr, status);

  if ( fits_close_file(ofptr, &status) )
      fits_report_error(stderr, status);

  free(buffer);
  free(fpix);

  return 0;
}