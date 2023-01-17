
/*

	@autor: Juan Pedro Cobos
	@Date: 31 Marzo 2011
	@loc: IAA- CSIC
	
	Convolucion para el caso Sophi: convolucion central de x con h.
	
	direct_convolution(x,h,delta)
	x spectro
	h gaussiana -perfil instrumental- ¡ ojo, solo con longitud impar!
	delta anchura de muestreo

	--nota: como h es simetrico no se invierte su orden 
	
	//result=calloc(nresult,sizeof(PRECISION*));
	//free();

	_juanp
*/

void direct_convolution(PRECISION * x, int nx,PRECISION * h, int nh,PRECISION delta){

	PRECISION *x_aux;
	int nresult,nx_aux;
	int k,j;
	
		
	nx_aux=nx+nh-1; //tamaño de toda la convolucion
	x_aux=calloc(nx_aux,sizeof(PRECISION));	
	
	int mitad_nh=nh/2;
	
	//rellenamos el vector auxiliar	
	for(k=0;k<nx_aux;k++){
		x_aux[k]=0;
	}	
	
	for(k=0;k<nx;k++){
		x_aux[k+mitad_nh]=x[k];		
	}
	
	//Condiciones de contorno reflexión
	/*
	//Relleno por la izq
	for(k=0;k<mitad_nh-1;k++){
		x_aux[k]=x[mitad_nh-k];		
	}

	//for(k=nx_aux-1;k>nx+mitad_nh-1;k++){
	for(k=0;k<mitad_nh-1;k++){
		x_aux[nx_aux-1-k]=x[mitad_nh+k];		
	}
	*/
	
	//vamos a tomar solo la convolucion central	
	
	for(k=0;k<nx;k++){
		x[k]=0;
		for(j=0;j<nh;j++){
			x[k]+=h[j]*x_aux[j+k];
		}
		x[k]*=delta;
	}	
	
	free(x_aux);

}

/*
	Genera una gaussiana de anchura FWHM (en A)
	centrada en 0 y muestreada por delta

	¡ojo  nmuestras_G debe ser un numero impar	!
*/
PRECISION * vgauss(PRECISION fwhm,int nmuestras_G,PRECISION delta){

	PRECISION * res;
	PRECISION sigma,alfa;
	PRECISION aux,line;
	int i;
	
	line=0; //es cero porque la gaussina esta centrada en cero
	sigma=fwhm / (2*sqrt(2*log(2)));
	alfa= 1   /  (sigma*sqrt(2*PI));
	
	res=calloc(nmuestras_G,sizeof(PRECISION));


	//se generan las posiciones de muestreo
	PRECISION *pos_muestras_G;
	pos_muestras_G=calloc(nmuestras_G,sizeof(PRECISION));

	int mitad_nmuestras_G; 		
	mitad_nmuestras_G=nmuestras_G/2;
	
	for(i=0;i<nmuestras_G;i++){
		pos_muestras_G[i]=-(mitad_nmuestras_G*delta)+(i*delta);
	}
	
	
	//se genera la gaussiana 
	for(i=0;i<nmuestras_G;i++){
		aux=pos_muestras_G[i]-line;  
		res[i]=alfa*exp(-( 	(aux*aux)/((sigma*sigma)*2) ) );
		res[i] = res[i]*delta;
	}
	
	//se normaliza su área
	
	float sum = 0;
	for(i=0;i<nmuestras_G;i++){
		sum+=res[i];
	}
	for(i=0;i<nmuestras_G;i++){
		res[i]=res[i]/sum;
	}

	
	free(pos_muestras_G);
	
	return res;


}







