#by juanp 2015 (IAA-CSIC)
mil: print calculosCompartidos.o create_cuantic.o fvoigt.o fgauss.o me_der.o mil_sinrf.o lib.o milos fin

print:
	clear
	#make clear
	@echo Compilando CMILOS...
	
create_cuantic.o: create_cuantic.c 
	gcc -c -O3   create_cuantic.c

fvoigt.o: fvoigt.c
	gcc -O3 -c     fvoigt.c

fgauss.o: fgauss.c
	gcc -O3 -c     fgauss.c

me_der.o: me_der.c 
	gcc -O3 -c    me_der.c

mil_sinrf.o: mil_sinrf.c 
	gcc -O3 -c     mil_sinrf.c 

milos.o: milos.c svdcordic.c
	gcc -O3 -c    milos.c

calculosCompartidos.o: calculosCompartidos.c
	gcc -c -O3   calculosCompartidos.c

lib.o: lib.c
	gcc -O3 -c   lib.c

milos:  calculosCompartidos.o fgauss.o fvoigt.o  milos.o me_der.o mil_sinrf.o lib.o create_cuantic.o
	gcc -o milos milos.o calculosCompartidos.o create_cuantic.o fgauss.o fvoigt.o me_der.o mil_sinrf.o  lib.o -lm  

fin:
	@echo --
	@echo All done.

clear:
	rm *.o
	rm milos
