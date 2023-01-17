#by juanp 2015 (IAA-CSIC)
CC = gcc
LIBS = -lm

UNAME := $(shell uname)
ifeq ($(UNAME), Linux)
CFLAGS = -O3 -fPIC
else
UNAME := $(shell uname)
ifeq ($(UNAME), Linux)
CFLAGS = -O3 -fPIC
else
CFLAGS = -O3
endif
endif

CXXFLAGS="-mmacosx-version-min=10.10"
HOSTCOMPUTER := $(shell hostname | cut -d . -f 1)

default: print create_cuantic.o fvoigt.o fgauss.o me_der.o mil_sinrf.o calculosCompartidos.o lib.o milos.o libmilos.a milos fin


print:
	@echo Compilando CMILOS in $(info HOSTCOMPUTER is $(HOSTCOMPUTER))

create_cuantic.o: create_cuantic.c 
	$(CC) $(CFLAGS) -c $< 

fvoigt.o: fvoigt.c
	$(CC) $(CFLAGS) -c $< 

fgauss.o: fgauss.c
	$(CC) $(CFLAGS) -c $< 

me_der.o: me_der.c 
	$(CC) $(CFLAGS) -c $< 

mil_sinrf.o: mil_sinrf.c 
	$(CC) $(CFLAGS) -c $<  

# svdcordic.o: svdcordic.c
# 	$(CC) $(CFLAGS) -c $< 

calculosCompartidos.o: calculosCompartidos.c
	$(CC) $(CFLAGS) -c $< 

lib.o: lib.c
	$(CC) $(CFLAGS) -c $< 

milos.o:  milos.c milos.h
	$(CC) $(CFLAGS) -c $<   

milos:  
	$(CC) -o milos milos.o create_cuantic.o fvoigt.o fgauss.o me_der.o mil_sinrf.o calculosCompartidos.o lib.o $(LIBS)  

libmilos.a: milos.o create_cuantic.o fvoigt.o fgauss.o me_der.o mil_sinrf.o calculosCompartidos.o lib.o 
	ar rcs $@ $^

fin:
	cp milos ../milos.$(HOSTCOMPUTER)
	@echo --
	@echo All done.

clean:
	rm -f *.o
	rm -f *.a 
	rm -f *.so
	rm -f milos
