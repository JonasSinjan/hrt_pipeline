compiler=$(shell (mpicc -show))
comp=$(shell echo $(compiler) | head -c 3)

CC=mpicc

CFLAGS=
ifeq ($(comp),icc)
	CFLAGS+=-ipo -xHost -mp1 -no-fma -no-multibyte-chars -qopt-dynamic-align -ansi-alias -fno-alias -parallel -ipp -align
endif
ifeq ($(comp),gcc)
	CFLAGS+=-march=native
endif
CFLAGS+=-O3

ifdef develop
	ifeq ($(develop),yes)
		CFLAGS+=-g 
		CFLAGS+=-Wall -Wextra		
	endif
endif
ifdef use_double
	ifeq ($(use_double),yes)
		CFLAGS+=-D USE_DOUBLE_PRECISION=double
	endif
endif
CFLAGS+=-fno-omit-frame-pointer 

HOST_SIZE   := $(shell getconf LONG_BIT)

CFLAGS+=-m${HOST_SIZE}

SRCDIR= src
DEPENCOMMON=$(SRCDIR)/calculosCompartidos.o $(SRCDIR)/fgauss.o $(SRCDIR)/fvoigt.o  $(SRCDIR)/me_der.o $(SRCDIR)/mil_sinrf.o $(SRCDIR)/lib.o $(SRCDIR)/create_cuantic.o $(SRCDIR)/utilsFits.o $(SRCDIR)/milosUtils.o $(SRCDIR)/convolution.o $(SRCDIR)/readConfig.o $(SRCDIR)/svdcmp.o 
DEPEN_SEQ=$(SRCDIR)/milos.o 
DEPEN_PAR=$(SRCDIR)/pmilos.o 
LDLIBS= -L/opt/local/cfitsio/cfitsio-3.350/lib/ -L/opt/local/cfitsio/cfitsio-3.350/include/ -lcfitsio -lm -lnsl -lgsl -lgslcblas -lfftw3 -ldl -lpthread
BIN= milos.x pmilos.x 


all: $(BIN)

milos.x: $(DEPENCOMMON) $(DEPEN_SEQ)
	$(CC) -o $@ $^ $(CFLAGS) $(LDLIBS)

pmilos.x: $(DEPENCOMMON) $(DEPEN_PAR)
	$(CC) -o $@ $^ $(CFLAGS) $(LDLIBS)

clean:
	rm -f  $(SRCDIR)/*.o $(BIN)
