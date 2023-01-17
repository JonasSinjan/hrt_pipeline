# cimport the Cython declarations for numpy
#cython: language_level=3
cimport numpy as np
import numpy as np

DTYPE_INT = np.intc
DTYPE_DOUBLE = np.float_

# if you want to use the Numpy-C-API from Cython
# (not strictly necessary for this example, but good practice)
np.import_array()

# cdefine the signature of our c function
cdef extern from "./lib/milos.h":
    void call_milos(const int *options, size_t size, const double *waveaxis, double *weight, double *initial_model, const double *inputdata, double *outputdata)

# create the wrapper code, with numpy type annotations
def py_milos(np.ndarray[int, ndim=1, mode="c"] options not None,
    np.ndarray[double, ndim=1, mode="c"] inputdata not None,
    np.ndarray[double, ndim=1, mode="c"] waveaxis not None,
    np.ndarray[double, ndim=1, mode="c"] weight not None,
    np.ndarray[double, ndim=1, mode="c"] initial_model not None,
    np.ndarray[double, ndim=1, mode="c"] outputdata not None):

    # assert inputdata.dtype == DTYPE_DOUBLE and outputdata.dtype == DTYPE_DOUBLE and waveaxis.dtype == DTYPE_DOUBLE
    # assert options.dtype == DTYPE_INT 
    # assert options[0] == len(waveaxis) 

    call_milos(<int*> np.PyArray_DATA(options),inputdata.shape[0], <double*> np.PyArray_DATA(waveaxis), <double*> np.PyArray_DATA(weight), <double*> np.PyArray_DATA(initial_model), <double*> np.PyArray_DATA(inputdata),<double*> np.PyArray_DATA(outputdata))

    return  

def pmilos(options,input_data,waveaxis,
           weight=np.asarray([1.,4.,5.4,4.1]),initial_model=np.asarray([400,30,120,1,0.05,1.5,.01,.22,.85])):

    #prepare input
    ny,nx,npol,nlon = input_data.shape
    input_data = input_data.flatten(order='C')

    if input_data.dtype != DTYPE_DOUBLE:
        input_data = input_data.astype(DTYPE_DOUBLE)

    if input_data.flags['C_CONTIGUOUS'] != True:
        print('non contiguous data')
        input_data = input_data.copy(order='c')
    print('input_data shape: ',input_data.shape)

    if waveaxis.dtype != DTYPE_DOUBLE:
        waveaxis = waveaxis.astype(DTYPE_DOUBLE)
    if waveaxis.flags['C_CONTIGUOUS'] != True:
        print('non contiguous waveaxis')
        waveaxis = waveaxis.copy(order='c')

    if options.dtype != DTYPE_INT:
        options = options.astype(DTYPE_INT)
    if options.flags['C_CONTIGUOUS'] != True:
        print('non contiguous options')
        options = options.copy(order='c')

    if options.shape[0] != 4 and options.shape[0] != 7:
        print("milos: Error en el numero de parametros: %d . Pruebe: milos NLAMBDA MAX_ITER CLASSICAL_ESTIMATES RFS [FWHM(in A) DELTA(in A) NPOINTS] perfil.txt\n")
        print("O bien: milos NLAMBDA MAX_ITER CLASSICAL_ESTIMATES RFS [DELTA(in A)] perfil.txt")
        print("Note : CLASSICAL_ESTIMATES=> 0: Disabled, 1: Enabled, 2: Only Classical Estimates.")
        print("RFS : 0: Disabled     1: Synthesis      2: Synthesis and Response Functions")
        print("Note when RFS>0: perfil.txt is considered as models.txt.")
        raise ValueError("Error in options")

    #	if(CLASSICAL_ESTIMATES!=0 && CLASSICAL_ESTIMATES != 1 && CLASSICAL_ESTIMATES != 2){#
    #		printf("milos: Error in CLASSICAL_ESTIMATES parameter. [0,1,2] are valid values. Not accepted: %d\n",CLASSICAL_ESTIMATES);
    #		return -1;
    #	}

    #	if(RFS != 0 && RFS != 1 && RFS != 2){
    #		printf("milos: Error in RFS parameter. [0,1,2] are valid values. Not accepted: %d\n",RFS);
    #		return -1;
    #	}

    assert options[0] == len(waveaxis) 

    length = len(input_data)
    output_data = np.zeros((length//len(waveaxis)//4 * 12),dtype=DTYPE_DOUBLE)

    py_milos(options,input_data,waveaxis,weight,initial_model,output_data)

    return np.reshape(output_data,(ny,nx,12))     
