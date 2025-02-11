"""
``phirte.py`` run MILOS inversion on provided data

:Project: Solar Orbiter Polarimetric and Helioseismic Imager (SoPHI - FDT)
:Date: 2023-09-15
:Authors: **David Orozco Suárez (orozco@iaa.es)**
:Contributors: **Alex Feller (feller@mps.mpg.de)**

"""

import concurrent.futures
import os, itertools, subprocess, time, functools
import numpy as np
import pymilos

class bcolors:
    """
    This is a simple class for colors.

    Available colors:

    * HEADER = '\033[95m'
    * OKBLUE = '\033[94m'
    * OKGREEN = '\033[92m'
    * YELLOW = '\033[93m'
    * WARNING = '\033[36m'
    * FAIL = '\033[91m'
    * ENDC = '\033[0m'
    * BOLD = '\033[1m'
    * UNDERLINE = '\033[4m'
    * RESET = '\u001b[0m'
    * CYAN = '\033[96m'
    * DARKCYAN = '\033[36m'

    """

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    YELLOW = '\033[93m'
    WARNING = '\033[36m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\u001b[0m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    VERBOSE = YELLOW


def printc(*args, color=bcolors.RESET, **kwargs):
    """

    This function wraps the python ``print()`` functions adding color capabilities.

    :param args: params to pass through to ``print()`` function.
    :param color: provide the text color, defaults to ``bcolors.RESET`` . Valid colors: any color or ``color=bcolors.KEYWORD`` where ``KEYWORD`` should be in :py:meth:`bcolors` class.
    :param kwargs: ``**kwargs`` enables printc to retain ``print()`` functionality

    """

    print(u"\u001b" + f"{color}", end='\r')
    print(*args, **kwargs)
    print(u"\u001b" + f"{bcolors.RESET}", end='\r')

    return


def countcalls(fn):
    """
    Decorator function count function calls. Use ``@countcalls`` above def.

    """

    @functools.wraps(fn)
    def wrapped(*args):
        wrapped.ncalls += 1
        return fn(*args)

    wrapped.ncalls = 0
    return wrapped


def timeit(method):
    """
    Decorator function to calculate executing time. Use ``@timeit`` above def.

    """

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            t = (te - ts) * 1000
            if t > 1e3:
                print('%r  %2.2f seconds' %
                      (method.__name__, t / 1000.))
            elif t > 1e6:
                print('%r  %2.2f mim' %
                      (method.__name__, t / 1000. / 60.))
            else:
                print('%r  %2.2f ms' %
                      (method.__name__, t))

        return result

    return timed

def phi_rte(
        data: np.ndarray, wave_axis: np.ndarray, rte_mode: str, temp_dir: str = './',
        cmd: str = str(), options: list = [],
        weight: np.ndarray = None,
        mask: np.ndarray = 0,
        initial_model: np.ndarray = None,
        parallel: bool = False, num_workers: int = 10,
        cavity: np.ndarray = np.empty([], dtype=float)):

    """ 
    Run ME inversion over a provided set of Stokes profiles with their wave axis

    .. note::
         An example of how to use this code can be found `in tests <milos_test_inversion.html>`_

    .. warning::
         This software is optimized for python. However, you can also use the ASCII version. For that, the milos code has to be previosly compiled. 
         In principle, a ``milos.x`` executable should be located in ``milos/lib`` folder. If not, run ``make clean`` and ``make`` in ``milos/lib``.
         This will create a copy of milos (milos.x ASCII version). 

    :param data: Stokes data

        * Dimensions are ``[l,p,x,y]`` where:

            * ``l`` is the wavelength dimension
            * ``p`` is the polarization dimension
            * ``x`` is the spatial dimension
            * ``y`` is the spatial dimension

        * In synthesis mode dimensions are ``[m, n]`` where:

            * ``m`` is the model dimension
            * ``n`` is the number of models

        * The output model in syntehsis inversion as well as the initial model are ordered as follows:

			0. magnetic field strength in gauss
			1. magnetic field inclination in degree
			2. magnetic field azimuth in degree
			3. Line absorption :math:`\\eta_0`
			4. Doppler width :math:`\\Delta\\lambda` in angstrom
			5. Damping :math:`a`
			6. LoS velocity in km/s
			7. Source function :math:`S_0`
			8. Source function :math:`S_1`

    :type data: np.ndarray
    :param wave_axis: wavelength axis, in Angstrom. Should be same length as ``data``, e.g., ``l``
    :type wave_axis: np.ndarray
    :param rte_mode: RTE mode

        * ``RTE`` Only ME
        * ``RTE+CE`` ME + Classical estimates
        * ``CE`` Classical estimates
        * ``RTE+CE+PSF``  ME + Classical estimates + spectral PSF
        * ``SYN``  ME in synthesis mode
        * ``SYN+PSF``  ME in synthesis mode + spectral PSF

    :type rte_mode: str
    :param temp_dir: output directory for storing temporal files when using milos C version (I/O ASCII files), defaults to './'
    :type temp_dir: str, optional
    :param cmd: command for executing ``milos ASCII``. If none ``phi_rte`` uses ``pmilos``, defaults to ``None``.
    :type cmd: str, optional

    .. note::
        Currently, both pmilos and cmilos support cavity map correction.

    :param options: inverter options, defaults to None

        * Depending on the ``RTE mode``

            * options[0] = wavelength axis dimension ``l``
            * options[1] = maximum number of iterations, defaults to 30
            * options[2] = 0: RTE, 1: CE + RTE, 2: CE [Only when inversion is activated]
            * options[3] = 0: Inversion, 1-> synthesis 2-> Response functions.

                # .. note:: From here, options are NOT mandatory although options input to milos.py should be a 9 element list.

            * options[4] = full width half maximum of spectral PSF (only if PSF is activated) in mA (integer)
            * options[5] = spectral PSF wavelength steps in mA (integer)
            * options[6] = spectral PSF number of sampling points
            * options[7] = If != from 0 and positive. Initiallyze S0/S1. If > 1, rerun pixels with bad chi2 ``chi2 > abs(options[7]) * previous_chi2``. If <0, deactivates S0/S1 but still rerun bad pixels.
            * options[8] = If != 0 and positive, it corresponds to initial lambda parameter. If < 0, initial lambda parameters is initiallyzed automatically using the gradient. -100 is a good value.
            
    :type options: np.ndarray
    :param weight: stokes profiles weights, defaults to ``[1,10,10,4]``; values currently used for HRT: ``[1, 4, 5.4, 4.1]``
    :type weight: np.ndarray, optional
    :param initial_model:

        Milne-Eddington initial model, defaults to ``[400,30,120, 50.,0.04,0.01,0.01,0.22,0.85]`` 

        * The initial model corresponds to (in order):

            0. Magnetic field strength. Defaults to 400 G.
            1. Magnetic field inclination. Defaults to 30 degree.
            2. Magnetic field azimuth. Defaults to 120 degree.
            3. Line absorption :math:`\\eta_0`. Defaults to 50.
            4. Doppler width :math:`\\Delta\\lambda`. Defaults to 0.04 Angstrom.
            5. Damping :math:`a`. Defaults to 0.01.
            6. LoS velocity. Defaults to 0.01 km/s.
            7. Source function :math:`S_0`. Defaults to 0.22.
            8. Source function :math:`S_1`. Defaults to 0.85.

    :type initial_model: list, optional
    :param parallel: run parallel version (python pmilos based only), defaults to ``False``.
    :type parallel: integer, optional
    :param num_workers: number of parallel instances, defaults to ``10``.
    :type num_workers: integer, optional
    :param mask: input bit mask (same dimensions as data input) specifing the pixels to invert. ``0 or 1``, defaults to ``0``.
    :type mask: np.ndarray, optional
    :param cavity: cavity map in Angstrom (default: None)
    :type cavity: np.ndarray, optional

    :return: results

        * in the inversion mode the results are stored in a ``[k,y,x]`` where ``k`` corresponds to the following 12 output model parameters

			0. pixel counter
			1. iterations number (with positive convergence)
			2. magnetic field strength in gauss
			3. magnetic field inclination in degree
			4. magnetic field azimuth in degree
			5. Line absorption :math:`\\eta_0`
			6. Doppler width :math:`\\Delta\\lambda` in angstrom
			7. Damping :math:`a`
			8. LoS velocity in km/s
			9. Source function :math:`S_0`
			10. Source function :math:`S_1`
			11. merit function final value.

        * In the synthesis mode the results are stored in a ``[s,m]`` where:

            * ``s`` is the polarization dimension [wave, Stokes I, Q, U, V]
            * ``n`` is the number of models

    :rtype: np.ndarray

    """

    options_set = 0
    try:
        if not(options): # so no options becouse the first one should be the length of the wave axis
            print('No input options. Setting for PHI only.')
            options = np.zeros((9))
            options[0] = len(wave_axis) #NLAMBDA wave axis dimension
            options[1] = 30 #MAX_ITER max number of iterations
            options[2] = 1 #CLASSICAL_ESTIMATES [0,1,2] 0=RTE, 1= CE+RTE, 2= CE
            options[3] = 0 #RFS [0,1,2] 0.-> Inversion, 1-> synthesis 2-> RFS
            options[4] = 0 #FWHM = atof(argv[5]);
            options[5] = 0 ##DELTA = atof(argv[6]);
            options[6] = 0 #NMUESTRAS_G = atoi(argv[7]);
            options[7] = 0 #INITIALIZE S0S1
            options[8] = 0 #Initial lambda. 0 means nothing changes (internal), negative auto, and positive, init value
            options_set = 0
        else:
            options = np.array(options)
            options_set = 1
            assert (options.size == 9)
    except TypeError:
        print('ups, options problem')

    if weight is None:
        print('Using defaults weights.')
        weight = np.array([1.,10.,10.,4])
    if initial_model is None:
        print('Using defaults init model.')
        initial_model = np.array([400,30,120, 50.,0.04,0.01,0.01,0.22,0.85])

    print('RTE_MODE ', rte_mode)

    if rte_mode in {'SYN', 'SYN+PSF','SYN+RFS','SYN+PSF+RFS'}:
        # RTE IN SYNTHESIS MODE
        # In synthesis mode, the input are not Stokes profiles but a model atmosphere
        data = data.flatten()
        if data.size % 9 not in [9, 0]:
            printc('Input data is not multiple of 9. Data length: ',data.size,color=bcolors.FAIL)
            return 0
        nmodels = len(data)//9

        #assumes cmd is cmilos
        # if cmd != 'cmilos':
        #     printc('Only for CMILOS yet: ',cmd,color=bcolors.FAIL)
        #     return

        #check if the wavelength axis coincides with the options
        if options[0] != wave_axis.size:
            printc('In synthesis mode, no options are necessary unless PSF.', colors=bcolors.FAIL)
            printc('   Hence options[0] should coincide with wave_axis length ',options[0] ,wave_axis.size,color=bcolors.FAIL)
            return 0

        #set synthesis mode
        options[3] = 2 if rte_mode in {'SYN+RFS','SYN+PSF+RFS'} else 1

        if rte_mode in {'SYN+PSF','SYN+PSF+RFS'} and not(options_set):
            options[4] = 105 # 0.105 #FWHM = atof(argv[5]);  ·INTEGER NEEDS TO BE CONVERTER TO FLOAT BECOUSE OF / 10000
            options[5] = 70 # 0.070  #DELTA = atof(argv[6]); ·INTEGER NEEDS TO BE CONVERTER TO FLOAT BECOUSE OF / 10000
            options[6] = wave_axis.size  #NMUESTRAS_G = atoi(argv[7]);

        if options[4] > 0:
            printc('PSF in synthesis version activated',color=bcolors.OKBLUE)

        if cmd:  # meaning you will use cmilos
            printc('Using CMILOS ASCII version')

            #loop over the input
            file_dummy_in = os.path.join(temp_dir, 'dummy_in.txt')
            file_dummy_out = os.path.join(temp_dir, 'dummy_out.txt')

            filename = file_dummy_in
            with open(filename,"w") as f:
                #loop in wavelength axis
                for waves in wave_axis:
                    f.write('%.10f \n' % (waves) )
                #loop in input model
                iter = 0
                for model in data:
                    if not(iter % 9):
                        f.write('%d \n' % (iter // 9) )
                    f.write('%.10f \n' % (model) )
                    iter += 1


            printc('  ---- >>>>> Synthesizing data.... ',color=bcolors.OKGREEN)

            trozo = f" {str(options[0].astype(int))} {str(options[1].astype(int))} {str(options[2].astype(int))} {str(options[3].astype(int))} {str(options[4].astype(int))} {str(options[5].astype(int))} {str(options[6].astype(int))}"

            cmd = cmd + trozo + " " + file_dummy_in + " > " + file_dummy_out
            printc(cmd, color=bcolors.OKGREEN)

            rte_on = subprocess.call(cmd,shell=True)
            printc(rte_on,color=bcolors.OKGREEN)

            printc('  ---- >>>>> Finishing.... ',color=bcolors.OKGREEN)
            printc('  ---- >>>>> Reading results.... ',color=bcolors.OKGREEN)

            res = np.loadtxt(file_dummy_out)
            if nmodels == 1:
                if options[3] == 1:  #SYN
                    res = np.einsum('ij->ji',res)
                    return res[1:,:]
                if options[3] == 2:  #SYN+RFS
                    res = np.reshape(res,(10,wave_axis.size,5))
                    res = np.einsum('kij->kji',res)
                    return res[:,1:,:]

            if nmodels >  1:
                if options[3] == 1:
                    res = np.reshape(res,(nmodels,wave_axis.size,5))
                    res = np.einsum('kij->kji',res)
                    return res[:,1:,:]
                if options[3] == 2:
                    res = np.reshape(res,(nmodels,10,wave_axis.size,5))
                    res = np.einsum('mkij->mkji',res)
                    return res[:,:,1:,:]
            return
        else:

            printc('Using PMILOS version')
            printc('  ---- >>>>> Synthesizing data.... ',color=bcolors.OKGREEN)

            res = pymilos.pymilos(options, data, wave_axis)
            # n_models,stokes,wave  we would need (wave, pol, y * x)

            #add wavelenth axis for having same behaviour as cmilos

            printc('  ---- >>>>> Finishing.... ',color=bcolors.OKGREEN)

            return res

    if rte_mode in {'RTE', 'RTE+PSF'}:
        options[2] = 0
    elif rte_mode == 'CE':
        options[2] = 2
    elif rte_mode in {'CE+RTE', 'CE+RTE+PSF'}:
        options[2] = 1
    else:
        printc('RET option not recognized: ',rte_mode,color=bcolors.FAIL)
        return

    if rte_mode in {'CE+RTE+PSF', 'RTE+PSF'}:
        options[4] = 105 # 0.105 #FWHM = atof(argv[5]);  ·INTEGER NEEDS TO BE CONVERTER TO FLOAT BECOUSE OF / 10000
        options[5] = 70 # 0.070  #DELTA = atof(argv[6]); ·INTEGER NEEDS TO BE CONVERTER TO FLOAT BECOUSE OF / 10000
        options[6] = 6  #NMUESTRAS_G = atoi(argv[7]);

    print('options: ', options)

    if cmd:  # meaning you will use cmilos
        printc('Using CMILOS ASCII version')

        wave, p, y, x = data.shape
        printc('   saving data into dummy_in.txt for RTE input. dimensions (l,p,y,x):', wave, p, y, x)
        # GV output_dir was include in the chain (generate_level_2, pft_pipe_modules_ phi_rte) to have it availabel here

        wave_axis = np.broadcast_to(wave_axis, (y, x, wave))
        wave_axis = np.einsum('ijl->lij', wave_axis)

        if cavity.shape:
            cavity = np.broadcast_to(cavity, (wave, y, x))
            wave_axis = wave_axis - cavity

        # AF: temporary input / output files for cmilos will be written / read from temp_dir
        # if output_dir is None:
        #     # get file path from calling script
        #     output_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

        file_dummy_in = os.path.join(temp_dir, 'dummy_in.txt')
        file_dummy_out = os.path.join(temp_dir, 'dummy_out.txt')

        filename = file_dummy_in
        with open(filename,"w") as f:
            for yi, xi in itertools.product(range(y), range(x)):  #NOTICE THAT x and y are exchanged !!!!
                if np.ndim(mask) == 0:
                    for k in range(wave):
                        f.write('%e %e %e %e %e \n' % (wave_axis[k, yi, xi],data[k,0,yi,xi],data[k,1,yi,xi],data[k,2,yi,xi],data[k,3,yi,xi]))
                else:
                    if mask[yi,xi] == 1:
                        for k in range(wave):
                            f.write('%e %e %e %e %e \n' % (wave_axis[k, yi, xi],data[k,0,yi,xi],data[k,1,yi,xi],data[k,2,yi,xi],data[k,3,yi,xi]))

        printc('  ---- >>>>> Inverting data.... ',color=bcolors.OKGREEN)

        trozo = f" {str(options[0].astype(int))} {str(options[1].astype(int))} {str(options[2].astype(int))} {str(options[3].astype(int))} {str(options[4].astype(int))} {str(options[5].astype(int))} {str(options[6].astype(int))}"

        cmd = cmd + trozo + " " + file_dummy_in + " > " + file_dummy_out
        printc(cmd, color=bcolors.OKGREEN)

        rte_on = subprocess.call(cmd,shell=True)
        printc(rte_on,color=bcolors.OKGREEN)

        printc('  ---- >>>>> Finishing.... ',color=bcolors.OKGREEN)
        printc('  ---- >>>>> Reading results.... ',color=bcolors.OKGREEN)

        res = np.loadtxt(file_dummy_out)

        # del_dummy = subprocess.call(f"rm {file_dummy_in}", shell=True)
        # print(del_dummy)
        # del_dummy = subprocess.call(f"rm {file_dummy_out}", shell=True)
        # print(del_dummy)

        if np.ndim(mask) == 0:
            result = np.zeros((12,y*x)).astype(float)
            for i in range(y*x):
                result[:,i] = res[i*12:(i+1)*12]
            return result.reshape(12,y,x)
        else:
            result = np.zeros((12,y,x)).astype(float)
            index = 0
            xii = 0
            for yi, xi in itertools.product(range(y), range(x)):  #goes through the image
                if mask[yi,xi] == 1:
                    result[:,yi,xi] = res[index*12:(index+1)*12]
                    index += 1
            return result

    else: #meaning you will use pmilos

        #TODO: make array one dimmension in space
        printc('Using PMILOS version')
        printc('   input shape in phi_rte: ', data.shape)

        # check data dimensions
        if data.ndim == 3: # one missing spatial dimension
            printc('   no reshaping needed: ', data.shape)
            nwave, npol, nx = data.shape
            ny = 1
        elif data.ndim == 4: #four dimensions need reshaping
            # Here we flatten the data to be one dimensinal and change the order (size first)
            nwave, npol, nx, ny = data.shape
            data = data.reshape(nwave, npol ,nx*ny)
            printc('   reshaping into: ', data.shape)

        data = np.einsum('ijk->kji',data)  #FROM  (wave, pol, y * x) TO (y * x, pol,wave) for C
        nyx, npol, nwave = data.shape
        print('   reshaping data. New data shape',data.shape, "should be (y * x,pol,wave) for C")

        if cavity.shape:
            print('   Cavity shape is',cavity.shape)
            cavity = cavity.flatten()
            print('   reshaping cavity (flatten). New cavity shape',cavity.shape)

        if (np.ndim(mask)) != 0:
            printc('   INPUT MASK... ')
            mask = mask.reshape(ny * nx)
            data = data[mask == 1, :, :]
            output_from_rte = np.zeros((nyx, 12))
            nyx, npol, nwave = data.shape  #OJO nyx
            print('   reshaping data to exclude masked pixels. New data shape',data.shape)
            if cavity.shape:
                cavity = cavity[mask == 1]
                print('   reshaping cavity to exclude masked pixels. New cavity shape',cavity.shape)

        if parallel:
            global phi_rte_map_func

            def phi_rte_map_func(args):
                stripe, data, cavity = args
                return stripe, pymilos.pymilos(options, data, wave_axis,weight = weight,initial_model = initial_model, cavity = cavity)
            #check if we need any worker to work harder
            if nyx // num_workers != nyx / num_workers:
                plus_one_worker = 1
                print('warning. A worker works a bit more than the rest')
                extra_data = data[nyx // num_workers * num_workers:,:,:]
                if cavity.shape:
                    extra_cavity_data = cavity[nyx // num_workers * num_workers:]
            else:
                plus_one_worker = 0

            data = np.reshape(data[:nyx // num_workers * num_workers,:,:], (num_workers, nyx // num_workers, npol, nwave))  # split data into pieces
            if cavity.shape:
                cavity = np.reshape(cavity[:nyx // num_workers * num_workers], (num_workers, nyx // num_workers))  # split cavity into pieces

            if cavity.shape:
                args_list = [(stripe, data[stripe, :],cavity[stripe]) for stripe in range(num_workers)]
            else:
                args_list = [(stripe, data[stripe, :],cavity) for stripe in range(num_workers)]

            if plus_one_worker:
                if cavity.shape:
                    args_list.append((num_workers+plus_one_worker,extra_data,extra_cavity_data))
                else:
                    args_list.append((num_workers+plus_one_worker,extra_data,cavity))

            with concurrent.futures.ProcessPoolExecutor(num_workers+plus_one_worker) as executor:
                results = executor.map(phi_rte_map_func, args_list)

            for count, result in enumerate(results):
                if count == 0:
                    stripes = result[0]
                    data = result[1]
                else:
                    stripes = np.append(stripes,result[0])
                    data = np.append(data,result[1],axis=0)
            if (np.ndim(mask)) != 0:
                output_from_rte[mask == 1, :] = np.reshape(np.array(data), (nyx, 12))
            else:
                output_from_rte = np.reshape(np.array(data), (nyx, 12))

        else:
            print('Entering pmilos')
            if (np.ndim(mask)) != 0:
                output_from_rte[mask == 1, :] = pymilos.pymilos(options, data, wave_axis,weight = weight,initial_model = initial_model, cavity = cavity)
            else:
                output_from_rte = pymilos.pymilos(options, data, wave_axis,weight = weight,initial_model = initial_model, cavity = cavity)


        return np.einsum('ijk->kij', np.reshape(output_from_rte,(nx,ny,12)))
