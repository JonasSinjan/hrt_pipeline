import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import subprocess
import time

# here import pmilos from the pymilos cython code
from .pymilos import pmilos
#although not necessaty these are the dtype to be passed to C
DTYPE_INT = np.intc
DTYPE_DOUBLE = np.float_

#test data in IDL save format
#from scipy.io import readsav
#s = readsav('cmilos_testdata/test_data.save')
#datos = s.data
#wave_axis = s.wave_axis

#datos = np.tile(datos, 4) 

print('shape input',datos.shape) 
# the pipeline has (for the moment being) the data in
# (4, 6, 298, 1176) (pol,wave, y,x)
# This has to be changed to (y,x,pol,wave) for C
                                                                                                                                                                                                                 
datos = np.einsum('ijkl->klij',datos)
y,x,p,l = datos.shape
print('New shape',datos.shape)                                                                                                                                                                                                                      

#the next input to pymilos is the options
#for the moment no PSF is included 
# and classical estimates are deactivated.
# these will be added in following versions

options = np.zeros((4))#,dtype=DTYPE_INT)
options[0] = len(wave_axis) #NLAMBDA wave axis dimension
options[1] = 15 #MAX_ITER max number of iterations
options[2] = 0 #CLASSICAL_ESTIMATES [0,1] classical estimates ON or OFF
options[3] = 0 #RFS [0,1,2] 0.-> Inversion, 1-> synthesis 0-> RFS
# Only inversion is now available
# options[4] = 0 #FWHM(in A)
# options[5] = 0 #DELTA(in A)
# options[6] = 0 #NPOINTS

npx = x
npy = y

print('----------- pmilos---------')
start = time.process_time()
out = pmilos(options,datos[0:npx,0:npy,:,:],wave_axis)
print(time.process_time() - start)
print(out.shape)
#output has npy*npx 
#out = np.reshape(out,(npy,npx,12))                                                                                                                                                                                        

#comparison using milos

sdata = datos[0:npx,0:npy,:,:] #stupid copy
y,x,p,l = sdata.shape

print('----------- milos---------')
start = time.process_time()

filename = 'dummy_in.txt'
with open(filename,"w") as f: 
    for i in range(x):
        for j in range(y):
            for k in range(l):
                f.write('%e %e %e %e %e \n' % (wave_axis[k],sdata[j,i,0,k],sdata[j,i,1,k],sdata[j,i,2,k],sdata[j,i,3,k]))
del sdata

rte_on = subprocess.call("time ./milos.bob 6 15 0 0 dummy_in.txt  >  dummy_out.txt",shell=True)
print(rte_on)
print(time.process_time() - start)


# cmd = subprocess.Popen('ls -l', shell=True, stdout=PIPE)
# for line in cmd.stdout.readlines():
#     print line


res = np.loadtxt('dummy_out.txt')
npixels = res.shape[0]/12.
result = np.zeros((y*x,12)).astype(float)
for i in range(y*x):
    result[i,:] = res[i*12:(i+1)*12]
result = np.einsum('ijk->jik',result.reshape(y,x,12))

fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(121)
im1 = ax1.imshow(result[:,:,8], interpolation='None')

divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')

ax2 = fig.add_subplot(122)
im2 = ax2.imshow(out[:,:,8], interpolation='None')

divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')

plt.show()

			# outputdata[cnt_model] = contador;
			# outputdata[cnt_model+1] = iter;
			# outputdata[cnt_model+2] = initModel.B;
			# outputdata[cnt_model+3] = initModel.gm;
			# outputdata[cnt_model+4] = initModel.az;
			# outputdata[cnt_model+5] = initModel.eta0;
			# outputdata[cnt_model+6] = initModel.dopp;
			# outputdata[cnt_model+7] = initModel.aa;
			# outputdata[cnt_model+8] = initModel.vlos;
			# outputdata[cnt_model+9] = initModel.S0;
			# outputdata[cnt_model+10] = initModel.S1;
			# outputdata[cnt_model+11] = chisqrf;
