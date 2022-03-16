#!/bin/bash

module load openmpi_gcc
module load anaconda/3-5.0.1
modlue load fftw/3.3.5

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/local/cfitsio/cfitsio-3.350/lib

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/local/cfitsio/cfitsio-3.350/include

#compile the rte codes

cd ./cmilos
make clear
make

cd ..

cd ./cmilos-fits
make clear 
make

cd ..

cd p-milos
make clean
make

cd ..

conda env create -f environment.yml
#pip install -r requrements.txt

source activate hrt_pipeline_env

pip install -e .

