"""
check for a few datasets that the scaling of the dark field and flat field is the same
"""
from astropy.io import fits
import numpy as np
import pytest
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.append('../')

from sophi_hrt_pipe.utils import get_data

def load_fits(path):
    hdul = fits.open(path)
    return np.array(hdul[0].data, dtype = np.float32)

def test_science_dark_scaling():
    feb_2021 = {'science_f': '/data/slam/home/sinjan/fits_files/solo_L1_phi-hrt-ilam_20210223T170002_V202109240858C_0142230201.fits', 'accum_scaling': True, 'bit_conversion': True, 'scale_data': True, 'dark_f': '/data/slam/home/sinjan/fits_files/solo_L1_phi-fdt-ilam_20200228T155103_V202109240737I_0022210004.fits'}
    sep_2021 = {'science_f': '/data/slam/home/sinjan/fits_files/solo_L1_phi-hrt-ilam_20210914T071515_V202110260809C_0149140401.fits', 'accum_scaling': True, 'bit_conversion': True, 'scale_data': False, 'dark_f': '/data/slam/home/sinjan/fits_files/solo_L1_phi-hrt-ilam_20210428T130238_V202109240900C_0164281001.fits'}

    sci_datasets = [feb_2021, sep_2021] #missing apr_2021


    for i, sci in enumerate(sci_datasets):

        science_f = sci['science_f']
        accum_scaling = sci['accum_scaling']
        bit_conversion = sci['bit_conversion']
        scale_data = sci['scale_data']
        dark_f = sci['dark_f']

        sci,_ = get_data(science_f, accum_scaling, bit_conversion, scale_data)
        dark,_ = get_data(dark_f, accum_scaling, bit_conversion, scale_data)

        print(sci.shape, i)

        avg_sci = np.mean(sci[0,:100,1920:])
        avg_dark = np.mean(dark[:100,1920:])

        std_sci = np.std(sci[0,:100,1920:])
        std_dark = np.std(dark[:100,1920:])

        print(f"Science: {avg_sci:.3g} +/- {std_sci:.3g}")
        print(f"Dark: {avg_dark:.3g} +/- {std_dark:.3g}")
        assert abs(avg_dark-avg_sci) < 1000