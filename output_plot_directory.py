"""
Generate standard plots for pipeline results in a specific directory
"""


import os
from argparse import ArgumentParser

from sophi_hrt_pipe.utils import plot_l2_pdf


# Parse command line arguments
arg_parser = ArgumentParser(description='Plot pipeline results')
arg_parser.add_argument('path', type=str, metavar='PATH', help='Path to inversion results')
# arg_parser.add_argument('did', type=str, metavar='DID', help='DID of the data')
# arg_parser.add_argument('version', type=str, metavar='V', help='version',default=None)
args = arg_parser.parse_args()

file_n = os.listdir(args.path)
file_n.sort()
icnt_n = [i for i in file_n if 'icnt' in i]; icnt_n.sort()

N = len(icnt_n)

for n in range(N):
    did = icnt_n[n].split('_')[-1][:10]
    version = icnt_n[n].split('_')[-2]
    print('Plotting',did)
    plot_l2_pdf(args.path,did,version)
# hmimag = spg.plot_lib.cmap_from_rgb_file('HMI', 'hmi_mag.csv')  # color map
# solo_L2_phi-hrt-binc_20221017T011503_V43_0250170102
# # Prepare saving of plots to pdf
