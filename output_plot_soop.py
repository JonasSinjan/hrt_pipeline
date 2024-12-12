"""
Generate one standard plot for each SOOP/day for pipeline results in a specific l2 directory
"""


import os
from argparse import ArgumentParser

from sophi_hrt_pipe.utils import plot_l2_pdf


# Parse command line arguments
arg_parser = ArgumentParser(description='Plot pipeline results')
arg_parser.add_argument('path', type=str, metavar='PATH', help='Path to l2 directory')
args = arg_parser.parse_args()

dirs = os.listdir(args.path); dirs.sort()
for d in dirs:
    daydir = args.path+d

    file_n = os.listdir(daydir)
    file_n.sort()
    icnt_n = [i for i in file_n if 'icnt' in i]; icnt_n.sort()

    N = len(icnt_n)
    did_old = 0
    for n in range(N):
        did = icnt_n[n].split('_')[-1][:10]
        version = icnt_n[n].split('_')[-2]
        if int(did) - int(did_old) > 2:
            print('Plotting',did)
            plot_l2_pdf(daydir,did,version)
        did_old = did
