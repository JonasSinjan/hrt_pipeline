
time ./milos 6 15 0 0.08 0.07 7 ./CRISP/fits_CRISP_1_6wl.txt > ./CRISP/result_C/fits_CRISP_1_6wl_15iter_noCE_conv_080_070_7_svdcordic27b.txt

echo "running...."
date
time ./milos 6 15 1 0.08 0.07 7 ./CRISP/fits_CRISP_1_6wl.txt > ./CRISP/result_C/fits_CRISP_1_6wl_15iter_CE_conv_080_070_7_svdcordic27b.txt
echo "running....1" 
date

time ./milos 6 25 1 0.08 0.07 7 ./CRISP/fits_CRISP_1_6wl.txt > ./CRISP/result_C/fits_CRISP_1_6wl_25iter_CE_conv_080_070_7_svdcordic27b.txt
echo "running....2" 
date

time ./milos 6 25 1 ./CRISP/fits_CRISP_1_6wl.txt > ./CRISP/result_C/fits_CRISP_1_6wl_25iter_CE_noconv_svdcordic27b.txt

echo "running...."
date

time ./milos 6 30 1 ./CRISP/fits_CRISP_1_6wl.txt > ./CRISP/result_C/fits_CRISP_1_6wl_30iter_noCE_noconv_svdcordic27b.txt

echo "running...."
date
