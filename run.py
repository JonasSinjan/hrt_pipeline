from hrt_pipe import phihrt_pipe
import numpy as np

sciencedata_fits_filenames = ['solo_L0_phi-hrt-ilam_20210421T120003_V202106080929C_0144210101.fits', 'solo_L0_phi-hrt-ilam_20210424T120003_V202106141014C_0144240101.fits', 
  'solo_L0_phi-hrt-ilam_20210425T120002_V202106141020C_0144250101.fits', 'solo_L0_phi-hrt-ilam_20210426T120002_V202106162118C_0144260101.fits',
  'solo_L0_phi-hrt-ilam_20210427T120002_V202106162052C_0144270101.fits', 'solo_L0_phi-hrt-ilam_20210427T120002_V202106171444C_0144270101.fits', 
  'solo_L0_phi-hrt-ilam_20210427T120002_V202106171517C_0144270101.fits']
  
#['solo_L0_phi-hrt-ilam_0667414748_V202103221851C_0142230201.fits']#['solo_L0_phi-hrt-ilam_0667414905_V202103221851C_0142230602.fits', 'solo_L0_phi-hrt-ilam_0667415054_V202103221851C_0142230603.fits', 'solo_L0_phi-hrt-ilam_0667415205_V202103221851C_0142230604.fits', 'solo_L0_phi-hrt-ilam_0667415354_V202103221851C_0142230605.fits', 'solo_L0_phi-hrt-ilam_0667415505_V202103221851C_0142230606.fits', 'solo_L0_phi-hrt-ilam_0667415654_V202103221851C_0142230607.fits', 'solo_L0_phi-hrt-ilam_0667415805_V202103221851C_0142230608.fits']#['../fits_files/solo_L0_phi-hrt-ilam_0667414748_V202103221851C_0142230201.fits']
#sciencedata_fits_filenames = ['solo_L0_phi-hrt-ilam_0667414748_V202103221851C_0142230201.fits']
#sciencedata_fits_filenames = ['solo_L0_phi-hrt-ilam_0667414905_V202103221851C_0142230602.fits', 'solo_L0_phi-hrt-ilam_0667415054_V202103221851C_0142230603.fits', 'solo_L0_phi-hrt-ilam_0667415205_V202103221851C_0142230604.fits', 'solo_L0_phi-hrt-ilam_0667415354_V202103221851C_0142230605.fits', 'solo_L0_phi-hrt-ilam_0667415505_V202103221851C_0142230606.fits', 'solo_L0_phi-hrt-ilam_0667415654_V202103221851C_0142230607.fits', 'solo_L0_phi-hrt-ilam_0667415805_V202103221851C_0142230608.fits']

flatfield_fits_filename = '../fits_files/solo_L0_phi-hrt-flat_0667134081_V202103221851C_0162201100.fits'
darkfield_fits_filename = '../fits_files/solo_L0_phi-fdt-ilam_20200228T155100_V202002281636_0022210004_000.fits'

sciencedata_fits_filenames = ['../fits_files/' + i for i in sciencedata_fits_filenames]

prefilter_f = '../fits_files/fitted_prefilter.fits'

c_talk_params = np.zeros((2,3))

q_slope = 0.0038
u_slope = -0.0077
v_slope = -0.0009

q_int = -0.0056 #the offset, normalised to I_c
u_int = 0.0031
v_int = -0.0002 

c_talk_params[0,0] = q_slope
c_talk_params[0,1] = u_slope
c_talk_params[0,2] = v_slope

c_talk_params[1,0] = q_int
c_talk_params[1,1] = u_int
c_talk_params[1,2] = v_int

phihrt_pipe(sciencedata_fits_filenames, darkfield_fits_filename, flatfield_fits_filename, flat_states = 24, norm_stokes = True, prefilter_f = prefilter_f, 
                   clean_f = True, ctalk_params = c_talk_params, ItoQUV = True, out_demod_file = True, out_dir = '/data/slam/home/sinjan/hrt_pipe_results/stp-136_ctalk_prefilter/', rte = 'False')
