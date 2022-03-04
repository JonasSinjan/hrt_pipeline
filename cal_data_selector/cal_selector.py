from datetime import datetime as dt
import pandas as pd

def return_cal_data(master_path, input_ilam_filename):
  """
  master_path: str "/path/to/master/calibration/folder/"
  input_ilam_filename: str "xxx.fits"
  """
  
  dark_fol = master_path + "dark/"
  flat_fol = master_path + "flat/"
  prefilter_fol = master_path + "prefilter/"

  #extract time for input file
  time = input_ilam_filename.split("T")[0][-8:]
  dt_time = dt.strptime(time, "%Y%m%d")
  print(dt_time)

  df = pd.read_csv(master_path + 'look_up_table.csv', skiprows=1)
  df['Start'] = pd.to_datetime(df['Start'], dayfirst = True)
  df['End'] = pd.to_datetime(df['End'], dayfirst = True)

  start_flat = df.index[df['Filename'] == 'Flat'].tolist()[0]
  start_prefilter = df.index[df['Filename'] == 'Prefilter'].tolist()[0]

  darks = df.iloc[0:start_flat+1] #rows that contain darks
  flats = df.iloc[start_flat+1:start_prefilter] #rows that contain flats
  prefilters = df.iloc[start_prefilter+1:]#rows that contain flats

  print(darks)
  print(flats)
  print(prefilters)

  #loop through dark time ranges, see if it fits
  for index, row in darks.iterrows():
    if dt_time >= row['Start'] and dt_time <= row['End']:
      dark = row['Filename']
      break
    else:
      dark = darks.iloc[-1]['Filename']

  #loop through flats
  for index, row in flats.iterrows():
    if dt_time >= row['Start'] and dt_time <= row['End']:
      flat = row['Filename']
      us_sig = row['USSig']
      us_mode = row['USmode']
      break
    else:
      flat = flats.iloc[-1]['Filename']
      us_sig = flats.iloc[-1]['USSig']
      us_mode = flats.iloc[-1]['USmode']

  #loop through prefilter
  for index, row in prefilters.iterrows():
    if dt_time >= row['Start'] and dt_time <= row['End']:
      prefilter = row['Filename']
      break
    else:
      prefilter = prefilters.iloc[-1]['Filename']

  #print(dark, flat, prefilter, us_sig, us_mode)
  return dark_fol + dark, flat_fol + flat, prefilter_fol + prefilter, us_sig, us_mode

#test
out = return_cal_data("./", "solo_L1_phi-hrt-ilam_20210914T034515_V202110211713C_0149140201.fits")
print(out)

