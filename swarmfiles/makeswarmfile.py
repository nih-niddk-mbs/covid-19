import glob
import os
import re

data_path = '/Users/schwartzao/Documents/GitHub/covid-sicr/data/'
US_rois = []
for name in glob.glob(data_path + 'covidtimeseries_US_??.csv'): # focusing on US
    roi = os.path.basename(name)
    roi = re.search('[US_..]', roi)
    print(roi)

# rois_path =  # remember to change for Biowulf
# # roi_files = os.listdir(rois_path)
# print(roi_files)
# rois =




#
# source /data/schwartzao/conda/etc/profile.d/conda.sh \
# && conda activate covid \
# && python /home/schwartzao/covid-sicr/scripts/run.py SICRdiscrete -r='Afghanistan' -mp='/data/schwartzao/covid-sicr/models/' -dp='/home/schwartzao/covid-sicr/data/' -fp='/data/schwartzao/covid-sicr/fits/' -it=10000	-wm=6000	-f=1	-ad=.85	-ft=1
