import glob
import os

# Create list of US ROIs found in data_path
data_path = './data/'
US_rois = []
for name in glob.glob(data_path + 'covidtimeseries_US_??.csv'):
    roi = os.path.basename(name)
    US_rois.append(roi[16:-4])
US_rois.sort()

swarmFile = open("run.swarm", "w")

line = ("source /data/schwartzao/conda/etc/profile.d/conda.sh "
        "&& conda activate covid "
        "&& python /home/schwartzao/covid-sicr/scripts/run.py SICRdiscrete "
        "-r='{}' "
        "-mp='/data/schwartzao/covid-sicr/models/' "
        "-fp='/data/schwartzao/covid-sicr/fits/' "
        "-it=10000 -wm=6000 -f=1 -ad=.85 -ft=1"
        )

for roi in US_rois:
    swarmFile.write(line.format(roi))
    swarmFile.write('\n')

swarmFile.close()
