import argparse
import glob
from pathlib import Path
import os
from os import mkdir
from datetime import datetime

# Parse all the command-line arguments
parser = argparse.ArgumentParser(description=('Creates swarm file for '
                                              'make-tables.py'))
today = datetime.today().strftime('%Y%m%d')
parser.add_argument('-fn', '--file-name', default='maketables',
                    help='Swarm filename')
parser.add_argument('-fp', '--fits-path', default=today,
                    help='Path to subdirectory to save fit files. Default is'
                    ' todays date')
parser.add_argument('-tp', '--tables-path', default=today,
                    help='Path to subdirectory to save tables. Default is'
                    ' todays date')
parser.add_argument('-ft', '--fixed-t', default=1,
                    help='Use a fixed time base (where 1/22/20 is t=0'))
args = parser.parse_args()

swarmFile = open(f"../{args.file_name}.swarm", "w")
line = ("source /data/schwartzao/conda/etc/profile.d/conda.sh "
        "&& conda activate covid "
        "&& python /home/schwartzao/covid-sicr/scripts/make-tables.py "
        f"-fp='/data/schwartzao/covid-sicr/fits/{args.fits_path}' "
        f"-tp='/data/schwartzao/covid-sicr/tables/{args.tables_path}' "
        f"-ft={args.fixed_t} --max-jobs=$SLURM_CPUS_PER_TASK"
        )
print(line)

swarmFile.write(line)
swarmFile.write('\n')
swarmFile.close()
