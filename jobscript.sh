#!/bin/bash
#SBATCH --time=00:01:00
#SBATCH --partition=regular

module purge
module load Python/3.11.3-GCCcore-12.3.0

source $HOME/venvs/ltp/bin/activate

python3 --version
which python3

deactivate