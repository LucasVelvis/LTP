#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16000
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

module purge
module load Python/3.11.3-GCCcore-12.3.0

source $HOME/venvs/ltp/bin/activate

python main.py

deactivate