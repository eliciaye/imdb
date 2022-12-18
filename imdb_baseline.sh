#!/bin/bash
#SBATCH --array=1-10
#SBATCH -p rise # partition (queue)
#SBATCH -D /home/eecs/eliciaye/imdb
#SBATCH --exclude=freddie,havoc,r4,r16,atlas,blaze,flaminio,manchester,bombe,pavia,como,luigi
##SBATCH --nodelist=luigi
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=1 # number of cores per task
#SBATCH --gres=gpu:1
#SBATCH -t 7-00:00 # time requested (D-HH:MM)
#SBATCH -o imdb_baseline_adam_%A_%a.out

pwd
hostname
date
echo starting job...
source ~/.bashrc
conda activate nlp

cd /home/eecs/eliciaye/imdb

cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p config.txt)
lr=$(echo $cfg | cut -f 1 -d ' ')

nlayers=3
epochs=200
NAME=baseline_$epochs-lr$lr-layers$nlayers

# python main.py --run_name $NAME --use_weightwatcher \
# --baseline --sample_evals \
# --lr $lr --nlayers $nlayers --epoch_num $epochs

python main_adam.py --run_name $NAME --use_weightwatcher \
--baseline \
--lr $lr --nlayers $nlayers --epoch_num $epochs

echo "All done."
