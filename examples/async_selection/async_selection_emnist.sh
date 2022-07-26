#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=72G
#SBATCH --output=Async_selection_EMNIST_lenet5_500_20eachRound_10least_10stalenss_sleep10_concen1_newSampler_seed1_concen1.out

filename='async_emnist_lenet5.yml'
echo "The configuration filename is: $filename"
echo " " 
while read line; do
# reading each line
echo "$line"

done < $filename
python async_selection.py -c async_emnist_lenet5.yml -b /data/ykang/plato


