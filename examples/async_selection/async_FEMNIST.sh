#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem=64G
#SBATCH --output=Async_selection_FEMNIST_lenet5_500_20eachRound_10least_10stalenss_sleep10.out
filename='async_femnist_lenet5.yml'
echo "The configuration filename is: $filename"
echo " " 
while read line; do
# reading each line
echo "$line"

done < $filename
python async_selection.py -c async_femnist_lenet5.yml -b /data/ykang/plato