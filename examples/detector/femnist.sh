#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=72G
#SBATCH --output=Attack_FEMNIST_lenet5_rand.out
filename='femnist_lenet5.yml'
echo "The configuration filename is: $filename"
echo " " 
while read line; do
# reading each line
echo "$line"

done < $filename
python detector.py -c femnist_lenet5.yml -b /data/ykang/plato
