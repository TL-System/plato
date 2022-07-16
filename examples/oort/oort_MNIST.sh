#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=oort_MNIST_lenet5_500_20eachRound_10least_no_test_stalenss_10.out
filename='oort_MNIST_lenet5.yml'
echo "The configuration filename is: $filename"
echo " " 
while read line; do
# reading each line
echo "$line"

done < $filename
python oort.py -c oort_MNIST_lenet5.yml -b /data/ykang/plato
