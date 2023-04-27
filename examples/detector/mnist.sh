#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=72G
#SBATCH --output=Attack_MNIST_lenet5_rand.out
filename='mnist_lenet5.yml'
echo "The configuration filename is: $filename"
echo " " 
while read line; do
# reading each line
echo "$line"

done < $filename
python detector.py -c mnist_lenet5.yml -b /data/ykang/plato
