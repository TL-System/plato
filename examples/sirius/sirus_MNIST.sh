#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=Sirius_MNSIT_lenent5_rand1.out
filename='Sirius_MNIST_lenet5.yml'
echo "The configuration filename is: $filename"
echo " " 
while read line; do
# reading each line
echo "$line"

done < $filename
python sirius.py -c sirius_MNIST_lenet5.yml -b /data/ykang/plato

