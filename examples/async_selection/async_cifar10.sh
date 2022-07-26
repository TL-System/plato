#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=72G
#SBATCH --output=async_CIFAR_resnet_500_20eachRound_concen5_skipBNlayer1.out


filename='./async_cifar10_resnet.yml'
echo "The configuration filename is: $filename"
echo " " 
while read line; do
# reading each line
echo "$line"

done < $filename

python async_selection.py -c async_cifar10_resnet.yml -b /data/ykang/plato
