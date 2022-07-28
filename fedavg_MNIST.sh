#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=FedAvg_MNIST_lenet5_500_20eachRound_10least_no_test_stalenss_10.out
echo "==============================================="
echo "This .out file is generated at: "
date
echo "==============================================="
echo " " 

filename1 = './fedavg_MNIST.sh'
echo "The bash filename is: $filename1"
echo " " 
while read line; do
# reading each line
echo "$line"

done < $filename1
echo " " 

filename2='./configs/MNIST/fedavg_lenet5_noiid.yml'
echo "The configuration filename is: $filename2"
echo " " 
while read line; do
# reading each line
echo "$line"

done < $filename2

./run -c ./configs/MNIST/fedavg_lenet5_noniid.yml -b /data/ykang/plato
