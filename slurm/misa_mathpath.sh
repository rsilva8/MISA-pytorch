#!/bin/bash
#SBATCH -e /data/users3/rsilva/MISA-pytorch/slurm_logs/error%A-%a.err
#SBATCH -o /data/users3/rsilva/MISA-pytorch/slurm_logs/out%A-%a.out
#SBATCH -e /data/users3/rsilva/MISA-pytorch/slurm_logs/error%A-%a.err
#SBATCH -o /data/users3/rsilva/MISA-pytorch/slurm_logs/out%A-%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rsilva@student.gsu.edu
#SBATCH --chdir=/data/users3/rsilva/MISA-pytorch
#SBATCH --mail-user=rsilva@student.gsu.edu
#SBATCH --chdir=/data/users3/rsilva/MISA-pytorch
#
#SBATCH -p qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --array=0-2
#SBATCH --account=trends53c17
#SBATCH --job-name=MISAtorch
#SBATCH --verbose
#SBATCH --time=7200
#
#SBATCH --nodes=1
#SBATCH --mem=128g
#SBATCH --cpus-per-task=32


sleep 5s
hostname
source ~/.bashrc
. ~/init_miniconda3.sh
conda activate pt2

seed=(7 14 21)
w=('wpca' 'w0' 'w1')

SEED=${seed[$((SLURM_ARRAY_TASK_ID % 3))]}
echo $SEED
W=${w[$((SLURM_ARRAY_TASK_ID / 3))]}
echo $W
declare -i n_dataset=100
declare -i n_source=12
declare -i n_sample=32768
lrs=(0.01 0.001 0.0001)

batch_size=(316 316 316)
patience=(10 10 10)

#Adam optimizer parameters
foreach=(0 0 0)
fused=(0 0 0)
beta1=(0.7 0.8 0.9)
beta2=(0.65 0.95 0.8)

experimenter="$USER"
configuration="/data/users3/rsilva/MISA-pytorch/configs/sim-siva.yaml"

data_file="sim-siva_dataset"$n_dataset"_source"$n_source"_sample"$n_sample"_seed"$SEED".mat"
declare -i num_experiments=${#lrs[@]}
for ((i=0; i<num_experiments; i++)); do
    python main.py -c "$configuration" -f "$data_file" -r results/MathPath2024/ -w "$W" -a -lr "${lrs[$i]}" -b1 "${beta1[$i]}" -b2 "${beta2[$i]}" -bs "${batch_size[$i]}" -e "$experimenter" -fu "${fused[$i]}" -fo "${foreach[$i]}" -p "${patience[$i]}" -s "$SEED"
    sleep 5s
done