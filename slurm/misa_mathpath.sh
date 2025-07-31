#!/bin/bash
#SBATCH -e /data/users3/rsilva/MISA-pytorch/slurm_logs/error%A-%a.err
#SBATCH -o /data/users3/rsilva/MISA-pytorch/slurm_logs/out%A-%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rsilva@gsu.edu
#SBATCH --chdir=/data/users3/rsilva/MISA-pytorch
#
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:A100:1
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

# seed=(7 14 21)
# w=('wpca' 'w0' 'w1')

declare -i SEED=7
echo $SEED
W="wpca"
echo $W

declare -i n_dataset=100
declare -i n_source=12
declare -i n_sample=32768
lrs=(0.01)
batch_size=(316)
patience=(10)
gpu=("A100")
#Adam optimizer parameters
adam_params=(0) #0 sets foreach and fused to false, 1 sets foreach=true and fused=false, 2 for foreach=false and fused=true
beta1=(0.7)
beta2=(0.65)

experimenter="$USER"
configuration="/data/users3/rsilva/MISA-pytorch/configs/sim-siva.yaml"

data_file="sim-siva_dataset"$n_dataset"_source"$n_source"_sample"$n_sample"_seed"$SEED".mat"
declare -i num_experiments=${#lrs[@]}
# for ((i=0; i<num_experiments; i++)); do
#     python main.py -c "$configuration" -f "$data_file" -r results/MathPath2024/ -w "$W" -a -lr "${lrs[$i]}" -b1 "${beta1[$i]}" -b2 "${beta2[$i]}" -bs "${batch_size[$i]}" -e "$experimenter" -ff "${adam_params[$i]}" -p "${patience[$i]}" -s "$SEED" -g "${gpu[$i]}" &
#     sleep 5s
#     wait 
# done

# echo parameters, including variable name and value:
echo ""
echo "slurm_job_id: $SLURM_JOB_ID"
echo "experimenter: $experimenter"
echo "SEED: $SEED"
echo "n_dataset: $n_dataset"
echo "n_source: $n_source"
echo "n_sample: $n_sample"
echo "data_file: $data_file"
echo "configuration: $configuration"
echo "W: $W"
echo "lrs: ${lrs[@]}"
echo "batch_size: ${batch_size[@]}"
echo "patience: ${patience[@]}"
echo "gpu: ${gpu[@]}"
echo "adam_params: ${adam_params[@]} #0 sets foreach and fused to false, 1 sets foreach=true and fused=false, 2 for foreach=false and fused=true"
echo "beta1: ${beta1[@]}"
echo "beta2: ${beta2[@]}"
echo "hostname: $(hostname)"
echo "date: $(date)"
echo "pwd: $(pwd)"
echo "slurm_job_name: $SLURM_JOB_NAME"
echo "slurm_job_cpus_per_node: $SLURM_JOB_CPUS_PER_NODE"
echo "slurm_job_partition: $SLURM_JOB_PARTITION"
echo "slurm_job_mem: $SLURM_JOB_MEM"
