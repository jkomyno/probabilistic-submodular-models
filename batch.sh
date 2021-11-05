#!/bin/bash

#SBATCH -J probabilistic-models
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --error=stderr.txt
#SBATCH --output=stdout.txt
#SBATCH --partition=cpufast
ml Python/3.8.6-GCCcore-10.2.0

# squeue --format="%.18i %.9P %.30j %.8T %.10M %.9l %.6D %R" --me

# generate samples
echo "generate samples"
python -u -m python.mcmc_sample -m \
	obj=demo_monotone,delta_5 \
	sample_size=small,middle,big \
	sampler=gibbs,metropolis,lovasz_projection,lovasz_projection_continuous,lovasz_projection_continuous_descending

# collect metrics
echo "collect metrics"
python -u -m python.metrics

# generate plots
echo "generate plots"
python -u -m python.plots
