#!/bin/bash

#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=1:00:00:00
#SBATCH --error=stderr.txt
#SBATCH --output=stdout.txt
#SBATCH --partition=cpu
ml Python/3.8.6-GCCcore-10.2.0

# generate samples
echo "generate samples"
python -u -m python.mcmc_sample -m \
	obj=demo_monotone,delta_5 \
	sample_size=small,middle,big \
	sampler=gibbs,metropolis,lovasz_projection

# collect metrics
echo "collect metrics"
python -u -m python.metrics

# generate plots
echo "generate plots"
python -u -m python.plots
