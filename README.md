# Probabilistic Submodular Models

## Tech stack

- Python 3.7 or superior
- [Hydra](https://hydra.cc) configuration framework


## Initialize project

Install the virtual environment:

```bash
python3 -m venv ./venv
```

Activate the virtual environment:

```bash
source ./venv/bin/activate
```

Install third-party dependencies:

```bash
python3 -m pip install -r python/requirements.txt
```

## How to run

This project is composed of 3 Python modules (one for each step of the application) that should be run in sequence.
The output of the application is stored in the [/out](/out) folder in a hierarchical fashion.

It's easier to explain the hierarchical structure of the output via an example.
Consider an experiment with the `demo_monotone` probabilistic submodular model with a `3`-dimensional ground set,
`M=10000` samples obtained with a `metropolis` sampler.
The ground truth density map, which associates each subset to the probability of being sampled from the ground truth distribution, is saved in [/out/demo_monotone/n-3/ground_truth.csv](/out/demo_monotone/n-3/ground_truth.csv).
The history of the samples obtained with the `metropolis` sampler is saved in [/out/demo_monotone/n-3/M-10000/metropolis/history.csv](/out/demo_monotone/n-3/M-10000/metropolis/history.csv).
The cumulative probability distances are saved in [/out/demo_monotone/n-3/M-10000/metropolis/cumulative_probability_distances.csv](/out/demo_monotone/n-3/M-10000/metropolis/cumulative_probability_distances.csv).
The plots are saved in [/out/demo_monotone/n-3/M-10000/metropolis/plots](/out/demo_monotone/n-3/M-10000/metropolis/plots).

### (1) Sample Step: python.mcmc_sample

- Compute the ground truth density map (which associates each subset to the probability of being sampled from the ground truth distribution) only once
- Run the specified samplers over the target probabilistic submodular models, saving the history of the samples obtained in CSV

```bash
python3 -u -m python.mcmc_sample -m \
  obj=demo_monotone,demo_non_monotone,delta_5 \
  sample_size=small,middle,big \
  sampler=gibbs,metropolis,lovasz_projection
```

### (2) Metrics collection Step: python.metrics

- Compute the cumulative probability distances for all experiments run in the previous step, saving them in CSV

```bash
python3 -u -m python.metrics
```

### (3) Plot generation Step: python.plotter

- Plot the cumulative probability distance between each sampler's outcome and the ground truth distribution

```bash
python3 -u -m python.plotter
```
## Configuration

Please read and edit [/python/conf/config.yaml](/python/conf/config.yaml)
