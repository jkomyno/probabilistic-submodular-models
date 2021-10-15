# Probabilistic Submodular Models

## Tech stack

- Python 3.7
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

Run:

```bash
python3 -u -m python.mcmc_sample -m \
  obj=demo_monotone,demo_non_monotone \
  sample_size=small,middle,big \
  sampler=gibbs,metropolis,lovasz_projection
```

```bash
python3 -u -m python.metrics
```

## Configuration

Please read and edit [/python/conf/config.yaml](/python/conf/config.yaml)
