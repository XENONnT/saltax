# saltax

[![DOI](https://zenodo.org/badge/646649000.svg)](https://doi.org/10.5281/zenodo.14247662)
[![Test package](https://github.com/XENONnT/saltax/actions/workflows/test_package.yml/badge.svg?branch=main)](https://github.com/XENONnT/saltax/actions/workflows/test_package.yml)
[![Coverage Status](https://coveralls.io/repos/github/XENONnT/saltax/badge.svg?branch=main&kill_cache=1)](https://coveralls.io/github/XENONnT/saltax?branch=main&kill_cache=1)
[![PyPI version shields.io](https://img.shields.io/pypi/v/saltax.svg)](https://pypi.python.org/pypi/saltax/)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/XENONnT/saltax/main.svg)](https://results.pre-commit.ci/latest/github/XENONnT/saltax/main)

Salting `fuse` into `strax` data, followed by comparison and analysis. Please check this [notes](https://xe1t-wiki.lngs.infn.it/doku.php?id=lanqing:ambience_interference_and_sprinkling#raw_records_simu) to see how it serves physics.

## Installation

```
cd saltax
pip install -e ./ --user
```

## Tutorial

Please check notebooks in `notebooks/`.

## Computing

Please see folder `jobs/` for slurm job submission. Below you can see a benchmark from a 26 seconds Ar37 run, sprinkled 50Hz flat beta band. You can roughly estimate the overhead by scaling it, neglecting the rate dependence in overhead.

| Step                  | Overhead [sec] (salt) | Overhead [sec] (simu) |
| :-------------------: | :-------------------: | :-------------------: |
| `microphysics_summary`| 19                    |                       |
| `raw_records_simu`    | 210                   |                       |
| `records`             | 15                    |                       |
| `peaklets`            | 68                    | 9                     |
| `peak_basics`         | 3                     | 1                     |
| `event_info`          | 63                    | 11                    |
