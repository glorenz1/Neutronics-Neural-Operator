# Usage

These scripts were designed for the UTK ISAAC HPC cluster.

* `data_gen.sh` -> generate OpenMC data
* `fno.sh` -> train model and make plots

## Before running

Edit all path placeholders to your desired paths:

* project folder
* samples folder
* results folder
* logs folder
* OpenMC `cross_sections.xml`

Make sure your conda env is set (`conda activate nno`).

## Run

```bash
sbatch data_gen.sh
sbatch fno.sh
```
