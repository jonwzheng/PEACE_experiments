# PEACE_experiments

Experiments for validation against experimental data (microstates + acid-base sites), as well as numerical benchmarks.
The results from running these codes are discussed in the preprint/article: 
> Zheng, J. W. & Heid, E. An Atomistic Workflow for Determining Acid-Base Prototropic Tautomer Microstate Populations.

Citations should correspond to the preprint/article rather than to this directory.

PEACE itself is expected as a sibling checkout at `../PEACE` relative to this repository. Once this is done, simply execute the Python scripts in the respective directories.

## Layout

Each experiment lives in its own subfolder with local `data/`, scripts, and `results/`:

- `SAMPL/` — SAMPL challenge microstate preparation and analysis
- `ampholyte_zwitterion_ratios/` — benchmark predicted zwitterion fractions against tautomer-ratio reference data
- `boltzmann_vis/` — visualize Boltzmann weighting as a function of relative free energy
- `common/` — shared benchmark helpers (`benchmark_common.py`) used across experiment scripts
- `pka_database_checks/` — check PEACE ionization-site identification against IUPAC pKa database species, to verify acid-base sites

