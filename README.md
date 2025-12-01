# A multi-region recurrent circuit for evidence accumulation in rats

Code repository for Neuron manuscript.

## Overview

This repository contains analysis code for investigating neural population dynamics in rat frontal orienting fields (FOF) and anterior dorsal striatum (ADS) during a perceptual decision-making task with accumulating sensory evidence. The analyses include neural encoding and decoding, generalized linear models (GLM), reduced-rank regression, optogenetic perturbation experiments, and multi-region recurrent neural network (RNN) modeling.

## Repository Structure

```
├── Code/
│   ├── figure_code/          # Analysis scripts organized by figure
│   │   ├── figure1/           # Population response characterization
│   │   ├── figure2/           # Neural encoding/decoding and GLM
│   │   ├── figure3/           # Optogenetic perturbation analysis
│   │   ├── figure4_6/         # RNN modeling
│   │   └── helpers/           # Shared utility functions
│   ├── saved_results/         # Intermediate analysis outputs (.npy, .pkl)
│   ├── run_params.py          # Global configuration and paths
│   └── my_imports.py          # Standard imports and plotting settings
├── figure_pdfs/               # Generated figures (PDF format)
├── requirements.txt           # Python dependencies
```


## Main Analyses

### Figure 1: Population Neural Responses
- **Scripts**: `figure1_popraster.py`, `figure1_trialraster.py`
- **Outputs**: Population PSTHs, single-trial rasters

```bash
cd Code/figure_code/figure1/
python figure1_popraster.py
python figure1_trialraster.py
```

### Figure 2: Neural Encoding and Decoding
- **Scripts**: `figure2_encodingfigs.py`, `figure2_decodingfigs.py`, `figure2/fig2_helpers/`
- **Analyses**:
  - Choice, evidence, and history decoding using logistic regression
  - Tuning curve analysis (`compute_hanks_tuning_curves.py`)
  - Reduced-rank GLM for inter-region communication (`neural_GLM/glmfits_rr.py`)
  - Cross-correlation of decision variables (`DVcc_sims.py`)

```bash
cd Code/figure_code/figure2/
python figure2_encodingfigs.py
python figure2_decodingfigs.py
```

### Figure 3: Optogenetic Perturbations
- **Script**: `figure3/figure1_metaopto.m` (MATLAB)
- **Analysis**: Effects of inactivation of FOF to ADS projections on behavior

### Figures 4-6: RNN Modeling
- **Scripts**: `figure4_6/train_nn_1_multiplicative_opto.py`, `analysis/fig4_save_*.py`
- **Architecture**: Multi-region RNN with recurrent connectivity trained on evidence accumulation task
- **Analyses**: Network inactivation simulations, decoding from RNN units, recovery mechanisms

## Configuration

Before running analyses, edit `Code/run_params.py`:

```python
BASEPATH = "/path/to/Code/"
```

All other paths are automatically configured relative to `BASEPATH`.


## Contact

For questions regarding code or analyses, please contact Diksha Gupta [diksha.gupta@ucl.ac.uk]

