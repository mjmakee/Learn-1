# CCAR Deposit NII & Margin — Full Shapley Analysis Package

## Contents
- `deposit_nii_shapley_demo.ipynb`: self-contained demonstration notebook. Run all cells to simulate data, perform Shapley calculations, verify reconciliations, and generate figures.
- `attribution_v2.py`: CLI implementation for CSV-based comparisons; see `METHODOLOGY.md` for command-line usage and column mapping.
- `data/sample_deposits_two_cycles.csv`: sample input with two CCAR cycles and scenario projections.
- `results/`: precomputed sample waterfall charts and CSV summaries.
- `METHODOLOGY.md`: calculation definitions and limitations.
- `requirements.txt`: Python dependencies.

## Quick start
```bash
python -m pip install -r requirements.txt
jupyter notebook deposit_nii_shapley_demo.ipynb
```
Run all notebook cells. Files contain simulated data only. This is a mathematical attribution of aggregate implied yields, not a causal or product-level FTP/mix separation.
