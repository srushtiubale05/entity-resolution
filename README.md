# Entity Resolution Pipeline — VS Code

This package is designed to run on the full training files without loading unnecessary
copies of the multi-million-row datasets into RAM.

Expected files in `data/`:
- train_source1.tsv
- train_source2.tsv
- train_source3.tsv
- train_ground_truth.tsv

Current training population observed:
- Source 1: ~2.2M rows
- Source 2: ~5.0M rows
- Source 3: ~5.3M rows
- GT: ~2.2M rows / ~7.64M positive references

IMPORTANT:
1. Put the four TSV files in `data/`.
2. Create a Python 3.11/3.12 virtual environment.
3. Install requirements.txt.
4. Run scripts in numeric order.
5. Outputs go into `outputs/`.

The scripts use chunked TSV reading where practical and avoid the Colab
in-memory representation that caused RAM crashes.

The pipeline currently covers:
01_eda.py
02_normalization.py
03_gt_validation.py
04_positive_pairs.py
05_blocking.py
06_features.py
07_train_model.py
08_threshold.py
09_output.py

The later model stages are intentionally conservative and can be adjusted after
checking the diagnostics produced by earlier stages.
