# Data Mining Project

| File | Task | Dataset |
|---|---|---|
| `heart_disease.py` | Classification | UCI Heart Disease |
| `student_performance.py` | Regression | UCI Student Performance (target G3) |
| `market_basket.py` | Association Rules | UCI Online Retail |
| `data_loader.py` | downloads + caches the datasets in `data/` | |

## Setup (PyCharm)
1. Open this folder as a project, create a Virtualenv interpreter.
2. In the Terminal: `pip install -r requirements.txt`
3. Run once (needs internet): `python data_loader.py`
4. Run each script (or run it cell by cell using the `# %%` markers).

Charts are saved in `outputs/`. `data/` and `outputs/` are in `.gitignore`.

## Notes
- Scaling, imputation, feature selection and SMOTE are fitted on the train set only (no data leakage).
- `student_performance.py`: run with `USE_G1_G2 = False` and `True`, then compare in the report.
- `market_basket.py`: if it is slow, raise `MIN_SUPPORT` or lower `TOP_ITEMS`.
