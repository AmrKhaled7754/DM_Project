"""
Loads the three datasets. The first call downloads from UCI (needs internet)
and saves a CSV copy in data/. Next calls read the local CSV.
"""
from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)


def load_heart() -> pd.DataFrame:
    """UCI Heart Disease. Returns features + binary 'target' (1 = disease)."""
    path = DATA / "heart.csv"
    if path.exists():
        return pd.read_csv(path)
    ds = fetch_ucirepo(id=45)
    df = ds.data.features.copy()
    df["target"] = (ds.data.targets["num"] > 0).astype(int)  # 0 = healthy, 1-4 = disease
    df.to_csv(path, index=False)
    return df


def load_student() -> pd.DataFrame:
    """UCI Student Performance. Target column is G3 (final grade, 0-20)."""
    path = DATA / "student.csv"
    if path.exists():
        return pd.read_csv(path)
    ds = fetch_ucirepo(id=320)
    df = ds.data.original.copy()
    df.to_csv(path, index=False)
    return df


def load_retail() -> pd.DataFrame:
    """UCI Online Retail (541,909 rows)."""
    path = DATA / "online_retail.csv"
    if path.exists():
        return pd.read_csv(path)
    ds = fetch_ucirepo(id=352)
    df = ds.data.original.copy()
    if "InvoiceNo" not in df.columns:
        df = df.reset_index()
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    # Run this file once to download everything: python data_loader.py
    for name, fn in [("heart", load_heart), ("student", load_student), ("retail", load_retail)]:
        d = fn()
        print(f"{name}: {d.shape}")
