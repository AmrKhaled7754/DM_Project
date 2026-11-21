"""
Loads the three datasets. Each project has its own folder inside data/:
  data/heart/    processed.cleveland.data, processed.hungarian.data,
                 processed.switzerland.data, processed.va.data
  data/student/  student-mat.csv
  data/retail/   Online Retail.xlsx
"""
from pathlib import Path
import pandas as pd

DATA = Path(__file__).parent / "data"

HEART_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
              "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]


def _path(folder: str, filename: str) -> Path:
    path = DATA / folder / filename
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}\nPut it inside data/{folder}/")
    return path


def _read_heart(filename: str) -> pd.DataFrame:
    return pd.read_csv(_path("heart", filename), header=None, names=HEART_COLS, na_values="?")


def load_heart(filename: str = "processed.cleveland.data") -> pd.DataFrame:
    """One hospital file (default Cleveland, 303 rows). Adds binary target: 1 = disease."""
    df = _read_heart(filename)
    df["target"] = (df["num"] > 0).astype(int)
    return df.drop(columns="num")


def load_heart_all() -> pd.DataFrame:
    """All 4 hospitals merged (920 rows). Column 'source' = hospital name (drop it before training)."""
    parts = []
    for name in ["cleveland", "hungarian", "switzerland", "va"]:
        d = _read_heart(f"processed.{name}.data")
        d["source"] = name
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["target"] = (df["num"] > 0).astype(int)
    return df.drop(columns="num")


def load_student(filename: str = "student-mat.csv") -> pd.DataFrame:
    """UCI Student Performance (separator ';'). Target: G3."""
    return pd.read_csv(_path("student", filename), sep=";")


def load_retail(filename: str = "Online Retail.xlsx") -> pd.DataFrame:
    """UCI Online Retail. A CSV copy is cached next to the Excel file after the first read."""
    xlsx = _path("retail", filename)
    cache = xlsx.with_suffix(".csv")
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["InvoiceDate"])
    df = pd.read_excel(xlsx)
    df.to_csv(cache, index=False)
    return df


if __name__ == "__main__":
    print("heart (Cleveland):", load_heart().shape)
    print("heart (all 4)    :", load_heart_all().shape)
    print("student          :", load_student().shape)
    print("retail           :", load_retail().shape)
