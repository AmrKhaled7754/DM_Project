"""
Loads the three datasets. Each project has its own folder inside data/:
  data/heart/    heart.csv                       (Cardiovascular disease, sep=';')
  data/student/  StudentPerformanceFactors.csv   (Target: Exam_Score)
  data/retail/   Online Retail.csv  (or Online Retail.xlsx)
"""
from pathlib import Path
import pandas as pd

DATA = Path(__file__).parent / "data"


def _path(folder: str, filename: str) -> Path:
    path = DATA / folder / filename
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}\nPut it inside data/{folder}/")
    return path


def load_heart(filename: str = "heart.csv") -> pd.DataFrame:
    """Cardiovascular dataset (10K rows). Separator is ';'. Target: 'cardio' (0/1)."""
    return pd.read_csv(_path("heart", filename), sep=";")


def load_student(filename: str = "StudentPerformanceFactors.csv") -> pd.DataFrame:
    """Student Performance Factors (Kaggle). Target: Exam_Score."""
    return pd.read_csv(_path("student", filename))


def _read_csv_safe(path: Path) -> pd.DataFrame:
    """Online Retail CSV is often latin-1 encoded, so fall back if utf-8 fails."""
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="ISO-8859-1")


def load_retail(filename: str = "Online Retail.csv") -> pd.DataFrame:
    """Online Retail. Reads the CSV; if it's missing, reads the xlsx and caches a CSV copy."""
    csv_path = DATA / "retail" / filename
    if csv_path.exists():
        df = _read_csv_safe(csv_path)
    else:
        xlsx = _path("retail", csv_path.with_suffix(".xlsx").name)
        df = pd.read_excel(xlsx)
        df.to_csv(csv_path, index=False)
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    return df


if __name__ == "__main__":
    print("heart  :", load_heart().shape)
    print("student:", load_student().shape)
    print("retail :", load_retail().shape)