"""Preprocessing and dataset preparation for Student Performance Predictor."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_COLUMNS = [
    "age",
    "study_hours",
    "attendance",
    "sleep_hours",
    "previous_grade",
    "assignments_completed",
    "participation",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "internet_access",
    "parent_education",
    "family_income",
    "extra_classes",
]


@dataclass
class DatasetSplit:
    """Container for regression and classification train/test splits."""
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_reg_train: pd.Series
    y_reg_test: pd.Series
    y_clf_train: pd.Series
    y_clf_test: pd.Series


def clean_student_data(data: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, standardize values, and fix invalid ranges."""
    df = data.copy().drop_duplicates().reset_index(drop=True)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    numeric_bounds = {
        "age": (10, 25),
        "study_hours": (0, 16),
        "attendance": (0, 100),
        "sleep_hours": (0, 14),
        "previous_grade": (0, 100),
        "assignments_completed": (0, 100),
        "participation": (0, 100),
    }

    for col, (lower, upper) in numeric_bounds.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower, upper)

    categorical_defaults = {
        "gender": "unknown",
        "internet_access": "unknown",
        "parent_education": "unknown",
        "family_income": "unknown",
        "extra_classes": "no",
    }
    for col, default in categorical_defaults.items():
        if col in df.columns:
            df[col] = df[col].astype("string").replace(["", "nan", "none"], default).fillna(default)

    if "final_score" in df.columns and "pass_fail" not in df.columns:
        df["pass_fail"] = np.where(df["final_score"] >= 40, 1, 0)

    return df


def build_preprocessor(
    numeric_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
) -> ColumnTransformer:
    """Build the sklearn preprocessing graph."""
    num_cols = numeric_columns or NUMERIC_COLUMNS
    cat_cols = categorical_columns or CATEGORICAL_COLUMNS

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
