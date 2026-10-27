"""Prediction utilities and inference logic for the Student Performance Predictor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.feature_engineering import add_engineered_features
from src.preprocessing import clean_student_data
from src.utils import MODELS_DIR, load_joblib


@dataclass
class PredictionResult:
    """Structured prediction response for one student."""
    predicted_score: float
    pass_fail: str
    pass_probability: float
    confidence_score: float
    contributions: pd.DataFrame


def load_models(regression_model=None, classifier_model=None):
    """Load trained regression and classification pipelines."""
    if regression_model is None:
        regression_model = load_joblib(MODELS_DIR / "regression.pkl")
    if classifier_model is None:
        classifier_model = load_joblib(MODELS_DIR / "classifier.pkl")
    return regression_model, classifier_model


def build_student_frame(
    age: int = 17,
    gender: str = "female",
    attendance: float = 85.0,
    study_hours: float = 5.0,
    sleep_hours: float = 7.5,
    assignments_completed: float = 80.0,
    previous_grade: float = 70.0,
    parent_education: str = "secondary",
    internet_access: str = "yes",
    family_income: str = "medium",
    extra_classes: str = "no",
    participation: float = 70.0,
) -> pd.DataFrame:
    """Create a single-row DataFrame from user input."""
    return pd.DataFrame([{
        "age": age,
        "gender": gender,
        "attendance": attendance,
        "study_hours": study_hours,
        "sleep_hours": sleep_hours,
        "assignments_completed": assignments_completed,
        "previous_grade": previous_grade,
        "parent_education": parent_education,
        "internet_access": internet_access,
        "family_income": family_income,
        "extra_classes": extra_classes,
        "participation": participation,
    }])


def prepare_inference_frame(student_data: pd.DataFrame) -> pd.DataFrame:
    """Apply the same cleaning and feature engineering used in training."""
    cleaned = clean_student_data(student_data)
    return add_engineered_features(cleaned)


def estimate_feature_contributions(model: Pipeline, data: pd.DataFrame) -> pd.DataFrame:
    """Estimate top local feature contributions from transformed input values."""
    try:
        preprocessor = model.named_steps["preprocessor"]
        estimator = model.named_steps["model"]
        transformed = preprocessor.transform(data)
        if hasattr(preprocessor, "get_feature_names_out"):
            feature_names = preprocessor.get_feature_names_out()
        else:
            feature_names = [f"f_{i}" for i in range(transformed.shape[1])]

        if hasattr(estimator, "feature_importances_"):
            weights = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            weights = np.ravel(estimator.coef_)
        else:
            return pd.DataFrame()

        # Handle sparse or dense transformed array
        x_dense = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
        signed_values = x_dense[0] * weights

        contributions = pd.DataFrame({
            "feature": feature_names,
            "contribution": signed_values,
        })
        contributions["abs_contribution"] = contributions["contribution"].abs()
        top_contributions = (
            contributions.sort_values("abs_contribution", ascending=False)
            .head(10)
            .drop(columns=["abs_contribution"])
        )
        return top_contributions
    except Exception:
        return pd.DataFrame()


def predict_student(
    student_data: pd.DataFrame,
    regression_model=None,
    classifier_model=None,
) -> PredictionResult:
    """Predict final score, pass/fail label, probability, and contributions."""
    reg_model, clf_model = load_models(regression_model, classifier_model)
    prepared = prepare_inference_frame(student_data)

    pred_score = float(np.clip(reg_model.predict(prepared)[0], 0, 100))

    if hasattr(clf_model, "predict_proba"):
        pass_prob = float(clf_model.predict_proba(prepared)[0][1])
        class_pred = int(clf_model.predict(prepared)[0])
    else:
        class_pred = int(clf_model.predict(prepared)[0])
        pass_prob = 1.0 if class_pred == 1 else 0.0

    pass_fail = "Pass" if class_pred == 1 or pass_prob >= 0.5 else "Fail"
    confidence_score = float(round(abs(pass_prob - 0.5) * 2, 4))
    contributions = estimate_feature_contributions(reg_model, prepared)

    return PredictionResult(
        predicted_score=pred_score,
        pass_fail=pass_fail,
        pass_probability=pass_prob,
        confidence_score=confidence_score,
        contributions=contributions,
    )
