"""
Machine Learning Module
========================
1. Price Prediction – regression model to predict auction cost in Crore.
2. Sold/Unsold Classifier – predict whether a player will be sold.

Models are trained on the cleaned dataset features derived from the actual
uploaded CSV columns. No external data is assumed.

Available features from cleaned dataset:
  - player_role        (categorical)
  - base_price_cr      (numeric)
  - player_origin      (categorical: Indian / Overseas)
  - current_team       (categorical – for classifier only, target for team model)
  - prev_team_2022     (categorical)
  - price_multiplier   (for post-hoc analysis only, not training)

Target:
  - cost_cr            → Regression
  - is_sold (binary)   → Classification
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, classification_report, confusion_matrix,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODELS_DIR = "models"
ASSETS_DIR = "assets"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0F1117",
    "axes.facecolor":   "#1A1D2E",
    "axes.edgecolor":   "#2E3250",
    "axes.labelcolor":  "#E0E0E0",
    "xtick.color":      "#A0A0A0",
    "ytick.color":      "#A0A0A0",
    "text.color":       "#E0E0E0",
    "grid.color":       "#2E3250",
})

NUMERIC_FEATURES   = ["base_price_cr"]
CATEGORICAL_FEATURES = ["player_role", "player_origin"]


# ─────────────────────────────────────────────────────────────────────────────
# Shared preprocessor
# ─────────────────────────────────────────────────────────────────────────────
def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 1. PRICE REGRESSION
# ─────────────────────────────────────────────────────────────────────────────
def train_price_model(df: pd.DataFrame) -> dict:
    """
    Train a Random Forest Regressor to predict auction price in Crore.
    Only auction-sold players (not retained) with cost_cr > 0 are used.
    """
    logger.info("Training price regression model …")
    data = df[df["is_sold"] & ~df["is_retained"] & (df["cost_cr"] > 0)].copy()

    if len(data) < 30:
        logger.warning("Insufficient data for regression.")
        return {}

    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data["cost_cr"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("preprocessor", _build_preprocessor()),
        ("model", RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    cv_r2 = cross_val_score(pipeline, X, y, cv=5, scoring="r2").mean()

    logger.info(f"Price Model — MAE: {mae:.3f}, R²: {r2:.3f}, CV-R²: {cv_r2:.3f}")

    # Save model
    model_path = os.path.join(MODELS_DIR, "price_regressor.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    # Feature importance chart
    ohe_cols = (pipeline.named_steps["preprocessor"]
                .named_transformers_["cat"]
                .get_feature_names_out(CATEGORICAL_FEATURES))
    all_feat = NUMERIC_FEATURES + list(ohe_cols)
    importances = pipeline.named_steps["model"].feature_importances_
    feat_df = pd.DataFrame({"feature": all_feat, "importance": importances}).sort_values("importance", ascending=True).tail(12)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(feat_df["feature"], feat_df["importance"], color="#2EC4B6", height=0.6)
    ax.set_title("Feature Importance – Price Prediction Model", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", alpha=0.4)
    plt.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "ml_feature_importance.png"), dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()

    # Actual vs Predicted scatter
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.7, color="#E9C46A", edgecolors="#0F1117", s=60)
    ax.plot([0, y_test.max()], [0, y_test.max()], "--", color="#E63946", linewidth=1.5, label="Perfect Fit")
    ax.set_xlabel("Actual Price (₹ Cr)", fontsize=12)
    ax.set_ylabel("Predicted Price (₹ Cr)", fontsize=12)
    ax.set_title(f"Price Prediction: Actual vs Predicted\n(R² = {r2:.2f}, MAE = ₹{mae:.2f} Cr)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "ml_actual_vs_predicted.png"), dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()

    return {
        "mae": round(mae, 3),
        "r2": round(r2, 3),
        "cv_r2": round(cv_r2, 3),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "model_path": model_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. SOLD / UNSOLD CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
def train_sold_classifier(df: pd.DataFrame) -> dict:
    """
    Train a Gradient Boosting Classifier to predict if a player gets sold.
    Uses all non-retained players (both sold and unsold at auction).
    """
    logger.info("Training sold/unsold classifier …")
    data = df[~df["is_retained"]].copy()

    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data["is_sold"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline([
        ("preprocessor", _build_preprocessor()),
        ("model", GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_acc = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy").mean()

    logger.info(f"Classifier — Accuracy: {acc:.3f}, CV-Accuracy: {cv_acc:.3f}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Unsold','Sold'])}")

    # Save model
    model_path = os.path.join(MODELS_DIR, "sold_classifier.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="YlOrRd", aspect="auto")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Unsold", "Sold"]); ax.set_yticklabels(["Unsold", "Sold"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=18, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted", fontsize=12); ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix – Sold Classifier\nAccuracy: {acc:.1%}", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "ml_confusion_matrix.png"), dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()

    return {
        "accuracy": round(acc, 3),
        "cv_accuracy": round(cv_acc, 3),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "model_path": model_path,
        "report": classification_report(y_test, y_pred, target_names=["Unsold", "Sold"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION FUNCTIONS (for Streamlit UI)
# ─────────────────────────────────────────────────────────────────────────────
def predict_price(player_role: str, base_price_cr: float, player_origin: str) -> float:
    """Load trained model and predict auction price in Crore."""
    model_path = os.path.join(MODELS_DIR, "price_regressor.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Price model not trained yet. Run train_price_model() first.")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    X = pd.DataFrame([{
        "base_price_cr": base_price_cr,
        "player_role":   player_role,
        "player_origin": player_origin,
    }])
    return round(float(model.predict(X)[0]), 2)


def predict_sold(player_role: str, base_price_cr: float, player_origin: str) -> dict:
    """Load trained classifier and predict sold probability."""
    model_path = os.path.join(MODELS_DIR, "sold_classifier.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Classifier not trained yet. Run train_sold_classifier() first.")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    X = pd.DataFrame([{
        "base_price_cr": base_price_cr,
        "player_role":   player_role,
        "player_origin": player_origin,
    }])
    proba = model.predict_proba(X)[0]
    return {
        "prediction": "Sold" if proba[1] >= 0.5 else "Unsold",
        "sold_probability": round(float(proba[1]) * 100, 1),
        "unsold_probability": round(float(proba[0]) * 100, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MASTER RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_ml_pipeline(df: pd.DataFrame) -> dict:
    price_metrics = train_price_model(df)
    sold_metrics  = train_sold_classifier(df)
    return {"price_model": price_metrics, "sold_classifier": sold_metrics}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.cleaning.pipeline import run_pipeline
    df = run_pipeline(save=False)
    results = run_ml_pipeline(df)
    print("\n=== ML Results ===")
    for model, metrics in results.items():
        print(f"\n{model}:")
        for k, v in metrics.items():
            if k != "report":
                print(f"  {k}: {v}")
