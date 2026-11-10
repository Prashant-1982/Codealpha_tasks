"""
Task 3: Car Price Prediction with Machine Learning

Run:
    pip install -r requirements.txt
    python car_price_prediction.py

The script reads the supplied compressed CSV, compares three regression
models, evaluates the selected model on a held-out test set, saves charts,
and exports the fitted preprocessing/model pipeline.
"""

from pathlib import Path
import re
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "car data.csv.gz"
OUTPUT_DIR = BASE_DIR / "outputs"
REFERENCE_YEAR = 2026
RANDOM_STATE = 42

NUMERIC_FEATURES = ["Car_Age", "Present_Price", "Driven_kms", "Owner"]
CATEGORICAL_FEATURES = ["Brand", "Fuel_Type", "Selling_type", "Transmission"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Selling_Price"


def clean_brand(car_name: str) -> str:
    """Extract a consistent first-token brand label from the vehicle name."""
    cleaned = re.sub(r"[^a-zA-Z0-9 ]", " ", str(car_name).lower())
    tokens = cleaned.split()
    return tokens[0] if tokens else "unknown"


def load_and_prepare_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the supplied data and create the engineered model features."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Keep 'car data.csv.gz' inside the data folder."
        )

    frame = pd.read_csv(path, compression="gzip")
    required = {
        "Car_Name", "Year", "Selling_Price", "Present_Price", "Driven_kms",
        "Fuel_Type", "Selling_type", "Transmission", "Owner"
    }
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    frame = frame.copy()
    for column in ["Car_Name", "Fuel_Type", "Selling_type", "Transmission"]:
        frame[column] = frame[column].astype("string").str.strip()

    # Brand is a compact representation of the manufacturer/model-name prefix.
    frame["Brand"] = frame["Car_Name"].map(clean_brand)
    frame["Car_Age"] = REFERENCE_YEAR - pd.to_numeric(frame["Year"], errors="coerce")

    # Exclude the raw name and manufacturing year after deriving the model features.
    # Selling_Price is the target and is never part of FEATURES.
    return frame


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def build_models():
    """Return a small, varied set of tree-based regression algorithms."""
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=1, max_features=1.0,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=400, min_samples_leaf=1, max_features=1.0,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=2,
            loss="huber", random_state=RANDOM_STATE
        ),
    }


def evaluate(y_true, predictions):
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_true, predictions)),
        "R2": r2_score(y_true, predictions),
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    data = load_and_prepare_data()
    print("=" * 68)
    print("CAR PRICE PREDICTION – DATA OVERVIEW")
    print("=" * 68)
    print(f"Rows: {data.shape[0]} | Columns: {data.shape[1]}")
    print("\nMissing values:\n", data.isna().sum())
    print(f"Duplicate rows: {data.duplicated().sum()}")
    print("\nTarget summary:\n", data[TARGET].describe().round(2))

    # Exploratory visualisations.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(data[TARGET], kde=True, ax=axes[0])
    axes[0].set_title("Distribution of selling prices")
    axes[0].set_xlabel("Selling price (source units)")
    sns.scatterplot(data=data, x="Present_Price", y=TARGET, hue="Fuel_Type", ax=axes[1])
    axes[1].set_title("Selling price vs present price")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "exploratory_analysis.png", dpi=160)
    plt.close(fig)

    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    print(f"\nTraining records: {len(X_train)} | Test records: {len(X_test)}")

    results = []
    fitted_models = {}
    for name, estimator in build_models().items():
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", estimator),
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        scores = evaluate(y_test, predictions)
        results.append({"Model": name, **scores})
        fitted_models[name] = (pipeline, predictions)
        print(f"\n{name}:")
        print(f"  MAE  : {scores['MAE']:.3f}")
        print(f"  RMSE : {scores['RMSE']:.3f}")
        print(f"  R²   : {scores['R2']:.3f}")

    comparison = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print("\nModel comparison:\n", comparison.round(3).to_string(index=False))

    best_name = comparison.loc[0, "Model"]
    best_pipeline, best_predictions = fitted_models[best_name]
    best_scores = evaluate(y_test, best_predictions)
    print(f"\nSelected model: {best_name}")

    # Actual-vs-predicted plot.
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(x=y_test, y=best_predictions, ax=ax)
    lower = min(float(y_test.min()), float(best_predictions.min()))
    upper = max(float(y_test.max()), float(best_predictions.max()))
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="red", label="Perfect prediction")
    ax.set_xlabel("Actual selling price")
    ax.set_ylabel("Predicted selling price")
    ax.set_title(f"Actual vs predicted – {best_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    # Residual analysis.
    residuals = y_test.to_numpy() - best_predictions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(residuals, kde=True, ax=axes[0])
    axes[0].set_title("Residual distribution")
    axes[0].set_xlabel("Actual - predicted")
    sns.scatterplot(x=best_predictions, y=residuals, ax=axes[1])
    axes[1].axhline(0, linestyle="--", color="red")
    axes[1].set_title("Residuals vs predicted price")
    axes[1].set_xlabel("Predicted selling price")
    axes[1].set_ylabel("Residual")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "residual_analysis.png", dpi=160)
    plt.close(fig)

    # Permutation importance is calculated against the untouched hold-out set.
    from sklearn.inspection import permutation_importance
    importance = permutation_importance(
        best_pipeline, X_test, y_test, n_repeats=20,
        random_state=RANDOM_STATE, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    importance_frame = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": importance.importances_mean,
        "Std": importance.importances_std,
    }).sort_values("Importance", ascending=False)
    importance_frame.to_csv(OUTPUT_DIR / "permutation_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=importance_frame, x="Importance", y="Feature", ax=ax)
    ax.set_title("Permutation importance on the test set")
    ax.set_xlabel("Decrease in negative MAE when shuffled")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=160)
    plt.close(fig)

    # Save the complete pipeline so future predictions use identical preprocessing.
    joblib.dump(
        {"pipeline": best_pipeline, "reference_year": REFERENCE_YEAR,
         "features": FEATURES, "model_name": best_name},
        OUTPUT_DIR / "car_price_model.joblib"
    )

    predictions_frame = X_test.copy()
    predictions_frame["Actual_Selling_Price"] = y_test.to_numpy()
    predictions_frame["Predicted_Selling_Price"] = best_predictions
    predictions_frame["Absolute_Error"] = np.abs(residuals)
    predictions_frame.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    print("\nFinal hold-out metrics:")
    for metric, value in best_scores.items():
        print(f"  {metric}: {value:.4f}")
    print(f"\nSaved charts, metrics, predictions and model in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
