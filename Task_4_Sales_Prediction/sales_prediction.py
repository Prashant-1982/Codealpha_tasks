"""Task 4: Sales Prediction using Python. Run with: python sales_prediction.py"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["TV", "Radio", "Newspaper"]
TARGET = "Sales"
RANDOM_STATE = 42


def load_data():
    path = Path(__file__).parent / "data" / "Advertising.csv"
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    if "Unnamed: 0" in frame.columns:
        frame = frame.drop(columns=["Unnamed: 0"])
    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    frame = frame[required].copy()
    for col in required:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if frame.isna().any().any():
        raise ValueError("Missing or non-numeric values found in the dataset.")
    return frame


def engineer_features(frame):
    result = frame.copy()
    result["Total_Advertising_Spend"] = result[FEATURES].sum(axis=1)
    total = result["Total_Advertising_Spend"].replace(0, float("nan"))
    for col in FEATURES:
        result[f"{col}_Share"] = result[col] / total
    return result


def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0))
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=400, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=400, random_state=RANDOM_STATE
        )
    }


def main():
    sns.set_theme(style="whitegrid")
    data = load_data()
    print("Dataset shape:", data.shape)
    print("\nFirst five rows:\n", data.head())
    print("\nMissing values:\n", data.isna().sum())
    print("\nDuplicate rows:", data.duplicated().sum())
    print("\nSummary statistics:\n", data.describe())

    engineered = engineer_features(data)
    print("\nEngineered features:\n", engineered.head())

    X, y = data[FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    # Cross-validation is run on training data only; the test set stays untouched.
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fitted = {}
    rows = []
    for name, model in get_models().items():
        scores = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring={"r2": "r2", "mae": "neg_mean_absolute_error"}
        )
        rows.append({
            "Model": name,
            "CV R2": scores["test_r2"].mean(),
            "CV MAE": -scores["test_mae"].mean()
        })
        model.fit(X_train, y_train)
        fitted[name] = model

    comparison = pd.DataFrame(rows).sort_values("CV R2", ascending=False)
    print("\nModel comparison:\n", comparison.to_string(index=False))

    best_name = comparison.iloc[0]["Model"]
    best_model = fitted[best_name]
    predictions = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)
    print(f"\nSelected model: {best_name}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2: {r2:.4f}")

    # Channel relationships.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, channel in zip(axes, FEATURES):
        sns.regplot(data=data, x=channel, y=TARGET,
                    scatter_kws={"alpha": 0.55}, ax=ax)
        ax.set_title(f"{channel} spend vs Sales")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.show()

    lower = min(y_test.min(), predictions.min())
    upper = max(y_test.max(), predictions.max())
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, predictions, alpha=0.75)
    plt.plot([lower, upper], [lower, upper], "--", color="black")
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title(f"Actual vs Predicted — {best_name}")
    plt.tight_layout()
    plt.show()

    residuals = y_test - predictions
    plt.figure(figsize=(6, 4))
    plt.scatter(predictions, residuals, alpha=0.75)
    plt.axhline(0, linestyle="--", color="black")
    plt.xlabel("Predicted Sales")
    plt.ylabel("Residual")
    plt.title("Residual plot")
    plt.tight_layout()
    plt.show()

    perm = permutation_importance(
        best_model, X_test, y_test, n_repeats=50, random_state=RANDOM_STATE
    )
    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": perm.importances_mean
    }).sort_values("Importance", ascending=False)
    print("\nPermutation importance:\n", importance.to_string(index=False))

    plt.figure(figsize=(6, 4))
    plt.barh(importance["Feature"], importance["Importance"])
    plt.xlabel("Mean decrease in model score")
    plt.title("Permutation feature importance")
    plt.tight_layout()
    plt.show()

    # Example prediction using the same feature order as the training data.
    example = pd.DataFrame([[150.0, 25.0, 20.0]], columns=FEATURES)
    print("\nExample predicted Sales:", best_model.predict(example)[0])


if __name__ == "__main__":
    main()
