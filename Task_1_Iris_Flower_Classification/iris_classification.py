"""
Iris Flower Classification
Train a KNN model using the supplied Iris.csv dataset.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    data_path = Path(__file__).parent / "data" / "Iris.csv"
    df = pd.read_csv(data_path)

    print("Dataset shape:", df.shape)
    print("\nFirst five rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())

    # The ID column is not a flower measurement, so it is left out.
    X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
    y = df["Species"].str.replace("Iris-", "", regex=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Scaling is fitted on training data only because it is inside the pipeline.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\nTest accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, predictions, labels=["setosa", "versicolor", "virginica"]))

    sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=X.columns)
    print("\nExample prediction:", model.predict(sample)[0])


if __name__ == "__main__":
    main()
