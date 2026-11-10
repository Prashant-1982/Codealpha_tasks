# Iris Flower Classification

This project predicts the species of an Iris flower from four sepal and petal measurements. It uses the Iris.csv file supplied for the internship task.

## Objective
Train a supervised machine-learning classifier to recognise setosa, versicolor and virginica.

## Dataset
The supplied dataset contains 150 records. The four measurement columns are used as input features. The `Id` column is only a row identifier and is not used for training.

## Workflow
1. Load and inspect the dataset.
2. Check missing values and explore the measurements with plots.
3. Split the data into 80% training and 20% testing records using stratified sampling.
4. Standardise the measurements with `StandardScaler`.
5. Train a K-Nearest Neighbours model with `k=5`.
6. Evaluate predictions using accuracy, precision, recall, F1-score and a confusion matrix.

Scaling and classification are combined in a Scikit-learn pipeline to ensure the scaler is fitted on training data only.

## Result
With `random_state=42` and a stratified 80/20 split, the model achieves 93.33% accuracy on the 30-record test set (28 correct predictions). This is the result of one split, not a guarantee for every future sample.

## Run locally
Use Python 3.9 or later:

```bash
pip install -r requirements.txt
jupyter notebook
```

Open `Iris_Flower_Classification.ipynb` and run all cells. The same workflow is available as a script:

```bash
python iris_classification.py
```

## Files
- `Iris_Flower_Classification.ipynb` — notebook with exploration, training and evaluation.
- `iris_classification.py` — script version.
- `data/Iris.csv` — dataset supplied for the task.
- `Project_Report.md` — project report.
- `requirements.txt` — required libraries.
