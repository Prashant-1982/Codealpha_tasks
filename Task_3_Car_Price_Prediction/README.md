# Task 3 – Car Price Prediction with Machine Learning

A regression project that estimates the selling price of a used vehicle from its available specifications and condition-related information. The workflow uses the dataset supplied for the internship task.

## Project objectives
- Explore and prepare the supplied car-sales dataset.
- Engineer useful features, including vehicle age and manufacturer/brand.
- Compare regression algorithms using the same train/test split.
- Evaluate predictions with MAE, RMSE and R².
- Explain the model's limitations and how price estimates can be used in practice.

## Dataset
The supplied file is stored at `data/car data.csv.gz` (a gzip-compressed CSV). It contains 301 observations and these original columns:

| Column | Meaning |
|---|---|
| Car_Name | Vehicle/model name |
| Year | Manufacturing year |
| Selling_Price | Observed selling price; prediction target |
| Present_Price | Listed/ex-showroom price in the source dataset |
| Driven_kms | Distance driven |
| Fuel_Type | Fuel category |
| Selling_type | Dealer or individual sale |
| Transmission | Manual or automatic |
| Owner | Previous-owner count/category |

The dataset has no missing values in its supplied form. Prices are in the units used by the source data (commonly lakh INR); the project does not convert currencies or units.

## Approach
1. Inspect data types, duplicates, missing values and target distribution.
2. Standardise text fields and derive `Brand` from the first token in `Car_Name`.
3. Derive `Car_Age` using the analysis reference year (2026).
4. Split data into 80% training and 20% testing rows with `random_state=42`.
5. Preprocess numeric and categorical features inside a Scikit-learn pipeline.
6. Compare Random Forest, Extra Trees and Gradient Boosting regressors.
7. Evaluate the selected model on the held-out test set and inspect residuals and feature importance.

The target `Selling_Price` is never included among the input features. The preprocessing pipeline is fitted on training data only to prevent test-set information from leaking into training.

## Held-out test result
With the included configuration, Random Forest is the best of the three models on the fixed test split:

- MAE: **0.608**
- RMSE: **0.922**
- R²: **0.963**

These figures describe one 20% hold-out split of a small dataset; they are not a guarantee of future market accuracy. The source dataset is limited in size and coverage, and `Present_Price` is a particularly influential input. For a live valuation product, validate on newer, geographically representative listings and avoid relying on fields that would not be known at prediction time.

## Run the project
Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook
```

Open `Car_Price_Prediction.ipynb` and run all cells. Or run the complete script:

```bash
python car_price_prediction.py
```

The script writes evaluation charts and a trained pipeline to the `outputs/` directory.

## Repository contents
- `Car_Price_Prediction.ipynb` – step-by-step analysis and model evaluation.
- `car_price_prediction.py` – reproducible command-line workflow.
- `Project_Report.md` – methodology, results, interpretation and limitations.
- `data/car data.csv.gz` – dataset supplied for the task, compressed without changing its records.
- `requirements.txt` – Python dependencies.

## Important interpretation
This is an educational model trained on historical sample data, not a professional vehicle appraisal. Real transaction prices depend on location, service history, accident/damage status, trim, condition, market timing and other factors not represented in this dataset.
