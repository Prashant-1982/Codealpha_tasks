# Task 4 — Sales Prediction using Python

This project analyses the relationship between advertising spend and sales, then trains regression models to estimate sales from TV, Radio and Newspaper advertising.

## Supplied dataset
The supplied Advertising.csv contains 200 observations. Its useful columns are TV, Radio, Newspaper and Sales. The original Unnamed: 0 field is only a row index and is removed. The file has no missing values.

The task description mentions target segment and platform, but those columns are not present in the supplied file. This project uses only the fields actually provided. No segment or platform values are fabricated. The dataset does not specify units or currency, so the original numeric scale is retained.

## Workflow
1. Load, inspect and clean the CSV.
2. Explore distributions, channel-vs-sales relationships and correlations.
3. Engineer total advertising spend and channel-share features for analysis.
4. Split data into training and held-out test sets.
5. Compare Linear Regression, Ridge, Random Forest, Gradient Boosting and Extra Trees using five-fold cross-validation on the training set.
6. Evaluate the selected model with MAE, RMSE and R-squared.
7. Inspect actual-vs-predicted values, residuals and permutation feature importance.
8. Discuss practical marketing insights and dataset limitations.

## Reproduced result
With an 80/20 split and random_state=42, Extra Trees is the top model in the training-only cross-validation comparison. On the held-out test set it achieves approximately:
- MAE: 0.438
- RMSE: 0.553
- R-squared: 0.990

These results are specific to this small dataset and split. They are not a guarantee of future campaign performance. Association is not proof of causation.

## Run locally
Python 3.9 or later is recommended.

~~~bash
pip install -r requirements.txt
jupyter notebook
~~~

Open Sales_Prediction.ipynb and run all cells. Alternatively:

~~~bash
python sales_prediction.py
~~~

## Files
- Sales_Prediction.ipynb — detailed notebook with EDA, feature engineering, model comparison and evaluation.
- sales_prediction.py — reproducible script version.
- Project_Report.md — internship report and business interpretation.
- data/Advertising.csv — cleaned copy of the supplied dataset.
- requirements.txt — required packages.
