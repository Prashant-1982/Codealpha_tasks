# Task 4: Sales Prediction using Python
## Internship Project Report

### 1. Introduction
Sales prediction is an important business analytics task. Organisations use historical sales and marketing information to understand demand, plan campaigns and allocate advertising budgets. This project uses regression models to estimate sales from spending across TV, Radio and Newspaper advertising channels.

### 2. Problem Statement and Objectives
The objective is to predict the Sales value using the advertising information in the supplied dataset. The project covers data cleaning, exploratory analysis, feature engineering, regression model training, model comparison and evaluation.

The task description mentions target segment and platform. However, the provided CSV does not contain customer-segment or platform columns. It contains TV, Radio, Newspaper and Sales, plus a row index. The analysis therefore uses the available fields and does not invent missing information.

### 3. Dataset Description
The supplied Advertising.csv contains 200 observations:
- TV: advertising spend on TV.
- Radio: advertising spend on radio.
- Newspaper: advertising spend on newspaper.
- Sales: observed sales value and prediction target.
- Unnamed: 0: source row index, removed during cleaning.

There are no missing values in the supplied dataset. The file does not define units or currency, so the original numeric scale is preserved.

### 4. Data Cleaning and Preparation
The file is loaded using Pandas. Column names are stripped of extra whitespace, the source index column is removed, and the four relevant columns are checked for numeric values and missing data. Sales is separated from the input features to prevent target leakage.

The data is split into 80% training and 20% testing records using random_state=42. The test set is held aside for final evaluation. Five-fold shuffled cross-validation is used on the training set only for model comparison.

### 5. Exploratory Data Analysis
The notebook includes distribution plots, scatter plots for each advertising channel against Sales, and a correlation heatmap. In this dataset, TV has the strongest simple correlation with Sales (approximately 0.782), followed by Radio (approximately 0.576). Newspaper has a weaker correlation (approximately 0.228). Correlation measures association and does not establish causation.

### 6. Feature Engineering and Selection
For analysis, the notebook creates Total_Advertising_Spend and each channel's share of total spend. These derived fields describe the marketing mix. The core model comparison uses the three original advertising variables; adding totals and shares to the same model would duplicate information already contained in those variables.

Permutation importance is calculated on the held-out set to estimate how much predictive performance depends on each input feature. This is a measure of model reliance, not causal impact.

### 7. Regression Models
Five algorithms are compared:
1. Linear Regression — a simple linear baseline.
2. Ridge Regression — linear regression with L2 regularisation.
3. Random Forest Regressor — an ensemble of decision trees.
4. Gradient Boosting Regressor — builds trees sequentially to reduce errors.
5. Extra Trees Regressor — a tree ensemble with additional randomness.

The average cross-validation R-squared and MAE are used to compare models on training data. The selected model is then evaluated on the held-out test data.

### 8. Evaluation Metrics
- MAE (Mean Absolute Error) is the average absolute difference between actual and predicted sales. Lower is better.
- RMSE (Root Mean Squared Error) gives larger errors more weight. Lower is better.
- R-squared measures how much target variation is explained on the evaluated data. Values closer to 1 indicate a better fit.

With the fixed 80/20 split, the Extra Trees model gives approximately:
- MAE: 0.438
- RMSE: 0.553
- R-squared: 0.990

The notebook prints the exact values when run. Actual-vs-predicted and residual plots provide additional checks beyond a single metric.

### 9. Business Insights
- TV and Radio are more strongly associated with Sales than Newspaper in this particular dataset.
- Newspaper contributes comparatively little predictive information in the fitted model, but that does not mean it is ineffective in every campaign or market.
- The model can support preliminary scenario analysis for spending combinations similar to those represented in the data.
- Business decisions should also consider campaign objective, customer audience, seasonality, pricing, competition and other information not included here.

### 10. Limitations and Future Improvements
The dataset is small, with 200 observations and only three advertising predictors. It has no dates, customer segments, platforms, product categories, seasonality, pricing or competitor information. Since no time variable is supplied, a genuine time-series forecast cannot be built from this file. Future work could collect these fields, use a larger dataset, tune model parameters within cross-validation and validate against a later campaign period.

### 11. Conclusion
This project demonstrates a complete regression workflow using the supplied advertising dataset. It includes cleaning, exploration, feature engineering, cross-validated model comparison and final evaluation. Extra Trees gives the strongest result among the tested models on the selected split. The findings suggest TV and Radio are more informative predictors than Newspaper in this dataset, while highlighting the additional data needed for stronger real-world sales forecasting.
