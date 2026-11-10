# Project Report: Car Price Prediction with Machine Learning

## 1. Introduction
The resale value of a used vehicle depends on several factors, including its age, original/listed price, distance travelled, fuel type, transmission and ownership history. Estimating a reasonable selling price from these details is a regression problem.

This project develops a machine-learning workflow that learns the relationship between vehicle characteristics and the selling prices recorded in the supplied dataset. The work covers data understanding, preprocessing, feature engineering, model training, evaluation and interpretation.

## 2. Objectives
- Study the supplied car-sales dataset and identify the prediction target.
- Prepare numerical and categorical features for machine learning.
- Create useful features from the original columns.
- Train and compare different regression algorithms.
- Measure prediction errors using standard regression metrics.
- Discuss how a price-estimation model could be used and where it may fail.

## 3. Dataset Description
The supplied `car data.csv` contains **301 rows and 9 original columns**.

| Feature | Description | Type |
|---|---|---|
| Car_Name | Vehicle/model name | Text |
| Year | Manufacturing year | Integer |
| Selling_Price | Actual selling price; target | Numeric |
| Present_Price | Present/listed price in source data | Numeric |
| Driven_kms | Distance driven | Numeric |
| Fuel_Type | Petrol, Diesel or CNG | Categorical |
| Selling_type | Dealer or Individual | Categorical |
| Transmission | Manual or Automatic | Categorical |
| Owner | Previous-owner count/category | Numeric |

The supplied data contains no missing values. The target price is retained in the source units (commonly lakh INR for this dataset); no unit conversion is performed.

## 4. Data Preprocessing
The following checks and transformations are applied:
1. Load the CSV with Pandas and inspect its shape, data types and summary statistics.
2. Check missing values and duplicate rows.
3. Remove leading/trailing spaces from text columns and standardise text handling.
4. Convert the vehicle name into a compact `Brand` feature using its first name token.
5. Create `Car_Age` as 2026 minus the manufacturing year.
6. Keep the target `Selling_Price` separate from the predictors.

The original `Car_Name` and `Year` are not directly passed to the model after feature engineering. The brand extraction is intentionally simple and may not correctly identify every multi-word manufacturer. A production dataset should use a verified manufacturer field.

## 5. Feature Engineering
Two derived features are used:
- **Brand:** captures broad manufacturer-level differences from the vehicle name.
- **Car_Age:** represents how old the vehicle is relative to the reference year 2026.

The final model inputs are `Car_Age`, `Present_Price`, `Driven_kms`, `Owner`, `Brand`, `Fuel_Type`, `Selling_type` and `Transmission`.

## 6. Model Building
The dataset is divided into 80% training data and 20% testing data with a fixed random state of 42. The same split is used for each algorithm to make the comparison fair.

A Scikit-learn preprocessing pipeline is used:
- Numerical features: median imputation followed by standardisation.
- Categorical features: most-frequent imputation followed by one-hot encoding.
- Unseen categories are safely ignored during transformation.

The preprocessing steps are fitted only on the training data. This prevents information from the test set from influencing the model.

Three regression algorithms are compared:
1. **Random Forest Regressor:** combines predictions from many decision trees and can model non-linear relationships.
2. **Extra Trees Regressor:** uses additional randomness while building trees, which can reduce variance in some settings.
3. **Gradient Boosting Regressor:** builds trees sequentially, with later trees correcting errors made by earlier ones.

## 7. Evaluation Metrics
- **Mean Absolute Error (MAE):** average absolute difference between actual and predicted prices. Lower is better.
- **Root Mean Squared Error (RMSE):** penalises larger errors more strongly. Lower is better.
- **R-squared (R²):** indicates how much variation in the target is explained by the model on the evaluated data. A value closer to 1 is generally better.

## 8. Results
With the supplied dataset, an 80/20 split and random state 42, the model comparison is:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 0.608 | 0.922 | 0.963 |
| Extra Trees | 0.685 | 1.533 | 0.898 |
| Gradient Boosting | 0.742 | 1.238 | 0.933 |

**Random Forest** gives the strongest result on this particular held-out test set, with an MAE of approximately 0.608, RMSE of 0.922 and R² of 0.963. The error values are expressed in the same price units as the source target.

These results should be interpreted carefully. The dataset is small, the test set contains only about 60 records, and one random split can produce an optimistic or pessimistic result. A stronger study would use repeated cross-validation and a newer independent test dataset.

## 9. Visual Analysis
The notebook and script generate the following plots:
- Selling-price distribution.
- Present price against selling price, grouped by fuel type.
- Actual versus predicted selling prices.
- Residual distribution and residuals against predictions.
- Permutation feature importance.

Points close to the diagonal in the actual-versus-predicted plot indicate more accurate predictions. Residual plots help reveal systematic over- or under-prediction. Permutation importance estimates how much test-set performance worsens when one input feature is shuffled.

## 10. Real-World Applications
A model of this type can support:
- Used-car marketplace price suggestions.
- Initial valuation estimates for dealerships.
- Comparing a seller's asking price with similar historical examples.
- Supporting buyers during preliminary price research.
- Internal inventory analysis for vehicle resellers.

The prediction should be treated as an estimate, not a final offer or certified valuation.

## 11. Limitations and Future Improvements
1. The dataset contains only 301 observations and may not represent the current used-car market.
2. Location, accident history, service records, vehicle condition, trim/variant and ownership documentation are not included.
3. The brand feature is extracted from the first token of the vehicle name, so some names may be grouped imperfectly.
4. The model uses `Present_Price`; this field must be available and consistently defined at the time a real prediction is made. If it is unavailable or represents information too close to the target, the model should be retrained without it.
5. The 2026 reference year is fixed for this exercise. A maintained application should calculate vehicle age from the current date and retrain or validate as the market changes.
6. Future work could add verified brand/model/variant fields, location, condition and service history, and compare the results using repeated cross-validation and time-based validation.

## 12. Conclusion
This project demonstrates an end-to-end supervised machine-learning regression workflow. After preprocessing the supplied data and creating brand and vehicle-age features, three regression models were trained and evaluated. Random Forest performed best on the selected test split, achieving an R² of approximately 0.963.

The task shows how Python, Pandas, Matplotlib, Seaborn and Scikit-learn can be combined to explore data, prepare features, train models and communicate evaluation results. The model is suitable as an educational demonstration; broader and more recent data would be required for dependable real-world price estimates.
