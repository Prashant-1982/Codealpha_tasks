# Internship Project Report: Iris Flower Classification

## 1. Introduction
This project applies supervised machine learning to identify the species of an Iris flower from measurements of its sepals and petals. It is a three-class classification problem and demonstrates the main stages of a basic machine-learning workflow.

## 2. Objectives
- Use flower measurements as input features.
- Train a Scikit-learn classification model.
- Predict setosa, versicolor or virginica.
- Evaluate the model using a separate test set.

## 3. Dataset
The supplied Iris.csv file contains 150 records, with 50 samples for each species. The four numerical features are sepal length, sepal width, petal length and petal width, measured in centimetres. The Id column is an identifier and is excluded from model training.

## 4. Tools and Libraries
Python, Pandas, NumPy, Matplotlib, Seaborn and Scikit-learn are used. Pandas handles the data, Matplotlib and Seaborn create visualisations, and Scikit-learn provides the train-test split, preprocessing, classifier and evaluation metrics.

## 5. Methodology
1. Load the CSV file and inspect its rows, columns, data types and missing values.
2. Remove the identifier from the input features and clean the species labels.
3. Split the data into 80% training and 20% testing records. Stratification keeps the class proportions balanced.
4. Standardise the four measurements with StandardScaler.
5. Train a K-Nearest Neighbours classifier with k = 5.
6. Predict the test labels and evaluate the results.

The scaler and classifier are placed in a Scikit-learn pipeline. This means the scaling parameters are learned from the training data only, avoiding information leakage from the test set.

## 6. Evaluation
Using test_size=0.20, random_state=42 and stratified sampling, the test set contains 30 flowers. The model correctly classified 28 records and achieved **93.33% accuracy**.

Confusion matrix (actual classes in rows, predicted classes in columns):

| Actual / Predicted | Setosa | Versicolor | Virginica |
|---|---:|---:|---:|
| Setosa | 10 | 0 | 0 |
| Versicolor | 0 | 10 | 0 |
| Virginica | 0 | 2 | 8 |

Setosa and versicolor were correctly classified in this split. Two virginica samples were predicted as versicolor, which suggests that these classes have some overlap in their measurements.

## 7. Conclusion
The project demonstrates a complete introductory classification workflow, from loading and exploring the data to training and evaluating a model. The KNN classifier achieved 93.33% accuracy on this test set. Since the dataset is small and the result comes from one split, further validation and comparison with other classifiers would be useful before relying on the model for new data.

## 8. How to Run
Install the packages in requirements.txt, start Jupyter Notebook, open Iris_Flower_Classification.ipynb and run all cells. The same workflow is also available in iris_classification.py.
