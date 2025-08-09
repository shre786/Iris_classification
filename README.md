# Iris_classification
 Overview
This project uses the Iris dataset to classify flowers into three species — Iris setosa, Iris versicolor, and Iris virginica — based on their morphological features. The dataset contains measurements for sepal length, sepal width, petal length, and petal width.
Machine learning models such as Logistic Regression and Random Forest Classifier are trained and evaluated to achieve accurate classification.

🎯 Objective
The goal is to:
Perform exploratory data analysis (EDA) to understand the dataset.
Train classification models using scikit-learn.
Evaluate model performance using metrics like accuracy, confusion matrix, and classification report.

📂 Dataset
File: Iris.csv
Source: UCI Machine Learning Repository / Kaggle

Features:
SepalLengthCm
SepalWidthCm
PetalLengthCm
PetalWidthCm

Target:
Species (Setosa, Versicolor, Virginica)

🛠️ Libraries Used
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn

🚀 Steps Performed
Importing Libraries – Loaded necessary Python libraries for data manipulation, visualization, and modeling.
Loading Dataset – Read Iris.csv into a Pandas DataFrame
Data Exploration – Checked head, tail, info, and summary statistics.
Data Visualization – Used Seaborn and Matplotlib for:
Pair plots
Heatmaps
Distribution plots
Data Preprocessing – Encoded categorical target labels using LabelEncoder.
Model Training – Trained Logistic Regression and Random Forest models.
Hyperparameter Tuning – Applied GridSearchCV and RandomizedSearchCV.
Evaluation – Used confusion matrix, accuracy score, and classification report.

📊 Results
Achieved high accuracy in classifying Iris flowers using both models.

Visualization clearly shows distinct separation between species in feature space.

📌 How to Run
Clone the repository or download the notebook.

Install dependencies:
pip install numpy pandas matplotlib seaborn scikit-learn
Place Iris.csv in the same directory as the notebook.
Run the notebook cells sequentially.

