# Diabetes Prediction Model

## Project Overview
This project focuses on predicting the likelihood of diabetes in individuals using machine learning techniques. The dataset used for this task is the **Pima Indians Diabetes Database**, which includes several medical predictor variables and one target variable, indicating whether or not the individual has diabetes.

## Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Technologies Used](#technologies-used)
3. [Modeling Pipeline](#modeling-pipeline)
4. [Results](#results)
5. [How to Run](#how-to-run)
6. [Future Improvements](#future-improvements)

## Dataset Overview
- **Source**: [Pima Indians Diabetes Database on Kaggle](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset)
- **Features**:
  - Pregnancies: Number of times pregnant
  - Glucose: Plasma glucose concentration
  - BloodPressure: Diastolic blood pressure (mm Hg)
  - SkinThickness: Triceps skinfold thickness (mm)
  - Insulin: 2-Hour serum insulin (mu U/ml)
  - BMI: Body mass index (weight in kg/(height in m)^2)
  - DiabetesPedigreeFunction: Diabetes pedigree function (a measure of diabetes likelihood based on family history)
  - Age: Age in years
- **Target**: Diabetes status (0 = Non-diabetic, 1 = Diabetic)

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - pandas, numpy: Data manipulation and analysis
  - matplotlib, seaborn: Data visualization
  - scikit-learn: Model building and evaluation
  - SVM: Advanced machine learning algorithm
  - XGBoost: Advanced machine learning algorithm

## Modeling Pipeline
1. **Data Preprocessing**:
   - Handled missing values (no one in this dataset).
   - Scaled features using StandardScaler.
2. **Exploratory Data Analysis (EDA)**:
   - Visualized feature distributions.
   - Checked correlations between features and target.
3. **Feature Engineering**:
   - Identified and selected the most relevant features.
   - Mutual Information
4. **Modeling**:
   - Used SVM and XGBoost for classification.
   - Applied GridSearchCV for hyperparameter tuning.
   - Evaluated models using metrics like Accuracy, Precision, Recall and F1-Score.

## Results (In progress)
- **Best Model**:
- **Key Metrics**:
  - Accuracy: `XX.XX%`
  - Precision: `YY.YY%`
  - Recall: `ZZ.ZZ%`
  - ROC-AUC: `WW.WW`
- The model demonstrated good performance, particularly in identifying diabetic cases.

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/Colo07/DiabetesPrediction.git
   ```
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:
   ```bash
   jupyter notebook DiabetesPrediction.ipynb
   ```

## Future Improvements
- **Data Augmentation**: Collect more diverse data to improve model generalization.
- **Advanced Techniques**: Experiment with neural networks for better performance.
