# 🚢 Titanic Survival ML Pipeline

An end-to-end Machine Learning pipeline built with **Scikit-Learn** for predicting passenger survival on the Titanic. This project demonstrates modular preprocessing, feature selection, model training, cross-validation, hyperparameter tuning, and inference serialization using Scikit-Learn's `Pipeline` and `ColumnTransformer`.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Complete Pipeline Flow & Architecture](#-complete-pipeline-flow--architecture)
- [Algorithms & Techniques Used](#-algorithms--techniques-used)
- [Project Directory Structure](#-project-directory-structure)
- [How to Start & Run the Project](#-how-to-start--run-the-project)
- [Input Feature Reference](#-input-feature-reference)
- [Making Predictions with Serialized Pipeline](#-making-predictions-with-serialized-pipeline)
- [Model Evaluation & Tuning](#-model-evaluation--tuning)

---

## 📖 Project Overview

The goal of this project is to predict whether a passenger survived the Titanic disaster (`0 = Did not survive`, `1 = Survived`) based on passenger demographics, ticket class, and boarding details.

Key highlights:
- **Clean Architecture**: Eliminates data leakage by encapsulating all transformations and estimators into an automated Scikit-Learn `Pipeline`.
- **Reproducible Pipeline**: Uses `ColumnTransformer` for selective column imputation and encoding.
- **Model Persistence**: Exports the trained pipeline to `pipe.pkl` for single-step inference on raw input data.

---

## 🔄 Complete Pipeline Flow & Architecture

```mermaid
flowchart TD
    A[Raw Input Data] --> B[Drop Non-Predictive Columns: PassengerId, Name, Ticket, Cabin]
    B --> C[Train / Test Split 80/20]
    C --> D[ML Pipeline: pipe]
    
    subgraph Pipeline Steps
        D --> E[Step 1: trf1 - SimpleImputer<br>Age: Mean | Embarked: Most Frequent]
        E --> F[Step 2: trf2 - OneHotEncoder<br>Encode Sex & Embarked]
        F --> G[Step 3: trf3 - MinMaxScaler<br>Scale All Features]
        G --> H[Step 4: trf4 - SelectKBest chi2<br>Select Top 8 Features]
        H --> I[Step 5: trf5 - DecisionTreeClassifier<br>Model Training & Classification]
    end

    I --> J[Hyperparameter Tuning: GridSearchCV]
    I --> K[Cross-Validation: 5-Fold CV]
    I --> L[Export: pipe.pkl]
    L --> M[Inference: predict-using-pipeline.ipynb]
```

### Detailed Pipeline Stages

1. **Data Ingestion & Cleaning**:
   - Ingests `train.csv`.
   - Drops non-predictive identifiers and high-cardinality/missing columns: `PassengerId`, `Name`, `Ticket`, and `Cabin`.
   - Remaining feature set: `['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']`.
2. **Train/Test Split**:
   - Splits data into 80% training (`X_train`, `y_train`) and 20% testing (`X_test`, `y_test`) with `random_state=42`.
3. **Transformer 1 (`trf1` - Imputation)**:
   - Imputes missing `Age` values using mean strategy (`SimpleImputer()`).
   - Imputes missing `Embarked` values using mode strategy (`SimpleImputer(strategy='most_frequent')`).
   - Remaining columns are passed through without modification (`remainder='passthrough'`).
4. **Transformer 2 (`trf2` - Categorical Encoding)**:
   - Converts categorical variables `Sex` and `Embarked` into numerical representation using `OneHotEncoder(sparse_output=False, handle_unknown='ignore')`.
5. **Transformer 3 (`trf3` - Feature Scaling)**:
   - Scales numeric and one-hot encoded features to range `[0, 1]` using `MinMaxScaler()`.
6. **Transformer 4 (`trf4` - Feature Selection)**:
   - Selects the top $k=8$ most informative features using the Chi-Square test (`SelectKBest(score_func=chi2, k=8)`).
7. **Transformer 5 (`trf5` - Classification Estimator)**:
   - Trains a `DecisionTreeClassifier` on the transformed feature subset.
8. **Evaluation & Tuning**:
   - Evaluates using `accuracy_score` and 5-fold cross-validation (`cross_val_score`).
   - Optimizes tree depth using `GridSearchCV` over `decisiontreeclassifier__max_depth: [1, 2, 3, 4, 5, None]`.
9. **Export & Inference**:
   - Serializes the entire end-to-end pipeline into `pipe.pkl` using Python's `pickle`.
   - Real-time prediction is executed directly on raw NumPy arrays or DataFrames without manual preprocessing.

---

## ⚙️ Algorithms & Techniques Used

| Component / Task | Algorithm / Technique | Library | Purpose |
| :--- | :--- | :--- | :--- |
| **Missing Value Imputation** | Mean Imputation & Most Frequent (Mode) | `sklearn.impute.SimpleImputer` | Handles missing values in `Age` and `Embarked` |
| **Categorical Encoding** | One-Hot Encoding (OHE) | `sklearn.preprocessing.OneHotEncoder` | Converts `Sex` and `Embarked` to binary dummy variables |
| **Feature Scaling** | Min-Max Normalization | `sklearn.preprocessing.MinMaxScaler` | Normalizes values into $[0, 1]$ interval |
| **Feature Selection** | Chi-Square ($\chi^2$) Independence Test | `sklearn.feature_selection.SelectKBest`, `chi2` | Selects $k=8$ most statistically significant features |
| **Classification Model** | Decision Tree Classifier | `sklearn.tree.DecisionTreeClassifier` | Non-linear tree-based binary classification |
| **Pipeline Construction** | Pipeline & Column Transformer | `sklearn.pipeline.make_pipeline`, `ColumnTransformer` | Chains preprocessing and modeling steps to avoid data leakage |
| **Hyperparameter Optimization** | Grid Search Cross-Validation | `sklearn.model_selection.GridSearchCV` | Finds the best `max_depth` parameter |
| **Cross-Validation** | K-Fold Stratified CV ($k=5$) | `sklearn.model_selection.cross_val_score` | Estimates model generalization performance |
| **Model Serialization** | Object Serialization | `pickle` | Saves trained pipeline for production deployment |

---

## 📁 Project Directory Structure

```text
titanic-survival-ml-pipeline/
├── train.csv                      # Titanic dataset for model training and evaluation
├── titanic-using-pipeline.ipynb   # Complete training, feature engineering, and tuning pipeline
├── predict-using-pipeline.ipynb   # Inference script loading pipe.pkl for new predictions
├── pipe.pkl                       # Serialized trained Scikit-Learn pipeline
└── README.md                      # Comprehensive project documentation and guide
```

---

## 🚀 How to Start & Run the Project

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your machine.

### 2. Clone / Open the Repository
```bash
git clone https://github.com/Suraj24-stack/titanic-survival-ml-pipeline.git
cd titanic-survival-ml-pipeline
```

### 3. Set Up Virtual Environment (Recommended)

#### Option A: Using `venv`
```bash
python3 -m venv venv
source venv/bin/activate    # On macOS/Linux
# or: venv\Scripts\activate # On Windows
```

#### Option B: Using Conda
```bash
conda create -n titanic-ml python=3.10 -y
conda activate titanic-ml
```

### 4. Install Dependencies
```bash
pip install pandas numpy scikit-learn jupyter
```

### 5. Running the Notebooks

#### Step A: Train the Pipeline
Launch Jupyter Lab / Notebook:
```bash
jupyter notebook
```
- Open `titanic-using-pipeline.ipynb`.
- Run all cells sequentially (**Cell -> Run All**).
- This will preprocess the data, train the decision tree classifier, evaluate performance, and generate/update the serialized model file `pipe.pkl`.

#### Step B: Test Inference on New Data
- Open `predict-using-pipeline.ipynb`.
- Run the notebook to load `pipe.pkl` and predict survival for sample passenger input.

---

## 📊 Input Feature Reference

When providing raw input for inference, the pipeline expects the following 7 features in exact order:

| Index | Feature Name | Data Type | Description | Example Values |
| :---: | :--- | :--- | :--- | :--- |
| `0` | `Pclass` | Integer | Ticket class / Socio-economic status | `1` (1st Class), `2` (2nd Class), `3` (3rd Class) |
| `1` | `Sex` | String | Gender of the passenger | `'male'`, `'female'` |
| `2` | `Age` | Float | Passenger age in years | `31.0`, `22.5` |
| `3` | `SibSp` | Integer | Number of siblings / spouses aboard | `0`, `1`, `2` |
| `4` | `Parch` | Integer | Number of parents / children aboard | `0`, `1`, `2` |
| `5` | `Fare` | Float | Passenger fare paid | `10.50`, `71.28` |
| `6` | `Embarked` | String | Port of embarkation | `'S'` (Southampton), `'C'` (Cherbourg), `'Q'` (Queenstown) |

---

## 🔮 Making Predictions with Serialized Pipeline

You can load `pipe.pkl` in any Python script to make immediate predictions on raw passenger data:

```python
import pickle
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# 1. Load the trained pipeline
with open('pipe.pkl', 'rb') as f:
    pipe = pickle.load(f)

# 2. Define raw passenger input [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]
sample_passenger = np.array([2, 'male', 31.0, 0, 0, 10.5, 'S'], dtype=object).reshape(1, 7)

# 3. Predict survival
prediction = pipe.predict(sample_passenger)
result = "Survived" if prediction[0] == 1 else "Did Not Survive"

print(f"Prediction: {prediction[0]} -> Passenger {result}")
```

---

## 📈 Model Evaluation & Tuning

- **Baseline Cross-Validation**: Evaluated using 5-fold CV (`cross_val_score`) to measure out-of-fold generalization.
- **Hyperparameter Optimization**:
  - Parameter grid: `decisiontreeclassifier__max_depth: [1, 2, 3, 4, 5, None]`
  - Search method: `GridSearchCV(pipe, params, cv=5, scoring='accuracy')`
  - The tuned pipeline produces optimal tree depth balance between bias and variance.

---

## � Author

- **Suraj Khadka**

---

## 📜 License

This project is licensed under the **[MIT License](LICENSE)**. See the [LICENSE](LICENSE) file for more details.
Copyright (c) 2026 Suraj Khadka.
