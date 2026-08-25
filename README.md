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

The end-to-end pipeline follows a streamlined transformation and modeling process:

1. **Data Cleaning & Split**: Drops non-predictive columns (`PassengerId`, `Name`, `Ticket`, `Cabin`) and performs an 80/20 train-test split.
2. **Missing Value Imputation (`trf1`)**: Imputes `Age` with mean and `Embarked` with mode.
3. **Categorical Encoding (`trf2`)**: One-Hot Encodes categorical features (`Sex`, `Embarked`).
4. **Feature Scaling (`trf3`)**: Scales all features to `[0, 1]` using `MinMaxScaler`.
5. **Feature Selection (`trf4`)**: Selects top 8 most informative features using Chi-Square (`SelectKBest`).
6. **Model Training & Export (`trf5`)**: Fits a `DecisionTreeClassifier`, tunes with `GridSearchCV`, and exports the trained pipeline to `pipe.pkl`.

---

## ⚙️ Algorithms & Techniques Used

| Component / Task | Algorithm / Technique | Purpose |
| :--- | :--- | :--- |
| **Missing Value Imputation** | Mean Imputation & Most Frequent (Mode) | Handles missing values in `Age` and `Embarked` |
| **Categorical Encoding** | One-Hot Encoding (OHE) | Converts `Sex` and `Embarked` to binary dummy variables |
| **Feature Scaling** | Min-Max Normalization | Normalizes values into $[0, 1]$ interval |
| **Feature Selection** | Chi-Square ($\chi^2$) Independence Test | Selects $k=8$ most statistically significant features |
| **Classification Model** | Decision Tree Classifier | Non-linear tree-based binary classification |
| **Pipeline Construction** | Pipeline & Column Transformer | Chains preprocessing and modeling steps to avoid data leakage |
| **Hyperparameter Optimization** | Grid Search Cross-Validation | Finds the best `max_depth` parameter |
| **Cross-Validation** | K-Fold Stratified CV ($k=5$) | Estimates model generalization performance |
| **Model Serialization** | Object Serialization | Saves trained pipeline for production deployment |

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

## 📈 Model Evaluation & Tuning

- **Baseline Cross-Validation**: Evaluated using 5-fold CV (`cross_val_score`) to measure out-of-fold generalization.
- **Hyperparameter Optimization**:
  - Parameter grid: `decisiontreeclassifier__max_depth: [1, 2, 3, 4, 5, None]`
  - Search method: `GridSearchCV(pipe, params, cv=5, scoring='accuracy')`
  - The tuned pipeline produces optimal tree depth balance between bias and variance.

---

## 👤 Author

- **Suraj Khadka**

---

## 📜 License

This project is licensed under the **[MIT License](LICENSE)**. See the [LICENSE](LICENSE) file for more details.
Copyright (c) 2026 Suraj Khadka.
