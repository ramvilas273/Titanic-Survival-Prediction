# 🚢 Titanic Survival Prediction

An end-to-end data science project that predicts passenger survival on the Titanic —
covering EDA, feature engineering, a leakage-safe preprocessing pipeline, multi-model
comparison with hyperparameter tuning, and a deployed Streamlit app.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Pipeline-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](#license)

---

## 📌 Overview

This project tackles the classic Titanic survival problem: predict whether a passenger
survived, based on attributes like class, sex, age, fare, and family relationships. The
goal wasn't just an accuracy number — it's a complete, reproducible workflow from raw
data to a usable app:

1. **Exploratory Data Analysis** — understand what actually drove survival
2. **Feature Engineering** — extract signal the raw columns don't expose directly
3. **Preprocessing Pipeline** — `ColumnTransformer` + `Pipeline`, fit only on training folds
4. **Model Comparison** — 6 classifiers tuned with `GridSearchCV`
5. **Evaluation** — confusion matrix, ROC-AUC, feature importance
6. **Deployment** — an interactive Streamlit app powered by the notebook's own trained model

## 🖥️ App Preview

The Streamlit app has three tabs:

| Tab | What it does |
|---|---|
| 🔮 **Predict** | Enter a passenger's details and get a survival prediction with probability |
| 📄 **Batch Prediction** | Upload a CSV (same shape as `test.csv`) and download predictions for every row |
| 📊 **EDA Dashboard** | Live survival-rate charts by sex, class, age, and family size |

## 🧠 Approach

### Feature Engineering
- **`Title`** — parsed from passenger name (Mr, Mrs, Miss, Master, Rare)
- **`FamilySize`** / **`IsAlone`** — derived from `SibSp` + `Parch`
- **`Deck`** — first letter of `Cabin` (`'Unknown'` when missing)
- **`TicketGroupSize`** — how many passengers share a ticket number

### Pipeline
```
ColumnTransformer
├── Numeric (Age, Fare, FamilySize, TicketGroupSize)
│   └── Median Imputation → Standard Scaling
└── Categorical (Pclass, Sex, Embarked, Title, Deck, IsAlone)
    └── Most-Frequent Imputation → One-Hot Encoding
```
Wrapped in a single `sklearn.Pipeline` with the classifier, so imputation/scaling
statistics are learned **only on the training fold** during cross-validation — no data
leakage.

### Model Comparison

Six models were tuned with `GridSearchCV` (5-fold stratified CV, scored on accuracy):

| Model | CV Accuracy | Validation Accuracy |
|---|---|---|
| **Logistic Regression** ✅ | 0.8273 | **0.8380** |
| SVM (RBF) | 0.8399 | 0.8268 |
| Gradient Boosting | 0.8343 | 0.8045 |
| XGBoost | 0.8343 | 0.8045 |
| Random Forest | 0.8315 | 0.7989 |
| KNN | 0.8160 | 0.8101 |

**Selected model:** Logistic Regression (`C=1`, L2 penalty) — highest validation accuracy,
Validation ROC-AUC **0.871**.

## 📂 Project Structure

```
titanic-survival-prediction/
├── titanic.ipynb              # Full EDA → modeling → evaluation notebook
├── app.py                     # Streamlit app (loads the notebook's saved model)
├── model.pkl                  # Trained pipeline, saved directly from the notebook
├── train.csv                  # Labeled training data
├── test.csv                   # Unlabeled test data
├── gender_submission.csv      # Baseline submission format
├── requirements.txt
└── README.md
```

## ⚙️ Setup & Usage

```bash
# Clone the repo
git clone https://github.com/ramvilas273/titanic-survival-prediction.git
cd titanic-survival-prediction

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Run the notebook:**
```bash
jupyter notebook titanic.ipynb
```

**Run the app:**
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`.

> `model.pkl` is saved directly from the notebook (`joblib.dump(final_pipeline, 'model.pkl')`,
> where `final_pipeline` is the `GridSearchCV` winner refit on the full training set) — the
> app doesn't retrain anything independently.

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `Matplotlib` · `Seaborn` ·
`Streamlit` · `Jupyter`

## 🚀 Possible Extensions

- Soft-voting/stacking ensemble of the top models
- Repeated cross-validation for lower-variance model comparison
- Group-wise (Title × Pclass) age imputation instead of a global median
- SHAP-based explanations in the app for individual predictions

## 👤 Author

**Ram Vilas**
Aspiring Data Scientist & ML Engineer, based in Chennai, India

- GitHub: [@ramvilas273](https://github.com/ramvilas273)
- LinkedIn: [in/ramvilas](https://www.linkedin.com/in/ram-vilas/)

## 📄 License

This project is licensed under the [MIT License](LICENSE).
