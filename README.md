# breast-cancer-diagnosis-with-ml-models

This project focuses on building predictive machine learning models for breast cancer diagnosis using the **Breast Cancer Wisconsin (Diagnostic) Dataset**. The goal is to classify tumors as **malignant** or **benign** based on features extracted from digitized images of fine needle aspirates (FNA) of breast masses.

## 🔍 Problem Statement

Early and accurate detection of breast cancer is critical for effective treatment. This project applies supervised learning algorithms to predict the presence of cancerous cells, helping support medical diagnosis.

## 📊 Dataset

- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- Instances: 569
- Features: 30 numerical features (mean, standard error, and worst of 10 real-valued features)
- Target: `diagnosis` (M = Malignant, B = Benign)

## 🧠 Machine Learning Models

The following models and techniques were used:

- **Logistic Regression** – as a baseline model
- **Random Forest Classifier** – to capture non-linear patterns and feature importance
- **RandomizedSearchCV** – for hyperparameter tuning and optimization

## ⚙️ Workflow

1. **Data Preprocessing**
   - Dropped `id` column
   - Encoded target variable (`M` as 1, `B` as 0)
   - Scaled feature values using StandardScaler
   - Train-test split for evaluation

2. **Model Training**
   - Logistic Regression and Random Forest trained and compared
   - Randomized Search used for hyperparameter tuning of Random Forest

3. **Evaluation Metrics**
   - Accuracy
   - Precision, Recall, F1-Score
   - ROC-AUC Curve
   - Confusion Matrix

4. **Feature Importance**
   - Visualized most influential features using Random Forest feature importances

## 📈 Results

- Logistic Regression and Random Forest both showed high accuracy, with Random Forest performing slightly better after tuning.
- Feature importance analysis highlighted key features such as `worst_concave_points`, `worst_radius`, and `worst_perimeter`.

## 📦 Tools and Libraries

- Python (NumPy, Pandas)
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

## ✅ Key Takeaways

- Ensemble models like Random Forest are highly effective for structured, tabular medical data.
- Hyperparameter tuning significantly improves model performance.
- Even simple models like Logistic Regression can be strong baselines in binary classification tasks.

## 📚 References

- Bennett, K. P., & Mangasarian, O. L. (1992). *Robust Linear Programming Discrimination of Two Linearly Inseparable Sets*. Optimization Methods and Software.
- UCI ML Repository – Breast Cancer Wisconsin (Diagnostic) Data Set
