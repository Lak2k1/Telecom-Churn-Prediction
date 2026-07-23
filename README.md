# Telecom-Churn-Prediction
Machine learning pipeline to predict telecom customer churn using statistical feature selection and SVM.

## 📌 Project Objective
Customer attrition is a major expense in the telecom industry. The objective of this project is to build a Machine Learning classification model to predict whether a customer will churn. Because the business cost of a **False Negative** (missing a churning customer) is drastically higher than a False Positive, this pipeline deliberately shifts focus away from standard accuracy to optimize strictly for **Class 1 Recall** (Churn Detection Rate).

## 📊 Dataset 
* **Size:** 7,032 customer records (after initial cleaning), 20+ features.
* **Source:** [Kaggle Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## 🛠️ Tech Stack
* **Data Manipulation & Visualization:** `pandas`, `numpy`, `matplotlib`, `seaborn`
* **Statistical Testing:** `scipy.stats` (Chi-Square, T-Test)
* **Machine Learning:** `scikit-learn` (Logistic Regression, Decision Trees, Support Vector Machines, GridSearchCV)

## 🚀 Key Methodologies & Workflow

1. **Statistical Feature Selection:** 
   * Conducted **Welch's T-Tests** to mathematically prove that `tenure` (p < 2.3e-234) and `MonthlyCharges` heavily impact churn.
   * Ran **Chi-Square Tests of Independence** to purge statistically insignificant features (e.g., `gender`, `PhoneService` yielded p > 0.05).
   * Checked **Spearman Correlation** to identify and drop multicollinear variables (`TotalCharges` had a 0.89 correlation with `tenure`).
   
2. **Advanced Feature Engineering:** 
   * Abstracted human behavior by creating an `Is_Family` flag (combining Partner/Dependents) and an `Is_Auto_Pay` flag to measure payment friction.
   * Engineered a `Total_Addons` ecosystem lock-in score (0 to 6) to mathematically measure a customer's dependence on the company's network.
   
3. **Data Leakage Prevention:** 
   * Applied Ordinal Encoding to hierarchical features (Contracts) and One-Hot Encoding to nominal features.
   * Performed `StandardScaler` transformations strictly *after* the stratified 80/20 train-test split to ensure zero test-data leakage.
   
4. **Handling Imbalanced Classes & Tuning:** 
   * The dataset was heavily imbalanced (73% Stay / 27% Churn). Applied `class_weight='balanced'` to mathematically penalize minority class classification errors.
   * Utilized `GridSearchCV` to exhaustively tune Logistic Regression, Decision Trees, and SVM models, optimizing explicitly for `scoring='recall'`.

## 📈 Final Evaluation (Test Set Recall)
By mathematically proving feature significance and tuning the cost function for the minority class, the final model achieved a massive jump in churn detection compared to the unweighted baseline.

* **Baseline Logistic Regression:** 57% Recall (Caught 215 / 374 actual churners)
* **Tuned Decision Tree:** 79% Recall (Caught 295 / 374 actual churners)
* **🏆 Tuned Linear SVM:** **82% Recall (Caught 305 / 374 actual churners)**

*Conclusion: Rigorous statistical testing, multicollinearity purging, and behavioral feature engineering allowed a highly-explainable Linear Kernel SVM to maximize minority-class recall and draw a highly robust decision boundary.*

## 💡 How to View
Simply open the `.ipynb` file in this repository to view the full Exploratory Data Analysis (EDA) and model training pipeline.

Or if you simply want to open in Colab use [this](https://colab.research.google.com/drive/1E9oB_ARCN9KUbKjeSmrKI7Qxg6RKpfoC#scrollTo=8VkAVEkBiYK8) link.
