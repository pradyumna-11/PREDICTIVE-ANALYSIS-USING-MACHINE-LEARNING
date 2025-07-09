# PREDICTIVE-ANALYSIS-USING-MACHINE-LEARNING

"COMPANY" : CODTECH IT SOLUTIONS

"NAME" : SIKHAKOLLI PRADYUMNA

"INTERN ID" : CT08DN583

"DOMAIN" : DATA ANALYSIS

"DURATION" : 8 WEEKS

"MENTOR" : NEELA SANTOSH

## 📌 Task 2: Build a Machine Learning Model to Predict Outcomes

### 📝 Overview

The objective of Task 2 is to **build and evaluate a machine learning model**, either for classification or regression, that can predict outcomes based on a structured dataset. This task demonstrates the ability to extract meaningful features from data, preprocess it for modeling, apply machine learning algorithms, and evaluate predictive performance using appropriate metrics.

For this task, we used a **classification approach** on the well-known **Titanic dataset**, where the goal is to predict whether a passenger survived the Titanic disaster based on features such as age, gender, class, fare, and embarkation point. This dataset is ideal for beginners and showcases the full machine learning pipeline from data cleaning to model evaluation.

---

### ⚙️ Tools and Technologies

* **Python 3**
* **Pandas** for data handling and preprocessing
* **Scikit-learn** for model building and evaluation
* **Matplotlib / Seaborn** for visualization
* **Google Colab / Jupyter Notebook** for execution environment

---

### 🎯 Problem Statement

The Titanic dataset is a binary classification problem:

> Given a set of features about passengers, predict whether each passenger survived (`1`) or not (`0`) the Titanic sinking.

This task reflects a common real-world scenario where organizations want to use past data to make future predictions (e.g., churn prediction, fraud detection, customer classification, etc.).

---

### 📂 Dataset Description

The dataset used contains the following features:

* `Pclass`: Passenger class (1st, 2nd, 3rd)
* `Sex`: Gender
* `Age`: Age in years
* `SibSp`: Number of siblings/spouses aboard
* `Parch`: Number of parents/children aboard
* `Fare`: Ticket fare
* `Embarked`: Port of embarkation
* `Survived`: Target variable (0 = did not survive, 1 = survived)

Some columns such as `Name`, `Ticket`, and `Cabin` were dropped due to irrelevance or excessive missing values.

---

### 🧠 Methodology

1. **Data Preprocessing**

   * Handled missing values using median and mode imputation.
   * Converted categorical variables (`Sex`, `Embarked`) into numeric format using `LabelEncoder`.
   * Removed irrelevant features.

2. **Feature-Target Split**

   * Separated independent features (X) and target variable (y).

3. **Train-Test Split**

   * Used an 80-20 split to train and evaluate the model using `train_test_split`.

4. **Model Selection**

   * Chose **Random Forest Classifier** for its robustness and accuracy with minimal tuning.
   * Other models (e.g., Logistic Regression, Decision Tree) can be substituted based on the use case.

5. **Model Training**

   * Trained the classifier on the training data using `.fit()` method.

6. **Model Evaluation**

   * Evaluated using:

     * **Accuracy Score**
     * **Confusion Matrix**
     * **Classification Report** (Precision, Recall, F1-Score)
   * Visualized feature importance and prediction quality.

---

### 📈 Results

* The model achieved an **accuracy of \~80–85%** depending on the random train-test split.
* **Feature importance** showed that `Sex`, `Fare`, and `Pclass` were the most influential factors in predicting survival.
* Confusion matrix and classification report indicated a good balance between precision and recall.

---

### 📊 Visualizations

* Bar plots of survival by gender and class
* Correlation heatmap of numerical features
* Confusion matrix heatmap
* Feature importance chart from the Random Forest model

These visualizations added interpretability and insights into the model’s decision-making process.

---

### 🚀 Applications and Extensions

* Use more advanced models like **Gradient Boosting, XGBoost, or LightGBM**
* Perform **hyperparameter tuning** using `GridSearchCV`
* Use **cross-validation** for more robust evaluation
* Integrate the model into a web application for real-time prediction

---

### 📤 Deliverables

* A `.ipynb` notebook or `.py` script containing:

  * Data loading and preprocessing
  * Model training and evaluation
  * Summary of insights and performance metrics
* Screenshots or plots showing key metrics and visualizations

---

This task highlights the core process of building predictive models, a crucial skill in data science and machine learning roles. It reflects your understanding of supervised learning, model evaluation, and real-world data challenges.
