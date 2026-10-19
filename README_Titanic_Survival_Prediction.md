# Titanic Survival Prediction

## 1. Project Overview

This project focuses on analyzing the Titanic passenger dataset and building machine learning classification models to predict whether a passenger survived.

The project follows a complete machine learning workflow:

**Data Loading → Data Understanding → EDA → Data Preprocessing → Feature Engineering → Model Training → Model Evaluation → Hyperparameter Tuning → Feature Importance → Final Conclusions**

### Objective

The main objective is to predict the target variable `survived` using passenger information such as gender, age, passenger class, fare, and family-related variables.

The target variable is:

- `0` = Did not survive
- `1` = Survived

---

## 2. Dataset

The Titanic dataset was loaded directly from Seaborn using:

```python
import seaborn as sns

df = sns.load_dataset("titanic")
```

The dataset contains **891 passengers and 15 columns**.

Important columns include:

| Column | Description |
|---|---|
| `survived` | Target variable indicating survival |
| `pclass` | Passenger class: 1st, 2nd, or 3rd |
| `sex` | Passenger gender |
| `age` | Passenger age |
| `sibsp` | Number of siblings/spouses aboard |
| `parch` | Number of parents/children aboard |
| `fare` | Passenger fare |
| `embarked` | Port of embarkation |

---

# 3. Data Understanding

The following data-understanding steps were performed:

- Displayed the first 10 rows.
- Checked the dataset shape.
- Checked data types using `dtypes` and `info()`.
- Checked missing values using `isnull().sum()`.
- Generated numerical statistics using `describe()`.
- Generated categorical statistics.
- Identified numerical and categorical columns.
- Identified `survived` as the target variable.

### Numerical Features

The main numerical features used were:

```text
pclass
age
sibsp
parch
fare
```

### Categorical Features

The main categorical features used were:

```text
sex
embarked
```

---

# 4. Exploratory Data Analysis (EDA)

EDA was performed to understand patterns and relationships in the Titanic dataset.

## 4.1 What percentage of passengers survived?

Approximately **38.38% of passengers survived**, while approximately **61.62% did not survive**.

This shows that the survived class was smaller than the non-survived class.

---

## 4.2 Does survival rate differ between males and females?

Yes. A substantial difference was observed.

Approximate survival rates:

- **Female:** 74.2%
- **Male:** 18.9%

Female passengers had a much higher survival rate than male passengers in this dataset.

---

## 4.3 Does survival rate differ between passenger classes?

Yes.

Approximate survival rates:

| Passenger Class | Survival Rate |
|---|---:|
| 1st Class | 62.96% |
| 2nd Class | 47.28% |
| 3rd Class | 24.24% |

The survival rate decreased from first class to third class.

---

## 4.4 How does age relate to survival?

The average age was approximately:

- **Survivors:** 28.34 years
- **Non-survivors:** 30.63 years

Survivors were somewhat younger on average, although the age distributions overlapped considerably. Therefore, age alone does not determine survival.

---

## 4.5 Does fare appear to have a relationship with survival?

Yes. A difference in fare was observed.

Approximate average fare:

- **Survivors:** 48.40
- **Non-survivors:** 22.12

Passengers who survived generally paid higher fares. This may partly reflect differences in passenger class, so the relationship should be interpreted as an association rather than a causal effect.

---

## 4.6 How does the number of siblings/spouses affect survival?

The survival rate varied across different `sibsp` values.

The relationship was not strictly linear. Passengers with different numbers of siblings/spouses showed different survival patterns, and some groups contained relatively few passengers.

Therefore, `sibsp` may provide useful information, but its effect is not a simple "more siblings means lower survival" relationship.

---

## 4.7 How does the number of parents/children affect survival?

Survival rates also varied according to `parch`.

The relationship was not straightforward or perfectly linear. Different family sizes produced different survival patterns, suggesting that `parch` can provide additional information when combined with other passenger characteristics.

---

# 5. EDA Visualizations

More than the required four meaningful visualizations were created.

### Visualization 1 — Survival Rate by Gender

A bar plot was used to compare survival rates between male and female passengers.

### Visualization 2 — Survival Rate by Passenger Class

A bar plot was used to compare survival rates across first, second, and third class.

### Visualization 3 — Age Distribution by Survival

A box plot was used to compare age distributions between survivors and non-survivors.

### Visualization 4 — Fare Distribution by Survival

A box plot was used to compare fare distributions between survivors and non-survivors.

### Additional Visualization — Correlation Heatmap

A correlation heatmap was created for numerical variables to inspect relationships among numerical features and the target.

---

# 6. Data Preprocessing

The raw dataset was prepared for machine learning using the following steps.

## 6.1 Removed unnecessary and redundant columns

The following columns were removed:

```python
[
    "alive",
    "class",
    "who",
    "adult_male",
    "deck",
    "embark_town",
    "alone"
]
```

### Reason for removing them

- `alive` was removed because it directly represents the same survival information as the target and could cause **data leakage**.
- `class` was removed because `pclass` already represents passenger class.
- `who` and `adult_male` contain information that overlaps with other available passenger variables.
- `deck` contains substantial missing information.
- `embark_town` duplicates the information represented by `embarked`.
- `alone` was removed because `IsAlone` was created as an engineered feature.

---

## 6.2 Missing-value handling

Missing values were handled inside a Scikit-learn pipeline.

### Numerical Features

Missing numerical values were replaced using the **median**:

```python
SimpleImputer(strategy="median")
```

Median was chosen because it is less affected by extreme values than the mean.

### Categorical Features

Missing categorical values were replaced using the **most frequent category**:

```python
SimpleImputer(strategy="most_frequent")
```

---

## 6.3 Categorical Encoding

Categorical variables were converted into numerical form using:

```python
OneHotEncoder(handle_unknown="ignore")
```

For example:

```text
sex
male
female
```

is transformed into numerical indicator features such as:

```text
sex_male
sex_female
```

This allows machine learning models to process categorical information.

---

## 6.4 Numerical Scaling

Numerical variables were standardized using:

```python
StandardScaler()
```

Scaling was included in the preprocessing pipeline, particularly to support Logistic Regression effectively.

---

## 6.5 Feature and Target Separation

The dataset was separated into:

```python
X = features
y = target
```

The target was:

```python
y = data["survived"]
```

---

## 6.6 Train/Test Split

The dataset was divided into:

- **80% training data**
- **20% testing data**

The following settings were used:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Why these settings were used

- `test_size=0.20` reserves 20% of the data for final evaluation.
- `random_state=42` makes the experiment reproducible.
- `stratify=y` helps maintain a similar survival-class distribution in training and testing sets.

---

# 7. Feature Engineering

Three engineered features were created.

## 7.1 FamilySize

```python
FamilySize = sibsp + parch + 1
```

The `+1` represents the passenger themselves.

### Purpose

This combines siblings/spouses and parents/children into one family-group-size feature.

---

## 7.2 IsAlone

```python
IsAlone = 1 if FamilySize == 1 else 0
```

Interpretation:

- `1` = Passenger was traveling alone
- `0` = Passenger was not traveling alone

### Purpose

This provides a direct binary representation of whether the passenger was traveling alone.

---

## 7.3 FarePerPerson

```python
FarePerPerson = fare / FamilySize
```

### Purpose

This provides an approximate fare per family member and gives the model another representation of the fare information.

Because this feature is derived from `fare` and `FamilySize`, it is not independent information.

---

# 8. Machine Learning Models

Three classification models were trained:

## 8.1 Logistic Regression

Logistic Regression estimates the probability of a passenger belonging to the survival class and converts that probability into a class prediction.

---

## 8.2 Decision Tree

Decision Tree creates a sequence of feature-based decision rules to classify passengers.

---

## 8.3 Random Forest

Random Forest is an ensemble of multiple Decision Trees. The trees produce predictions that are combined to produce the final classification.

---

# 9. Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Final Baseline Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.45% | 78.33% | 68.12% | 72.87% |
| Decision Tree | 75.42% | 68.12% | 68.12% | 68.12% |
| Random Forest | **81.01%** | 77.78% | **71.01%** | **74.24%** |

The values above are from the final comparison produced in the notebook.

---

# 10. Understanding the Evaluation Metrics

## Accuracy

Accuracy measures the proportion of all predictions that were correct.

```text
Accuracy = Correct Predictions / Total Predictions
```

It gives an overall view of model correctness.

---

## Precision

Precision answers:

> Of the passengers predicted as survivors, how many were actually survivors?

```text
Precision = TP / (TP + FP)
```

---

## Recall

Recall answers:

> Of the passengers who actually survived, how many did the model correctly identify?

```text
Recall = TP / (TP + FN)
```

---

## F1 Score

F1 Score combines precision and recall:

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

It is useful when both precision and recall need to be considered together.

---

# 11. Which Metric Is More Informative?

For this binary classification problem, F1 Score is useful because it balances precision and recall.

Recall is also important because it tells us how many actual survivors were successfully identified.

Accuracy provides a general performance measure, but using accuracy alone does not show the complete error pattern.

Therefore, accuracy, precision, recall, F1 Score, and the confusion matrix were considered together.

---

# 12. False Positives and False Negatives

## False Positive

A passenger actually did **not survive**, but the model predicts:

```text
survived = 1
```

In this case, the model incorrectly predicts survival.

## False Negative

A passenger actually **survived**, but the model predicts:

```text
survived = 0
```

In this case, the model fails to identify an actual survivor.

---

# 13. Model Improvement

Random Forest was selected for hyperparameter tuning.

`GridSearchCV` was used with **5-fold cross-validation**.

The following hyperparameters were explored:

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
```

The search used:

```python
scoring="f1"
```

because F1 balances precision and recall.

The search was performed only using the training data, while the test set was kept for final evaluation.

## Before vs After Tuning

| Metric | Before Tuning | After Tuning |
|---|---:|---:|
| Accuracy | 81.01% | 81.01% |
| Precision | 77.78% | **80.70%** |
| Recall | **71.01%** | 66.67% |
| F1 Score | **74.24%** | 73.02% |

### Tuning Result

After tuning:

- Accuracy remained unchanged.
- Precision increased.
- Recall decreased.
- F1 Score decreased slightly.

Therefore, the tuned configuration did **not** improve the overall test-set F1 performance compared with the baseline Random Forest.

This is a useful machine learning result because hyperparameter tuning does not necessarily improve every evaluation metric.

> The exact best hyperparameter combination selected by `GridSearchCV` is available in the notebook output (`grid_search.best_params_`).

---

# 14. Feature Importance

Feature importance was extracted from the trained Random Forest model.

The original feature-level importance values were:

| Feature | Importance |
|---|---:|
| `sex` | 0.269857 |
| `age` | 0.206426 |
| `FarePerPerson` | 0.169713 |
| `fare` | 0.163926 |
| `pclass` | 0.064333 |
| `FamilySize` | 0.036543 |
| `embarked` | 0.032735 |
| `sibsp` | 0.022310 |
| `parch` | 0.021056 |
| `IsAlone` | 0.013101 |

### Interpretation

According to the trained Random Forest:

- `sex` was the most influential original feature.
- `age` was the second most influential.
- `FarePerPerson` and `fare` were also highly influential.
- `pclass` provided additional predictive information.
- Among the engineered features, `FarePerPerson` had greater importance than `FamilySize` and `IsAlone`.

Feature importance describes how the trained model used the features for prediction. It should **not** be interpreted as proof that a feature caused survival.

---

# 15. Key Observations

1. Survival rates differed substantially between males and females.
2. Passenger class was strongly associated with survival rates.
3. Survivors were somewhat younger on average than non-survivors.
4. Survivors generally had higher recorded fares.
5. Family-related variables showed more complex, non-linear patterns.
6. `FamilySize`, `IsAlone`, and `FarePerPerson` added engineered information.
7. Random Forest achieved the highest baseline accuracy, recall, and F1 Score in the final comparison.
8. Hyperparameter tuning improved precision but reduced recall and F1 on the held-out test set.
9. Random Forest feature importance identified `sex`, `age`, `FarePerPerson`, `fare`, and `pclass` as influential features.

---

# 16. Final Conclusion

This project implemented a complete machine learning workflow for Titanic survival prediction.

The dataset was explored using statistical analysis and visualizations, followed by missing-value handling, categorical encoding, scaling, feature engineering, and an 80/20 train-test split.

Three classification models were trained: Logistic Regression, Decision Tree, and Random Forest. Among the baseline models in the final comparison, Random Forest achieved an accuracy of **81.01%** and an F1 Score of **74.24%**.

Feature engineering produced `FamilySize`, `IsAlone`, and `FarePerPerson`, with `FarePerPerson` providing substantial model importance.

Random Forest hyperparameter tuning using GridSearchCV and 5-fold cross-validation increased precision from **77.78% to 80.70%**, but recall decreased from **71.01% to 66.67%**, causing F1 Score to decrease from **74.24% to 73.02%**.

The project demonstrates the importance of comparing multiple models, using appropriate evaluation metrics, preventing data leakage, applying cross-validation, and interpreting model behavior through feature importance.

---

# 17. Possible Future Improvements

Future work could include:

- Testing additional classification algorithms.
- Using `RandomizedSearchCV` for a larger hyperparameter search.
- Experimenting with different feature-engineering strategies.
- Performing more extensive cross-validation.
- Carrying out detailed error analysis.
- Comparing additional evaluation metrics.
- Testing the final approach on external or unseen Titanic-related data.

---

# 18. Technologies Used

- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn
- Jupyter Notebook / Google Colab

---

# 19. Project Structure

A recommended project structure is:

```text
Titanic-Survival-Prediction/
│
├── Titanic_Survival_Prediction.ipynb
├── README.md
└── screenshots/
```

---

## Author

**Waleed Asrar**

Machine Learning Task — Titanic Survival Prediction
