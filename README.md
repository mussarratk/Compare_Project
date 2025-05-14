
# Experiment-Driven Comparison of Four Regression Algorithms Models for Bike Rental Prediction - Azure ML 
### Comparative Analysis of Regression Models for Bike Rental Prediction using Azure ML Experiments

* Used Python 3.8 AzureML - azureml.core - SDK V1


<details>

# Of Tunning.ipynb
Great! Let’s go through each part of your code and understand the **purpose**, **concepts**, **uses**, **advantages**, and **disadvantages** of each step.

---

## 🔧 **Hyperparameter Tuning Using Grid Search**

```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, r2_score
from sklearn.ensemble import GradientBoostingRegressor
```

### ✅ Purpose:

* You're setting up a **grid search** to find the best combination of hyperparameters for the `GradientBoostingRegressor`.

### 🔍 Key Concepts:

* **Hyperparameters**: Settings like `learning_rate`, `n_estimators` that you choose before training.
* **GridSearchCV**: Tries every combination of hyperparameter values and evaluates using cross-validation.
* **make\_scorer**: Wraps the `r2_score` function so GridSearchCV knows how to score models.

### 🧠 Uses:

* Systematically finds the best model configuration.

### ✅ Advantages:

* Simple to implement and understand.
* Exhaustive — ensures best config (within tested values).

### ❌ Disadvantages:

* Very **slow** with many parameters/values.
* Doesn’t scale well with large datasets.

---

## 🔍 **Specify Search Parameters**

```python
params = {
  'learning_rate': [0.1, 0.5, 1.0],
  'n_estimators' : [50, 100, 150]
}
```

### 📘 Uses:

* Tests all combinations of `learning_rate` and `n_estimators` (3×3 = 9 combinations).

### ✅ Pros:

* Gives insight into how each parameter affects performance.

### ❌ Cons:

* Can become computationally expensive with more parameters.

---

## 🚀 **Run the Grid Search**

```python
gridsearch = GridSearchCV(alg, params, scoring=score, cv=3, return_train_score=True)
gridsearch.fit(X_train, y_train)
```

* `cv=3`: 3-fold cross-validation — splits data into 3 parts to validate model performance.
* `return_train_score=True`: Also keeps training score for analysis.

---

## ✅ **Select and Evaluate the Best Model**

```python
model = gridsearch.best_estimator_
```

* `best_estimator_`: The model with the best performance on validation folds.

```python
predictions = model.predict(X_test)
```

### 📊 Metrics

```python
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)
```

* **MSE**: Penalizes larger errors more.
* **RMSE**: Easier to interpret because it's in original units.
* **R²**: Shows how well predictions explain variability in the target.

---

## 📈 **Plot Predictions**

```python
plt.scatter(y_test, predictions)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Daily Bike Share Predictions')
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), color='magenta')
```

### 🔍 Purpose:

* Visualize how close predictions are to actual values.
* Ideal: Points should fall close to the magenta line (perfect prediction).

---

## 🏗️ **Pipeline with Preprocessing**

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
```

### 🔧 What is a Pipeline?

* Combines **preprocessing** and **model training** into one streamlined process.

---

### 🔄 ColumnTransformer

```python
numeric_features = [6,7,8,9]
categorical_features = [0,1,2,3,4,5]

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)
```

### ✅ Uses:

* Scales numeric values (StandardScaler → mean=0, std=1).
* Encodes categorical features (OneHotEncoder → binary vectors).

### 🧠 Advantages:

* Keeps preprocessing clean and reusable.
* Helps avoid **data leakage** (transformers only see training data during `fit`).

### ❌ Disadvantage:

* Slight learning curve to properly format and map columns.

---

## 🏁 **Train a Model with Pipeline**

```python
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor())
])

model = pipeline.fit(X_train, y_train)
```

### ✅ Purpose:

* Combines all preprocessing and training steps.

### 👍 Pros:

* Cleaner, less error-prone.
* Great for production use (consistent transforms).

---

## 🔁 **Change Estimator — Try Random Forest**

```python
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])
```

### 🤔 Why Try a Different Estimator?

* Helps compare model performance (Random Forest vs Gradient Boosting).

### 📊 Evaluation:

* Run same metrics and plot to compare model behavior.

---

## 🔚 Summary Table

| Step                      | Purpose                    | Pros                         | Cons                         |
| ------------------------- | -------------------------- | ---------------------------- | ---------------------------- |
| GridSearchCV              | Find best hyperparams      | Thorough, simple             | Slow with large grids        |
| make\_scorer + r2\_score  | Customize evaluation       | Flexible                     | Manual setup                 |
| Pipeline                  | Combine preprocess + model | Clean, consistent            | Needs careful column mapping |
| ColumnTransformer         | Handle different types     | Flexible                     | Slight complexity            |
| StandardScaler            | Normalize features         | Essential for some models    | Not all models need it       |
| OneHotEncoder             | Encode categories          | Needed for ML algorithms     | Can create many features     |
| GradientBoostingRegressor | Boosting trees             | Accurate, handles non-linear | Slower, sensitive to params  |
| RandomForestRegressor     | Bagging trees              | Robust, less tuning          | Less accurate in some cases  |

---

If you'd like, I can help you automate hyperparameter tuning using **RandomizedSearchCV** (faster) or integrate these models into **Azure ML pipelines** for deployment. Let me know!

  
</details>

  
------------------------------------------------------------------------------------------
- compute cpu
![image](https://github.com/user-attachments/assets/2cacaab3-5002-437d-89d1-ef0ada89a1e6)
- Notebook - sdk
![image](https://github.com/user-attachments/assets/fc7c3357-6523-466a-86bc-6254c16f0d60)
![image](https://github.com/user-attachments/assets/b880c266-6eda-4375-9006-113734bb0c4e)
![image](https://github.com/user-attachments/assets/6689fc33-3261-4d2e-9cbe-d839b98278a8)
![image](https://github.com/user-attachments/assets/89f2c368-a620-48ea-a44e-d5a2a46b0085)

 - Jobs
![image](https://github.com/user-attachments/assets/f8bcd366-d971-4f3a-844a-fd847be80b66)

- experiment
![image](https://github.com/user-attachments/assets/789383a6-2a9f-4440-925f-b7360b3d8c7d)
- Metrics
![image](https://github.com/user-attachments/assets/d84db244-0aa1-4c3d-aed7-46be9c9bb810)

- Image
![image](https://github.com/user-attachments/assets/053e6ea1-efc2-49ff-b01f-9bfb122c1fd1)
 - Charts
![image](https://github.com/user-attachments/assets/9eedb7fe-7565-441a-a3bf-190a96a64495)
![image](https://github.com/user-attachments/assets/47e48b6a-80a2-462c-9deb-8380a2464756)
![image](https://github.com/user-attachments/assets/7c32a949-8778-4e03-9388-7f4e9f462055)




-------------------------------------------------------------------------------------------------------
