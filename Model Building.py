# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==============================
# 2. Load Dataset
# ==============================
train = pd.read_csv("Titanic_train.csv")
test = pd.read_csv("Titanic_test.csv")

# ==============================
# 3. Data Preprocessing
# ==============================

# Drop unnecessary columns
train.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True, errors='ignore')
test.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True, errors='ignore')

# Fill missing values
train["Age"].fillna(train["Age"].median(), inplace=True)
test["Age"].fillna(test["Age"].median(), inplace=True)

train["Embarked"].fillna(train["Embarked"].mode()[0], inplace=True)
test["Embarked"].fillna(test["Embarked"].mode()[0], inplace=True)

# Convert categorical to numeric
train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
test["Sex"] = test["Sex"].map({"male": 0, "female": 1})

train = pd.get_dummies(train, columns=["Embarked"], drop_first=True)
test = pd.get_dummies(test, columns=["Embarked"], drop_first=True)

# Align train and test columns
train, test = train.align(test, join='left', axis=1, fill_value=0)

# ==============================
# 4. Define Features & Target
# ==============================
X = train.drop("Survived", axis=1)
y = train["Survived"]

# ==============================
# 5. Split Data
# ==============================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 6. Model Selection
# ==============================
model = LogisticRegression(max_iter=1000)

# ==============================
# 7. Train Model
# ==============================
model.fit(X_train, y_train)

# ==============================
# 8. Prediction
# ==============================
y_pred = model.predict(X_val)

# ==============================
# 9. Evaluation
# ==============================
print("Accuracy:", accuracy_score(y_val, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))

# ==============================
# 10. Test Prediction
# ==============================
test_predictions = model.predict(test)

# ==============================
# 11. Save Output File
# ==============================
output = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_predictions
})

output.to_csv("submission.csv", index=False)

print("\n✅ Model Built Successfully & Output Saved as submission.csv")