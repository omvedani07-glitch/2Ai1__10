# =========================================
# 1. IMPORT LIBRARIES
# =========================================
import pandas as pd
import numpy as np

# =========================================
# 2. LOAD DATA
# =========================================
train = pd.read_csv("Titanic_train.csv")
test = pd.read_csv("Titanic_test.csv")

# =========================================
# 3. VIEW DATA (OPTIONAL)
# =========================================
print(train.head())
print(train.info())

# =========================================
# 4. HANDLE MISSING VALUES
# =========================================

# Fill Age with median
train["Age"].fillna(train["Age"].median(), inplace=True)
test["Age"].fillna(test["Age"].median(), inplace=True)

# Fill Embarked with most frequent value
train["Embarked"].fillna(train["Embarked"].mode()[0], inplace=True)
test["Embarked"].fillna(test["Embarked"].mode()[0], inplace=True)

# Fill Fare (only in test usually)
test["Fare"].fillna(test["Fare"].median(), inplace=True)

# =========================================
# 5. DROP UNNECESSARY COLUMNS
# =========================================
train.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True, errors='ignore')
test.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True, errors='ignore')

# =========================================
# 6. ENCODE CATEGORICAL DATA
# =========================================

# Convert Sex to numeric
train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
test["Sex"] = test["Sex"].map({"male": 0, "female": 1})

# One-hot encoding for Embarked
train = pd.get_dummies(train, columns=["Embarked"], drop_first=True)
test = pd.get_dummies(test, columns=["Embarked"], drop_first=True)

# =========================================
# 7. ALIGN TRAIN & TEST DATA
# =========================================
train, test = train.align(test, join='left', axis=1, fill_value=0)

# =========================================
# 8. DEFINE FEATURES & TARGET
# =========================================
X = train.drop("Survived", axis=1)
y = train["Survived"]

# =========================================
# 9. FEATURE SCALING (IMPORTANT FOR LOGISTIC)
# =========================================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)

# Convert back to DataFrame (optional)
X = pd.DataFrame(X_scaled, columns=X.columns)
test = pd.DataFrame(test_scaled, columns=test.columns)

# =========================================
# 10. FINAL OUTPUT
# =========================================
print("\n✅ Data Preprocessing Completed Successfully!")
print("Shape of X:", X.shape)
print("Shape of Test:", test.shape)