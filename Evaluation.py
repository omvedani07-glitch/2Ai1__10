# =========================================
# 1. IMPORT LIBRARIES
# =========================================
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc

# =========================================
# 2. PREDICTION (if not already done)
# =========================================
y_pred = model.predict(X_val)

# =========================================
# 3. ACCURACY
# =========================================
accuracy = accuracy_score(y_val, y_pred)
print("✅ Accuracy:", accuracy)
print("✅ Accuracy (%):", accuracy * 100)

# =========================================
# 4. CONFUSION MATRIX
# =========================================
cm = confusion_matrix(y_val, y_pred)
print("\n✅ Confusion Matrix:\n", cm)

# -----------------------------------------
# Confusion Matrix Visualization (IMPORTANT)
# -----------------------------------------
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# =========================================
# 5. CLASSIFICATION REPORT
# =========================================
cr = classification_report(y_val, y_pred)
print("\n✅ Classification Report:\n", cr)

# =========================================
# 6. ROC CURVE & AUC
# =========================================
# Get probability scores
y_prob = model.predict_proba(X_val)[:, 1]

# Calculate ROC
fpr, tpr, thresholds = roc_curve(y_val, y_prob)

# Calculate AUC
roc_auc = auc(fpr, tpr)
print("\n✅ AUC Score:", roc_auc)

# -----------------------------------------
# ROC Curve Visualization
# -----------------------------------------
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label="AUC = " + str(round(roc_auc, 3)))
plt.plot([0, 1], [0, 1], linestyle='--')  # random line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# =========================================
# 7. EXTRA: FEATURE IMPORTANCE (if supported)
# =========================================
try:
    import pandas as pd
    
    feature_importance = model.coef_[0]
    features = X_val.columns

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": feature_importance
    }).sort_values(by="Importance", ascending=False)

    print("\n✅ Feature Importance:\n", importance_df)
except:
    print("\n⚠ Feature importance not available for this model")

# =========================================
# 8. FINAL MESSAGE
# =========================================
print("\n🎯 Model Evaluation Completed Successfully!")