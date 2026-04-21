# 🚢 Titanic Survival Prediction

---

## 👥 Team Members (7 Members)

| Name    |           Role           |
| ------  |--------------------------|   
| Om    | Project Lead             |
| om   | feature engineering      |
| on   | Frontend Developer       |
| om   | EDA              |
| shashivardhan   | Data Preprocessing       |
| K U Ashvwandh| model building | 
| Afrath isma   | model evalution|  
| om  | model deployment            |
| om   | Documentation & Testing  |

---

## 🎯 Problem Statement

Predict whether a passenger survived or not using machine learning.

---

## 📊 Dataset Used

* titanic-train.csv
* titanic-test.csv

👉 These datasets contain passenger details like:

* Age
* Sex
* Fare
* Passenger Class

---

## 🧹 Data Preprocessing

We cleaned the data before training the model:

* Filled missing Age values using average
* Filled missing Embarked values using most frequent value
* Removed unnecessary columns like Cabin
* Converted text data into numbers (Male/Female → 0/1)

👉 Simple meaning:
Cleaning data like fixing mistakes before using it

---

## 🤖 Model Building

We used **Logistic Regression** to build the model.

Steps:

* Selected important features (Age, Sex, Fare, etc.)
* Split data into training and testing
* Trained model using training dataset

👉 Simple meaning:
Teaching the computer using past data

---

## 📈 Evaluation

We checked how well the model works:

* Accuracy Score
* Confusion Matrix

👉 Result:
Model gives around **80% accuracy** (may change)

👉 Simple meaning:
Like checking exam results after studying

---

## 🔗 Conclusion

* Model successfully predicts survival
* Data preprocessing improved performance
* Logistic Regression is simple and effective

---

## 🔥 Simple Flow

Data → Cleaning → Model → Result
