# 🧮 MNIST Digit Classifier  
### RandomForest model for the Kaggle Digit Recognizer challenge

A compact but complete machine learning pipeline that trains a model to recognize handwritten digits from raw pixel data.  
This project was built for the **Kaggle Digit Recognizer** competition and includes the exact steps used to generate a submission.

---
<p float="left">
  <img src="https://github.com/user-attachments/assets/bd387090-007e-4389-911a-addac0283965" width="300" height="180" />
  <img src="https://github.com/user-attachments/assets/55ce722e-218c-4799-8517-31257416434c" width="300" height="180" />
</p>




## 📘 Competition  


**Kaggle:** https://www.kaggle.com/competitions/digit-recognizer

**Dataset:**  
- `train.csv` → 42,000 labeled digit images  
- `test.csv` → 28,000 unlabeled images  
- Pixel values range from `0–255`  
- 28×28 grayscale images  
- Dataset not included due to Kaggle rules

---

## ⚙️ What This Project Does  

**Data Processing**  
- Loads train/test CSVs  
- Removes duplicate rows  
- Handles missing values  
- Splits data: 80% training / 20% validation  

**Model Training**  
- Uses `RandomForestClassifier` (300 trees)  
- Prints validation accuracy  
- Retrains on full dataset  

**Output**  
- Generates a **Kaggle-ready** CSV  
- Saved as: `Submission_real_new.csv`

---

## 🚀 How to Run  

Make sure your working folder has these files:

train.csv
test.csv
digit_classifier.py
requirements.txt (optional)


Run the script:



python digit_classifier.py


The script will:

✔ Train the model  
✔ Show validation accuracy  
✔ Create `Submission_real_new.csv`

---

## 📁 Repository Structure  



├── digit_classifier.py # main training + inference pipeline
├── Submission_real_new.csv # final Kaggle submission
├── requirements.txt # optional
└── README.md



---

## 🧠 Model Details  
- **Algorithm:** RandomForestClassifier  
- **Trees:** 300  
- Works well on MNIST without heavy preprocessing  
- Fast, stable, and dependable baseline model  

---

## 👤 Author  
**Puneet Poddar**  
Kaggle Profile: [(https://www.kaggle.com/puneet2769)]

---

## 📌 Notes  
- The project is intentionally simple and clean.  
- No deep learning required.  
- Great baseline for experimenting with:
  - PCA  
  - scaling  
  - hyperparameter tuning  
  - alternative classifiers  
