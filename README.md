# Bank Marketing Prediction App

## Overview

This project is a machine learning web application that predicts whether a customer will subscribe to a term deposit based on data from bank marketing campaigns.

It leverages an **XGBoost model** integrated into a pipeline and provides real-time predictions through an interactive web interface built with **Streamlit**.



## Features

* Predicts customer subscription outcome (Yes/No)
* Real-time predictions via web interface
* Clean and user-friendly UI
* End-to-end ML pipeline (preprocessing + model)
* Fast and efficient predictions using XGBoost



## Machine Learning Model

* **Algorithm:** XGBoost Classifier
* **Pipeline includes:**

  * Data preprocessing
  * Feature encoding
  * Model training and prediction
* **Model file:** `xgboost_bank_pipeline.pkl`



## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Joblib



## 📂 Project Structure

```
Bank Marketing Prediction/
│── app.py
│── xgboost_bank_pipeline.pkl
│── requirements.txt
│── README.md
```



## ⚙️ Installation & Setup

1. Clone the repository:

```
git clone https://github.com/yourusername/bank-marketing-app.git
cd bank-marketing-app
```

2. Create environment (optional but recommended):

```
conda create -n bank_app python=3.10 -y
conda activate bank_app
```

3. Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```
streamlit run app.py
```

Then open the local URL shown in your browser.

---

## 🎯 Objective

To help financial institutions identify potential customers who are more likely to subscribe to term deposits, improving marketing efficiency and reducing costs.

---

## 🌍 Deployment

This app can be deployed using:

* Streamlit Cloud
* Heroku
* Render

---

## 👤 Author

Oyekemi Tawakalt
GitHub: [https://github.com/yourusername](https://github.com/Oyekem/bank_marketing_app)

