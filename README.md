# Viral Fever Prediction

## Introduction

This Flask application serves as a predictive tool for viral fevers, specifically focusing on Dengue and Typhoid. It provides a user-friendly web interface that allows individuals or healthcare professionals to input various clinical and laboratory parameters, upon which the system generates predictions using machine learning models. The aim is to offer preliminary diagnostic insights to aid in the early detection and management of these common fevers.

## Problem Statement

Accurate and timely diagnosis of viral fevers like Dengue and Typhoid is crucial for effective treatment and preventing complications. Traditional diagnostic methods can sometimes be time-consuming or require specialized lab equipment. There is a need for an accessible, preliminary screening tool that can leverage common blood test parameters to assist in the early identification and assessment of the potential presence and severity of these conditions, thereby supporting faster decision-making.

## Components Required

### Software Components:

* **Python 3.x:** The core programming language.
* **Flask:** A micro web framework for building the application's web interface.
* **Pandas:** Used for data manipulation, particularly for loading and processing the dataset CSV files.
* **Scikit-learn:** Utilized for `LabelEncoder` to transform categorical features into numerical formats suitable for machine learning models.
* **XGBoost:** The machine learning library used to build the classification models for both Dengue and Typhoid prediction.
* **Dataset Files:**
    * `Dengue-Dataset.csv`
    * `typhoid 1.csv`
    *(Note: These CSV files are expected to be present at a specific local path as defined in `app.py`, e.g., `C:\Users\dkath\Desktop\project\`)*
* **HTML Template (`index.html`):** Required for the web interface, it must be located in a `templates` directory within the application's root.

## Features

* **Dengue Prediction:** The application takes various blood parameters (e.g., Hemoglobin, Neutrophils, Lymphocytes, Platelets, WBC count, etc.) as input and predicts whether the patient is "POSITIVE" or "NEGATIVE" for Dengue.
* **Typhoid Severity Prediction:** If the Dengue prediction is "NEGATIVE", the system then proceeds to predict the "Severity" of Typhoid (e.g., "LOW", "MEDIUM", "HIGH") based on relevant parameters such as Gender, Age, Hemoglobin, Platelet Count, Blood Culture, Urine Culture, Calcium, and Potassium.
* **Interactive Web Interface:** Provides a user-friendly web form for seamless data input and immediate display of prediction results.
* **Machine Learning Models:** Employs robust XGBoost classifiers, trained on specific datasets, to perform accurate predictions for both conditions.

## Conclusion

This Flask-based application provides a practical and accessible tool for the preliminary prediction of viral fevers like Dengue and Typhoid. By automating the analysis of common laboratory parameters through machine learning models, it offers valuable insights that can assist in quicker diagnostic pathways. While not a substitute for professional medical diagnosis, this project demonstrates a feasible approach to leveraging data science and web technologies to support healthcare efforts in managing prevalent diseases, ultimately aiming to facilitate earlier intervention and improved patient outcomes.
