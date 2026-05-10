# 🎓 Smart College Placement Predictor

A machine learning web app that predicts student placement probability 
and provides personalized improvement suggestions.

## 🔴 Live Demo
👉 [Click here to try the app](https://smart-placement-predictor.streamlit.app/)

## 🔥 What makes this different
- Predicts placement probability (not just yes/no)
- Explains WHY using SHAP feature importance
- Suggests WHAT to improve using Skill Gap Analyzer
- What-If Analysis to simulate improvements live

## 📊 Dataset
- 5000 Indian engineering students
- 22 features including CGPA, skills, internships, backlogs

## 🤖 Model Performance
- Random Forest → Accuracy: 89.3% | ROC-AUC: 0.90
- XGBoost       → Accuracy: 85.3% | ROC-AUC: 0.90

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- SHAP (explainability)
- Streamlit (dashboard)

## 📁 Project Structure
placement-predictor/
├── data/          ← dataset files
├── models/        ← trained model files
├── notebooks/     ← EDA, cleaning, training, explainability
├── app.py         ← Streamlit dashboard
└── requirements.txt
