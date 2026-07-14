# ✈️ Flight Price Prediction

Predict flight ticket prices using Machine Learning.

This project is an end-to-end machine learning application that predicts airline ticket prices based on various flight details such as airline, source, destination, departure time, arrival time, duration, number of stops, and journey date. The project covers the complete ML workflow, including data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, and deployment using Streamlit.

---

## 🚀 Live Demo

**🌐 Streamlit App:** https://flight-prices-predictions.streamlit.app

---

## 📌 Features

- Flight fare prediction using Machine Learning
- Exploratory Data Analysis (EDA)
- Data preprocessing and cleaning
- Feature engineering and encoding
- Comparison of multiple regression models
- Interactive Streamlit web application
- Model training using Amazon SageMaker
- User-friendly prediction interface

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Machine Learning
- Scikit-learn
- XGBoost

### Data Analysis
- Pandas
- NumPy

### Visualization
- Matplotlib
- Seaborn

### Deployment
- Streamlit
- Amazon SageMaker

### Version Control
- Git
- GitHub

---

## 📂 Project Structure

```
flight-prices-prediction/
│
├── dataset/                 # Dataset files
├── notebooks/               # EDA and model development
├── models/                  # Trained models
├── app.py                   # Streamlit application
├── requirements.txt         # Dependencies
├── README.md
└── images/
```

---

## 📊 Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. Hyperparameter Tuning
8. Deployment using Streamlit

---

## 🤖 Machine Learning Models

The following regression models were trained and compared:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

The best-performing model was selected based on evaluation metrics.

---

## 📈 Model Evaluation

The models were evaluated using:

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

---

## 📷 Application Preview

> Add screenshots of your Streamlit application here.

Example:

```
images/homepage.png
images/prediction.png
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/rahuldasila/flight-prices-prediction.git
```

Move into the project directory

```bash
cd flight-prices-prediction
```

Create a virtual environment (optional)

```bash
python -m venv venv
```

Activate the environment

Windows

```bash
venv\Scripts\activate
```

Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📋 Input Features

- Airline
- Source
- Destination
- Journey Date
- Departure Time
- Arrival Time
- Duration
- Total Stops

---

## 💡 Future Improvements

- Add real-time flight APIs
- Improve prediction accuracy with ensemble learning
- Deploy using Docker
- CI/CD pipeline
- Cloud deployment using AWS

---

## 👨‍💻 Author

**Rahul Dasila**

GitHub: https://github.com/rahuldasila

LinkedIn: *(Add your LinkedIn URL here)*

---

## ⭐ If you found this project useful, consider giving it a Star!
