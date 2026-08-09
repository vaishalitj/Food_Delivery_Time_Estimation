# 🍔 Food Delivery Time Estimation

## 📌 Project Overview

Food Delivery Time Estimation is a Machine Learning project that predicts the estimated time required to deliver a food order.

The project analyzes delivery-related factors such as delivery person information, ratings, weather conditions, road traffic, vehicle condition, order type, location, multiple deliveries, festival conditions, and city.

A Random Forest Regression model is trained to predict the delivery time in minutes.

---

## 🎯 Objectives

- Analyze food delivery data.
- Clean and preprocess the dataset.
- Identify important factors affecting delivery time.
- Train Machine Learning regression models.
- Compare Linear Regression and Random Forest.
- Evaluate model performance.
- Visualize predictions and feature importance.
- Predict delivery time for a new order.
- Deploy the trained model using Streamlit.

---

## 📊 Dataset

The dataset contains **45,584 records and 20 columns** before preprocessing.

The target variable is:

`Time_taken (min)`

Important input features include:

- Delivery Person Age
- Delivery Person Ratings
- Restaurant Latitude
- Restaurant Longitude
- Delivery Location Latitude
- Delivery Location Longitude
- Order Date
- Time Ordered
- Time Order Picked
- Weather Conditions
- Road Traffic Density
- Vehicle Condition
- Type of Order
- Type of Vehicle
- Multiple Deliveries
- Festival
- City

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Joblib
- Streamlit
- Jupyter Notebook
- Visual Studio Code
- GitHub

---

## 🤖 Machine Learning Algorithms

### 1. Linear Regression

Linear Regression was used as a baseline regression model.

Performance:

- MAE: 4.75 minutes
- RMSE: 5.95 minutes
- R² Score: 0.60

### 2. Random Forest Regression

Random Forest was trained to capture more complex relationships between the input features and delivery time.

Performance:

- MAE: 3.45 minutes
- RMSE: 4.41 minutes
- R² Score: 0.78

### 🏆 Best Model

Random Forest performed better than Linear Regression based on the evaluation results.

---

## 📈 Visualizations

The project includes:

### Interactive 3D Prediction Visualization

The 3D visualization compares:

- Actual Delivery Time
- Predicted Delivery Time
- Prediction Error

### Feature Importance Visualization

The feature importance chart shows which processed features were most useful to the Random Forest model.

---

## 🔮 New Delivery Prediction

The trained Random Forest model can predict the delivery time for a new food order.

Example:

`Estimated Delivery Time: 19.62 minutes`

The prediction is an estimated value and is not a guaranteed delivery time.

---

## 🌐 Streamlit Application

The project includes an interactive Streamlit web application.

Users can enter delivery information such as:

- Delivery person age
- Delivery rating
- Weather
- Traffic
- Vehicle condition
- Order type
- Vehicle type
- Multiple deliveries
- Festival
- City
- Restaurant location
- Delivery location

The application then predicts the estimated delivery time.

---

## 📁 Project Structure

```text
Food_Delivery_Time_Estimation/
│
├── app.py
├── food_delivery_prediction.py
├── Zomato Dataset.csv
├── random_forest_model.pkl
├── preprocessor.pkl
├── requirements.txt
├── README.md
├── .gitignore
├── feature_importance.html
├── food_delivery_3d_visualization.html
└── venv/