import pandas as pd

# Load the dataset
df = pd.read_csv("Zomato Dataset.csv")

# Display the first 5 rows
print(df.head())

# Check the number of rows and columns
print("\nDataset Shape:")
print(df.shape)

# Display all column names
print("\nColumn Names:")
print(df.columns)

# Display information about the dataset
print("\nDataset Information:")
print(df.info())

# Display statistical summary
print("\nStatistical Summary:")
print(df.describe())

# -----------------------------------------
# STEP 9: DATA CLEANING - INSPECTION
# -----------------------------------------

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())

# Check unique values in important categorical columns
print("\nWeather Conditions:")
print(df["Weather_conditions"].unique())

print("\nRoad Traffic Density:")
print(df["Road_traffic_density"].unique())

print("\nType of Order:")
print(df["Type_of_order"].unique())

print("\nType of Vehicle:")
print(df["Type_of_vehicle"].unique())

print("\nFestival:")
print(df["Festival"].unique())

print("\nCity:")
print(df["City"].unique())

# -----------------------------------------
# STEP 10: REMOVE UNNECESSARY ID COLUMNS
# -----------------------------------------

# Remove identifier columns
df = df.drop(columns=["ID", "Delivery_person_ID"])

print("\nColumns after removing ID columns:")
print(df.columns)

print("\nNew Dataset Shape:")
print(df.shape)

# -----------------------------------------
# STEP 11: SEPARATE INPUTS AND TARGET
# -----------------------------------------

# X contains the input features
X = df.drop(columns=["Time_taken (min)"])

# y contains the target variable
y = df["Time_taken (min)"]

print("\nInput Features (X):")
print(X.columns)

print("\nTarget (y):")
print(y.name)

print("\nX Shape:")
print(X.shape)

print("\ny Shape:")
print(y.shape)

# -----------------------------------------
# STEP 12: IDENTIFY COLUMN TYPES
# -----------------------------------------

# Identify numerical columns
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns

# Identify categorical columns
categorical_features = X.select_dtypes(include=["str"]).columns

print("\nNumerical Features:")
print(list(numerical_features))

print("\nCategorical Features:")
print(list(categorical_features))

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Numerical preprocessing
numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

# Categorical preprocessing
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# Combine both preprocessing methods
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("\nPreprocessing pipeline created successfully.")

# -----------------------------------------
# STEP 13: TRAIN / TEST SPLIT
# -----------------------------------------

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

print("\nTraining Target Shape:")
print(y_train.shape)

print("\nTesting Target Shape:")
print(y_test.shape)

# -----------------------------------------
# STEP 14: APPLY PREPROCESSING
# -----------------------------------------

# Fit the preprocessing pipeline using training data
X_train_processed = preprocessor.fit_transform(X_train)

# Apply the same preprocessing to testing data
X_test_processed = preprocessor.transform(X_test)

print("\nPreprocessing completed successfully.")

print("\nProcessed Training Data Shape:")
print(X_train_processed.shape)

print("\nProcessed Testing Data Shape:")
print(X_test_processed.shape)

# -----------------------------------------
# STEP 15: LINEAR REGRESSION
# -----------------------------------------

from sklearn.linear_model import LinearRegression

# Create the Linear Regression model
linear_model = LinearRegression()

# Train the model
linear_model.fit(X_train_processed, y_train)

# Make predictions on the test data
linear_predictions = linear_model.predict(X_test_processed)

print("\nLinear Regression training completed successfully.")

print("\nFirst 10 Predictions:")
print(linear_predictions[:10])

print("\nFirst 10 Actual Values:")
print(y_test.iloc[:10].values)

# -----------------------------------------
# STEP 16: EVALUATE LINEAR REGRESSION
# -----------------------------------------

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Calculate evaluation metrics
linear_mae = mean_absolute_error(y_test, linear_predictions)

linear_rmse = np.sqrt(
    mean_squared_error(y_test, linear_predictions)
)

linear_r2 = r2_score(y_test, linear_predictions)

print("\nLinear Regression Evaluation:")
print(f"MAE  : {linear_mae:.2f}")
print(f"RMSE : {linear_rmse:.2f}")
print(f"R²   : {linear_r2:.2f}")

# -----------------------------------------
# STEP 17: RANDOM FOREST REGRESSION
# -----------------------------------------

from sklearn.ensemble import RandomForestRegressor

# Create the Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train the model
rf_model.fit(X_train_processed, y_train)

print("\nRandom Forest training completed successfully.")

# Make predictions
rf_predictions = rf_model.predict(X_test_processed)

print("\nFirst 10 Random Forest Predictions:")
print(rf_predictions[:10])

print("\nFirst 10 Actual Values:")
print(y_test.iloc[:10].values)

# -----------------------------------------
# STEP 18: EVALUATE RANDOM FOREST
# -----------------------------------------

# Calculate Random Forest evaluation metrics
rf_mae = mean_absolute_error(y_test, rf_predictions)

rf_rmse = np.sqrt(
    mean_squared_error(y_test, rf_predictions)
)

rf_r2 = r2_score(y_test, rf_predictions)

print("\nRandom Forest Evaluation:")
print(f"MAE  : {rf_mae:.2f}")
print(f"RMSE : {rf_rmse:.2f}")
print(f"R²   : {rf_r2:.2f}")

# -----------------------------------------
# STEP 19: MODEL COMPARISON
# -----------------------------------------

comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [linear_mae, rf_mae],
    "RMSE": [linear_rmse, rf_rmse],
    "R2": [linear_r2, rf_r2]
})

print("\nModel Comparison:")
print(comparison.round(2))

# -----------------------------------------
# STEP 20: INTERACTIVE 3D VISUALIZATION
# -----------------------------------------

import plotly.graph_objects as go

# Calculate prediction error
prediction_error = y_test.values - rf_predictions

# Create interactive 3D scatter plot
fig = go.Figure()

fig.add_trace(
    go.Scatter3d(
        x=y_test.values,
        y=rf_predictions,
        z=prediction_error,
        mode="markers",
        marker=dict(
            size=4,
            opacity=0.7
        ),
        text=[
            f"Actual: {actual:.1f} min<br>"
            f"Predicted: {predicted:.1f} min<br>"
            f"Error: {error:.1f} min"
            for actual, predicted, error
            in zip(y_test.values, rf_predictions, prediction_error)
        ],
        hovertemplate="%{text}<extra></extra>",
        name="Delivery Predictions"
    )
)

# Add a zero-error reference plane/line
fig.add_trace(
    go.Scatter3d(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        z=[0, 0],
        mode="lines",
        line=dict(
            width=6
        ),
        name="Zero Error Reference"
    )
)

# Customize the chart
fig.update_layout(
    title={
        "text": "Food Delivery Time Prediction - Random Forest",
        "x": 0.5
    },
    scene=dict(
        xaxis_title="Actual Delivery Time (minutes)",
        yaxis_title="Predicted Delivery Time (minutes)",
        zaxis_title="Prediction Error (minutes)"
    ),
    template="plotly_dark",
    width=1100,
    height=750
)

# Display the interactive chart
#fig.show()
fig.write_html("food_delivery_3d_visualization.html", auto_open=True)


# -----------------------------------------
# STEP 21: INTERACTIVE FEATURE IMPORTANCE
# -----------------------------------------

import plotly.express as px

# Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()

# Get importance values from Random Forest
feature_importance = rf_model.feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importance
})

# Sort by importance
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# Select top 20 features
top_features = importance_df.head(20)

print("\nTop 20 Important Features:")
print(top_features)

# Create interactive horizontal bar chart
fig = px.bar(
    top_features.sort_values("Importance"),
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top 20 Features Influencing Food Delivery Time",
    labels={
        "Importance": "Feature Importance",
        "Feature": "Feature"
    },
    template="plotly_dark"
)

fig.update_layout(
    width=1100,
    height=750,
    title={
        "x": 0.5
    }
)

# Save interactive chart
fig.write_html(
    "feature_importance.html",
    auto_open=True
)


# -----------------------------------------
# STEP 22: NEW DELIVERY TIME PREDICTION
# -----------------------------------------

# Create a new delivery order
new_order = pd.DataFrame([{
    "Delivery_person_Age": 30,
    "Delivery_person_Ratings": 4.7,
    "Restaurant_latitude": 12.9716,
    "Restaurant_longitude": 77.5946,
    "Delivery_location_latitude": 12.9352,
    "Delivery_location_longitude": 77.6245,
    "Order_Date": "15-03-2022",
    "Time_Orderd": "08:30",
    "Time_Order_picked": "08:45",
    "Weather_conditions": "Sunny",
    "Road_traffic_density": "Medium",
    "Vehicle_condition": 2,
    "Type_of_order": "Meal",
    "Type_of_vehicle": "motorcycle",
    "multiple_deliveries": 1,
    "Festival": "No",
    "City": "Urban"
}])

print("\nNew Delivery Order:")
print(new_order)

# Step 22 Preprocess the new order
new_order_processed = preprocessor.transform(new_order)

# Predict delivery time
new_prediction = rf_model.predict(new_order_processed)

print("\n-----------------------------------------")
print("NEW DELIVERY TIME PREDICTION")
print("-----------------------------------------")

print(
    f"Estimated Delivery Time: {new_prediction[0]:.2f} minutes"
)

# -----------------------------------------
# STEP 23: SAVE TRAINED MODEL
# -----------------------------------------

import joblib

# Save the trained Random Forest model
joblib.dump(rf_model, "random_forest_model.pkl")

# Save the preprocessing pipeline
joblib.dump(preprocessor, "preprocessor.pkl")

print("\n-----------------------------------------")
print("MODEL SAVING COMPLETED")
print("-----------------------------------------")
print("Random Forest model saved as: random_forest_model.pkl")
print("Preprocessor saved as: preprocessor.pkl")