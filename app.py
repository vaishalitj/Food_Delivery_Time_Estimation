import streamlit as st
import pandas as pd
import joblib

# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="Food Delivery Time Estimation",
    page_icon="🍔",
    layout="wide"
)

# -----------------------------------------
# LOAD MODEL
# -----------------------------------------

model = joblib.load("random_forest_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# -----------------------------------------
# CUSTOM CSS
# -----------------------------------------

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.section-title {
    font-size: 25px;
    font-weight: 600;
    margin-top: 20px;
}

.prediction-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 25px;
}

.prediction-value {
    font-size: 42px;
    font-weight: bold;
}

.info-box {
    padding: 18px;
    border-radius: 12px;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# TITLE
# -----------------------------------------

st.markdown(
    '<div class="main-title">🍔 Food Delivery Time Estimation</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict food delivery time using a trained Random Forest Machine Learning model'
    '</div>',
    unsafe_allow_html=True
)

# -----------------------------------------
# MODEL PERFORMANCE
# -----------------------------------------

st.markdown(
    '<div class="section-title">📊 Model Performance</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MAE", "3.45 min")

with col2:
    st.metric("RMSE", "4.41 min")

with col3:
    st.metric("R² Score", "0.78")

st.info(
    "The Random Forest model achieved an R² score of 0.78, "
    "meaning it explains a substantial portion of the variation "
    "in delivery time in the test data."
)

# -----------------------------------------
# INPUT SECTION
# -----------------------------------------

st.markdown(
    '<div class="section-title">🚴 Delivery Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    age = st.slider(
        "Delivery Person Age",
        18,
        50,
        30
    )

    rating = st.slider(
        "Delivery Person Rating",
        1.0,
        5.0,
        4.7,
        0.1
    )

    vehicle_condition = st.slider(
        "Vehicle Condition",
        0,
        3,
        2
    )

with col2:

    weather = st.selectbox(
        "Weather Conditions",
        [
            "Sunny",
            "Cloudy",
            "Fog",
            "Stormy",
            "Sandstorms",
            "Windy"
        ]
    )

    traffic = st.selectbox(
        "Road Traffic Density",
        [
            "Low",
            "Medium",
            "High",
            "Jam"
        ]
    )

    multiple = st.selectbox(
        "Multiple Deliveries",
        [0, 1, 2, 3]
    )

with col3:

    order_type = st.selectbox(
        "Type of Order",
        [
            "Meal",
            "Snack",
            "Drinks",
            "Buffet"
        ]
    )

    vehicle = st.selectbox(
        "Type of Vehicle",
        [
            "motorcycle",
            "scooter",
            "electric_scooter",
            "bicycle"
        ]
    )

    festival = st.selectbox(
        "Festival",
        [
            "No",
            "Yes"
        ]
    )

city = st.selectbox(
    "City",
    [
        "Urban",
        "Metropolitian",
        "Semi-Urban"
    ]
)

# -----------------------------------------
# LOCATION INFORMATION
# -----------------------------------------

st.markdown(
    '<div class="section-title">📍 Location Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    rest_lat = st.number_input(
        "Restaurant Latitude",
        value=12.9716,
        format="%.4f"
    )

    rest_long = st.number_input(
        "Restaurant Longitude",
        value=77.5946,
        format="%.4f"
    )

with col2:

    del_lat = st.number_input(
        "Delivery Latitude",
        value=12.9352,
        format="%.4f"
    )

    del_long = st.number_input(
        "Delivery Longitude",
        value=77.6245,
        format="%.4f"
    )

# -----------------------------------------
# PREDICTION BUTTON
# -----------------------------------------

st.markdown("---")

predict_button = st.button(
    "🚀 Predict Delivery Time",
    use_container_width=True
)

# -----------------------------------------
# PREDICTION
# -----------------------------------------

if predict_button:

    new_order = pd.DataFrame([{

        "Delivery_person_Age": age,

        "Delivery_person_Ratings": rating,

        "Restaurant_latitude": rest_lat,

        "Restaurant_longitude": rest_long,

        "Delivery_location_latitude": del_lat,

        "Delivery_location_longitude": del_long,

        "Order_Date": "15-03-2022",

        "Time_Orderd": "08:30",

        "Time_Order_picked": "08:45",

        "Weather_conditions": weather,

        "Road_traffic_density": traffic,

        "Vehicle_condition": vehicle_condition,

        "Type_of_order": order_type,

        "Type_of_vehicle": vehicle,

        "multiple_deliveries": multiple,

        "Festival": festival,

        "City": city

    }])

    # Preprocess input

    processed = preprocessor.transform(new_order)

    # Prediction

    prediction = model.predict(processed)[0]

    # -----------------------------------------
    # DISPLAY RESULT
    # -----------------------------------------

    st.markdown("---")

    st.markdown(
        '<div class="section-title">🎯 Prediction Result</div>',
        unsafe_allow_html=True
    )

    st.success(
        f"Estimated Delivery Time: {prediction:.2f} minutes"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Estimated Time",
            f"{prediction:.2f} min"
        )

    with col2:
        st.metric(
            "Model",
            "Random Forest"
        )

    with col3:
        st.metric(
            "R² Score",
            "0.78"
        )

# -----------------------------------------
# FOOTER
# -----------------------------------------

st.markdown("---")

st.caption(
    "Food Delivery Time Estimation | Machine Learning Project | Random Forest"
)