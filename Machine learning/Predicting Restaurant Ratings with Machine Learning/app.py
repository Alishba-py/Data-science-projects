import streamlit as st
import pandas as pd
import joblib

# Load scaler and model
scaler = joblib.load('scaler.pkl')
model = joblib.load('mlmodel.pkl')

# Mapping function for ratings
def map_rating_to_text(rating):
    if rating < 2.0:
        return "Poor"
    elif 2.0 <= rating < 3.0:
        return "Average"
    elif 3.0 <= rating < 4.0:
        return "Good"
    elif 4.0 <= rating < 4.6:
        return "Very Good"
    else:
        return "Excellent"

# Streamlit app input
st.title("Restaurant Rating Prediction App")

cost = st.number_input("Enter the average cost for two", min_value=0, value=500,step=200)
table_booking = st.selectbox("Does the restaurant have table booking?", ["Yes", "No"])
online_delivery = st.selectbox("Does the restaurant have online delivery?", ["Yes", "No"])
price_range = st.selectbox("Price range (1=Cheapest, 4=Most Expensive)", [1, 2, 3, 4])

# Prepare input data
table_booking = 1 if table_booking == "Yes" else 0
online_delivery = 1 if online_delivery == "Yes" else 0

input_data = pd.DataFrame([{
    "Average Cost for two": cost,
    "Has Table booking": table_booking,
    "Has Online delivery": online_delivery,
    "Price range": price_range
}])

# Scale input data
input_data_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    numerical_prediction = model.predict(input_data_scaled)[0]
    text_prediction = map_rating_to_text(numerical_prediction)
    st.success(f"The predicted rating is: {text_prediction}")


