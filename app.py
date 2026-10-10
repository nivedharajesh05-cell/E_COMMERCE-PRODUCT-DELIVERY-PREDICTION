import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page Layout
st.set_page_config(page_title="E-Commerce Delivery Predictor", page_icon="📦", layout="wide")

st.title("📦 E-Commerce Delivery Prediction App")
st.write("Predict whether a product shipment will reach on time based on order parameters.")

# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["Data Overview", "Predict Shipment"])

# Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv("E_Commerce.csv")

# Load Machine Learning Model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

# ----------------------------------------
# PAGE 1: DATA OVERVIEW
# ----------------------------------------
if page == "Data Overview":
    st.subheader("📊 Dataset Overview")
    try:
        df = load_data()
        col1, col2 = st.columns(2)
        col1.metric("Total Records", df.shape[0])
        col2.metric("Total Features", df.shape[1])
        
        st.write("### Preview")
        st.dataframe(df.head())
    except FileNotFoundError:
        st.error("`E_Commerce.csv` file not found in directory.")

# ----------------------------------------
# PAGE 2: PREDICT SHIPMENT
# ----------------------------------------
elif page == "Predict Shipment":
    st.subheader("🎯 Input Order Details")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            customer_care_calls = st.number_input("Customer Care Calls", min_value=1, max_value=10, value=3)
            customer_rating = st.slider("Customer Rating", 1, 5, 3)
            cost_of_product = st.number_input("Cost of Product ($)", min_value=1, max_value=1000, value=150)
            prior_purchases = st.number_input("Prior Purchases", min_value=1, max_value=20, value=3)

        with col2:
            discount_offered = st.number_input("Discount Offered (%)", min_value=0, max_value=100, value=10)
            weight_in_gms = st.number_input("Weight (grams)", min_value=100, max_value=10000, value=2000)

        submit_btn = st.form_submit_button("Predict Delivery Status")

    if submit_btn:
        try:
            model = load_model()

            input_df = pd.DataFrame([{
                'Customer_care_calls': customer_care_calls,
                'Customer_rating': customer_rating,
                'Cost_of_the_Product': cost_of_product,
                'Prior_purchases': prior_purchases,
                'Discount_offered': discount_offered,
                'Weight_in_gms': weight_in_gms
            }])

            prediction = model.predict(input_df)[0]

            if prediction == 1:
                st.success("✅ **Prediction:** The shipment will arrive **ON TIME**!")
            else:
                st.warning("⚠️ **Prediction:** The shipment is expected to be **DELAYED**.")

        except FileNotFoundError:
            st.error("`model.pkl` not found! Please save your trained model first.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")