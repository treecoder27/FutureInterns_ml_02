import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# load model and encoders
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
feature_names = model_data["feature_names"]

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# helper function for encoding
def apply_encodings(input_df, encoders):
    for column, encoder in encoders.items():
        if column in input_df.columns:
            input_df[column] = encoder.transform(input_df[column])
    return input_df

#plot feature importance
def plot_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
        fig, ax = plt.subplots(figsize=(6, 8))
        importances.plot(kind="barh", ax=ax)
        ax.set_title("Feature Importance")
        st.pyplot(fig)
    else:
        st.info("Feature importance not available for this model.")

# streamlit user interface layout
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("Customer Churn Prediction System")
st.write("Provide customer details to predict churn:")

with st.form("prediction_form"):
    user_input = {}

    for feature in feature_names:
        if feature in encoders.keys():
            options = encoders[feature].classes_.tolist()
            user_input[feature] = st.selectbox(f"{feature}:", options)
        elif feature in ["SeniorCitizen", "tenure"]:
            user_input[feature] = st.number_input(f"{feature}:", min_value=0, value=0)
        elif feature in ["MonthlyCharges", "TotalCharges"]:
            user_input[feature] = st.number_input(f"{feature}:", min_value=0.0, value=0.0, step=0.01)
        else:
            user_input[feature] = st.text_input(f"{feature}:")

    submitted = st.form_submit_button("Click to Predict")

    if submitted:
        input_df = pd.DataFrame([user_input])
        input_df = input_df[feature_names]
        input_df = apply_encodings(input_df, encoders)

        prediction = model.predict(input_df)[0]
        result = "Customer is likely to churn." if prediction == 1 else "Customer is not likely to churn."
        st.success(result)

# show feature importance
st.sidebar.header("Model Insights")
if st.sidebar.checkbox("Show Feature Importance"):
    plot_feature_importance(model, feature_names)
