import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Load or Train the Model
# -----------------------------

# If you have a saved model:
# model = joblib.load('loopcart_model.pkl')

# Or define a mock training for demo (if needed):
@st.cache_data
def train_model():
    # Synthetic training (use actual model in practice)
    df = pd.read_csv('product_returns.csv')
    df['Decision_encoded'] = df['Decision'].astype('category').cat.codes
    df_encoded = pd.get_dummies(df, columns=['Product_Category', 'Return_Reason'], drop_first=True)
    X = df_encoded.drop(['Decision', 'Decision_encoded', 'Product_Code'], axis=1)
    y = df_encoded['Decision_encoded']
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model, X.columns, dict(enumerate(df['Decision'].astype('category').cat.categories))

model, feature_order, label_map = train_model()

# -----------------------------
# Streamlit App UI
# -----------------------------

st.set_page_config(page_title="LoopCart Predictor", layout="centered")

st.title("♻️ LoopCart: Return Decision Predictor")
st.markdown("""
Predict whether a returned product should be **Resold**, **Recycled**, or **Donated** based on its condition and usage.
""")

# User Inputs
col1, col2 = st.columns(2)

with col1:
    product_category = st.selectbox("Product Category", ['Electronics', 'Clothing', 'Groceries'])
    return_reason = st.selectbox("Return Reason", ['Damaged', 'Expired', 'Size Issue', 'Defective'])

with col2:
    condition_score = st.slider("Condition Score (1 - 10)", 1, 10, 5)
    usage_days = st.slider("Usage Days", 1, 60, 30)
    product_value = st.number_input("Product Value (₹)", min_value=50, max_value=50000, value=1000, step=100)

# Prepare input for prediction
def prepare_input():
    data = {
        'Condition_Score': condition_score,
        'Usage_Days': usage_days,
        'Product_Value': product_value,
        f'Product_Category_{product_category}': 1,
        f'Return_Reason_{return_reason}': 1
    }
    # Add missing dummy columns as 0
    full_input = {feat: data.get(feat, 0) for feat in feature_order}
    return pd.DataFrame([full_input])

# Predict
if st.button("Predict Return Decision"):
    input_df = prepare_input()
    pred = model.predict(input_df)[0]
    decision = label_map[pred]

    st.success(f"🔍 Recommended Action: **{decision}**")

    if decision == 'Resell':
        st.info("✅ Product is in good condition. Consider resale on secondary markets.")
    elif decision == 'Recycle':
        st.warning("♻️ Product should be sent to recycling centers.")
    elif decision == 'Donate':
        st.info("🎁 Product can be donated to a suitable organization.")

# -----------------------------
# Footer or Insights
# -----------------------------

st.markdown("---")
st.markdown("💡 This AI model uses condition score, usage, value, and return reason to reduce waste and improve circular economy logistics.")


