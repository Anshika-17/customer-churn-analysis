import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📉 Customer Churn Analysis Dashboard")
st.markdown("Analyze customer behavior and identify churn risks using data-driven insights.")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader("Upload Customer Churn CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding="latin1")
    st.sidebar.success("Dataset loaded successfully")
else:
    st.warning("Please upload a Customer Churn CSV file to continue.")
    st.stop()


# -----------------------------
# Data Preview
# -----------------------------
with st.expander("📄 Preview Dataset"):
    st.dataframe(df.head())

# -----------------------------
# Data Cleaning
# -----------------------------
df = df.copy()
df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.fillna(df.median(numeric_only=True), inplace=True)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df_encoded = pd.get_dummies(df, drop_first=True)

# -----------------------------
# KPIs
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", df.shape[0])

with col2:
    st.metric("Churn Rate (%)", round(df["Churn"].mean() * 100, 2))

with col3:
    st.metric("Retention Rate (%)", round((1 - df["Churn"].mean()) * 100, 2))

st.divider()


# -----------------------------
# Model Training
# -----------------------------
X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=2000)
model.fit(X_scaled, y)

st.success("Model trained successfully")

# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("🔍 Key Factors Influencing Churn")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.coef_[0]
})

importance["AbsImportance"] = importance["Importance"].abs()
top_features = importance.sort_values(by="AbsImportance", ascending=False).head(10)

st.bar_chart(top_features.set_index("Feature")["AbsImportance"])

# -----------------------------
# Business Insights
# -----------------------------
st.subheader("💡 Business Insights")
st.markdown("""
- Month-to-month contract customers show higher churn risk  
- Higher monthly charges increase churn probability  
- Long-term customers are more likely to stay  
""")
st.markdown(
    """
    <hr>
    <div style="text-align: center; color: gray;">
        Made by <b>Anshika</b>
    </div>
    """,
    unsafe_allow_html=True
)
