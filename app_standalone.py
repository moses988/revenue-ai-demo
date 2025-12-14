import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="Revenue Intelligence AI", page_icon="📈", layout="wide")

# --- 1. AI ENGINE (Integrated directly for easy hosting) ---
@st.cache_resource
def train_model(df):
    """Trains the AI on the uploaded data instantly."""
    # Simple Cleaning
    df = df.fillna(0)
    
    # Feature Engineering (Simplified for Generic Data)
    # We look for common columns or create dummy ones
    if 'Month' not in df.columns:
        df['Month'] = np.random.randint(1, 13, size=len(df))
    if 'Quarter' not in df.columns:
        df['Quarter'] = df['Month'].apply(lambda x: (x-1)//3 + 1)
        
    # Encoder for Customer Names
    le = LabelEncoder()
    # Assume first string column is Customer Name if not explicit
    cust_col = [c for c in df.columns if 'Customer' in c or 'Name' in c][0]
    df['CustomerID_Encoded'] = le.fit_transform(df[cust_col].astype(str))
    
    # Target: Try to find a 'Total', 'Amount', or 'DealValue' column
    target_col = [c for c in df.columns if 'Total' in c or 'Amount' in c or 'Price' in c][0]
    
    X = df[['Month', 'Quarter', 'CustomerID_Encoded']]
    y = df[target_col]
    
    # Train Model
    model = HistGradientBoostingRegressor()
    model.fit(X, y)
    
    return model, le, cust_col, target_col

# --- 2. THE DASHBOARD ---
st.title("🤖 Enterprise Revenue Intelligence System")
st.markdown("### Upload your Sales Data (CSV) to generate AI predictions.")

uploaded_file = st.file_uploader("Upload Invoice Data CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Data Loaded: {len(df)} invoices detected.")
        
        # Train AI Live
        with st.spinner("Training Custom AI Model on your data..."):
            model, le, cust_col, target_col = train_model(df)
        
        st.divider()
        
        # --- PREDICTION INTERFACE ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔮 Predict Future Spending")
            # Get unique customers
            unique_customers = df[cust_col].unique()
            selected_customer = st.selectbox("Select a Customer to Analyze:", unique_customers)
            
            # Inputs
            selected_month = st.slider("Forecast Month:", 1, 12, 6)
            
            if st.button("Generate Revenue Forecast"):
                # Prepare Input
                cust_id = le.transform([selected_customer])[0]
                quarter = (selected_month-1)//3 + 1
                input_data = [[selected_month, quarter, cust_id]]
                
                # Predict
                prediction = model.predict(input_data)[0]
                
                st.metric(label=f"Predicted Spend for {selected_customer}", 
                          value=f"₹{prediction:,.2f}")
                
                if prediction > df[df[cust_col] == selected_customer][target_col].mean():
                    st.success("🚀 Insight: This customer is expected to spend ABOVE their average.")
                else:
                    st.warning("⚠️ Insight: Churn Risk? Predicted spend is LOWER than average.")

        with col2:
            st.subheader("📊 Historical Analysis")
            cust_data = df[df[cust_col] == selected_customer]
            st.line_chart(cust_data[target_col].reset_index(drop=True))
            st.caption("Past Transaction History")

    except Exception as e:
        st.error(f"Error processing data: {e}")
        st.info("Make sure your CSV has columns for 'Customer Name' and 'Total Amount'.")

else:
    st.info("👆 Upload the 'real_sales_data.csv' file to test the demo.")