import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIGURATION (Neutral & Professional) ---
st.set_page_config(
    page_title="Growth Intelligence Dashboard", 
    page_icon="📈", 
    layout="wide"
)

# --- 1. INTELLIGENCE ENGINE (The Logic) ---

@st.cache_resource
def process_data_and_train(df):
    """
    Analyzes Client Data securely.
    """
    df = df.copy()
    
    # Smart Column Detection
    cols = df.columns
    date_col = next((c for c in cols if 'Date' in c or 'Time' in c), None)
    cust_col = next((c for c in cols if 'Customer' in c or 'Name' in c), None)
    amt_col = next((c for c in cols if 'Total' in c or 'Amount' in c or 'Price' in c or 'Value' in c), None)
    prod_col = next((c for c in cols if 'Product' in c or 'Item' in c or 'Description' in c or 'SKU' in c), None)
    
    if not date_col or not cust_col or not amt_col:
        return None, f"❌ Data Error: Please ensure CSV has Date, Customer, and Amount columns."
    
    # Cleaning
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col]) 
    df[amt_col] = pd.to_numeric(df[amt_col], errors='coerce').fillna(0)
    
    # RFM Analysis (Retention Logic)
    snapshot_date = df[date_col].max() + pd.Timedelta(days=1)
    rfm = df.groupby(cust_col).agg({
        date_col: lambda x: (snapshot_date - x.max()).days,
        amt_col: ['count', 'sum', 'mean']
    }).reset_index()
    rfm.columns = ['Customer', 'Recency_Days', 'Frequency_Count', 'Total_LTV', 'Avg_Order_Value']
    
    # Risk Calculation
    rfm['Churn_Risk_Score'] = (rfm['Recency_Days'] / rfm['Recency_Days'].max()) * 100
    
    return {
        'df_clean': df,
        'rfm_data': rfm,
        'cols': {'cust': cust_col, 'prod': prod_col}
    }, None

def generate_cross_sell(df, cust_col, prod_col):
    """
    Market Basket Analysis (Expansion Logic) - SAFE VERSION
    """
    if not prod_col or df.empty: return pd.DataFrame()
    if df[prod_col].nunique() < 2: return pd.DataFrame()
        
    top_products = df[prod_col].value_counts().head(50).index
    df_top = df[df[prod_col].isin(top_products)]
    
    basket = pd.crosstab(df_top[cust_col], df_top[prod_col])
    basket = (basket > 0).astype(int)
    
    if basket.shape[1] < 2: return pd.DataFrame()

    cooc = basket.T.dot(basket)
    opportunities = []
    
    for product_A in cooc.columns:
        correlations = cooc[product_A].sort_values(ascending=False)
        if len(correlations) > 1:
            top_match = correlations.index[1]
            targets = basket[(basket[product_A] == 1) & (basket[top_match] == 0)].index.tolist()
            if len(targets) > 0:
                opportunities.append({
                    "Customer Bought": product_A,
                    "Likely Needs": top_match,
                    "Missed Opportunities": len(targets),
                    "Target Customers": ", ".join(str(x) for x in targets[:3])
                })
    
    if not opportunities: return pd.DataFrame()
    return pd.DataFrame(opportunities).sort_values('Missed Opportunities', ascending=False)

# --- 2. THE CLIENT DASHBOARD ---

st.sidebar.title("📊 Partner Portal")
st.sidebar.info("Upload Sales Data (CSV) to unlock insights.")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

st.title("Business Intelligence Overview")
st.markdown("### Revenue Recovery & Growth System")

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    with st.spinner("Analyzing purchase patterns..."):
        data_bundle, error = process_data_and_train(raw_df)
        
    if error:
        st.error(error)
    else:
        df = data_bundle['df_clean']
        rfm = data_bundle['rfm_data']
        cols = data_bundle['cols']
        
        # KEY METRICS (Client Focused)
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Active Customers", len(rfm))
        kpi2.metric("Total Revenue Analyzed", f"₹{rfm['Total_LTV'].sum():,.0f}")
        high_risk = len(rfm[rfm['Churn_Risk_Score'] > 70])
        kpi3.metric("⚠️ Clients at Risk", high_risk, delta="Needs Attention", delta_color="inverse")
        
        st.divider()
        
        tab1, tab2 = st.tabs(["🛡️ Retention (Save Money)", "🚀 Expansion (Make Money)"])
        
        with tab1:
            col1, col2 = st.columns([2,1])
            with col1:
                st.subheader("Customer Health Matrix")
                chart = alt.Chart(rfm).mark_circle(size=100).encode(
                    x=alt.X('Total_LTV', title='Customer Value'),
                    y=alt.Y('Recency_Days', title='Days Since Last Order'),
                    color=alt.Color('Churn_Risk_Score', scale=alt.Scale(scheme='redyellowgreen', reverse=True)),
                    tooltip=['Customer', 'Recency_Days', 'Total_LTV']
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
            with col2:
                st.subheader("🚨 Priority Call List")
                st.caption("High Value customers who haven't ordered recently.")
                st.dataframe(
                    rfm.sort_values(['Total_LTV', 'Recency_Days'], ascending=[False, False])
                    .head(10)[['Customer', 'Recency_Days']],
                    hide_index=True
                )
                
        with tab2:
            st.subheader("📦 Hidden Cross-Sell Opportunities")
            if cols['prod']:
                upsell_df = generate_cross_sell(df, cols['cust'], cols['prod'])
                if not upsell_df.empty:
                    st.dataframe(upsell_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No strong product correlations found yet.")
            else:
                st.warning("Upload a file with 'Product' or 'Item' columns to see Upsell data.")

else:
    st.info("👋 Welcome. Please upload the sales data to begin the audit.")
