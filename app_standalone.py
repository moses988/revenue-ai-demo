import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from fpdf import FPDF
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
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


# --- NEW: PDF GENERATION LOGIC ---
class AuditReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Business Intelligence Audit Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d")} | Page {self.page_no()}', 0, 0, 'C')

def create_pdf(rfm_df, total_revenue, active_cust, at_risk,at_risk_df):
    pdf = AuditReport()
    pdf.add_page()
    
    # Summary Section
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "Executive Summary", 0, 1, 'L')
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, f"Total Revenue Analyzed: {total_revenue}", 0, 1)
    pdf.cell(0, 8, f"Active Customers: {active_cust}", 0, 1)
    pdf.set_text_color(200, 0, 0)
    pdf.cell(0, 8, f"Clients at High Risk: {at_risk}", 0, 1)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)

    # Priority Call List Table
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "Priority Retention List (Top 10 High Value/Risk)", 0, 1, 'L')
    
    # Table Header
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(60, 8, 'Customer Name', 1)
    pdf.cell(40, 8, 'Status', 1)
    pdf.cell(40, 8, 'Total LTV', 1)
    pdf.cell(30, 8, 'Recency (Days)', 1,1)
    
    
    
    # Table Data
    pdf.set_font('Arial', '', 10)
    priority_list = rfm_df.sort_values(['Total_LTV', 'Recency_Days'], ascending=[False, False]).head(10)
    # for index, row in priority_list.iterrows():
    #     pdf.cell(80, 8, str(row['Customer']), 1)
    #     pdf.cell(50, 8, str(row['Recency_Days']), 1)
    #     pdf.cell(50, 8, f"{row['Total_LTV']:,.0f}", 1, 1)
    for _, row in priority_list.iterrows():
        is_high_risk = row['Churn_Risk_Score'] > 70
        
        if is_high_risk:
            pdf.set_text_color(200, 0, 0)
            status = "!!! FLAG !!!"
        else:
            status = "Stable"
            
        pdf.cell(60, 8, str(row['Customer']), 1)
        pdf.cell(40, 8, status, 1)
        pdf.cell(40, 8, f"{row['Total_LTV']:,.0f}", 1)
        pdf.cell(30, 8, f"{int(row['Recency_Days'])}d", 1, 1)
        pdf.set_text_color(0, 0, 0)    
    # return pdf.output(dest='S')
    # Page 2: Dedicated At-Risk Table
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(200, 0, 0) # Red Heading
    pdf.cell(0, 10, "HIGH-RISK CLIENT RECOVERY LIST", 0, 1, 'L')
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 8, f"The following clients have a Churn Risk Score > 70%:", 0, 1)
    pdf.ln(5)

    # Table Header
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(70, 10, 'Client Name', 1, 0, 'C', True)
    pdf.cell(40, 10, 'Potential Loss', 1, 0, 'C', True)
    pdf.cell(40, 10, 'Days Inactive', 1, 0, 'C', True)
    pdf.cell(30, 10, 'Risk %', 1, 1, 'C', True)

    # Table Rows
    pdf.set_font('Arial', size=10)
    for _, row in at_risk_df.iterrows():
        pdf.cell(70, 8, str(row['Customer']), 1)
        pdf.cell(40, 8, f"INR {row['Total_LTV']:,.0f}", 1)
        pdf.cell(40, 8, f"{int(row['Recency_Days'])} days", 1)
        pdf.cell(30, 8, f"{int(row['Churn_Risk_Score'])}%", 1, 1)
    return bytes(pdf.output(dest='S'))



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

        # PDF Download Button in Sidebar
        total_rev_str = f"INR {rfm['Total_LTV'].sum():,.0f}"
        active_cust = len(rfm)
        high_risk = len(rfm[rfm['Churn_Risk_Score'] > 70])
        at_risk_df = rfm[rfm['Churn_Risk_Score'] > 70].copy()
        at_risk_df = at_risk_df.sort_values('Total_LTV', ascending=False)
        pdf_bytes = create_pdf(rfm, total_rev_str, active_cust, high_risk,at_risk_df)
        st.sidebar.download_button(
            label="📥 Download Audit Report (PDF)",
            data=pdf_bytes,
            file_name=f"Growth_Audit_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
        )
        
        # KEY METRICS (Client Focused)
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Active Customers", len(rfm))
        kpi2.metric("Total Revenue Analyzed", f"₹{rfm['Total_LTV'].sum():,.0f}")
        high_risk = len(rfm[rfm['Churn_Risk_Score'] > 70])
        # kpi3.metric("⚠️ Clients at Risk", high_risk, delta="Needs Attention", delta_color="inverse")
        at_risk_mask = rfm['Churn_Risk_Score'] > 70
        risk_df = rfm[at_risk_mask]
        total_loss_at_risk = risk_df['Total_LTV'].sum()
        kpi3.metric("⚠️ Revenue at Risk", f"₹{total_loss_at_risk:,.0f}", delta=f"{len(risk_df)} Clients", delta_color="inverse")
        st.divider()
        at_risk_df = rfm[rfm['Churn_Risk_Score'] > 70].copy()
        at_risk_df = at_risk_df.sort_values('Total_LTV', ascending=False)
        total_loss = at_risk_df['Total_LTV'].sum()
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
                # st.subheader("🚨 Priority Call List")
                # st.caption("High Value customers who haven't ordered recently.")
                # st.dataframe(
                #     rfm.sort_values(['Total_LTV', 'Recency_Days'], ascending=[False, False])
                #     .head(10)[['Customer', 'Recency_Days']],
                #     hide_index=True
                # )
                st.subheader("🚨 Priority Call List")
                display_df = rfm.sort_values(['Total_LTV', 'Recency_Days'], ascending=[False, False]).head(50).copy()
                
                # Create the Flag
                display_df['Risk_Flag'] = display_df['Churn_Risk_Score'].apply(
                    lambda x: "🚩 HIGH RISK" if x > 70 else "✅ STABLE"
                )
                
                st.dataframe(
                    display_df[['Customer', 'Total_LTV', 'Recency_Days' , 'Risk_Flag']],
                    column_config={
                        "Total_LTV": st.column_config.NumberColumn("Potential Loss", format="₹%d"),
                        "Risk_Flag": "Status"
                    },
                    hide_index=True,
                    use_container_width=True
                )

            st.divider()
            st.subheader("🚩 At-Risk Client Details")
            if not at_risk_df.empty:
                st.dataframe(
                    at_risk_df[['Customer', 'Total_LTV', 'Recency_Days', 'Churn_Risk_Score']],
                    column_config={
                        "Total_LTV": st.column_config.NumberColumn("Potential Loss", format="₹%d"),
                        "Churn_Risk_Score": st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=100)
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.success("No high-risk clients detected.")    
                
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
