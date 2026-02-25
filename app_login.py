# =============================================================================
#   PROFITGUARD AI - FULL DEPLOYMENT VERSION (Supabase + Streamlit)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from fpdf import FPDF
from sklearn.ensemble import HistGradientBoostingRegressor
from datetime import datetime, timedelta
import vl_convert as vlc
import bcrypt
from sqlalchemy import text  # Required for SQL params
import re

import smtplib
from email.mime.text import MIMEText
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from tempfile import NamedTemporaryFile



def send_email_alert(name, company, phone, category):
    sender_email = st.secrets["email"]["address"]
    sender_password = st.secrets["email"]["password"] # Paste the 16-digit App Password
    receiver_email = st.secrets["email"]["address"]  # Send to yourself

    msg_body = f"""
    NEW USER SIGNUP:
    Name: {name}
    Company: {company}
    Phone: {phone}
    Category: {category}
    """
    
    msg = MIMEText(msg_body)
    msg['Subject'] = f"🚀 New User: {name}"
    msg['From'] = sender_email
    msg['To'] = receiver_email

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"Email Error: {e}")
        return False
# ────────────────────────────────────────────────
#  1. SECURE DATABASE CONNECTION (Supabase/PostgreSQL)
# ────────────────────────────────────────────────

def get_db_connection():
    """Connects to Supabase using Streamlit Secrets."""
    # Ensure you have set [connections.supabase] in your secrets.toml
    return st.connection("supabase", type="sql")

def validate_email(email):
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(pattern, email)

def save_user(email, password, name, company, phone, category, tier="free"):
    """Creates a new user in the SQL database."""
    conn = get_db_connection()
    
    # 1. Validation
    if not validate_email(email):
        return False, "⚠️ Invalid email address format."
    if len(password) < 6:
        return False, "⚠️ Password must be at least 6 characters."

    # 2. Check if email exists
    try:
        existing = conn.query("SELECT email FROM users WHERE email = :email;", params={"email": email}, ttl=0)
        if not existing.empty:
            return False, "⚠️ Email already exists. Please login."
    except Exception as e:
        return False, f"Database Error: {str(e)}"

    # 3. Hash Password (Bcrypt)
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    # 4. Insert User securely
    try:
        with conn.session as session:
            session.execute(
                text("""
                    INSERT INTO users (email, password_hash, name, company, phone, category, tier, signup_date)
                    VALUES (:email, :pwd, :name, :comp, :phone, :cat, :tier, :date);
                """),
                {
                    "email": email,
                    "pwd": hashed_pw,
                    "name": name,
                    "comp": company,
                    "phone": phone,
                    "cat": category,
                    "tier": tier,
                    "date": datetime.now()
                }
            )
            session.commit()
            send_email_alert(name, company, phone, category)
            # REDIRECT LOGIC
            # time.sleep(2)  # Wait 2 seconds so user sees the message
            # st.session_state.auth_mode = "Login"  # Switch mode back to Login
            # st.rerun()  # Reload the app to show the Login screen
        return True, "✅ Account created! Please login."
    except Exception as e:
        return False, f"Registration Failed: {str(e)}"

def check_login(email, password):
    """Verifies user credentials."""
    conn = get_db_connection()
    try:
        user_df = conn.query("SELECT * FROM users WHERE email = :email;", params={"email": email}, ttl=0)
        
        if user_df.empty:
            return False, None
        
        stored_hash = user_df.iloc[0]['password_hash']
        
        # Verify Password
        if bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
            return True, user_df.iloc[0].to_dict()
        else:
            return False, None
    except Exception as e:
        st.error(f"Login Error: {str(e)}")
        return False, None

# ────────────────────────────────────────────────
#  2. BUSINESS LOGIC (Tiers & Calculations)
# ────────────────────────────────────────────────

TIERS = {
    "free": {
        "title": "Free Revenue Audit (Preview)",
        "max_customers": 10,
        "allow_clv": False,
        "allow_cross_sell": False,
        "msg": "🔒 Showing Top 10 Only. Upgrade for Full List."
    },
    "tier1": {
        "title": "Audit Pro (₹25k/mo)",
        "max_customers": None,  # Unlimited
        "allow_clv": True,
        "allow_cross_sell": True,
        "msg": "✅ Full Access Active"
    },
    "tier2": {
        "title": "Growth Partner (₹75k/mo)",
        "max_customers": None,
        "allow_clv": True,
        "allow_cross_sell": True,
        "msg": "🚀 Enterprise Mode Active"
    }
}

# Codes to upgrade users instantly (bypass payment for now)
CODE_MAP = {
    "audit25": "tier1", 
    "growth75": "tier2"
}

@st.cache_resource
def process_data(df):
    """Core Logic: RFM Analysis & Data Cleaning"""
    df = df.copy()
    # Auto-detect columns
    cols = df.columns
    date_col = next((c for c in cols if 'Date' in c or 'Time' in c), None)
    cust_col = next((c for c in cols if 'Customer' in c or 'Name' in c), None)
    amt_col  = next((c for c in cols if 'Total' in c or 'Amount' in c or 'Value' in c), None)
    prod_col = next((c for c in cols if 'Product' in c or 'Item' in c or 'SKU' in c), None)

    if not all([date_col, cust_col, amt_col]):
        return None, "❌ Data Error: CSV must have Date, Customer, and Amount columns."

    # Data Type Conversion
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df[amt_col] = pd.to_numeric(df[amt_col], errors='coerce').fillna(0)

    # RFM Calculation
    snapshot_date = df[date_col].max() + timedelta(days=1)
    rfm = df.groupby(cust_col).agg({
        date_col: lambda x: (snapshot_date - x.max()).days,
        amt_col: ['count', 'sum']
    }).reset_index()
    rfm.columns = ['Customer', 'Recency_Days', 'Frequency', 'Total_LTV']
    
    # Simple Churn Score (0-100)
    rfm['Churn_Risk'] = (rfm['Recency_Days'] / rfm['Recency_Days'].max()) * 100
    
    return {
        'df': df, 'rfm': rfm, 
        'cols': {'d': date_col, 'c': cust_col, 'a': amt_col, 'p': prod_col}
    }, None

# def get_predictive_clv(df, customer_id_col, date_col, amt_col):
#     """
#     Train BG/NBD and Gamma-Gamma models to predict future purchasing behavior.
#     """
#     # 1. Filter out non-positive data (returns/errors)
#     df = df[df[amt_col] > 0]

#     # 2. Transform data into RFM format required by lifetimes
#     # (frequency, recency, T, monetary_value)
#     data = summary_data_from_transaction_data(
#         df, customer_id_col, date_col, 
#         monetary_value_col=amt_col, 
#         observation_period_end=df[date_col].max()
#     )
    
#     # 3. Fit BG/NBD Model (Predicts Frequency & Churn)
#     bgf = BetaGeoFitter(penalizer_coef=0.01)
#     bgf.fit(data['frequency'], data['recency'], data['T'])
    
#     # 4. Predict expected transactions in the next 30 days
#     data['predicted_purchases_30d'] = bgf.conditional_expected_number_of_purchases_up_to_time(
#         30, data['frequency'], data['recency'], data['T']
#     )
    
#     # 5. Fit Gamma-Gamma Model (Predicts Average Order Value)
#     # We only use customers with at least one repeat purchase for this
#     returning_customers = data[data['frequency'] > 0]
    
#     if len(returning_customers) > 0:
#         ggf = GammaGammaFitter(penalizer_coef=0.01)
#         ggf.fit(returning_customers['frequency'], returning_customers['monetary_value'])
        
#         # Calculate Predicted CLV (Customer Lifetime Value)
#         data['predicted_clv'] = ggf.customer_lifetime_value(
#             bgf, data['frequency'], data['recency'], data['T'], 
#             data['monetary_value'], time=1, discount_rate=0.01 # time=1 means 1 month
#         )
#     else:
#         data['predicted_clv'] = 0

#     return data.sort_values(by='predicted_purchases_30d', ascending=False)

def get_predictive_clv(df, customer_id_col, date_col, amt_col):
    """
    Train BG/NBD and Gamma-Gamma models to predict future purchasing behavior
    with an auto-scaling penalizer to prevent ConvergenceErrors.
    """
    # 1. Filter out non-positive data (returns/errors)
    df = df[df[amt_col] > 0]

    # 2. Transform data into RFM format required by lifetimes
    data = summary_data_from_transaction_data(
        df, customer_id_col, date_col, 
        monetary_value_col=amt_col, 
        observation_period_end=df[date_col].max()
    )
    
    if len(data) == 0:
        return pd.DataFrame()

    # List of penalizers to try, from smallest to largest
    penalizers_to_try = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    
    # 3. Fit BG/NBD Model (Predicts Frequency & Churn)
    bgf = None
    for p in penalizers_to_try:
        try:
            temp_bgf = BetaGeoFitter(penalizer_coef=p)
            temp_bgf.fit(data['frequency'], data['recency'], data['T'])
            bgf = temp_bgf
            break  # Success! Exit the loop.
        except Exception:
            continue  # Failed, try the next higher penalizer
            
    if bgf is None:
        raise RuntimeError("Dataset lacks enough repeat purchase patterns for AI modeling. Please upload a larger dataset.")

    # 4. Predict expected transactions in the next 30 days
    data['predicted_purchases_30d'] = bgf.conditional_expected_number_of_purchases_up_to_time(
        30, data['frequency'], data['recency'], data['T']
    )
    
    # 5. Fit Gamma-Gamma Model (Predicts Average Order Value)
    returning_customers = data[data['frequency'] > 0]
    
    if len(returning_customers) > 0:
        ggf = None
        for p in penalizers_to_try:
            try:
                temp_ggf = GammaGammaFitter(penalizer_coef=p)
                temp_ggf.fit(returning_customers['frequency'], returning_customers['monetary_value'])
                ggf = temp_ggf
                break # Success!
            except Exception:
                continue
                
        if ggf is not None:
            # Calculate Predicted CLV
            data['predicted_clv'] = ggf.customer_lifetime_value(
                bgf, data['frequency'], data['recency'], data['T'], 
                data['monetary_value'], time=1, discount_rate=0.01 
            )
        else:
            data['predicted_clv'] = 0  # Fallback if Gamma-Gamma completely fails
    else:
        data['predicted_clv'] = 0

    return data.sort_values(by='predicted_purchases_30d', ascending=False)

def get_aggregate_forecast(df, date_col, amt_col):
    """Predicts total revenue for the next month using Gradient Boosting."""
    # Resample to monthly revenue
    monthly = df.set_index(date_col).resample('ME')[amt_col].sum().reset_index()
    
    if len(monthly) < 3: 
        return None # Not enough data points
        
    # Prepare data for simple regression (X=Month Index, Y=Revenue)
    X = np.arange(len(monthly)).reshape(-1, 1)
    y = monthly[amt_col]
    
    model = HistGradientBoostingRegressor()
    model.fit(X, y)
    
    # Predict next month (Index = len)
    next_month_pred = model.predict([[len(monthly)]])[0]
    return max(0, next_month_pred) # Ensure no negative revenue

def get_seasonality_analysis(df, date_col, amt_col):
    """Identifies strongest and weakest months based on historical averages."""
    df = df.copy()
    df['Month'] = df[date_col].dt.month_name()
    df['Month_Num'] = df[date_col].dt.month
    
    # Group by Month Name and calculate average revenue
    seasonal = df.groupby(['Month_Num', 'Month'])[amt_col].sum().reset_index()
    seasonal = seasonal.sort_values('Month_Num')
    
    if seasonal.empty:
        return None, None, None
        
    strongest = seasonal.loc[seasonal[amt_col].idxmax()]
    weakest = seasonal.loc[seasonal[amt_col].idxmin()]
    
    return seasonal, strongest, weakest

@st.cache_resource
def get_cohort_analysis(df, cust_col, date_col):
    """
    Creates a Cohort Analysis Heatmap to track retention over time.
    """
    df = df.copy()
    
    # 1. Convert Date to Month Period (e.g., 2023-01)
    df['OrderPeriod'] = df[date_col].dt.to_period('M')
    
    # 2. Determine the user's "Cohort" (Month of first purchase)
    df['Cohort'] = df.groupby(cust_col)[date_col].transform('min').dt.to_period('M')
    
    # 3. Group data
    cohort_data = df.groupby(['Cohort', 'OrderPeriod']).agg(n_customers=(cust_col, 'nunique')).reset_index()
    
    # 4. Calculate "Cohort Index" (Months since first purchase)
    # Result is integer: 0 (first month), 1, 2, etc.
    cohort_data['PeriodNumber'] = (cohort_data.OrderPeriod - cohort_data.Cohort).apply(lambda x: x.n)
    
    # 5. Pivot the table
    cohort_pivot = cohort_data.pivot_table(index='Cohort', columns='PeriodNumber', values='n_customers')
    
    # 6. Calculate Retention Percentage
    cohort_size = cohort_pivot.iloc[:, 0]
    retention_matrix = cohort_pivot.divide(cohort_size, axis=0)
    
    return retention_matrix

# ────────────────────────────────────────────────
#  MARKET BASKET ANALYSIS (Upsell Logic)
# ────────────────────────────────────────────────

@st.cache_resource
def generate_cross_sell(df, cust_col, prod_col):
    """
    Identifies products frequently bought together to generate upsell recommendations.
    """
    # 1. Safety Checks
    if not prod_col or df.empty: 
        return None, "⚠️ No 'Product' column detected in your CSV."
    
    # 2. Filter for Top 50 Products (Speed Optimization)
    # We focus on top items to prevent memory crashes with huge catalogs
    top_products = df[prod_col].value_counts().head(50).index
    df_top = df[df[prod_col].isin(top_products)]
    
    if df_top[prod_col].nunique() < 2:
        return None, "⚠️ Not enough different products to find patterns."

    # 3. Create Basket Matrix (Customer x Product)
    basket = pd.crosstab(df_top[cust_col], df_top[prod_col])
    basket = (basket > 0).astype(int) # Convert to 1s and 0s
    
    if basket.shape[1] < 2: 
        return None, "⚠️ Not enough data to calculate correlations."

    # 4. Calculate Co-occurrence Matrix
    cooc = basket.T.dot(basket)
    opportunities = []
    
    # 5. Find Missed Opportunities
    for product_A in cooc.columns:
        # Find the product most correlated with Product A
        correlations = cooc[product_A].sort_values(ascending=False)
        
        if len(correlations) > 1:
            top_match = correlations.index[1] # Index 0 is the product itself
            
            # Find customers who bought A but NOT the match
            targets = basket[(basket[product_A] == 1) & (basket[top_match] == 0)].index.tolist()
            
            if len(targets) > 0:
                opportunities.append({
                    "If they bought...": product_A,
                    "They likely need...": top_match,
                    "Potential Sales": len(targets),
                    "Call These Customers": ", ".join(str(x) for x in targets[:3]) # Show top 3 examples
                })
    
    if not opportunities: 
        return None, "No strong correlations found yet."
        
    return pd.DataFrame(opportunities).sort_values('Potential Sales', ascending=False), None

def render_scenario_simulator(rfm_df):
    st.subheader("🧪 Strategy Simulator")
    st.markdown("Simulate how small improvements impact your bottom line.")
    
    # 1. User Inputs
    col1, col2 = st.columns(2)
    with col1:
        churn_reduction = st.slider("📉 If we reduce Churn by...", 0, 50, 10, format="%d%%")
    with col2:
        upsell_increase = st.slider("📈 If we increase Avg Order Value by...", 0, 50, 5, format="%d%%")
        
    # 2. Calculations
    current_rev = rfm_df['Total_LTV'].sum()
    
    # Revenue recovered from churn (Simplified: Assuming we save X% of at-risk revenue)
    at_risk_rev = rfm_df[rfm_df['Churn_Risk'] > 70]['Total_LTV'].sum()
    saved_rev = at_risk_rev * (churn_reduction / 100)
    
    # Revenue gained from upsell
    new_rev_base = current_rev + saved_rev
    upsell_gain = new_rev_base * (upsell_increase / 100)
    
    total_projected = current_rev + saved_rev + upsell_gain
    net_gain = total_projected - current_rev
    
    # 3. Visualization
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Current Revenue", f"₹{current_rev:,.0f}")
    m2.metric("Projected Revenue", f"₹{total_projected:,.0f}", delta=f"+{churn_reduction + upsell_increase}% Growth")
    m3.metric("💰 Net Profit Increase", f"₹{net_gain:,.0f}", delta="Extra Cash")
    
    # Simple Chart
    chart_data = pd.DataFrame({
        'Scenario': ['Current', 'With Strategy'],
        'Revenue': [current_rev, total_projected]
    })
    st.bar_chart(chart_data, x='Scenario', y='Revenue', color='#00CC96')


# --- CHART 1: SEASONALITY (Bar Chart) ---
def create_seasonality_chart(seasonal_data):
    if seasonal_data is None or seasonal_data.empty: return None
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 4))
    ax = sns.barplot(x='Month', y='Total_LTV', data=seasonal_data, color='#2196F3')
    plt.title('Monthly Revenue Trends', fontsize=12, fontweight='bold', color='#333333')
    plt.xlabel(''); plt.ylabel('')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    sns.despine(left=True, bottom=True)
    
    temp_file = NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_file.name, bbox_inches='tight', dpi=150)
    plt.close()
    return temp_file.name

# --- CHART 2: RISK DISTRIBUTION (Donut Chart) ---
def create_risk_chart(total_rev, risk_rev):
    safe_rev = total_rev - risk_rev
    labels = ['Safe Revenue', 'At Risk']
    sizes = [safe_rev, risk_rev]
    colors = ['#4CAF50', '#F44336'] # Green, Red
    
    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, pctdistance=0.85, textprops={'fontsize': 12})
    
    # Draw circle for Donut shape
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    plt.title('Revenue Health Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    temp_file = NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_file.name, bbox_inches='tight', dpi=150)
    plt.close()
    return temp_file.name

# --- CHART 3: TOP UPSELL PRODUCTS (Horizontal Bar) ---
def create_upsell_chart(upsell_df):
    if upsell_df is None or upsell_df.empty: return None
    # Prepare Top 5 Data
    top_5 = upsell_df.head(5).copy()
    # Ensure 'Potential Deals' is numeric
    if 'Potential Deals' not in top_5.columns and 'Missed Sales Count' in top_5.columns:
        top_5['Potential Deals'] = top_5['Missed Sales Count']
        
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(x='Potential Deals', y='Needs', data=top_5, palette='viridis')
    
    plt.title('Top 5 Missed Product Opportunities', fontsize=12, fontweight='bold')
    plt.xlabel('Missed Customers')
    plt.ylabel('')
    sns.despine(left=True, bottom=True)
    
    temp_file = NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_file.name, bbox_inches='tight', dpi=150)
    plt.close()
    return temp_file.name

def increment_download_count(email):
    """Updates the download count in DB and Session State."""
    conn = get_db_connection()
    try:
        with conn.session as session:
            session.execute(
                text("UPDATE users SET pdf_downloads = Coalesce(pdf_downloads, 0) + 1 WHERE email = :email;"),
                {"email": email}
            )
            session.commit()
            
        # Update local session immediately so the UI refreshes
        current = st.session_state.user.get('pdf_downloads', 0)
        st.session_state.user['pdf_downloads'] = current + 1
        
    except Exception as e:
        print(f"Error updating count: {e}")

class AuditReport(FPDF):
    def __init__(self, tier):
        super().__init__()
        self.tier = tier

    def header(self):
        if self.page_no() > 1:
            self.set_font('Arial', 'B', 9)
            self.set_text_color(180, 180, 180)
            self.cell(0, 10, 'ProfitGuard AI - Strategic Growth Audit', 0, 0, 'L')
            self.cell(0, 10, f'Page {self.page_no()}', 0, 1, 'R')
            self.line(10, 20, 200, 20)
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Confidential Analysis | Generated by ProfitGuard AI', 0, 0, 'C')

    def draw_locked_box(self, height=40, text="PREMIUM FEATURE LOCKED"):
        x = self.get_x()
        y = self.get_y()
        self.set_fill_color(245, 245, 245) # Light Grey
        self.rect(x, y, 190, height, 'F')
        
        self.set_y(y + (height/2) - 5)
        self.set_font('Arial', 'B', 12)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, f"[LOCKED] {text}", 0, 1, 'C')
        
        self.set_font('Arial', '', 9)
        self.ln(5)
        self.cell(0, 10, "Upgrade to Audit Pro to unlock this data.", 0, 1, 'C')
        self.set_y(y + height + 10) # Reset cursor below box

    # def add_watermark(self):
    #     if self.tier == 'free':
    #         self.set_font('Arial', 'B', 60)
    #         self.set_text_color(240, 240, 240)
    #         with self.rotation(45, 105, 148):
    #             self.text(40, 190, "FREE PREVIEW")
    #         self.set_text_color(0, 0, 0) # Reset
    #         self.set_font('Arial', '', 10) # Reset
    def add_watermark(self):
        if self.tier == 'free':
            self.set_font('Arial', 'B', 25) # Size of the repeating text
            self.set_text_color(200, 200, 200) # Light Grey (Visible but readable)
            
            # Loop vertically and horizontally to cover the A4 page (approx 210x297mm)
            # We step by 80mm to create a grid pattern
            for x_pos in range(0, 250, 80):
                for y_pos in range(0, 350, 80):
                    # Rotate 45 degrees around the current position (x_pos, y_pos)
                    with self.rotation(45, x_pos, y_pos):
                        self.text(x_pos, y_pos, "FREE PREVIEW")
            
            # Reset colors/fonts for the actual report content
            self.set_text_color(0, 0, 0)
            self.set_font('Arial', '', 10)

def create_professional_pdf(rfm_df, upsell_df, seasonal_data, forecast_val, user_name, company_name, tier):
    pdf = AuditReport(tier)
    pdf.set_auto_page_break(auto=True, margin=15)
    risk_value = rfm_df[rfm_df['Churn_Risk'] > 75]['Total_LTV'].sum()
    
    # B. Calculate Upsell Value (Money you find them)
    # We estimate 'Potential Deals' * 'Average Ticket Size' (Approx ₹5,000 per deal if unknown)
    upsell_value = 0
    if upsell_df is not None and not upsell_df.empty:
        # Sum of (Potential Deals * Estimated Unit Price)
        # Assuming average order value is roughly total_revenue / total_transactions
        avg_order_val = rfm_df['Total_LTV'].sum() / rfm_df['Frequency'].sum() if rfm_df['Frequency'].sum() > 0 else 2000
        
        # Check if 'Potential Deals' column exists (it might be named 'Missed Sales Count')
        count_col = 'Potential Deals' if 'Potential Deals' in upsell_df.columns else 'Missed Sales Count'
        
        if count_col in upsell_df.columns:
            total_missed_deals = upsell_df[count_col].sum()
            upsell_value = total_missed_deals * avg_order_val

    # C. TOTAL VALUE
    total_audit_value = risk_value + upsell_value
    # --- PAGE 1: COVER PAGE (Perfectly Centered) ---
    pdf.add_page()
    pdf.set_fill_color(24, 33, 47) # Navy Blue
    pdf.rect(0, 0, 210, 297, 'F') 
    
    pdf.ln(90)
    pdf.set_font('Arial', 'B', 36)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 15, 'REVENUE', 0, 1, 'C')
    pdf.cell(0, 15, 'INTELLIGENCE AUDIT', 0, 1, 'C')
    
    pdf.ln(20)
    pdf.set_font('Arial', '', 12)
    pdf.set_text_color(200, 200, 200)
    # Ensure company name is a string to prevent errors
    comp_str = str(company_name).upper() if company_name else "VALUED CLIENT"
    pdf.cell(0, 10, f'PREPARED FOR: {comp_str}', 0, 1, 'C')
    pdf.cell(0, 10, f'DATE: {datetime.now().strftime("%B %d, %Y")}', 0, 1, 'C')
    # --- NEW: DISPLAY THE MONEY VALUE ---
    pdf.ln(20)
    pdf.set_fill_color(46, 125, 50) # Green Box
    pdf.set_draw_color(255, 255, 255)
    
    # Center the box (approx width 120)
    x_pos = (210 - 140) / 2
    y_pos = pdf.get_y()
    
    pdf.rect(x_pos, y_pos, 140, 35, 'FD') # Filled with border
    
    pdf.set_y(y_pos + 5)
    pdf.set_font('Arial', 'B', 10)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "TOTAL IDENTIFIED OPPORTUNITY", 0, 1, 'C')
    
    pdf.set_font('Arial', 'B', 22)
    pdf.cell(0, 15, f"INR {total_audit_value:,.0f}", 0, 1, 'C')
    # ------------------------------------
    if tier == 'free':
        pdf.ln(10)
        pdf.set_text_color(255, 100, 100)
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "[ FREE PREVIEW MODE ]", 0, 1, 'C')

    # --- PAGE 2: EXECUTIVE SUMMARY (Side-by-Side Alignment Fixed) ---
    pdf.add_page()
    # pdf.add_watermark()
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', 'B', 18)
    pdf.cell(0, 2.5, '1. Executive Health Score', 0, 1)
    pdf.line(10, 30, 200, 30)
    pdf.ln(10)
    
    # -- METRICS & CHART LAYOUT --
    total_rev = rfm_df['Total_LTV'].sum()
    risk_rev = rfm_df[rfm_df['Churn_Risk'] > 75]['Total_LTV'].sum()
    
    # Capture starting Y position for side-by-side alignment
    start_y = pdf.get_y()
    
    # 1. LEFT COLUMN: Text Metrics (Width = 100mm)
    pdf.set_left_margin(10)
    pdf.set_font('Arial', '', 11)
    
    # Metric 1
    pdf.set_fill_color(240, 248, 255) # Light Blue
    pdf.cell(90, 10, "  Total Revenue Analyzed", 0, 1, 'L', True)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(90, 10, f"  INR {total_rev:,.0f}", 0, 1, 'L', True)
    pdf.ln(4)
    
    # Metric 2
    pdf.set_font('Arial', '', 11)
    pdf.cell(90, 10, "  Revenue at Churn Risk", 0, 1, 'L', True)
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(200, 0, 0) # Red Text
    pdf.cell(90, 10, f"  INR {risk_rev:,.0f}", 0, 1, 'L', True)
    pdf.set_text_color(0, 0, 0)
    
    # 2. RIGHT COLUMN: Donut Chart (Width = 90mm)
    # Reset Y to top, Move X to 110
    pdf.set_y(start_y) 
    pdf.set_x(110)
    
    try:
        risk_chart = create_risk_chart(total_rev, risk_rev)
        if risk_chart:
            # Place image at X=115 to centre it in the right column
            pdf.image(risk_chart, x=115, y=start_y, w=75) 
    except:
        pdf.cell(80, 40, "Chart Error", 1, 1, 'C')

    # Reset Cursor to below the chart/metrics
    pdf.set_y(start_y + 85) # Move down explicitly (Chart height approx 80)
    pdf.set_x(10) # Reset margin
    
    # 2. Forecast (Locked if Free)
    pdf.set_font('Arial', '', 11)
    pdf.cell(95, 12, "  Predicted Next Month Revenue", 1, 0, 'L', True)
    pdf.set_font('Arial', 'B', 11)
    
    if tier == 'free':
        pdf.set_text_color(150, 150, 150)
        pdf.cell(95, 12, " LOCKED ", 1, 1, 'R', True)
        pdf.set_text_color(0, 0, 0)
    else:
        val = f" INR {forecast_val:,.0f}" if forecast_val else " Not Enough Data"
        pdf.cell(95, 12, val, 1, 1, 'R', True)
    # pdf.ln(14)
    
    # 3. Risk
    pdf.set_font('Arial', '', 11)
    pdf.set_fill_color(255, 235, 238) # Red background
    pdf.set_text_color(200, 0, 0)
    pdf.cell(95, 12, "  Revenue at Churn Risk", 1, 0, 'L', True)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(95, 12, f" INR {risk_rev:,.0f}", 1, 1, 'R', True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(20)

    # Seasonality Section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Seasonal Revenue Strategy', 0, 1)
    pdf.ln(2)
    
    if tier == 'free':
        pdf.draw_locked_box(height=50, text="SEASONALITY CHART LOCKED")
    else:
        # PAID TIER: SHOW CHART + TEXT
        pdf.set_font('Arial', '', 10)
        if seasonal_data is not None:
            best_mo = seasonal_data.loc[seasonal_data['Total_LTV'].idxmax()]['Month'] 
            worst_mo = seasonal_data.loc[seasonal_data['Total_LTV'].idxmin()]['Month']
            pdf.multi_cell(0, 6, f"Insight: Your sales peak in {best_mo} and dip in {worst_mo}. Use this chart to plan inventory.")
            pdf.ln(5)
            
            # INSERT CHART
            try:
                chart_path = create_seasonality_chart(seasonal_data)
                if chart_path:
                    pdf.image(chart_path, x=15, w=180)
                    pdf.ln(5)
            except Exception as e:
                pdf.cell(0, 10, f"Chart could not be generated: {e}", 0, 1)
        else:
            pdf.cell(0, 10, "Insufficient data for seasonality.", 0, 1)

    # --- PAGE 3: HIGH RISK CLIENTS (Table Alignment Fixed) ---
    pdf.add_watermark()
    pdf.add_page()
    # pdf.add_watermark()
    
    pdf.set_font('Arial', 'B', 18)
    pdf.cell(0, 2.5, '2. Retention Alert (High Risk)', 0, 1)
    pdf.line(10, 30, 200, 30)
    pdf.ln(10)
    
    # Table Header Definition (Total Width = 190mm)
    w_name = 85
    w_ltv = 40
    w_days = 30
    w_status = 35
    
    pdf.set_font('Arial', 'B', 10)
    pdf.set_fill_color(200, 50, 50)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(w_name, 10, 'Customer Name', 1, 0, 'L', True)
    pdf.cell(w_ltv, 10, 'LTV (INR)', 1, 0, 'C', True)
    pdf.cell(w_days, 10, 'Days', 1, 0, 'C', True)
    pdf.cell(w_status, 10, 'Status', 1, 1, 'C', True)
    
    pdf.set_font('Arial', '', 9)
    pdf.set_text_color(0, 0, 0)
    # pdf.ln(10)
    
    # Table Rows
    risk_df = rfm_df[rfm_df['Churn_Risk'] > 75].sort_values('Total_LTV', ascending=False)
    if tier == 'free': risk_df = risk_df.head(10)
    else: risk_df = risk_df.head(25)

    for _, row in risk_df.iterrows():
        # Truncate strings to prevent overlap
        name = str(row['Customer']).replace('"', '')[:35] # Limit to 35 chars
        ltv = f"{row['Total_LTV']:,.0f}"
        days = str(int(row['Recency_Days']))
        
        pdf.cell(w_name, 8, name, 1)
        pdf.cell(w_ltv, 8, ltv, 1, 0, 'R')
        pdf.cell(w_days, 8, days, 1, 0, 'C')
        
        pdf.set_text_color(200, 0, 0)
        pdf.cell(w_status, 8, "CRITICAL", 1, 1, 'C')
        pdf.set_text_color(0, 0, 0)
        
    if tier == 'free':
        pdf.ln(5)
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 10, "... Upgrade to Audit Pro for full list.", 0, 1, 'C')

    # --- PAGE 4: GROWTH OPPORTUNITIES (Table Alignment Fixed) ---
    pdf.add_watermark()
    pdf.add_page()
    # pdf.add_watermark()
    
    pdf.set_font('Arial', 'B', 18)
    pdf.cell(0, 2.5, '3. Growth Opportunities', 0, 1)
    pdf.line(10, 30, 200, 30)
    pdf.ln(10)
    
    if tier == 'free':
        pdf.draw_locked_box(height=80, text="GROWTH ENGINE LOCKED")
    else:
        if upsell_df is not None and not upsell_df.empty:
            # 1. Chart (Upsell Bar)
            try:
                upsell_chart = create_upsell_chart(upsell_df)
                if upsell_chart:
                    pdf.image(upsell_chart, x=10, w=190)
                    pdf.ln(5)
            except: pass
            
            # 2. Table Header (Total Width = 190mm)
            w_bought = 70
            w_pitch = 70
            w_pot = 50
            
            pdf.ln(5)
            pdf.set_font('Arial', 'B', 9)
            pdf.set_fill_color(46, 125, 50) # Green
            pdf.set_text_color(255, 255, 255)
            
            pdf.cell(w_bought, 10, 'If they bought...', 1, 0, 'L', True)
            pdf.cell(w_pitch, 10, 'Pitch this Upsell...', 1, 0, 'L', True)
            pdf.cell(w_pot, 10, 'Opportunity', 1, 1, 'C', True)
            
            pdf.set_font('Arial', '', 9)
            pdf.set_text_color(0, 0, 0)
            # pdf.ln(10)
            
            for _, row in upsell_df.head(15).iterrows():
                # Extract values safely
                bought = str(row.get('If they bought...', row.get('Bought', '')))[:35]
                needs = str(row.get('They likely need...', row.get('Needs', '')))[:35]
                
                # Format 'Potential' properly
                raw_count = row.get('Missed Sales Count', row.get('Potential Deals', 0))
                pot_str = f"{raw_count} Clients"
                
                pdf.cell(w_bought, 8, bought, 1)
                pdf.cell(w_pitch, 8, needs, 1)
                pdf.cell(w_pot, 8, pot_str, 1, 1, 'C')

        else:
            pdf.cell(0, 10, "No significant cross-sell data found.", 0, 1)
    # --- PAGE 5: ACTIONABLE CALL LIST (New Section) ---
    pdf.add_watermark()
    pdf.add_page()
    
    pdf.set_font('Arial', 'B', 18)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 2.5, '4. High Priority Call List', 0, 1)
    pdf.line(10, 30, 200, 30)
    pdf.ln(10)

    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, "Action Required: The following customers have a high churn probability (>70%). Contact them immediately with a re-engagement offer.")
    pdf.ln(5)

    # Table Header
    w_name = 90
    w_rec = 40
    w_risk = 40
    
    pdf.set_fill_color(220, 220, 220)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(w_name, 10, 'Customer / Company', 1, 0, 'L', True)
    pdf.cell(w_rec, 10, 'Days Inactive', 1, 0, 'C', True)
    pdf.cell(w_risk, 10, 'Risk Level', 1, 1, 'C', True)
    
    # Table Rows
    pdf.set_font('Arial', '', 10)
    call_list = rfm_df[rfm_df['Churn_Risk'] > 70].sort_values('Churn_Risk', ascending=False)
    
    # Tier limit
    if tier == 'free': call_list = call_list.head(5)
    else: call_list = call_list.head(30)

    if call_list.empty:
        pdf.cell(0, 10, "No high-risk customers detected.", 1, 1, 'C')
    else:
        for _, row in call_list.iterrows():
            name = str(row['Customer'])[:35]
            days = str(int(row['Recency_Days']))
            risk = f"{row['Churn_Risk']:.1f}%"
            
            pdf.cell(w_name, 8, name, 1)
            pdf.cell(w_rec, 8, days, 1, 0, 'C')
            pdf.set_text_color(200, 0, 0)
            pdf.cell(w_risk, 8, risk, 1, 1, 'C')
            pdf.set_text_color(0, 0, 0)

    if tier == 'free':
        pdf.ln(5)
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 10, "... List truncated. Upgrade to Audit Pro for full list.", 0, 1, 'C')

    # Return statement remains at the very end
    # return bytes(pdf.output(dest='S'))
    return bytes(pdf.output(dest='S'))
# ────────────────────────────────────────────────
#  3. LOGIN PAGE UI
# ────────────────────────────────────────────────

def render_login_page():
    st.markdown("""
    <style>
        div[data-testid="stForm"] {
            background-color: #f9f9f9;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)

    # Hide the Streamlit header and footer
    hide_st_style = """
            <style>
            /* Hides the entire top header bar */
            
            
            /* Specifically hides the "Made with Streamlit" footer */
            footer {
                visibility: hidden !important;
            }

            /* Optional: Hides the "Manage App" floating button for you as well */
            .stAppDeployButton {
                display: none !important;
            }
            .st-emotion-cache-scp8yw{
            display:none !important;
            }
            [data-testid="manage-app-button"]{
            display:none !important;
            }
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    st.title("ProfitGuard AI")
    st.markdown("### Distributor Intelligence & Revenue Recovery System")
    st.caption("Secure Portal • 256-bit Encryption")

    tab_login, tab_signup = st.tabs(["🔐 Login", "📝 Start Free Audit"])

    # ─── LOGIN TAB ───
    with tab_login:
        st.subheader("Client Access")
        with st.form("login_form"):
            email = st.text_input("Business Email")
            password = st.text_input("Password", type="password")
            
            submit_login = st.form_submit_button("Access Dashboard", type="primary")
            
            if submit_login:
                if email and password:
                    success, user_data = check_login(email, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user = user_data
                        st.session_state.tier = user_data.get('tier', 'free')
                        st.success(f"Welcome back, {user_data['name']}!")
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")
                else:
                    st.warning("Please enter your credentials.")

    # ─── SIGNUP TAB (The Lead Magnet) ───
    with tab_signup:
        st.subheader("New User Registration")
        st.markdown("""
        **Get your Free Revenue Audit:**
        * 🔍 Identify High-Risk Churn Customers
        * 📊 See ₹ Lakhs in Hidden Opportunities
        * 🔒 Zero Cost. No Credit Card Required.
        """)
        
        with st.form("signup_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_email = st.text_input("Email Address*")
                new_pass = st.text_input("Create Password*", type="password", help="Min 6 characters")
                new_phone = st.text_input("Phone Number*", help="For account verification")
            
            with col2:
                new_name = st.text_input("Full Name*")
                new_company = st.text_input("Company Name*")
                # Specific Categories for your Target Market
                new_category = st.selectbox("Business Type", [
                    "Wholesale Distributor",
                    "C&F Agent", 
                    "Stockist / Super Stockist",
                    "Pharma Distributor",
                    "FMCG Trader",
                    "Manufacturer",
                    "Other"
                ])

            st.markdown("---")
            submit_signup = st.form_submit_button("Create Free Account & Upload Data", type="primary")

            if submit_signup:
                if new_email and new_pass and new_name and new_company:
                    success, msg = save_user(new_email, new_pass, new_name, new_company, new_phone, new_category)
                    if success:
                        st.balloons()
                        st.success(msg)
                    else:
                        st.error(msg)
                else:
                    st.warning("Please fill in all fields marked with *")

# ────────────────────────────────────────────────
#  4. MAIN DASHBOARD UI (Gated by Tier)
# ────────────────────────────────────────────────

def main_dashboard():
     # Hide the Streamlit header and footer
    hide_st_style = """
            <style>
            /* Hides the entire top header bar */
            
            
            /* Specifically hides the "Made with Streamlit" footer */
            footer {
                visibility: hidden !important;
            }

            /* Optional: Hides the "Manage App" floating button for you as well */
            .stAppDeployButton {
                display: none !important;
            }
            .st-emotion-cache-scp8yw{
            display:none !important;
            }
            [data-testid="manage-app-button"]{
            display:none !important;
            }
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    user = st.session_state.user
    tier_info = TIERS.get(st.session_state.tier, TIERS['free'])

    # Sidebar
    with st.sidebar:
        st.markdown(f"👤 **{user['name']}**")
        st.caption(f"{user['company']} • {st.session_state.tier.upper()}")
        
        st.divider()
        
        # # Upgrade System
        # code = st.text_input("Enter Upgrade Code", type="password")
        # if st.button("Apply Code"):
        #     if code.lower() in CODE_MAP:
        #         # Update session
        #         st.session_state.tier = CODE_MAP[code.lower()]
        #         # Update Database (Optional: Add SQL Update here to persist upgrade)
        #         st.success(f"Upgraded to {CODE_MAP[code.lower()].upper()}!")
        #         st.rerun()
        #     else:
        #         st.error("Invalid Code")
        
        
        
        
    # Main Content
    st.title("ProfitGuard AI")
    st.markdown(f"### {tier_info['title']}")
    
    if st.session_state.tier == "free":
        st.info("💡 **Tip:** You are in Preview Mode. Upload data to see your Top 10 Risks.")

    uploaded_file = st.file_uploader("Upload Sales CSV", type=['csv'])
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        res, err = process_data(data)
        
        if err:
            st.error(err)
        else:
            rfm = res['rfm']
            cols = res['cols']
            with st.sidebar:
                st.subheader("📥 Audit Report")
                
                # --- NEW LOGIC: Check Limits ---
                current_count = st.session_state.user.get('pdf_downloads', 0)
                limit = 3
                is_free = (st.session_state.tier == 'free')
                
                # If Limit Reached, Block Access
                if is_free and current_count >= limit:
                    st.error(f"🔒 Free Limit Reached ({current_count}/{limit})")
                    st.caption("Upgrade to Pro for unlimited reports.")
                
                else:
                    # --- EXISTING LOGIC (Wrapped in Else) ---
                    # Determine button label based on Tier
                    if is_free:
                        label = f"📄 Generate PDF (Used {current_count}/{limit})"
                    else:
                        label = "📄 Generate Full Audit PDF"
                        
                    # 1. GENERATE BUTTON
                    if st.button(label):
                        with st.spinner("Generating Report..."):
                            # ... [YOUR EXISTING DATA GATHERING LOGIC REMAINS UNCHANGED] ...
                            rfm = res['rfm']
                            seasonal_data = None
                            forecast_val = 0
                            upsell_df = pd.DataFrame()
                            
                            # (Keep your existing if len(res['df']) > 10 block here...)
                            if len(res['df']) > 10:
                                try:
                                    seasonal_data, _, _ = get_seasonality_analysis(res['df'], cols['d'], cols['a'])
                                    seasonal_data.columns = ['Month_Num', 'Month', 'Total_LTV'] 
                                except: pass
                                forecast_val = get_aggregate_forecast(res['df'], cols['d'], cols['a'])
                            
                            if cols['p']:
                                upsell_result = generate_cross_sell(res['df'], cols['c'], cols['p'])
                                if isinstance(upsell_result, tuple): upsell_df = upsell_result[0]
                                else: upsell_df = upsell_result
                                if not upsell_df.empty and "If they bought..." in upsell_df.columns:
                                    upsell_df = upsell_df.rename(columns={
                                        "If they bought...": "Bought",
                                        "They likely need...": "Needs",
                                        "Missed Sales Count": "Potential Deals",
                                        "Potential Sales": "Potential Deals"
                                    })

                            # Generate PDF
                            pdf_bytes = create_professional_pdf(
                                rfm_df=rfm,
                                upsell_df=upsell_df,
                                seasonal_data=seasonal_data,
                                forecast_val=forecast_val,
                                user_name=user['name'],
                                company_name=user['company'],
                                tier=st.session_state.tier
                            )
                            
                            # SAVE TO SESSION STATE (Crucial for Streamlit flow)
                            st.session_state['pdf_ready'] = pdf_bytes

                    # 2. DOWNLOAD BUTTON (Only appears after generation)
                    if 'pdf_ready' in st.session_state:
                        st.download_button(
                            label="⬇️ Click to Save PDF",
                            data=st.session_state['pdf_ready'],
                            file_name=f"ProfitGuard_Report_{st.session_state.tier}.pdf",
                            mime="application/pdf",
                            # --- NEW: Increment Count on Click ---
                            on_click=increment_download_count,
                            args=(user['email'],)
                        )

                st.divider()
                if st.button("Logout"):
                    st.session_state.logged_in = False
                    st.rerun()  
                    
            # TIER LOGIC: Filter Data
            display_rfm = rfm.copy()
            if tier_info['max_customers']:
                display_rfm = rfm.sort_values('Total_LTV', ascending=False).head(tier_info['max_customers'])
                st.warning(tier_info['msg'])

            # KPI Row
            k1, k2, k3 = st.columns(3)
            k1.metric("Total Customers", len(rfm))
            k1.caption("Analyzed")
            k2.metric("Total Revenue", f"₹{rfm['Total_LTV'].sum():,.0f}")
            risk_val = rfm[rfm['Churn_Risk'] > 75]['Total_LTV'].sum()
            k3.metric("⚠️ Revenue at Risk", f"₹{risk_val:,.0f}")
            k3.caption("High Churn Probability")

            # Tabs
            tabs = st.tabs(["📉 Retention", "🔮 Predictions", "📦 Inventory/Cross-Sell","🧪 Strategy"])
            
            with tabs[0]:
                st.subheader("High Priority Call List")
                st.dataframe(
                    display_rfm[['Customer', 'Total_LTV', 'Recency_Days', 'Churn_Risk']]
                    .sort_values('Churn_Risk', ascending=False)
                    .style.format({'Total_LTV': '₹{:,.0f}', 'Churn_Risk': '{:.1f}%'})
                    .background_gradient(subset=['Churn_Risk'], cmap='Reds'),
                    use_container_width=True
                )
                st.divider()
                st.subheader("📅 Cohort Retention Analysis")
                if st.session_state.tier in ['tier1', 'tier2']:
                    cohort_matrix = get_cohort_analysis(res['df'], cols['c'], cols['d'])
                    
                    # Display as a Heatmap styled dataframe
                    st.dataframe(
                        cohort_matrix.style.format("{:.0%}").background_gradient(cmap="Blues", axis=None),
                        use_container_width=True
                    )
                    st.caption("Read this: 'In Month 0 (start), 100% are here. In Month 1, X% returned.'")
                else:
                    st.info("🔒 Upgrade to Audit Pro to see your Customer Retention Heatmap.")

            with tabs[1]:
                if tier_info['allow_clv']:
                    st.subheader("AI Revenue Forecast (Next 30 Days)")
                    # Placeholder for advanced model
                    # st.info("✅ Prediction Module Active. (Connect ML model here)")
                    # Check if we have enough data to run the model
                    if len(res['df']) < 10:
                        st.warning("⚠️ Not enough data to train the AI model. Please upload a larger dataset.")
                    else:
                        with st.spinner("Training AI Models (BG/NBD + Gamma-Gamma)..."):
                            try:
                                # Retrieve raw data and column names from the processed result
                                raw_df = res['df'] 
                                c_col = res['cols']['c'] # Customer Column
                                d_col = res['cols']['d'] # Date Column
                                a_col = res['cols']['a'] # Amount Column

                                # Run the Prediction Function
                                preds = get_predictive_clv(raw_df, c_col, d_col, a_col)
                                
                                # Display Top Predicted Customers
                                top_pred = preds.head(10)[['predicted_purchases_30d', 'predicted_clv']]
                                
                                st.success("✅ AI Analysis Complete")
                                st.markdown("### Customers Most Likely to Buy Next Month")
                                
                                st.dataframe(
                                    top_pred.style.format({
                                        'predicted_purchases_30d': '{:.2f}', 
                                        'predicted_clv': '₹{:,.2f}'
                                    }).background_gradient(cmap='Greens'),
                                    use_container_width=True
                                )
                                
                                # Download Button for Full Forecast
                                st.download_button(
                                    "📥 Download Full Prediction Report",
                                    preds.to_csv().encode('utf-8'),
                                    "ai_revenue_forecast.csv",
                                    "text/csv"
                                )
                                
                            except Exception as e:
                                st.error(f"Model Error: {str(e)}")
                                st.caption("Ensure your data has valid positive transaction values.")
                        st.subheader("🔮 Financial Outlook & Intelligence")
                
                        # 1. Calculate Insights
                        with st.spinner("Crunching numbers..."):
                            preds_clv = get_predictive_clv(res['df'], cols['c'], cols['d'], cols['a'])
                            next_mo_rev = get_aggregate_forecast(res['df'], cols['d'], cols['a'])
                            seasonal_data, best_mo, worst_mo = get_seasonality_analysis(res['df'], cols['d'], cols['a'])

                        # 2. Display Top Level Metrics
                        col_pred1, col_pred2, col_pred3 = st.columns(3)
                        
                        if next_mo_rev:
                            col_pred1.metric("📅 Next Month Forecast", f"₹{next_mo_rev:,.0f}", help="Predicted total revenue based on trend.")
                        
                        if best_mo is not None:
                            col_pred2.metric("🔥 Best Season", best_mo['Month'], f"Avg: ₹{best_mo[cols['a']]:,.0f}")
                            col_pred3.metric("❄️ Slowest Season", worst_mo['Month'], f"Avg: ₹{worst_mo[cols['a']]:,.0f}", delta_color="inverse")

                        st.divider()

                        # 3. Two-Column Layout: Seasonality Chart & Customer List
                        c1 = st.columns([1])

                        # with c1:
                        st.markdown("### 📅 Seasonal Trends")
                        if seasonal_data is not None:
                            chart = alt.Chart(seasonal_data).mark_bar(color='#2196F3').encode(
                                x=alt.X('Month:N', sort=None, title=None),
                                y=alt.Y(cols['a'], title='Total Revenue'),
                                tooltip=['Month', cols['a']]
                            ).properties(height=300)
                            st.altair_chart(chart, use_container_width=True)
                            st.caption("💡 **Tip:** Stock up before the blue bars peak.")

                        # with c2:
                        #     st.markdown("### 👥 Top Customers (Next 30 Days)")
                        #     top_pred = preds_clv.head(8)[['predicted_purchases_30d', 'predicted_clv']]
                        #     st.dataframe(
                        #         top_pred.style.format({
                        #             'predicted_purchases_30d': '{:.1f} Orders', 
                        #             'predicted_clv': '₹{:,.0f}'
                        #         }).background_gradient(cmap='Greens'),
                        #         use_container_width=True
                        #     )
                        #     st.caption("Customers with highest probability to buy next month.")
                else:
                    st.markdown("### 🔒 Locked Feature")
                    st.warning("Upgrade to **Audit Pro** to see which customers will buy next month.")
            
            with tabs[2]:
                if tier_info['allow_cross_sell']:
                    st.subheader("Upsell Opportunities")
                    st.subheader("📦 Inventory & Cross-Sell Intelligence")
                    
                    # Check if Product Column exists (cols['p'] comes from process_data)
                    if not cols['p']:
                        st.error("❌ 'Product' or 'Item' column not found in CSV.")
                        st.info("Please ensure your CSV has a column named 'Product Name', 'Item Description', or 'SKU'.")
                    else:
                        with st.spinner("Analyzing purchase patterns..."):
                            # Run the Cross-Sell Logic
                            upsell_df, err = generate_cross_sell(res['df'], cols['c'], cols['p'])
                            
                            if err:
                                st.warning(err)
                            else:
                                st.success(f"✅ Found {len(upsell_df)} Upsell Opportunities")
                                
                                # Display the opportunities
                                st.dataframe(
                                    upsell_df,
                                    column_config={
                                        "Potential Sales": st.column_config.NumberColumn(
                                            "Missed Customers",
                                            help="Number of customers who bought Item A but missed Item B"
                                        )
                                    },
                                    use_container_width=True,
                                    hide_index=True
                                )
                                
                                st.markdown("---")
                                st.caption("💡 **Strategy:** Call the customers listed in the right column and offer them the 'Likely Need' product.")
                else:
                    st.markdown("### 🔒 Locked Feature")
                    st.warning("Upgrade to **Audit Pro** to see Product Recommendations.")
            with tabs[3]:
                if st.session_state.tier == "tier2":
                    # Feature 1: The Simulator
                    render_scenario_simulator(res['rfm'])
                    
                    st.divider()
                    
                    # Feature 2: PDF Reporting (From your standalone file)
                    st.subheader("📄 Executive Reporting")
                    st.write("Download professional audit reports for your sales team.")
                    
                    # (Mock PDF generation for speed - connect your full PDF function here)
                    if st.button("📥 Generate Board Report"):
                        with st.spinner("Compiling PDF..."):
                            # logic from app_stanalone.py would go here
                            st.success("Report Generated! (Connect PDF logic here)")
                            
                else:
                    # THE UPSELL LOCK SCREEN
                    st.empty()
                    st.info("💎 **Growth Partner Feature**")
                    st.markdown("""
                    ### Unlock Enterprise Strategy Tools
                    
                    **Tier 2 Users Get:**
                    * ✅ **What-If Simulator:** Calculate impact of price changes & churn reduction.
                    * ✅ **PDF Audit Reports:** Download branded reports for your sales team.
                    * ✅ **Dedicated Account Manager.**
                    """)
                    st.warning("Current Plan: " + tier_info['title'])
# ────────────────────────────────────────────────
#  5. APP ENTRY POINT
# ────────────────────────────────────────────────

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'tier' not in st.session_state:
    st.session_state.tier = "free"

if st.session_state.logged_in:
    main_dashboard()

else:
    render_login_page()
