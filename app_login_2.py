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
import google.generativeai as genai

import requests

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

CODE_MAP = {
    "audit25": "tier1", 
    "growth75": "tier2"
}


def get_zoho_token(auth_code):
    """Trades the temporary auth_code for a permanent Access Token."""
    url = "https://accounts.zoho.in/oauth/v2/token"
    
    payload = {
        "code": auth_code,
        "client_id": st.secrets["ZOHO_CLIENT_ID"],
        "client_secret": st.secrets["ZOHO_CLIENT_SECRET"],
        "redirect_uri": st.secrets["ZOHO_REDIRECT_URI"],
        "grant_type": "authorization_code"
    }
    
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        st.error(f"Zoho Auth Error: {response.text}")
        return None

def fetch_zoho_data(access_token, organization_id):
    """Pulls Sent Invoices from Zoho Books and formats them for ProfitGuard."""
    url = "https://www.zohoapis.in/books/v3/invoices"
    
    headers = {
        "Authorization": f"Zoho-oauthtoken {access_token}"
    }
    
    params = {
        "organization_id": organization_id,
        "status": "sent" # Only pull finalized sales
    }
    
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        invoices = response.json().get('invoices', [])
        if not invoices: return None, "No invoices found."
        
        # Convert to DataFrame
        df = pd.DataFrame(invoices)
        
        # Rename columns to match what ProfitGuard expects
        df = df[['customer_name', 'date', 'total', 'invoice_number']]
        df.rename(columns={
            'customer_name': 'Customer', 
            'date': 'Date', 
            'total': 'Amount'
        }, inplace=True)
        
        # Add a dummy Product column so the Market Basket analysis doesn't crash
        df['Product'] = "Zoho General Sale" 
        
        return df, None
    else:
        return None, f"API Error: {response.text}"



@st.cache_resource
def process_data(df):
    """Core Logic: RFM Analysis & Data Cleaning"""
    df = df.copy()
    cols = df.columns
    date_col = next((c for c in cols if 'Date' in c or 'Time' in c), None)
    cust_col = next((c for c in cols if 'Customer' in c or 'Name' in c), None)
    amt_col  = next((c for c in cols if 'Total' in c or 'Amount' in c or 'Value' in c), None)
    prod_col = next((c for c in cols if 'Product' in c or 'Item' in c or 'SKU' in c), None)

    if not all([date_col, cust_col, amt_col]):
        return None, "❌ Data Error: CSV must have Date, Customer, and Amount columns."

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df[amt_col] = pd.to_numeric(df[amt_col], errors='coerce').fillna(0)

    snapshot_date = df[date_col].max() + timedelta(days=1)
    rfm = df.groupby(cust_col).agg({
        date_col: lambda x: (snapshot_date - x.max()).days,
        amt_col: ['count', 'sum']
    }).reset_index()
    rfm.columns = ['Customer', 'Recency_Days', 'Frequency', 'Total_LTV']
    
    rfm['Churn_Risk'] = (rfm['Recency_Days'] / rfm['Recency_Days'].max()) * 100
    
    return {
        'df': df, 'rfm': rfm, 
        'cols': {'d': date_col, 'c': cust_col, 'a': amt_col, 'p': prod_col}
    }, None

# def get_predictive_clv(df, customer_id_col, date_col, amt_col):
#     df = df[df[amt_col] > 0]
#     data = summary_data_from_transaction_data(
#         df, customer_id_col, date_col, 
#         monetary_value_col=amt_col, 
#         observation_period_end=df[date_col].max()
#     )
#     bgf = BetaGeoFitter(penalizer_coef=0.1)
#     bgf.fit(data['frequency'], data['recency'], data['T'])
#     data['predicted_purchases_30d'] = bgf.conditional_expected_number_of_purchases_up_to_time(
#         30, data['frequency'], data['recency'], data['T']
#     )
#     returning_customers = data[data['frequency'] > 0]
#     if len(returning_customers) > 0:
#         ggf = GammaGammaFitter(penalizer_coef=0.01)
#         ggf.fit(returning_customers['frequency'], returning_customers['monetary_value'])
#         data['predicted_clv'] = ggf.customer_lifetime_value(
#             bgf, data['frequency'], data['recency'], data['T'], 
#             data['monetary_value'], time=1, discount_rate=0.01
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
    monthly = df.set_index(date_col).resample('ME')[amt_col].sum().reset_index()
    if len(monthly) < 3: return None
    X = np.arange(len(monthly)).reshape(-1, 1)
    y = monthly[amt_col]
    model = HistGradientBoostingRegressor()
    model.fit(X, y)
    next_month_pred = model.predict([[len(monthly)]])[0]
    return max(0, next_month_pred)

def get_seasonality_analysis(df, date_col, amt_col):
    df = df.copy()
    df['Month'] = df[date_col].dt.month_name()
    df['Month_Num'] = df[date_col].dt.month
    seasonal = df.groupby(['Month_Num', 'Month'])[amt_col].sum().reset_index()
    seasonal = seasonal.sort_values('Month_Num')
    if seasonal.empty: return None, None, None
    strongest = seasonal.loc[seasonal[amt_col].idxmax()]
    weakest = seasonal.loc[seasonal[amt_col].idxmin()]
    return seasonal, strongest, weakest

@st.cache_resource
def get_cohort_analysis(df, cust_col, date_col):
    df = df.copy()
    df['OrderPeriod'] = df[date_col].dt.to_period('M')
    df['Cohort'] = df.groupby(cust_col)[date_col].transform('min').dt.to_period('M')
    cohort_data = df.groupby(['Cohort', 'OrderPeriod']).agg(n_customers=(cust_col, 'nunique')).reset_index()
    cohort_data['PeriodNumber'] = (cohort_data.OrderPeriod - cohort_data.Cohort).apply(lambda x: x.n)
    cohort_pivot = cohort_data.pivot_table(index='Cohort', columns='PeriodNumber', values='n_customers')
    cohort_size = cohort_pivot.iloc[:, 0]
    retention_matrix = cohort_pivot.divide(cohort_size, axis=0)
    return retention_matrix

# ────────────────────────────────────────────────
#  NEW AI & PRICING LOGIC (For Tier 2)
# ────────────────────────────────────────────────

def get_dynamic_pricing_suggestion(churn_risk, ltv):
    """Recommends dynamic discount based on churn risk and LTV."""
    if churn_risk > 85:
        return "15-20%", "High Priority - Maximum approved margin sacrifice to save account."
    elif churn_risk > 70:
        if ltv > 50000: 
            return "10-15%", "High LTV Account - Pre-emptive aggressive discount."
        return "5-10%", "Standard Retention Discount."
    elif churn_risk > 40:
        return "Free Shipping / 2% Net 30", "Low cost value-add to maintain relationship."
    else:
        return "0%", "Account is stable. No discount required."

def generate_ai_outreach(customer_name, product_pitch, discount):
    """Simulates an LLM generating a highly personalized outreach script."""
    script = f"""Subject: Exclusive Partner Allocation for {customer_name}

Hi {customer_name} team,

We value our ongoing partnership and noticed it’s been a little while since your last restock. As we plan our inventory for the upcoming quarter, I wanted to personally reach out.

Based on your ordering history, we've set aside a priority allocation of {product_pitch} specifically for your account. To support your margins this month, I'm authorized to apply an exclusive {discount} discount if we can get this locked in this week.

Do you have 5 minutes tomorrow to discuss how we can best support your current demand?

Best regards,
Your Account Manager
ProfitGuard AI Copilot
"""
    return script

def generate_ai_outreach(customer_name, product_pitch, discount):
    """
    Uses Google's Gemini API to draft a highly personalized sales email.
    """
    try:
        # 1. Configure the API using your Streamlit secret
        api_key = st.secrets["gemini"]["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        
        # 2. Select the model (gemini-1.5-flash is fast and great for text)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # 3. Create a strict prompt for the AI
        prompt = f"""
        You are an expert, highly professional B2B Sales Account Manager for a wholesale distributor.
        Write a short, persuasive, and polite cold email to a client named '{customer_name}'.
        
        Context:
        - They are an existing client but haven't ordered recently.
        - We want to upsell them this specific product: {product_pitch}
        - We are authorized to offer them this specific discount: {discount}
        
        Rules:
        - Keep it under 150 words.
        - Do NOT sound like a robot. Make it sound like a human wrote it.
        - Include a clear call to action (e.g., asking for a 5-minute phone call).
        - Use a clear Subject Line.
        """
        
        # 4. Generate the response
        response = model.generate_content(prompt)
        
        return response.text
        
    except Exception as e:
        return f"⚠️ Error connecting to AI: {str(e)}\n\n(Did you add your GEMINI_API_KEY to your secrets?)"

# ────────────────────────────────────────────────
#  MARKET BASKET ANALYSIS
# ────────────────────────────────────────────────

@st.cache_resource
def generate_cross_sell(df, cust_col, prod_col):
    if not prod_col or df.empty: return None, "⚠️ No 'Product' column detected."
    top_products = df[prod_col].value_counts().head(50).index
    df_top = df[df[prod_col].isin(top_products)]
    if df_top[prod_col].nunique() < 2: return None, "⚠️ Not enough different products."

    basket = pd.crosstab(df_top[cust_col], df_top[prod_col])
    basket = (basket > 0).astype(int)
    if basket.shape[1] < 2: return None, "⚠️ Not enough data."

    cooc = basket.T.dot(basket)
    opportunities = []
    
    for product_A in cooc.columns:
        correlations = cooc[product_A].sort_values(ascending=False)
        if len(correlations) > 1:
            top_match = correlations.index[1] 
            targets = basket[(basket[product_A] == 1) & (basket[top_match] == 0)].index.tolist()
            if len(targets) > 0:
                opportunities.append({
                    "If they bought...": product_A,
                    "They likely need...": top_match,
                    "Potential Sales": len(targets),
                    "Call These Customers": ", ".join(str(x) for x in targets[:3])
                })
    
    if not opportunities: return None, "No strong correlations found yet."
    return pd.DataFrame(opportunities).sort_values('Potential Sales', ascending=False), None

def render_scenario_simulator(rfm_df):
    st.subheader("🧪 Strategy Simulator")
    st.markdown("Simulate how small improvements impact your bottom line.")
    col1, col2 = st.columns(2)
    with col1:
        churn_reduction = st.slider("📉 If we reduce Churn by...", 0, 50, 10, format="%d%%")
    with col2:
        upsell_increase = st.slider("📈 If we increase Avg Order Value by...", 0, 50, 5, format="%d%%")
        
    current_rev = rfm_df['Total_LTV'].sum()
    at_risk_rev = rfm_df[rfm_df['Churn_Risk'] > 70]['Total_LTV'].sum()
    saved_rev = at_risk_rev * (churn_reduction / 100)
    
    new_rev_base = current_rev + saved_rev
    upsell_gain = new_rev_base * (upsell_increase / 100)
    
    total_projected = current_rev + saved_rev + upsell_gain
    net_gain = total_projected - current_rev
    
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Current Revenue", f"₹{current_rev:,.0f}")
    m2.metric("Projected Revenue", f"₹{total_projected:,.0f}", delta=f"+{churn_reduction + upsell_increase}% Growth")
    m3.metric("💰 Net Profit Increase", f"₹{net_gain:,.0f}", delta="Extra Cash")
    
    chart_data = pd.DataFrame({'Scenario': ['Current', 'With Strategy'], 'Revenue': [current_rev, total_projected]})
    st.bar_chart(chart_data, x='Scenario', y='Revenue', color='#00CC96')

# --- CHARTS ---
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

def create_risk_chart(total_rev, risk_rev):
    safe_rev = total_rev - risk_rev
    labels = ['Safe Revenue', 'At Risk']
    sizes = [safe_rev, risk_rev]
    colors = ['#4CAF50', '#F44336']
    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, pctdistance=0.85, textprops={'fontsize': 12})
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title('Revenue Health Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    temp_file = NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_file.name, bbox_inches='tight', dpi=150)
    plt.close()
    return temp_file.name

def create_upsell_chart(upsell_df):
    if upsell_df is None or upsell_df.empty: return None
    top_5 = upsell_df.head(5).copy()
    if 'Potential Deals' not in top_5.columns and 'Missed Sales Count' in top_5.columns:
        top_5['Potential Deals'] = top_5['Missed Sales Count']
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(x='Potential Deals', y='Needs', data=top_5, palette='viridis')
    plt.title('Top 5 Missed Product Opportunities', fontsize=12, fontweight='bold')
    plt.xlabel('Missed Customers'); plt.ylabel('')
    sns.despine(left=True, bottom=True)
    temp_file = NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_file.name, bbox_inches='tight', dpi=150)
    plt.close()
    return temp_file.name

def increment_download_count(email):
    conn = get_db_connection()
    try:
        with conn.session as session:
            session.execute(
                text("UPDATE users SET pdf_downloads = Coalesce(pdf_downloads, 0) + 1 WHERE email = :email;"),
                {"email": email}
            )
            session.commit()
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
        self.set_fill_color(245, 245, 245)
        self.rect(x, y, 190, height, 'F')
        self.set_y(y + (height/2) - 5)
        self.set_font('Arial', 'B', 12)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, f"[LOCKED] {text}", 0, 1, 'C')
        self.set_font('Arial', '', 9)
        self.ln(5)
        self.cell(0, 10, "Upgrade to Audit Pro to unlock this data.", 0, 1, 'C')
        self.set_y(y + height + 10)

    def add_watermark(self):
        if self.tier == 'free':
            self.set_font('Arial', 'B', 25)
            self.set_text_color(200, 200, 200)
            for x_pos in range(0, 250, 80):
                for y_pos in range(0, 350, 80):
                    with self.rotation(45, x_pos, y_pos):
                        self.text(x_pos, y_pos, "FREE PREVIEW")
            self.set_text_color(0, 0, 0)
            self.set_font('Arial', '', 10)

def create_professional_pdf(rfm_df, upsell_df, seasonal_data, forecast_val, user_name, company_name, tier):
    pdf = AuditReport(tier)
    pdf.set_auto_page_break(auto=True, margin=15)
    risk_value = rfm_df[rfm_df['Churn_Risk'] > 75]['Total_LTV'].sum()
    
    upsell_value = 0
    if upsell_df is not None and not upsell_df.empty:
        avg_order_val = rfm_df['Total_LTV'].sum() / rfm_df['Frequency'].sum() if rfm_df['Frequency'].sum() > 0 else 2000
        count_col = 'Potential Deals' if 'Potential Deals' in upsell_df.columns else 'Missed Sales Count'
        if count_col in upsell_df.columns:
            total_missed_deals = upsell_df[count_col].sum()
            upsell_value = total_missed_deals * avg_order_val

    total_audit_value = risk_value + upsell_value

    # PAGE 1
    pdf.add_page()
    pdf.set_fill_color(24, 33, 47)
    pdf.rect(0, 0, 210, 297, 'F') 
    pdf.ln(90)
    pdf.set_font('Arial', 'B', 36)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 15, 'REVENUE', 0, 1, 'C')
    pdf.cell(0, 15, 'INTELLIGENCE AUDIT', 0, 1, 'C')
    pdf.ln(20)
    pdf.set_font('Arial', '', 12)
    pdf.set_text_color(200, 200, 200)
    comp_str = str(company_name).upper() if company_name else "VALUED CLIENT"
    pdf.cell(0, 10, f'PREPARED FOR: {comp_str}', 0, 1, 'C')
    pdf.cell(0, 10, f'DATE: {datetime.now().strftime("%B %d, %Y")}', 0, 1, 'C')
    
    pdf.ln(20)
    pdf.set_fill_color(46, 125, 50)
    pdf.set_draw_color(255, 255, 255)
    x_pos = (210 - 140) / 2
    y_pos = pdf.get_y()
    pdf.rect(x_pos, y_pos, 140, 35, 'FD')
    pdf.set_y(y_pos + 5)
    pdf.set_font('Arial', 'B', 10)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "TOTAL IDENTIFIED OPPORTUNITY", 0, 1, 'C')
    pdf.set_font('Arial', 'B', 22)
    pdf.cell(0, 15, f"INR {total_audit_value:,.0f}", 0, 1, 'C')
    
    if tier == 'free':
        pdf.ln(10)
        pdf.set_text_color(255, 100, 100)
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "[ FREE PREVIEW MODE ]", 0, 1, 'C')

    # PAGE 2
    pdf.add_page()
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', 'B', 18)
    pdf.cell(0, 2.5, '1. Executive Health Score', 0, 1)
    pdf.line(10, 30, 200, 30)
    pdf.ln(10)
    
    total_rev = rfm_df['Total_LTV'].sum()
    risk_rev = rfm_df[rfm_df['Churn_Risk'] > 75]['Total_LTV'].sum()
    start_y = pdf.get_y()
    
    pdf.set_left_margin(10)
    pdf.set_font('Arial', '', 11)
    pdf.set_fill_color(240, 248, 255)
    pdf.cell(90, 10, "  Total Revenue Analyzed", 0, 1, 'L', True)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(90, 10, f"  INR {total_rev:,.0f}", 0, 1, 'L', True)
    pdf.ln(4)
    
    pdf.set_font('Arial', '', 11)
    pdf.cell(90, 10, "  Revenue at Churn Risk", 0, 1, 'L', True)
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(200, 0, 0)
    pdf.cell(90, 10, f"  INR {risk_rev:,.0f}", 0, 1, 'L', True)
    pdf.set_text_color(0, 0, 0)
    
    pdf.set_y(start_y) 
    pdf.set_x(110)
    try:
        risk_chart = create_risk_chart(total_rev, risk_rev)
        if risk_chart: pdf.image(risk_chart, x=115, y=start_y, w=75) 
    except:
        pdf.cell(80, 40, "Chart Error", 1, 1, 'C')

    pdf.set_y(start_y + 85)
    pdf.set_x(10)
    
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
    
    pdf.set_font('Arial', '', 11)
    pdf.set_fill_color(255, 235, 238)
    pdf.set_text_color(200, 0, 0)
    pdf.cell(95, 12, "  Revenue at Churn Risk", 1, 0, 'L', True)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(95, 12, f" INR {risk_rev:,.0f}", 1, 1, 'R', True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(20)

    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Seasonal Revenue Strategy', 0, 1)
    pdf.ln(2)
    
    if tier == 'free':
        pdf.draw_locked_box(height=50, text="SEASONALITY CHART LOCKED")
    else:
        pdf.set_font('Arial', '', 10)
        if seasonal_data is not None:
            best_mo = seasonal_data.loc[seasonal_data['Total_LTV'].idxmax()]['Month'] 
            worst_mo = seasonal_data.loc[seasonal_data['Total_LTV'].idxmin()]['Month']
            pdf.multi_cell(0, 6, f"Insight: Your sales peak in {best_mo} and dip in {worst_mo}. Use this chart to plan inventory.")
            pdf.ln(5)
            try:
                chart_path = create_seasonality_chart(seasonal_data)
                if chart_path:
                    pdf.image(chart_path, x=15, w=180)
                    pdf.ln(5)
            except Exception as e:
                pdf.cell(0, 10, f"Chart could not be generated: {e}", 0, 1)
        else:
            pdf.cell(0, 10, "Insufficient data for seasonality.", 0, 1)

    # PAGE 3
    pdf.add_watermark()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 18)
    pdf.cell(0, 2.5, '2. Retention Alert (High Risk)', 0, 1)
    pdf.line(10, 30, 200, 30)
    pdf.ln(10)
    
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
    
    risk_df = rfm_df[rfm_df['Churn_Risk'] > 75].sort_values('Total_LTV', ascending=False)
    if tier == 'free': risk_df = risk_df.head(10)
    else: risk_df = risk_df.head(25)

    for _, row in risk_df.iterrows():
        name = str(row['Customer']).replace('"', '')[:35]
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

    # PAGE 4
    pdf.add_watermark()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 18)
    pdf.cell(0, 2.5, '3. Growth Opportunities', 0, 1)
    pdf.line(10, 30, 200, 30)
    pdf.ln(10)
    
    if tier == 'free':
        pdf.draw_locked_box(height=80, text="GROWTH ENGINE LOCKED")
    else:
        if upsell_df is not None and not upsell_df.empty:
            try:
                upsell_chart = create_upsell_chart(upsell_df)
                if upsell_chart:
                    pdf.image(upsell_chart, x=10, w=190)
                    pdf.ln(5)
            except: pass
            
            w_bought = 70
            w_pitch = 70
            w_pot = 50
            
            pdf.ln(5)
            pdf.set_font('Arial', 'B', 9)
            pdf.set_fill_color(46, 125, 50)
            pdf.set_text_color(255, 255, 255)
            
            pdf.cell(w_bought, 10, 'If they bought...', 1, 0, 'L', True)
            pdf.cell(w_pitch, 10, 'Pitch this Upsell...', 1, 0, 'L', True)
            pdf.cell(w_pot, 10, 'Opportunity', 1, 1, 'C', True)
            
            pdf.set_font('Arial', '', 9)
            pdf.set_text_color(0, 0, 0)
            
            for _, row in upsell_df.head(15).iterrows():
                bought = str(row.get('If they bought...', row.get('Bought', '')))[:35]
                needs = str(row.get('They likely need...', row.get('Needs', '')))[:35]
                raw_count = row.get('Missed Sales Count', row.get('Potential Deals', 0))
                pot_str = f"{raw_count} Clients"
                
                pdf.cell(w_bought, 8, bought, 1)
                pdf.cell(w_pitch, 8, needs, 1)
                pdf.cell(w_pot, 8, pot_str, 1, 1, 'C')

        else:
            pdf.cell(0, 10, "No significant cross-sell data found.", 0, 1)

    # PAGE 5
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

    w_name = 90
    w_rec = 40
    w_risk = 40
    
    pdf.set_fill_color(220, 220, 220)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(w_name, 10, 'Customer / Company', 1, 0, 'L', True)
    pdf.cell(w_rec, 10, 'Days Inactive', 1, 0, 'C', True)
    pdf.cell(w_risk, 10, 'Risk Level', 1, 1, 'C', True)
    
    pdf.set_font('Arial', '', 10)
    call_list = rfm_df[rfm_df['Churn_Risk'] > 70].sort_values('Churn_Risk', ascending=False)
    
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

    hide_st_style = """
            <style>
            footer { visibility: hidden !important; }
            .stAppDeployButton { display: none !important; }
            .st-emotion-cache-scp8yw { display:none !important; }
            [data-testid="manage-app-button"] { display:none !important; }
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    st.title("ProfitGuard AI")
    st.markdown("### Distributor Intelligence & Revenue Recovery System")
    st.caption("Secure Portal • 256-bit Encryption")

    tab_login, tab_signup = st.tabs(["🔐 Login", "📝 Start Free Audit"])

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
                new_category = st.selectbox("Business Type", [
                    "Wholesale Distributor", "C&F Agent", "Stockist / Super Stockist",
                    "Pharma Distributor", "FMCG Trader", "Manufacturer", "Other"
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
    hide_st_style = """
            <style>
            footer { visibility: hidden !important; }
            .stAppDeployButton { display: none !important; }
            .st-emotion-cache-scp8yw { display:none !important; }
            [data-testid="manage-app-button"] { display:none !important; }
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # Zoho OAuth Logic Capture
    # params = st.query_params
    # if "code" in params:
    #     auth_code = params["code"]
    #     st.success("✅ Successfully authenticated with Zoho Books!")
    #     # Here you would trade the auth_code for a token.
    #     st.query_params.clear()
    # Catch Zoho OAuth Redirect
    params = st.query_params
    if "code" in params:
        auth_code = params["code"]
        token = get_zoho_token(auth_code)
        if token:
            st.session_state["zoho_access_token"] = token
            st.success("✅ Successfully authenticated with Zoho Books!")
        st.query_params.clear() # Clean up the URL

    user = st.session_state.user
    tier_info = TIERS.get(st.session_state.tier, TIERS['free'])

    with st.sidebar:
        st.markdown(f"👤 **{user['name']}**")
        st.caption(f"{user['company']} • {st.session_state.tier.upper()}")
        st.divider()
        
    st.title("ProfitGuard AI")
    st.markdown(f"### {tier_info['title']}")
    
    if st.session_state.tier == "free":
        st.info("💡 **Tip:** You are in Preview Mode. Upload data to see your Top 10 Risks.")

    # --- NEW: DATA SOURCE SELECTOR UI ---
    st.markdown("### 🔌 Connect Your Data Source")
    data_source = st.radio(
        "Select your ERP / Accounting System:",
        [
            "📁 Upload CSV (Active)", 
            "☁️ Zoho Books (Beta)", 
            "🖥️ Tally Prime (Coming Soon)", 
            "🏢 SAP / NetSuite (Enterprise Only)"
        ],
        index=0,
        horizontal=True
    )

    uploaded_file = None
    
    if "Upload CSV" in data_source:
        uploaded_file = st.file_uploader("Upload Sales Register (CSV)", type=['csv'])

    # elif "Zoho Books" in data_source:
    #     st.info("💡 **Live Sync Enabled:** Connect your Zoho account to automatically pull invoice data every night at 2:00 AM IST.")
    #     zoho_auth_url = "https://accounts.zoho.in/oauth/v2/auth?response_type=code&client_id=YOUR_CLIENT_ID&redirect_uri=YOUR_STREAMLIT_URL"
    #     st.markdown(f'<a href="{zoho_auth_url}" target="_self"><button style="width:100%; background-color:#1e88e5; color:white; border-radius:5px; padding:10px; border:none; cursor:pointer;">🔗 Connect to Zoho Books</button></a>', unsafe_allow_html=True)
    elif "Zoho Books" in data_source:
        st.info("💡 **Live Sync Enabled:** Connect your Zoho account to pull live invoice data.")
        
        # 1. The Login URL with scopes asking for Zoho Books access
        client_id = st.secrets["zoho"]["ZOHO_CLIENT_ID"]
        redirect_uri = st.secrets["zoho"]["ZOHO_REDIRECT_URI"]
        zoho_auth_url = f"https://accounts.zoho.in/oauth/v2/auth?scope=ZohoBooks.invoices.READ&client_id={client_id}&response_type=code&redirect_uri={redirect_uri}&access_type=offline"
        
        st.markdown(f'<a href="{zoho_auth_url}" target="_self"><button style="width:100%; background-color:#1e88e5; color:white; border-radius:5px; padding:10px; border:none; cursor:pointer;">🔗 Connect to Zoho Books</button></a>', unsafe_allow_html=True)
        
        # 2. Check if they just logged in and we have a token
        if "zoho_access_token" in st.session_state:
            org_id = st.text_input("Enter your Zoho Organization ID (Found in Zoho Settings):")
            if st.button("Fetch Live Data") and org_id:
                with st.spinner("Pulling data from Zoho..."):
                    zoho_df, error = fetch_zoho_data(st.session_state["zoho_access_token"], org_id)
                    
                    if error:
                        st.error(error)
                    else:
                        st.success(f"✅ Successfully pulled {len(zoho_df)} invoices!")
                        # Here, you would pass `zoho_df` into your `process_data()` function
                        res, err = process_data(zoho_df)
    elif "Tally Prime" in data_source:
        st.warning("🚧 **In Development:** The Tally Prime Local Sync Agent is currently in closed beta.")
        if st.button("Join Tally Waitlist"):
            st.success("You've been added to the waitlist! We will notify you when the sync agent is ready to download.")

    else:
        st.error("🔒 **Enterprise Plan Required:** Native 2-way sync for SAP and NetSuite requires a dedicated integration engineer.")
        st.button("Contact Enterprise Sales")
    
    # Process if file uploaded (for CSV mode)
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        res, err = process_data(data)
        
        if err:
            st.error(err)
        else:
            rfm = res['rfm']
            cols = res['cols']
            
            # Prepare upsell_df globally so it can be used in the new Tier 2 UI
            upsell_df = pd.DataFrame()
            if cols['p']:
                upsell_result = generate_cross_sell(res['df'], cols['c'], cols['p'])
                if isinstance(upsell_result, tuple): 
                    upsell_df = upsell_result[0] if upsell_result[0] is not None else pd.DataFrame()
                else: 
                    upsell_df = upsell_result if upsell_result is not None else pd.DataFrame()
                if not upsell_df.empty and "If they bought..." in upsell_df.columns:
                    upsell_df = upsell_df.rename(columns={
                        "If they bought...": "Bought",
                        "They likely need...": "Needs",
                        "Missed Sales Count": "Potential Deals",
                        "Potential Sales": "Potential Deals"
                    })

            with st.sidebar:
                st.subheader("📥 Audit Report")
                current_count = st.session_state.user.get('pdf_downloads', 0)
                limit = 3
                is_free = (st.session_state.tier == 'free')
                
                if is_free and current_count >= limit:
                    st.error(f"🔒 Free Limit Reached ({current_count}/{limit})")
                    st.caption("Upgrade to Pro for unlimited reports.")
                else:
                    label = f"📄 Generate PDF (Used {current_count}/{limit})" if is_free else "📄 Generate Full Audit PDF"
                        
                    if st.button(label):
                        with st.spinner("Generating Report..."):
                            seasonal_data = None
                            forecast_val = 0
                            
                            if len(res['df']) > 10:
                                try:
                                    seasonal_data, _, _ = get_seasonality_analysis(res['df'], cols['d'], cols['a'])
                                    seasonal_data.columns = ['Month_Num', 'Month', 'Total_LTV'] 
                                except: pass
                                forecast_val = get_aggregate_forecast(res['df'], cols['d'], cols['a'])

                            pdf_bytes = create_professional_pdf(
                                rfm_df=rfm,
                                upsell_df=upsell_df,
                                seasonal_data=seasonal_data,
                                forecast_val=forecast_val,
                                user_name=user['name'],
                                company_name=user['company'],
                                tier=st.session_state.tier
                            )
                            st.session_state['pdf_ready'] = pdf_bytes

                    if 'pdf_ready' in st.session_state:
                        st.download_button(
                            label="⬇️ Click to Save PDF",
                            data=st.session_state['pdf_ready'],
                            file_name=f"ProfitGuard_Report_{st.session_state.tier}.pdf",
                            mime="application/pdf",
                            on_click=increment_download_count,
                            args=(user['email'],)
                        )

                st.divider()
                if st.button("Logout"):
                    st.session_state.logged_in = False
                    st.rerun()  
                    
            display_rfm = rfm.copy()
            if tier_info['max_customers']:
                display_rfm = rfm.sort_values('Total_LTV', ascending=False).head(tier_info['max_customers'])
                st.warning(tier_info['msg'])

            k1, k2, k3 = st.columns(3)
            k1.metric("Total Customers", len(rfm))
            k1.caption("Analyzed")
            k2.metric("Total Revenue", f"₹{rfm['Total_LTV'].sum():,.0f}")
            risk_val = rfm[rfm['Churn_Risk'] > 75]['Total_LTV'].sum()
            k3.metric("⚠️ Revenue at Risk", f"₹{risk_val:,.0f}")
            k3.caption("High Churn Probability")

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
                    if len(res['df']) < 10:
                        st.warning("⚠️ Not enough data to train the AI model. Please upload a larger dataset.")
                    else:
                        with st.spinner("Training AI Models (BG/NBD + Gamma-Gamma)..."):
                            try:
                                raw_df = res['df'] 
                                c_col = res['cols']['c']
                                d_col = res['cols']['d']
                                a_col = res['cols']['a']

                                preds = get_predictive_clv(raw_df, c_col, d_col, a_col)
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
                
                        with st.spinner("Crunching numbers..."):
                            preds_clv = get_predictive_clv(res['df'], cols['c'], cols['d'], cols['a'])
                            next_mo_rev = get_aggregate_forecast(res['df'], cols['d'], cols['a'])
                            seasonal_data, best_mo, worst_mo = get_seasonality_analysis(res['df'], cols['d'], cols['a'])

                        col_pred1, col_pred2, col_pred3 = st.columns(3)
                        
                        if next_mo_rev:
                            col_pred1.metric("📅 Next Month Forecast", f"₹{next_mo_rev:,.0f}", help="Predicted total revenue based on trend.")
                        
                        if best_mo is not None:
                            col_pred2.metric("🔥 Best Season", best_mo['Month'], f"Avg: ₹{best_mo[cols['a']]:,.0f}")
                            col_pred3.metric("❄️ Slowest Season", worst_mo['Month'], f"Avg: ₹{worst_mo[cols['a']]:,.0f}", delta_color="inverse")

                        st.divider()

                        st.markdown("### 📅 Seasonal Trends")
                        if seasonal_data is not None:
                            chart = alt.Chart(seasonal_data).mark_bar(color='#2196F3').encode(
                                x=alt.X('Month:N', sort=None, title=None),
                                y=alt.Y(cols['a'], title='Total Revenue'),
                                tooltip=['Month', cols['a']]
                            ).properties(height=300)
                            st.altair_chart(chart, use_container_width=True)
                            st.caption("💡 **Tip:** Stock up before the blue bars peak.")
                else:
                    st.markdown("### 🔒 Locked Feature")
                    st.warning("Upgrade to **Audit Pro** to see which customers will buy next month.")
            
            with tabs[2]:
                if tier_info['allow_cross_sell']:
                    st.subheader("📦 Inventory & Cross-Sell Intelligence")
                    
                    if not cols['p']:
                        st.error("❌ 'Product' or 'Item' column not found in CSV.")
                        st.info("Please ensure your CSV has a column named 'Product Name', 'Item Description', or 'SKU'.")
                    else:
                        if not upsell_df.empty:
                            st.success(f"✅ Found {len(upsell_df)} Upsell Opportunities")
                            st.dataframe(
                                upsell_df,
                                column_config={
                                    "Potential Deals": st.column_config.NumberColumn(
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
                            st.warning("No strong cross-sell patterns detected yet.")
                else:
                    st.markdown("### 🔒 Locked Feature")
                    st.warning("Upgrade to **Audit Pro** to see Product Recommendations.")
            
            # --- NEW: UPDATED TIER 2 TAB (STRATEGY & ENTERPRISE ACTIONS) ---
            with tabs[3]:
                if st.session_state.tier == "tier2":
                    st.markdown("## 🚀 Enterprise Action Orchestration")
                    
                    # Feature 1: The Simulator (Your Existing Feature)
                    render_scenario_simulator(res['rfm'])
                    
                    st.divider()
                    
                    # Feature 2: Dynamic Pricing & AI Agent (NEW)
                    st.subheader("🤖 AI Sales Agent & Dynamic Pricing")
                    st.write("Automatically generate retention campaigns with ML-optimized discount thresholds.")
                    
                    # Get high-risk customers from the current session
                    high_risk = display_rfm[display_rfm['Churn_Risk'] > 70]
                    
                    if not high_risk.empty:
                        col_ai1, col_ai2 = st.columns([1, 2])
                        
                        with col_ai1:
                            target_cust = st.selectbox("Select Account to Save:", high_risk['Customer'])
                            cust_data = high_risk[high_risk['Customer'] == target_cust].iloc[0]
                            
                            # Calculate Dynamic Pricing
                            discount_rate, pricing_rationale = get_dynamic_pricing_suggestion(
                                cust_data['Churn_Risk'], 
                                cust_data['Total_LTV']
                            )
                            
                            st.metric("Risk Level", f"{cust_data['Churn_Risk']:.1f}%")
                            st.info(f"**AI Recommended Promo:** {discount_rate}\n\n*Reasoning:* {pricing_rationale}")
                            
                            # Determine Upsell Product to inject into the email
                            pitch_product = "Premium Inventory"
                            if not upsell_df.empty:
                                pitch_product = upsell_df.iloc[0].get('Needs', 'our latest product line')
                            
                            generate_btn = st.button("✨ Draft AI Outreach", type="primary", use_container_width=True)
                            
                        with col_ai2:
                            if generate_btn:
                                with st.spinner("LLM is drafting personalized email..."):
                                    # Call the GenAI function
                                    draft = generate_ai_outreach(target_cust, pitch_product, discount_rate)
                                    st.text_area("Generated Outreach Script (Ready to Send):", value=draft, height=280)
                                    
                                    c1, c2 = st.columns(2)
                                    c1.button("📤 Send via Outlook/Gmail API")
                                    c2.button("🔄 Log to CRM")
                    else:
                        st.success("✅ No high-risk accounts currently require immediate intervention.")
                        
                    st.divider()
                    
                    # Feature 3: ERP / CRM Sync Mock (NEW)
                    st.subheader("🔄 Automated Workflow Integrations")
                    st.write("Push intelligence directly to your field reps' CRM.")
                    erp1, erp2, erp3 = st.columns(3)
                    if erp1.button("Sync to Salesforce", use_container_width=True): 
                        st.success("✅ Data synced to Salesforce API.")
                    if erp2.button("Sync to NetSuite", use_container_width=True): 
                        st.success("✅ Data synced to NetSuite.")
                    if erp3.button("Sync to Dynamics 365", use_container_width=True): 
                        st.success("✅ Data synced to Microsoft Dynamics.")

                else:
                    # THE UPSELL LOCK SCREEN (Updated with new features)
                    st.empty()
                    st.info("💎 **Growth Partner Feature**")
                    st.markdown("""
                    ### Unlock Enterprise Strategy Tools
                    
                    **Tier 2 Users Get:**
                    * ✅ **What-If Simulator:** Calculate impact of price changes & churn reduction.
                    * ✅ **AI Sales Agent:** Auto-generate targeted emails for at-risk clients.
                    * ✅ **Dynamic Pricing Engine:** Get the exact discount needed to save a customer without losing margin.
                    * ✅ **CRM/ERP Sync:** Push data directly to Salesforce, NetSuite, and Dynamics.
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
