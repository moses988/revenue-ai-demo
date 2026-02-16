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



def send_email_alert(name, company, phone, category):
    sender_email = "profitguardai@gmail.com"
    sender_password = "wulfnrpomptjttqo" # Paste the 16-digit App Password
    receiver_email = "profitguardai@gmail.com"  # Send to yourself

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

    st.title("ProfitGuard AI")
    st.markdown("### Distributor Intelligence & Revenue Recovery System")
    st.caption("Secure Portal • 256-bit Encryption • Hosted in Mumbai")

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
    user = st.session_state.user
    tier_info = TIERS.get(st.session_state.tier, TIERS['free'])

    # Sidebar
    with st.sidebar:
        st.markdown(f"👤 **{user['name']}**")
        st.caption(f"{user['company']} • {st.session_state.tier.upper()}")
        st.divider()
        
        # Upgrade System
        code = st.text_input("Enter Upgrade Code", type="password")
        if st.button("Apply Code"):
            if code.lower() in CODE_MAP:
                # Update session
                st.session_state.tier = CODE_MAP[code.lower()]
                # Update Database (Optional: Add SQL Update here to persist upgrade)
                st.success(f"Upgraded to {CODE_MAP[code.lower()].upper()}!")
                st.rerun()
            else:
                st.error("Invalid Code")
        
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

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
            tabs = st.tabs(["📉 Retention", "🔮 Predictions", "📦 Inventory/Cross-Sell"])
            
            with tabs[0]:
                st.subheader("High Priority Call List")
                st.dataframe(
                    display_rfm[['Customer', 'Total_LTV', 'Recency_Days', 'Churn_Risk']]
                    .sort_values('Churn_Risk', ascending=False)
                    .style.format({'Total_LTV': '₹{:,.0f}', 'Churn_Risk': '{:.1f}%'})
                    .background_gradient(subset=['Churn_Risk'], cmap='Reds'),
                    use_container_width=True
                )

            with tabs[1]:
                if tier_info['allow_clv']:
                    st.subheader("AI Revenue Forecast (Next 30 Days)")
                    # Placeholder for advanced model
                    st.info("✅ Prediction Module Active. (Connect ML model here)")
                else:
                    st.markdown("### 🔒 Locked Feature")
                    st.warning("Upgrade to **Audit Pro** to see which customers will buy next month.")
            
            with tabs[2]:
                if tier_info['allow_cross_sell']:
                    st.subheader("Upsell Opportunities")
                    st.success("✅ Basket Analysis Active")
                else:
                    st.markdown("### 🔒 Locked Feature")
                    st.warning("Upgrade to **Audit Pro** to see Product Recommendations.")

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
