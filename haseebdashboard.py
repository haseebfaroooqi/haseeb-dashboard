# Haseeb Data_dashboard.py
import warnings
warnings.filterwarnings("ignore")

import io
import os
import base64  # <--- Zaroori library image read karne k liye
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, confusion_matrix
import joblib

# =========================
# 1. APP CONFIGURATION & STYLE
# =========================
st.set_page_config(
    page_title="Haseeb Data Analytics Pro",
    page_icon="🔹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === FINAL CSS ===
st.markdown(
    """
    <style>
    /* 1. MAIN BACKGROUND */
    .stApp {
        background: linear-gradient(135deg, #E0F2FE 0%, #F0F9FF 100%);
        background-attachment: fixed;
    }

    /* 2. SIDEBAR STYLING */
    [data-testid="stSidebar"] {
        background: #669bbc !important;
        border-right: 1px solid #334155;
    }
    
    /* SIDEBAR TEXT */
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    /* 3. MAIN DASHBOARD TEXT */
    .block-container h1, 
    .block-container h2, 
    .block-container h3, 
    .block-container p, 
    .block-container span, 
    .block-container label, 
    .block-container div,
    .block-container li {
        color: #000000;
    }

    /* === 4. DROPDOWN MENU FIX === */
    div[data-baseweb="popover"], 
    div[data-baseweb="menu"],
    ul[data-baseweb="menu"] {
        background-color: #2C3E50 !important; 
    }
    div[data-baseweb="popover"] div,
    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] span {
        color: #FFFFFF !important; 
    }
    li[aria-selected="true"], li:hover {
        background-color: #4CA1AF !important; 
        color: #FFFFFF !important;
    }

    /* 5. GLASS CARD */
    .glass-card {
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 48, 73, 0.6);
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
    }

    /* 6. BUTTONS */
    .stButton > button {
        background: #780000 !important;
        color: #FFFFFF !important;
        border: none;
        font-weight: bold;
        border-radius: 10px;
    }
    
    /* 7. SKILL BADGES */
    .skill-badge {
        background-color: #2563EB;
        color: white !important;
        padding: 5px 12px;
        border-radius: 15px;
        font-size: 0.9rem;
        margin: 3px;
        display: inline-block;
        font-weight: 500;
    }
    
    /* 8. CIRCULAR PROFILE IMAGE */
    .profile-pic {
        width: 170px;
        height: 170px;
        border-radius: 50%;
        object-fit: cover;
        border: 4px solid #2563EB;
        display: block;
        margin-left: auto;
        margin-right: auto;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* 9. DATAFRAME FIX */
    [data-testid="stDataFrame"] {
        background-color: white;
    }
    [data-testid="stDataFrame"] * {
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# 2. SESSION STATE
# =========================
if "df" not in st.session_state:
    st.session_state.df = None
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None

# =========================
# 3. HELPER FUNCTIONS
# =========================
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def get_col_types(df):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return num_cols, cat_cols

# --- FUNCTION TO READ LOCAL IMAGE ---
def get_img_as_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# =========================
# 4. SIDEBAR NAVIGATION
# =========================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: white;'>🔹 Haseeb Data Pro</h2>", unsafe_allow_html=True)
    st.caption("v3.4 • Best Data Analysis Tool")
    st.markdown("---")
    
    page = st.radio("Navigation", [
        "🏠 Home & Upload",
        "🔍 Smart Analysis",
        "🤖 AutoML",
        "📉 Custom Charts",
        "💾 Export Data",
        "📄 My CV / Resume"
    ])
    
    st.markdown("---")
    st.info("💡 **Tip:** Sidebar controls.")

# =========================
# 5. PAGES
# =========================

# --- HOME PAGE ---
if page == "🏠 Home & Upload":
    st.markdown("<h1 style='text-align: center;'>📊 Haseeb Data Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-weight: 500;'>Universal Data Analysis & Machine Learning Solution</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📂 Upload Data")
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])
        
        if uploaded_file:
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.df = df
                st.session_state.df_raw = df.copy()
                st.success("✅ File Loaded Successfully!")
                
        if st.session_state.df is not None:
            if st.button("🔄 Reset Data"):
                st.session_state.df = None
                st.rerun()
                
    with col2:
        if st.session_state.df is not None:
            st.subheader("📊 Data Snapshot")
            st.dataframe(st.session_state.df.head(8), use_container_width=True)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", st.session_state.df.shape[0])
            c2.metric("Columns", st.session_state.df.shape[1])
            c3.metric("Missing Values", st.session_state.df.isnull().sum().sum())
        else:
            st.info("👈 Please upload your file.")
            st.markdown("### Features:")
            st.markdown("""
            - 🔹 **Smart Analysis:** Auto-generates insights.
            - 🔹 **AutoML:** Finds the best model automatically.
            - 🔹 **Export:** Download clean data and results.
            """)
    st.markdown("</div>", unsafe_allow_html=True)

# --- SMART ANALYSIS ---
elif page == "🔍 Smart Analysis":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data on the Home page first.")
    else:
        df = st.session_state.df
        num_cols, cat_cols = get_col_types(df)
        
        st.title("🔍 Smart Analysis Report")
        
        # Section 1: Regression Analysis
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("1. Analysis Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Regression Analysis**")
            if len(num_cols) > 1:
                x_val = st.selectbox("Select X-Axis", num_cols, index=0, key='reg_x_box')
                y_index = 1 if len(num_cols) > 1 else 0
                y_val = st.selectbox("Select Y-Axis", num_cols, index=y_index, key='reg_y_box')
            
                try:
                    # Calculate Score
                    clean_data = df[[x_val, y_val]].dropna()
                    X_reg = clean_data[[x_val]].values
                    y_reg = clean_data[y_val].values
                    
                    reg_model = LinearRegression()
                    reg_model.fit(X_reg, y_reg)
                    r2_val = reg_model.score(X_reg, y_reg)
                    corr_val = clean_data[x_val].corr(clean_data[y_val])
                    
                    # Show Score
                    m1, m2 = st.columns(2)
                    m1.metric("R² Score", f"{r2_val:.4f}")
                    m2.metric("Correlation", f"{corr_val:.4f}")

                    # Chart
                    fig = px.scatter(df, x=x_val, y=y_val, trendline="ols", 
                                    color_discrete_sequence=['#2563EB']) 
                    
                    fig.update_layout(
                        template="plotly_white",
                        font=dict(color="black"), 
                        xaxis=dict(color="black"), 
                        yaxis=dict(color="black"),
                        paper_bgcolor="white",
                        plot_bgcolor="white",
                        margin=dict(l=20, r=20, t=30, b=20),
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.info("Not enough numeric columns.")

        with col2:
            st.write("**Data Types Distribution**")
            dtypes = df.dtypes.value_counts().reset_index()
            dtypes.columns = ['Type', 'Count']
            dtypes['Type'] = dtypes['Type'].astype(str)
            fig = px.pie(dtypes, names='Type', values='Count', hole=0.4, 
                         color_discrete_sequence=px.colors.sequential.Blues)
            fig.update_layout(font=dict(color="black"), paper_bgcolor="white", plot_bgcolor="white", height=300)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Section 2: Automated Insights
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("2. Key Drivers (Correlation)")
        if len(num_cols) > 1:
            corr = df[num_cols].corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='Blues', zmin=-1, zmax=1)
            fig.update_layout(font=dict(color="black"), paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numeric columns.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Section 3: Distribution
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("3. Variable Distributions")
        selected_col = st.selectbox("Select Column to Visualize", df.columns)
        if selected_col in num_cols:
            fig = px.histogram(df, x=selected_col, marginal="box", color_discrete_sequence=['#2563EB'])
            fig.update_layout(template="plotly_white", font=dict(color="black"), paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.bar(df[selected_col].value_counts(), color_discrete_sequence=['#2563EB'])
            fig.update_layout(template="plotly_white", font=dict(color="black"), paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# --- AUTO ML ---
elif page == "🤖 AutoML":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first.")
    else:
        df = st.session_state.df
        num_cols, cat_cols = get_col_types(df)
        st.title("🤖 Haseeb Data AutoML")
        st.markdown("Automated Model Selection & Training.")
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            target = st.selectbox("🎯 Target Variable", df.columns)
        with col2:
            task_type = "Regression" if target in num_cols else "Classification"
            st.info(f"Task Detected: **{task_type}**")
        if st.button("🚀 Start Training"):
            with st.spinner("🤖 Training models..."):
                try:
                    X = df.drop(columns=[target]).dropna(axis=1, how='all').replace('Not Provided', np.nan)
                    y = df[target]
                    imputer = SimpleImputer(strategy='most_frequent')
                    X_imputed = imputer.fit_transform(X)
                    X_clean = pd.DataFrame(X_imputed, columns=X.columns)
                    le = LabelEncoder()
                    for col in X_clean.select_dtypes(include='object').columns:
                        X_clean[col] = le.fit_transform(X_clean[col].astype(str))
                    if task_type == "Classification":
                        y = le.fit_transform(y.astype(str))
                    else:
                        y = pd.to_numeric(y, errors='coerce')
                        mask = ~np.isnan(y)
                        X_clean = X_clean[mask]
                        y = y[mask]
                    X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42)
                    results = {}
                    if task_type == "Classification":
                        models = {"Logistic Regression": LogisticRegression(max_iter=1000), "Random Forest": RandomForestClassifier(), "Gradient Boosting": GradientBoostingClassifier()}
                        for name, model in models.items():
                            model.fit(X_train, y_train)
                            preds = model.predict(X_test)
                            results[name] = accuracy_score(y_test, preds)
                    else:
                        models = {"Linear Regression": LinearRegression(), "Random Forest": RandomForestRegressor(), "Gradient Boosting": GradientBoostingRegressor()}
                        for name, model in models.items():
                            model.fit(X_train, y_train)
                            preds = model.predict(X_test)
                            results[name] = r2_score(y_test, preds)
                    st.success("Training Complete!")
                    st.subheader("🏆 Leaderboard")
                    res_df = pd.DataFrame(list(results.items()), columns=['Model', 'Score']).sort_values('Score', ascending=False)
                    best_model = res_df.iloc[0]['Model']
                    best_score = res_df.iloc[0]['Score']
                    col_a, col_b = st.columns([1, 2])
                    with col_a:
                        st.metric("Best Model", best_model)
                        st.metric("Accuracy/R2", f"{best_score:.4f}")
                    with col_b:
                        fig = px.bar(res_df, x='Score', y='Model', orientation='h', color='Score', color_continuous_scale='Blues')
                        fig.update_layout(template="plotly_white", font=dict(color="black"), paper_bgcolor="white", plot_bgcolor="white")
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

# --- CUSTOM CHARTS ---
elif page == "📉 Custom Charts":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first.")
    else:
        df = st.session_state.df
        st.title("🎨 Custom Visualizer")
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        chart_type = st.selectbox("Chart Type", ["Scatter Plot", "Bar Chart", "Line Chart", "Box Plot", "Violin Plot", "3D Scatter"])
        col1, col2, col3 = st.columns(3)
        with col1:
            x_axis = st.selectbox("X-Axis", df.columns)
        with col2:
            y_axis = st.selectbox("Y-Axis", df.columns)
        with col3:
            color_dim = st.selectbox("Color By (Optional)", [None] + list(df.columns))
        if st.button("Generate Chart"):
            template = "plotly_white"
            if chart_type == "Scatter Plot": fig = px.scatter(df, x=x_axis, y=y_axis, color=color_dim, template=template)
            elif chart_type == "Bar Chart": fig = px.bar(df, x=x_axis, y=y_axis, color=color_dim, template=template)
            elif chart_type == "Line Chart": fig = px.line(df, x=x_axis, y=y_axis, color=color_dim, template=template)
            elif chart_type == "Box Plot": fig = px.box(df, x=x_axis, y=y_axis, color=color_dim, template=template)
            elif chart_type == "Violin Plot": fig = px.violin(df, x=x_axis, y=y_axis, color=color_dim, template=template)
            elif chart_type == "3D Scatter": 
                z_axis = st.selectbox("Z-Axis", df.columns)
                fig = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis, color=color_dim, template=template)
            fig.update_layout(font=dict(color="black"), xaxis=dict(color="black"), yaxis=dict(color="black"), paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# --- EXPORT ---
elif page == "💾 Export Data":
    if st.session_state.df is None:
        st.warning("⚠️ No data to export.")
    else:
        st.title("💾 Data Export")
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        csv = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV", csv, "Haseeb Data_data.csv", "text/csv", key='download-csv')
        st.markdown("</div>", unsafe_allow_html=True)

# --- CV / RESUME PAGE (LAST POSITION) ---
elif page == "📄 My CV / Resume":
    st.title("📄 resume")
    
    # 1. Header with Circular Image
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    
    col_img, col_text = st.columns([1, 3])
    
    with col_img:
        # === PERMANENT IMAGE LOGIC ===
        # File ka naam "profile.jpg" ya "profile.png" hona chahiye
        profile_image_path = "profile.jpg" # <--- Yahan apni file ka naam check karein
        
        if os.path.exists(profile_image_path):
            img_base64 = get_img_as_base64(profile_image_path)
            st.markdown(f'<img src="data:image/png;base64,{img_base64}" class="profile-pic">', unsafe_allow_html=True)
        else:
            # Agar file na mile to ye dummy image show karega
            st.markdown('<img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" class="profile-pic">', unsafe_allow_html=True)
            st.caption("Save 'profile.jpg' in folder to change.")
    
    with col_text:
        st.markdown("# Haseeb")
        st.markdown("### 🎬 Video Editor | 📊 Data Analyst")
        st.write("Passionate about storytelling through video editing and uncovering insights through data analysis. Experienced in creating engaging content and building interactive data dashboards.")
        st.download_button("⬇️ Download CV PDF", data="Haseeb CV Content", file_name="Haseeb_CV.pdf")
        
    st.markdown("</div>", unsafe_allow_html=True)

    # 2. Main Content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📞 Contact Info")
        st.write("📧 **Email:** fhaseeb297@gmail.com")
        st.write("🔗 **LinkedIn:** Haseeb Farooqi")
        st.write("📍 **Location:** Pakistan")
        st.markdown("---")
        
        st.subheader("🛠 Technical Skills")
        st.markdown("""
        <span class='skill-badge'>Python</span>
        <span class='skill-badge'>Data Analysis</span>
        <span class='skill-badge'>Streamlit</span>
        <span class='skill-badge'>Plotly</span>
        <span class='skill-badge'>Machine Learning</span>
        <span class='skill-badge'>SQL</span>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("🎬 Creative Skills")
        st.markdown("""
        <span class='skill-badge'>Video Editing</span>
        <span class='skill-badge'>Adobe Premiere Pro</span>
        <span class='skill-badge'>After Effects</span>
        <span class='skill-badge'>Storytelling</span>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("🎓 Education")
        st.write("**BS Data Science / CS**")
        st.caption("University Name | 202X - Present")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("💼 Work Experience")
        
        st.write("#### 🎬 Freelance Video Editor")
        st.caption("Self-Employed | 2024 - Present")
        st.write("""
        - Edited high-quality videos for YouTube channels and documentaries.
        - Managed post-production workflow including color grading and sound design.
        - Delivered projects under tight deadlines (average 7-8 hours work daily).
        """)
        
        st.divider()
        
        st.write("#### 📊 Data Analyst Intern / Projects")
        st.caption("Various Projects | 2025 - Present")
        st.write("""
        - Developed **Nexus Analytics Pro**, a comprehensive automated data dashboard using Python & Streamlit.
        - Performed EDA and Regression Analysis on complex datasets.
        - Built predictive models using Scikit-Learn (Random Forest, Linear Regression).
        """)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🚀 Featured Projects")
        st.write("**1. Automated Data Dashboard (Haseeb Data Pro)**")
        st.write("A tool that automates EDA, Cleaning, and ML Model training with a single click.")
        
        st.write("**2. YouTube Documentary Editing**")
        st.write("Produced engaging documentary-style videos covering global economics and history.")
        st.markdown("</div>", unsafe_allow_html=True)
