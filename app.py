import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import cvxpy as cp

# Set page configuration
st.set_page_config(
    page_title="Customer Segmentation & Optimization",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Models (Cached)
@st.cache_resource
def load_artifacts():
    # Check if 'models' directory exists, otherwise look in current directory '.'
    if os.path.exists("models") and os.path.isdir("models"):
        model_dir = "models"
    else:
        model_dir = "."
        
    try:
        imputer = joblib.load(os.path.join(model_dir, "imputer.joblib"))
        scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
        kmeans = joblib.load(os.path.join(model_dir, "kmeans.joblib"))
        model_columns = joblib.load(os.path.join(model_dir, "model_columns.joblib"))
        cluster_stats = joblib.load(os.path.join(model_dir, "cluster_stats.joblib"))
        return imputer, scaler, kmeans, model_columns, cluster_stats
    except FileNotFoundError:
        st.error(f"Model artifacts not found in '{model_dir}'. Please ensure .joblib files are uploaded.")
        return None, None, None, None, None

imputer, scaler, kmeans, model_columns, cluster_stats = load_artifacts()

def preprocess_data(df, imputer, scaler, model_columns):
    # Create copy to avoid modifying original
    data = df.copy()
    
    # Clean string columns (strip whitespace)
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].astype(str).str.strip()
    
    # 1. Log Transformation
    for col in ["Age", "Credit amount", "Duration"]:
        if col in data.columns:
            # Handle potential negative or zero values by clipping
            data[col] = np.log(data[col].clip(lower=0) + 1)
            
    # 2. Impute (Saving/Checking)
    # Ensure columns exist, if not create them with NaN so imputer handles them
    if 'Saving accounts' not in data.columns:
        data['Saving accounts'] = np.nan
    if 'Checking account' not in data.columns:
        data['Checking account'] = np.nan
        
    # Replace 'unknown' strings or empty strings with NaN for Imputer
    cols_to_impute = ['Saving accounts', 'Checking account']
    for col in cols_to_impute:
        data[col] = data[col].replace(['unknown', 'nan', 'None', ''], np.nan)
        
    data[cols_to_impute] = imputer.transform(data[cols_to_impute])
        
    # 3. Manual Mapping
    if 'Sex' in data.columns:
        data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})
    if 'Saving accounts' in data.columns:
        # Ensure mapping handles values that might have been imputed or existing
        data['Saving accounts'] = data['Saving accounts'].map({'little': 0, 'moderate': 1, 'quite rich': 2, 'rich': 3})
    if 'Checking account' in data.columns:
        data['Checking account'] = data['Checking account'].map({'little': 0, 'moderate': 1, 'rich': 2})
        
    # Fill any mapping failures (NaNs) with 0 or mode? 
    # With imputer above, they should be valid categories. 
    # But if map fails (unexpected category), fill with 0 to be safe.
    data['Sex'] = data['Sex'].fillna(0)
    data['Saving accounts'] = data['Saving accounts'].fillna(0)
    data['Checking account'] = data['Checking account'].fillna(0)

    # 4. One-Hot Encoding
    data = pd.get_dummies(data, columns=["Purpose", "Housing"])
    
    # Align columns
    for col in model_columns:
        if col not in data.columns:
            data[col] = 0
            
    data = data[model_columns]
    
    # 5. Scaling
    scale_col = ["Age", "Credit amount", "Duration"]
    data[scale_col] = scaler.transform(data[scale_col])
    
    return data

def run_optimization(cluster_stats, total_budget):
    # Calculate returns vector
    # Notebook Logic: Returns = 1 / Duration (Proxy for Turnover/ROI)
    # This assumes lower duration is better (higher turnover).
    n = len(cluster_stats)
    
    # cluster_stats is indexed by cluster ID
    # We use raw stats. Avoid division by zero.
    durations = np.array([cluster_stats.loc[i, 'Duration'] for i in range(n)])
    returns = 1 / (durations + 1e-6) # Adding epsilon for safety
    
    # Scale returns to be order of magnitude ~1 for numerical stability with entropy
    # (Optional, but helps Solver). Let's keep raw 1/Duration (~0.04) first as per notebook logic guess.
    
    # Optimization Variables
    p = cp.Variable(n)
    
    # Constants
    lambda_reg = 0.10
    
    # Objective
    objective = cp.Maximize(returns @ p + lambda_reg * cp.sum(cp.entr(p)))
    
    # Constraints
    constraints = [
        cp.sum(p) == 1,
        p >= 0.05,
        p <= 0.50
    ]
    
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve()
    except Exception as e:
        return None, str(e)
        
    if problem.status == 'optimal':
        return p.value * total_budget, None
    else:
        return None, f"Optimization failed: {problem.status}"

if imputer is not None:
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Select Mode", ["Single Prediction", "Batch Optimization"])
    
    if app_mode == "Single Prediction":
        st.header("Single Customer Prediction")
        
        # Input Form
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        sex = st.selectbox("Sex", ["male", "female"])
        job = st.selectbox("Job (Skill Level)", [0, 1, 2, 3])
        housing = st.selectbox("Housing", ["own", "rent", "free"])
        saving_accounts = st.selectbox("Saving accounts", ["little", "moderate", "quite rich", "rich", "unknown"])
        checking_account = st.selectbox("Checking account", ["little", "moderate", "rich", "unknown"])
        credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
        duration = st.number_input("Duration (months)", min_value=1, value=12)
        purpose = st.selectbox("Purpose", [
            "radio/TV", "education", "furniture/equipment", "car", "business", 
            "domestic appliances", "repairs", "vacation/others"
        ])
        
        if st.button("Predict"):
            df_single = pd.DataFrame({
                "Age": [age],
                "Sex": [sex],
                "Job": [job],
                "Housing": [housing],
                "Saving accounts": [saving_accounts if saving_accounts != "unknown" else np.nan],
                "Checking account": [checking_account if checking_account != "unknown" else np.nan],
                "Credit amount": [credit_amount],
                "Duration": [duration],
                "Purpose": [purpose]
            })
            
            processed_data = preprocess_data(df_single, imputer, scaler, model_columns)
            cluster = kmeans.predict(processed_data)[0]
            
            st.success(f"Predicted Cluster: {cluster}")
            
            # Show stats
            if cluster in cluster_stats.index:
                stats = cluster_stats.loc[cluster]
                st.write(f"Cluster Avg Duration (Return Proxy): {stats['Duration']:.2f}")

    elif app_mode == "Batch Optimization":
        st.header("Batch Optimization & Budget Allocation")
        
        # Budget Input
        total_budget = st.number_input("Total Marketing Budget ($)", min_value=1000, value=1000000, step=1000)
        
        # File Uploader
        uploaded_file = st.file_uploader("Upload Excel/CSV Sheet", type=["csv", "xlsx"])
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.xlsx'):
                    df_batch = pd.read_excel(uploaded_file)
                else:
                    df_batch = pd.read_csv(uploaded_file)
                    
                # Pre-processing for Display (Cleaning & Imputation)
                # 1. Drop Unnamed
                if "Unnamed: 0" in df_batch.columns:
                    df_batch = df_batch.drop("Unnamed: 0", axis=1)
                    
                # 2. Clean Strings & Normalize "None"
                # Strip whitespace
                for col in df_batch.select_dtypes(include=['object']).columns:
                    df_batch[col] = df_batch[col].astype(str).str.strip()
                    
                # Replace visual artifacts with NaN for imputation
                cols_to_clean = ['Saving accounts', 'Checking account']
                for col in cols_to_clean:
                    if col in df_batch.columns:
                        df_batch[col] = df_batch[col].replace(['nan', 'None', 'unknown', ''], np.nan)
                
                # 3. Impute for Display
                # Use the loaded imputer to fill NaNs with the Mode
                if all(col in df_batch.columns for col in cols_to_clean):
                    filled_values = imputer.transform(df_batch[cols_to_clean])
                    df_batch[cols_to_clean] = filled_values
                
                st.write("First 5 rows of uploaded data (Cleaned & Imputed):")
                st.dataframe(df_batch.head())
                
                # Check for required columns
                required_cols = ["Age", "Sex", "Job", "Housing", "Saving accounts", 
                                 "Checking account", "Credit amount", "Duration", "Purpose"]
                missing = [col for col in required_cols if col not in df_batch.columns]
                
                if missing:
                    st.error(f"Missing columns in uploaded file: {missing}")
                else:
                    # Predict Clusters for Batch
                    with st.spinner("Classifying customers..."):
                        # preprocess_data will do scaling/encoding on this already cleaned df
                        processed_batch = preprocess_data(df_batch, imputer, scaler, model_columns)
                        clusters = kmeans.predict(processed_batch)
                        df_batch['Predicted Cluster'] = clusters
                    
                    st.success("Classification Complete!")
                    st.write("Data with Predicted Clusters:")
                    st.dataframe(df_batch.head())
                    
                    # Cluster Counts
                    counts = df_batch['Predicted Cluster'].value_counts().sort_index()
                    st.bar_chart(counts)
                    
                    st.divider()
                    
                    # Optimization
                    st.subheader("Optimal Budget Allocation (Per Cluster)")
                    st.write("Calculating optimal allocation based on cluster 'Duration' metrics (Maximizing Return + Entropy)...")
                    
                    allocation, error = run_optimization(cluster_stats, total_budget)
                    
                    if allocation is not None:
                        # Display Results
                        res_data = []
                        for i, amount in enumerate(allocation):
                            count = counts.get(i, 0)
                            per_user = amount / count if count > 0 else 0
                            
                            # Get duration stats for display context causes trust
                            dur = cluster_stats.loc[i, 'Duration']
                            
                            res_data.append({
                                "Cluster": i,
                                "Allocated Budget": f"${amount:,.2f}",
                                "User Count": count,
                                "Budget Per User": f"${per_user:,.2f}",
                                "Allocation %": f"{(amount/total_budget)*100:.1f}%",
                                "Avg Duration": f"{dur:.1f} m"
                            })
                        
                        st.table(pd.DataFrame(res_data))
                        
                    else:
                        st.error(f"Optimization Error: {error}")
                        
            except Exception as e:
                st.error(f"Error processing file: {e}")
