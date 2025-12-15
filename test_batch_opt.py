import pandas as pd
import numpy as np
import joblib
import cvxpy as cp
import os

def test_batch_opt():
    print("Testing Updated Batch Optimization Logic...")
    
    # 1. Load Artifacts
    model_dir = "models"
    try:
        imputer = joblib.load(os.path.join(model_dir, "imputer.joblib"))
        scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
        kmeans = joblib.load(os.path.join(model_dir, "kmeans.joblib"))
        model_columns = joblib.load(os.path.join(model_dir, "model_columns.joblib"))
        cluster_stats = joblib.load(os.path.join(model_dir, "cluster_stats.joblib"))
        print("Artifacts loaded.")
    except Exception as e:
        print(f"Artifact error: {e}")
        return

    # 2. Simulate Upload (Create dummy CSV)
    data = {
        "Age": [22, 45, 33, 50],
        "Sex": ["female", "male ", "male", "female"], # Added space to test trimming
        "Job": [2, 1, 3, 2],
        "Housing": ["own", "free", "own", "rent"],
        "Saving accounts": ["little", "moderate", "unknown", "rich"],
        "Checking account": ["moderate", "unknown", "rich", "little"],
        "Credit amount": [5951, 2096, 7882, 4870],
        "Duration": [48, 12, 42, 24],
        "Purpose": ["radio/TV", "education", "furniture/equipment", "car"]
    }
    df_batch = pd.DataFrame(data)
    
    # 3. Preprocess Logic (Replicating app.py update)
    processed = df_batch.copy()
    
    # Clean Strings
    for col in processed.select_dtypes(include=['object']).columns:
        processed[col] = processed[col].astype(str).str.strip()

    # Log 
    processed["Age"] = np.log(processed["Age"] + 1)
    processed["Credit amount"] = np.log(processed["Credit amount"] + 1)
    processed["Duration"] = np.log(processed["Duration"] + 1)
    
    # Impute
    cols_to_impute = ['Saving accounts', 'Checking account']
    for col in cols_to_impute:
        processed[col] = processed[col].replace(['unknown', 'nan', 'None', ''], np.nan)
        
    processed[cols_to_impute] = imputer.transform(processed[cols_to_impute])
    
    # Map
    processed['Sex'] = processed['Sex'].map({'male': 1, 'female': 0})
    processed['Saving accounts'] = processed['Saving accounts'].map({'little': 0, 'moderate': 1, 'quite rich': 2, 'rich': 3})
    processed['Checking account'] = processed['Checking account'].map({'little': 0, 'moderate': 1, 'rich': 2})
    
    processed['Sex'] = processed['Sex'].fillna(0)
    processed['Saving accounts'] = processed['Saving accounts'].fillna(0)
    processed['Checking account'] = processed['Checking account'].fillna(0)
    
    # Encode
    processed = pd.get_dummies(processed, columns=["Purpose", "Housing"])
    
    for col in model_columns:
        if col not in processed.columns:
            processed[col] = 0
    processed = processed[model_columns]
    
    # Scale
    scale_col = ["Age", "Credit amount", "Duration"]
    processed[scale_col] = scaler.transform(processed[scale_col])
    
    clusters = kmeans.predict(processed)
    print(f"Predicted Clusters: {clusters}")
    
    # 4. Optimization (Updated Logic)
    print("Running Optimization...")
    total_budget = 1000000
    n = len(cluster_stats)
    
    # Use 1 / Duration as proxy for Return (Turnover)
    durations = np.array([cluster_stats.loc[i, 'Duration'] for i in range(n)])
    returns = 1 / (durations + 1e-6)
    
    p = cp.Variable(n)
    lambda_reg = 0.10
    objective = cp.Maximize(returns @ p + lambda_reg * cp.sum(cp.entr(p)))
    
    constraints = [
        cp.sum(p) == 1,
        p >= 0.05,
        p <= 0.50
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    if prob.status == 'optimal':
        allocation = p.value * total_budget
        print("Optimal Allocation:")
        for i, val in enumerate(allocation):
            percentage = p.value[i] * 100
            print(f"Cluster {i}: ${val:,.2f} ({percentage:.1f}%) [Duration: {durations[i]:.1f}]")
    else:
        print("Optimization failed.")

if __name__ == "__main__":
    test_batch_opt()
