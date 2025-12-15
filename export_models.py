import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import joblib
import os

# 1. Load Data
print("Loading data...")
df = pd.read_csv(r"c:/Users/zuz12/OneDrive/Desktop/pa project new/german_credit_data.csv")
df = df.drop("Unnamed: 0", axis=1)

# 2. Preprocessing Steps (Replicating Notebook)

# Log Transformation
print("Applying log transformation...")
df["Age"] = np.log(df["Age"] + 1)
df["Credit amount"] = np.log(df["Credit amount"] + 1)
df["Duration"] = np.log(df["Duration"] + 1)

# Handling Missing Values
print("Imputing missing values...")
null_col = ['Saving accounts', 'Checking account']
imputer = SimpleImputer(strategy="most_frequent")
df[null_col] = imputer.fit_transform(df[null_col])

# Manual Categorical Mapping
print("Encoding categorical variables...")
df['Sex'] = df['Sex'].replace({'male': 1, 'female': 0})
df['Saving accounts'] = df['Saving accounts'].replace({'little': 0, 'moderate': 1, 'quite rich': 2, 'rich': 3})
df['Checking account'] = df['Checking account'].replace({'little': 0, 'moderate': 1, 'rich': 2})

# One-Hot Encoding
ohe_col = ["Purpose", "Housing"]
df = pd.get_dummies(data=df, columns=ohe_col)

# Scaling
print("Scaling features...")
scale_col = ["Age", "Credit amount", "Duration"]
scaler = StandardScaler()
df[scale_col] = scaler.fit_transform(df[scale_col])

# 3. Model Training
print("Training KMeans model...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(df)

# 4. Save Artifacts
print("Saving artifacts...")
output_dir = "models"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

joblib.dump(imputer, os.path.join(output_dir, "imputer.joblib"))
joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
joblib.dump(kmeans, os.path.join(output_dir, "kmeans.joblib"))

# Save model columns to ensure One-Hot Encoding alignment during inference
model_columns = list(df.columns)
# Remove 'cluster' from model columns as it is the target
if 'cluster' in model_columns:
    model_columns.remove('cluster')
joblib.dump(model_columns, os.path.join(output_dir, "model_columns.joblib"))

# Calculate and save cluster statistics for optimization
# Note: Re-calculating stats on original scale might be better for display, 
# but for consistency with the notebook's simple optimization, we'll use the transformed/scaled?
# Wait, the notebook calculated stats on 'df_pres' which had log transform but NOT scaling?
# Let's check the notebook again. 
# In the notebook: 
# 1. df is fully processed (log, encoded, scaled).
# 2. kmeans fits on df.
# 3. df['cluster'] is assigned.
# 4. separate df_pres is created from raw csv.
# 5. df_pres gets log transform, encode, impute BUT NO SCALING (according to snippets seen earlier).
# 6. df_pres['cluster'] = df['cluster']
# 7. Cluster stats are calculated on df_pres.

# To simplify for the app, we will calculating the stats on a version that represents "real" values or at least interpretable ones.
# Actually, for the Portfolio, the notebook usually uses Return/Risk.
# The user wants "Prescriptive Analytics". 
# The optimization snippet showed: `cluster_stats = df_pres.groupby("cluster")["Duration"].mean()`
# This implies we need the stats from the UN-SCALED data (but log transformed? or maybe raw?).
# In the notebook, df_pres had: drop unnamed, manual map Sex/Checking/Saving, OneHot, Impute.
# It did NOT seem to have Scaling or Log transform explicitly shown in the snippet near the end, 
# BUT `df["Age"]=np.log(df["Age"]+1)` was done on `df`.
# Let's assume for the simplest valid display, we save the stats of the FULLY PROCESSED df OR 
# we save a separate dictionary derived from the raw data + cluster labels.

# Let's save a simple dictionary of cluster stats from the Training DF (which IS scaled/logged).
# Or better, let's behave like the notebook:
# The notebook used 'Duration' means from 'df_pres' (which likely wasn't scaled).
# We will save the cluster centroids or specific stats needed for the visual.
# Since we have `df` here which is scaled, we can just save `kmeans.cluster_centers_` 
# OR `df.groupby('cluster').mean()`.
# However, for the USER to understand "Cluster 0 has average Credit 5000", we want UN-SCALED values.
# So, let's reload raw data, assign clusters, and calculate interpretable stats.

df_raw = pd.read_csv(r"c:/Users/zuz12/OneDrive/Desktop/pa project new/german_credit_data.csv")
df_raw = df_raw.drop("Unnamed: 0", axis=1)
df_raw['cluster'] = df['cluster'] # Use the labels we just learned

# Get means for the optimization/display
cluster_stats = df_raw.groupby('cluster')[['Age', 'Credit amount', 'Duration']].mean()
joblib.dump(cluster_stats, os.path.join(output_dir, "cluster_stats.joblib"))

print("Export complete.")
