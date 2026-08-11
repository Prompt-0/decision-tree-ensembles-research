import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score
import xgboost as xgb

os.makedirs('/root/.gemini/antigravity-cli/brain/daefbb3c-184f-4666-85af-a288927fd2d7/figures', exist_ok=True)
scratch_dir = '/root/.gemini/antigravity-cli/brain/daefbb3c-184f-4666-85af-a288927fd2d7/scratch'
os.makedirs(scratch_dir, exist_ok=True)

print("Starting Benchmark Experiments...")

# 1. Classification Dataset: Breast Cancer Wisconsin
cancer = load_breast_cancer()
X_c, y_c = cancer.data, cancer.target
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.2, random_state=42)

clf_models = {
    "Single Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42)
}

clf_results = []
for name, model in clf_models.items():
    t0 = time.time()
    model.fit(X_train_c, y_train_c)
    fit_time = time.time() - t0
    
    t0 = time.time()
    preds = model.predict(X_test_c)
    proba = model.predict_proba(X_test_c)[:, 1] if hasattr(model, "predict_proba") else preds
    infer_time = time.time() - t0
    
    acc = accuracy_score(y_test_c, preds)
    f1 = f1_score(y_test_c, preds)
    auc = roc_auc_score(y_test_c, proba)
    
    clf_results.append({
        "Model": name,
        "Accuracy": acc,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "Train Time (s)": fit_time,
        "Infer Time (s)": infer_time
    })

df_clf = pd.DataFrame(clf_results)
print("\n--- Classification Results (Breast Cancer Dataset) ---")
print(df_clf.to_string(index=False))

# 2. Regression Dataset: California Housing
housing = fetch_california_housing()
X_r, y_r = housing.data, housing.target
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

reg_models = {
    "Single Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42)
}

reg_results = []
for name, model in reg_models.items():
    t0 = time.time()
    model.fit(X_train_r, y_train_r)
    fit_time = time.time() - t0
    
    t0 = time.time()
    preds = model.predict(X_test_r)
    infer_time = time.time() - t0
    
    rmse = np.sqrt(mean_squared_error(y_test_r, preds))
    r2 = r2_score(y_test_r, preds)
    
    reg_results.append({
        "Model": name,
        "RMSE": rmse,
        "R2 Score": r2,
        "Train Time (s)": fit_time,
        "Infer Time (s)": infer_time
    })

df_reg = pd.DataFrame(reg_results)
print("\n--- Regression Results (California Housing Dataset) ---")
print(df_reg.to_string(index=False))

# 3. Sensitivity / Ablation Analysis: Tree Depth vs Performance
depths = list(range(1, 16))
rf_train_acc, rf_test_acc = [], []
xgb_train_acc, xgb_test_acc = [], []

for d in depths:
    # RF
    rf = RandomForestClassifier(n_estimators=50, max_depth=d, random_state=42)
    rf.fit(X_train_c, y_train_c)
    rf_train_acc.append(accuracy_score(y_train_c, rf.predict(X_train_c)))
    rf_test_acc.append(accuracy_score(y_test_c, rf.predict(X_test_c)))
    
    # XGB
    xb = xgb.XGBClassifier(n_estimators=50, max_depth=d, eval_metric='logloss', random_state=42)
    xb.fit(X_train_c, y_train_c)
    xgb_train_acc.append(accuracy_score(y_train_c, xb.predict(X_train_c)))
    xgb_test_acc.append(accuracy_score(y_test_c, xb.predict(X_test_c)))

# Generate Plot
plt.figure(figsize=(10, 5))
plt.plot(depths, rf_train_acc, 'b--', label='Random Forest (Train)')
plt.plot(depths, rf_test_acc, 'b-o', label='Random Forest (Test)')
plt.plot(depths, xgb_train_acc, 'r--', label='XGBoost (Train)')
plt.plot(depths, xgb_test_acc, 'r-s', label='XGBoost (Test)')
plt.title('Effect of Max Tree Depth on Overfitting & Generalization')
plt.xlabel('Max Tree Depth')
plt.ylabel('Accuracy')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
fig_path = '/root/.gemini/antigravity-cli/brain/daefbb3c-184f-4666-85af-a288927fd2d7/figures/tree_depth_ablation.png'
plt.savefig(fig_path, dpi=300)
plt.close()

# Save results to JSON/CSV for artifact generation
df_clf.to_csv(f"{scratch_dir}/clf_benchmark.csv", index=False)
df_reg.to_csv(f"{scratch_dir}/reg_benchmark.csv", index=False)
print(f"\nExperiments completed successfully! Saved figure to {fig_path}")
