import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.datasets import (
    load_breast_cancer, load_wine, load_digits, load_diabetes,
    fetch_california_housing, make_classification, make_regression
)
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings('ignore')

os.makedirs('/root/Projects/decision-tree-ensembles-research/data', exist_ok=True)
os.makedirs('/root/Projects/decision-tree-ensembles-research/figures', exist_ok=True)

print("==========================================================================")
print("  MULTI-DATASET EMPIRICAL BENCHMARK & STATISTICAL SIGNIFICANCE SUITE")
print("==========================================================================")
print(f"Loaded Libraries: XGBoost={xgb.__version__}, LightGBM={HAS_LGBM}, CatBoost={HAS_CATBOOST}")

# --------------------------------------------------------------------------
# 1. DATASET PREPARATION (10 Diverse Tabular Benchmarks)
# --------------------------------------------------------------------------

def get_classification_datasets():
    cancer = load_breast_cancer()
    wine = load_wine()
    digits = load_digits()
    
    # Synthetic High-Dim
    X_hd, y_hd = make_classification(n_samples=2000, n_features=50, n_informative=10, random_state=42)
    # Synthetic Imbalanced (90:10)
    X_imb, y_imb = make_classification(n_samples=3000, n_features=20, weights=[0.9, 0.1], random_state=42)
    # Synthetic Noisy Labels (15% noise)
    X_ns, y_ns = make_classification(n_samples=2000, n_features=20, flip_y=0.15, random_state=42)
    
    return {
        "Breast Cancer": (cancer.data, cancer.target, False),
        "Wine": (wine.data, wine.target, True),
        "Digits": (digits.data, digits.target, True),
        "High-Dim (50-feat)": (X_hd, y_hd, False),
        "Imbalanced (90:10)": (X_imb, y_imb, False),
        "Noisy Labels (15%)": (X_ns, y_ns, False)
    }

def get_regression_datasets():
    housing = fetch_california_housing()
    diabetes = load_diabetes()
    
    # Synthetic Non-linear
    X_nl, y_nl = make_regression(n_samples=3000, n_features=20, n_informative=10, noise=5.0, random_state=42)
    # Synthetic High-Variance
    X_hv, y_hv = make_regression(n_samples=2500, n_features=30, n_informative=15, noise=25.0, random_state=42)
    
    return {
        "California Housing": (housing.data, housing.target),
        "Diabetes": (diabetes.data, diabetes.target),
        "Nonlinear Reg (3k)": (X_nl, y_nl),
        "High-Variance Reg": (X_hv, y_hv)
    }

# --------------------------------------------------------------------------
# 2. MODEL DEFINITIONS
# --------------------------------------------------------------------------

def get_classification_models():
    models = {
        "Single CART": lambda: DecisionTreeClassifier(random_state=42),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Extra Trees": lambda: ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "AdaBoost": lambda: AdaBoostClassifier(n_estimators=100, random_state=42),
        "XGBoost": lambda: xgb.XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42, n_jobs=-1)
    }
    if HAS_LGBM:
        models["LightGBM"] = lambda: lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42, n_jobs=-1)
    if HAS_CATBOOST:
        models["CatBoost"] = lambda: cb.CatBoostClassifier(n_estimators=100, verbose=0, random_state=42)
    return models

def get_regression_models():
    models = {
        "Single CART": lambda: DecisionTreeRegressor(random_state=42),
        "Random Forest": lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Extra Trees": lambda: ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "AdaBoost": lambda: AdaBoostRegressor(n_estimators=100, random_state=42),
        "XGBoost": lambda: xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    if HAS_LGBM:
        models["LightGBM"] = lambda: lgb.LGBMRegressor(n_estimators=100, verbose=-1, random_state=42, n_jobs=-1)
    if HAS_CATBOOST:
        models["CatBoost"] = lambda: cb.CatBoostRegressor(n_estimators=100, verbose=0, random_state=42)
    return models

# --------------------------------------------------------------------------
# 3. 5-FOLD CROSS-VALIDATION EVALUATION LOOP
# --------------------------------------------------------------------------

def evaluate_classification_benchmarks():
    datasets = get_classification_datasets()
    model_factories = get_classification_models()
    
    cv_records = []
    
    for d_name, (X, y, is_multiclass) in datasets.items():
        print(f"\nEvaluating Classification Benchmark: {d_name} (N={len(X)}, D={X.shape[1]})...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for m_name, factory in model_factories.items():
            fold_accs, fold_f1s, fold_aucs = [], [], []
            fit_times, infer_times = [], []
            
            for train_idx, val_idx in skf.split(X, y):
                X_tr, X_va = X[train_idx], X[val_idx]
                y_tr, y_va = y[train_idx], y[val_idx]
                
                model = factory()
                
                t0 = time.time()
                model.fit(X_tr, y_tr)
                t_fit = time.time() - t0
                
                t0 = time.time()
                preds = model.predict(X_va)
                t_infer = time.time() - t0
                
                acc = accuracy_score(y_va, preds)
                f1 = f1_score(y_va, preds, average='macro' if is_multiclass else 'binary')
                
                # ROC-AUC computation
                try:
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X_va)
                        if is_multiclass:
                            auc = roc_auc_score(y_va, proba, multi_class='ovr')
                        else:
                            auc = roc_auc_score(y_va, proba[:, 1])
                    else:
                        auc = acc
                except Exception:
                    auc = acc
                
                fold_accs.append(acc)
                fold_f1s.append(f1)
                fold_aucs.append(auc)
                fit_times.append(t_fit)
                infer_times.append(t_infer)
            
            cv_records.append({
                "Dataset": d_name,
                "Model": m_name,
                "Acc_Mean": np.mean(fold_accs),
                "Acc_Std": np.std(fold_accs),
                "F1_Mean": np.mean(fold_f1s),
                "F1_Std": np.std(fold_f1s),
                "AUC_Mean": np.mean(fold_aucs),
                "AUC_Std": np.std(fold_aucs),
                "Train_Time_Mean": np.mean(fit_times),
                "Infer_Time_Mean": np.mean(infer_times),
                "Fold_Accs": fold_accs
            })
            
    return pd.DataFrame(cv_records)

def evaluate_regression_benchmarks():
    datasets = get_regression_datasets()
    model_factories = get_regression_models()
    
    cv_records = []
    
    for d_name, (X, y) in datasets.items():
        print(f"\nEvaluating Regression Benchmark: {d_name} (N={len(X)}, D={X.shape[1]})...")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for m_name, factory in model_factories.items():
            fold_rmses, fold_r2s = [], []
            fit_times, infer_times = [], []
            
            for train_idx, val_idx in kf.split(X, y):
                X_tr, X_va = X[train_idx], X[val_idx]
                y_tr, y_va = y[train_idx], y[val_idx]
                
                model = factory()
                
                t0 = time.time()
                model.fit(X_tr, y_tr)
                t_fit = time.time() - t0
                
                t0 = time.time()
                preds = model.predict(X_va)
                t_infer = time.time() - t0
                
                rmse = np.sqrt(mean_squared_error(y_va, preds))
                r2 = r2_score(y_va, preds)
                
                fold_rmses.append(rmse)
                fold_r2s.append(r2)
                fit_times.append(t_fit)
                infer_times.append(t_infer)
            
            cv_records.append({
                "Dataset": d_name,
                "Model": m_name,
                "RMSE_Mean": np.mean(fold_rmses),
                "RMSE_Std": np.std(fold_rmses),
                "R2_Mean": np.mean(fold_r2s),
                "R2_Std": np.std(fold_r2s),
                "Train_Time_Mean": np.mean(fit_times),
                "Infer_Time_Mean": np.mean(infer_times),
                "Fold_RMSEs": fold_rmses
            })
            
    return pd.DataFrame(cv_records)

# --------------------------------------------------------------------------
# 4. STATISTICAL SIGNIFICANCE TESTS (Paired t-test / Wilcoxon)
# --------------------------------------------------------------------------

def compute_statistical_pvalues(df_clf):
    # Paired comparison across all classification dataset folds between XGBoost and Random Forest
    rf_folds = []
    xgb_folds = []
    
    for dataset in df_clf["Dataset"].unique():
        sub = df_clf[df_clf["Dataset"] == dataset]
        rf_row = sub[sub["Model"] == "Random Forest"]
        xgb_row = sub[sub["Model"] == "XGBoost"]
        
        if not rf_row.empty and not xgb_row.empty:
            rf_folds.extend(rf_row.iloc[0]["Fold_Accs"])
            xgb_folds.extend(xgb_row.iloc[0]["Fold_Accs"])
            
    t_stat, p_val = stats.ttest_rel(xgb_folds, rf_folds)
    w_stat, w_pval = stats.wilcoxon(xgb_folds, rf_folds)
    
    print("\n--------------------------------------------------------------------------")
    print("  STATISTICAL SIGNIFICANCE TESTS (XGBoost vs. Random Forest across 30 CV Folds)")
    print("--------------------------------------------------------------------------")
    print(f"Paired t-test t-statistic = {t_stat:.4f}, p-value = {p_val:.6f}")
    print(f"Wilcoxon signed-rank test statistic = {w_stat:.4f}, p-value = {w_pval:.6f}")
    
    return {
        "t_statistic": float(t_stat),
        "t_pvalue": float(p_val),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_pvalue": float(w_pval)
    }

# --------------------------------------------------------------------------
# 5. VISUALIZATION & FIGURE GENERATION
# --------------------------------------------------------------------------

def generate_publication_plots(df_clf, df_reg):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Figure 1: Classification Accuracy Comparison across Datasets
    plt.figure(figsize=(12, 6))
    chart = sns.barplot(
        data=df_clf, x="Dataset", y="Acc_Mean", hue="Model", palette="viridis"
    )
    plt.title("Multi-Dataset Classification Accuracy Benchmark (5-Fold Stratified CV)", fontsize=14, fontweight='bold')
    plt.ylabel("Mean Accuracy", fontsize=12)
    plt.xlabel("Benchmark Dataset", fontsize=12)
    plt.ylim(0.70, 1.00)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("/root/Projects/decision-tree-ensembles-research/figures/multi_dataset_classification.png", dpi=300)
    plt.close()
    
    # Figure 2: Regression R2 Score Comparison
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=df_reg, x="Dataset", y="R2_Mean", hue="Model", palette="magma"
    )
    plt.title("Multi-Dataset Regression $R^2$ Score Benchmark (5-Fold CV)", fontsize=14, fontweight='bold')
    plt.ylabel("Mean $R^2$ Score", fontsize=12)
    plt.xlabel("Benchmark Dataset", fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("/root/Projects/decision-tree-ensembles-research/figures/multi_dataset_regression.png", dpi=300)
    plt.close()
    
    # Figure 3: Training & Inference Execution Time Trade-offs (Log Scale)
    df_clf_avg_time = df_clf.groupby("Model")[["Train_Time_Mean", "Infer_Time_Mean"]].mean().reset_index()
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = 'tab:blue'
    ax1.set_xlabel('Model Architecture', fontsize=12)
    ax1.set_ylabel('Mean Fit Time (seconds, log scale)', color=color, fontsize=12)
    ax1.bar(df_clf_avg_time['Model'], df_clf_avg_time['Train_Time_Mean'], color=color, alpha=0.6, width=0.4)
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Mean Inference Time (seconds, log scale)', color=color, fontsize=12)
    ax2.plot(df_clf_avg_time['Model'], df_clf_avg_time['Infer_Time_Mean'], color=color, marker='o', linewidth=2)
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Computational Complexity: Training vs Inference Latency Trade-off', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig("/root/Projects/decision-tree-ensembles-research/figures/latency_tradeoff.png", dpi=300)
    plt.close()
    
    print("\nPublication plots successfully generated in /root/Projects/decision-tree-ensembles-research/figures/")

# --------------------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------------------

if __name__ == "__main__":
    df_clf = evaluate_classification_benchmarks()
    df_reg = evaluate_regression_benchmarks()
    
    pvalues = compute_statistical_pvalues(df_clf)
    
    # Save CSVs
    df_clf_clean = df_clf.drop(columns=["Fold_Accs"])
    df_reg_clean = df_reg.drop(columns=["Fold_RMSEs"])
    
    df_clf_clean.to_csv("/root/Projects/decision-tree-ensembles-research/data/clf_multi_dataset_benchmark.csv", index=False)
    df_reg_clean.to_csv("/root/Projects/decision-tree-ensembles-research/data/reg_multi_dataset_benchmark.csv", index=False)
    
    with open("/root/Projects/decision-tree-ensembles-research/data/statistical_tests.json", "w") as f:
        json.dump(pvalues, f, indent=2)
        
    generate_publication_plots(df_clf, df_reg)
    print("\nMulti-dataset empirical evaluation complete!")
