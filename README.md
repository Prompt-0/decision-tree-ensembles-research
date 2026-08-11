# Decision Tree Ensembles Research: Bagging vs. Boosting

This repository contains a Master's-level research study evaluating Decision Tree Ensembles (**Random Forests** vs. **XGBoost / Gradient Boosted Trees**) with a focus on mathematical derivations, bias-variance trade-offs, and empirical benchmarks accessible to Computer Science undergraduates.

---

## 📂 Repository Structure

```
decision-tree-ensembles-research/
├── README.md                           # Main repository landing page & reproduction guide
├── paper.md                            # Full research paper in GitHub-Flavored Markdown
├── paper.tex                           # Full research paper in IEEE LaTeX format
├── src/
│   └── run_experiments.py              # Reproducible benchmark script (Python 3.14)
├── data/
│   ├── clf_benchmark.csv               # Empirical results for Breast Cancer classification
│   └── reg_benchmark.csv               # Empirical results for California Housing regression
└── figures/
    └── tree_depth_ablation.png         # Generated ablation study chart (Depth vs Accuracy)
```

---

## 📄 Key Research Findings

1. **Variance Floor Derivation**:
   - Bagging reduces single-tree variance down to an irreducible floor:
     $$\text{Var}(\bar{X}) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$$
   - Random Forests lower $\rho$ by evaluating a random subset $m = \sqrt{d}$ of features at each split.

2. **2nd-Order Taylor Optimization in XGBoost**:
   - XGBoost solves for optimal leaf weights $w_j^*$ using exact first-order gradients ($g_i$) and second-order Hessians ($h_i$):
     $$w_j^* = -\frac{\sum g_i}{\sum h_i + \lambda}$$

3. **Empirical Performance**:
   - **Classification**: XGBoost achieves $0.9934$ ROC-AUC while training ~30% faster than Random Forests.
   - **Regression ($N=20,640$)**: XGBoost achieves $R^2 = 0.8367$ (vs. $0.8046$ for RF) while training **>45× faster** (0.35s vs. 15.94s).

---

## 🚀 How to Reproduce Experiments

### Prerequisites
- Python 3.10+
- `scikit-learn`, `xgboost`, `pandas`, `matplotlib`, `numpy`

```bash
# Install dependencies
pip install scikit-learn xgboost pandas matplotlib numpy

# Run the benchmark experiment suite
python3 src/run_experiments.py
```

---

## 📖 Citation

```bibtex
@article{antigravity2026treeensembles,
  title={Demystifying Decision Tree Ensembles: Rigorous Bias-Variance Analysis, Mathematical Foundations, and Empirical Benchmarks of Bagging vs. Boosting},
  author={Antigravity AI Research \& Pedagogical Systems},
  year={2026},
  journal={Antigravity Open Research}
}
```
