# Decision Trees & Ensembles

Model selection and performance evaluation on a tabular **binary classification** task using scikit-learn. The notebook compares multiple models with **GridSearchCV** and reports standard metrics; ROC curves are plotted for visual comparison.

---

## 🧪 Data
- Synthetic dataset via `make_classification`.
- Train/test split using `train_test_split` (70/30 as defined in the notebook).
- Features are used as-is (no scaler pipeline detected).

---

## 🤖 Models & Tuning
- **Decision Tree**
- **Random Forest**
- **Gradient Boosting**
- **AdaBoost**
- (XGBoost is imported in the environment but not detected in use in this notebook run.)

**Hyperparameter search:** `GridSearchCV` with parameter grids such as `param_grid`, `param_grid_ada`, and `param_grid_gb` defined in the notebook. Each model is fit with 5-fold CV across the specified candidate settings.

---

## 📈 Evaluation
- **Metrics implemented in code:** `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`
- **Visualization:** ROC curves (use these to compare ranking performance across models).

_Example console outputs from GridSearchCV (abbrev.):_
```
Fitting 5 folds for each of 616 candidates, totalling 3080 fits
{'criterion': 'entropy', 'max_depth': 7, 'min_samples_leaf': 6, 'min_samples_split': 6}
DecisionTreeClassifier(criterion='entropy', max_depth=7, min_samples_leaf=6,
                       min_samples_split=6)
...
```
> Re-run the notebook to generate your exact best-parameter dicts and test scores for each model.

---

## 📁 Repository Structure
```text
├── DecisionTreesAndEnsembles.ipynb   # Main notebook
└── README.md         # This file
```

---

## ⚙️ Setup & Run
```bash
pip install -U numpy pandas scikit-learn matplotlib
jupyter notebook "DecisionTreesAndEnsembles.ipynb"
```

---

## 💡 Tips & Next Steps
- Add a **confusion matrix** and **classification report** per best model for clearer error analysis.
- Wrap preprocessing + model into a **`Pipeline`** to avoid leakage and ease grid searches.
