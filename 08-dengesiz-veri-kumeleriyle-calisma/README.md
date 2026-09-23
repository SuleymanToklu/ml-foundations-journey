# Dengesiz Veri Kümeleri: SMOTE ve Class Weights

Doğruluk paradoksu (Accuracy Paradox), Chawla SMOTE sentetik azınlık aşırı örnekleme matematiği, `class_weight='balanced'` cezalandırma mimarisi ve Precision/Recall/F1-Score analizi.

## İlgili Notebook
- [imbalanced-data-smote-and-class-weights.ipynb](imbalanced-data-smote-and-class-weights.ipynb)

## Temel Kütüphaneler
- `imblearn.over_sampling.SMOTE`
- `sklearn.ensemble.RandomForestClassifier(class_weight='balanced')`
- `classification_report`, `ConfusionMatrixDisplay`
